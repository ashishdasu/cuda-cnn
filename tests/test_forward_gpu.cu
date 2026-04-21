// End-to-end correctness test for the CUDA forward pass. Mirrors
// test_reference.cpp, but swaps cnn::forward_ref for
// cnn::cuda::forward_naive_gpu. This is the minimum-tier milestone:
// passing here means the composed naive-kernel pipeline reproduces
// PyTorch's log-softmax outputs on every MNIST fixture within the
// project's 1e-4 tolerance.

#include "host/weights.h"
#include "kernels/forward.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load_bin(const std::string& path, std::size_t expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open " + path);
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    if (bytes != expected_count * sizeof(float)) {
        throw std::runtime_error(path + ": size mismatch");
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(bytes));
    return data;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;
    constexpr int N_FIXTURES = 5;

    try {
        const cnn::Weights weights = cnn::load_weights("weights/");

        int failures = 0;
        float overall_max = 0.0f;
        for (int i = 0; i < N_FIXTURES; ++i) {
            const std::string idx = std::to_string(i);
            const auto input    = load_bin("fixtures/input_"  + idx + ".bin", 1 * 28 * 28);
            const auto expected = load_bin("fixtures/output_" + idx + ".bin", 10);

            std::vector<float> actual(10);
            cnn::cuda::forward_naive_gpu(weights, input.data(), actual.data(), /*N=*/1);

            float max_err = 0.0f;
            int actual_argmax = 0;
            int expected_argmax = 0;
            for (int c = 0; c < 10; ++c) {
                max_err = std::max(max_err, std::fabs(actual[c] - expected[c]));
                if (actual[c]   > actual[actual_argmax])     actual_argmax = c;
                if (expected[c] > expected[expected_argmax]) expected_argmax = c;
            }

            const bool ok = (max_err < TOLERANCE) && (actual_argmax == expected_argmax);
            std::printf(
                "fixture %d: %s  max_abs_err=%.3e  predicted=%d  expected=%d\n",
                i, ok ? "PASS" : "FAIL", max_err, actual_argmax, expected_argmax);
            if (!ok) ++failures;
            overall_max = std::max(overall_max, max_err);
        }

        std::printf("----\n%d/%d fixtures passed (tolerance %.1e, overall max err %.3e)\n",
                    N_FIXTURES - failures, N_FIXTURES, TOLERANCE, overall_max);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
