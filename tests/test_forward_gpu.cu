// End-to-end correctness test for the CUDA forward pass. Mirrors
// test_reference.cpp but swaps cnn::forward_ref for
// cnn::cuda::forward_naive_gpu. Passing here means the composed naive-kernel
// pipeline reproduces PyTorch's log-softmax outputs on every MNIST fixture
// within the 1e-4 tolerance that defines numerical parity for this project.

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

using ForwardFn = void(*)(const cnn::Weights&, const float*, float*, int);

int run_variant(const char* label,
                ForwardFn forward,
                const cnn::Weights& weights,
                int n_fixtures,
                float tolerance) {
    int failures = 0;
    float overall_max = 0.0f;
    for (int i = 0; i < n_fixtures; ++i) {
        const std::string idx = std::to_string(i);
        const auto input    = load_bin("fixtures/input_"  + idx + ".bin", 1 * 28 * 28);
        const auto expected = load_bin("fixtures/output_" + idx + ".bin", 10);

        std::vector<float> actual(10);
        forward(weights, input.data(), actual.data(), /*N=*/1);

        float max_err = 0.0f;
        int actual_argmax = 0;
        int expected_argmax = 0;
        for (int c = 0; c < 10; ++c) {
            max_err = std::max(max_err, std::fabs(actual[c] - expected[c]));
            if (actual[c]   > actual[actual_argmax])     actual_argmax = c;
            if (expected[c] > expected[expected_argmax]) expected_argmax = c;
        }
        const bool ok = (max_err < tolerance) && (actual_argmax == expected_argmax);
        std::printf("[%s] fixture %d: %s  max_abs_err=%.3e  predicted=%d  expected=%d\n",
                    label, i, ok ? "PASS" : "FAIL", max_err, actual_argmax, expected_argmax);
        if (!ok) ++failures;
        overall_max = std::max(overall_max, max_err);
    }
    std::printf("[%s] ---- %d/%d passed (overall max err %.3e)\n",
                label, n_fixtures - failures, n_fixtures, overall_max);
    return failures;
}

int main() {
    constexpr float TOLERANCE = 1e-4f;
    constexpr int N_FIXTURES = 5;

    try {
        const cnn::Weights weights = cnn::load_weights("weights/");
        int failures = 0;
        failures += run_variant("naive", cnn::cuda::forward_naive_gpu,
                                weights, N_FIXTURES, TOLERANCE);
        failures += run_variant("tiled", cnn::cuda::forward_tiled_gpu,
                                weights, N_FIXTURES, TOLERANCE);
        std::printf("----\n%s (tolerance %.1e)\n",
                    failures == 0 ? "ALL VARIANTS PASS" : "FAILURES", TOLERANCE);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
