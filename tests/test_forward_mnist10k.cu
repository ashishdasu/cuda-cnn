// Full-MNIST-test-set parity harness. Reads the three bulk fixture
// files produced by `tools/dump_fixtures.py --bulk --count 10000`
// (inputs, PyTorch log-softmax outputs, integer labels), runs the
// naive and tiled CUDA forward pipelines on every image, and reports:
//
//   * max absolute error vs. PyTorch log-softmax over all 100k logits
//   * argmax agreement with PyTorch (should be 100% — same math,
//     float32 summation-order noise only)
//   * engine accuracy on the MNIST test labels (should match the
//     PyTorch eval-mode accuracy to the last digit for the same reason)
//
// This is not an add_test target. Running it requires the ~32 MB bulk
// fixture file, which is gitignored. Invoke manually after a build to
// produce the row that shows up in the report's parity table.

#include "host/weights.h"
#include "kernels/forward.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load_floats(const std::string& path, std::size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open " + path);
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    if (bytes != expected * sizeof(float)) {
        throw std::runtime_error(path + ": size mismatch (got "
                                 + std::to_string(bytes) + " bytes, expected "
                                 + std::to_string(expected * sizeof(float)) + ")");
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> data(expected);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(bytes));
    return data;
}

std::vector<std::int32_t> load_i32(const std::string& path, std::size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open " + path);
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    if (bytes != expected * sizeof(std::int32_t)) {
        throw std::runtime_error(path + ": int32 size mismatch");
    }
    f.seekg(0, std::ios::beg);
    std::vector<std::int32_t> data(expected);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(bytes));
    return data;
}

using ForwardFn = void(*)(const cnn::Weights&, const float*, float*, int);

struct Summary {
    float overall_max_err;
    int argmax_matches;
    int correct_vs_label;
    int n;
};

Summary run(const char* label, ForwardFn forward, const cnn::Weights& w,
            const std::vector<float>& inputs,
            const std::vector<float>& py_outputs,
            const std::vector<std::int32_t>& labels) {
    const int n = static_cast<int>(labels.size());
    Summary s{0.0f, 0, 0, n};
    std::vector<float> actual(10);
    for (int i = 0; i < n; ++i) {
        const float* xi = inputs.data() + static_cast<std::size_t>(i) * 784;
        const float* yi = py_outputs.data() + static_cast<std::size_t>(i) * 10;
        forward(w, xi, actual.data(), /*N=*/1);

        int actual_argmax = 0, py_argmax = 0;
        float local_max = 0.0f;
        for (int c = 0; c < 10; ++c) {
            local_max = std::max(local_max, std::fabs(actual[c] - yi[c]));
            if (actual[c] > actual[actual_argmax]) actual_argmax = c;
            if (yi[c]     > yi[py_argmax])         py_argmax     = c;
        }
        s.overall_max_err = std::max(s.overall_max_err, local_max);
        if (actual_argmax == py_argmax)         ++s.argmax_matches;
        if (actual_argmax == labels[i])         ++s.correct_vs_label;
    }
    std::printf("[%s] n=%d  max_abs_err=%.3e  argmax_match=%d/%d (%.2f%%)  "
                "acc_vs_label=%d/%d (%.2f%%)\n",
                label, s.n, s.overall_max_err,
                s.argmax_matches, s.n, 100.0 * s.argmax_matches / s.n,
                s.correct_vs_label, s.n, 100.0 * s.correct_vs_label / s.n);
    return s;
}

}  // namespace

int main() {
    try {
        const cnn::Weights w = cnn::load_weights("weights/");
        const auto inputs  = load_floats("fixtures/mnist10k_inputs.bin",  10000UL * 784);
        const auto outputs = load_floats("fixtures/mnist10k_outputs.bin", 10000UL * 10);
        const auto labels  = load_i32   ("fixtures/mnist10k_labels.bin",  10000);

        const Summary naive = run("naive", cnn::cuda::forward_naive_gpu, w,
                                  inputs, outputs, labels);
        const Summary tiled = run("tiled", cnn::cuda::forward_tiled_gpu, w,
                                  inputs, outputs, labels);

        constexpr float TOLERANCE = 1e-4f;
        const bool ok =
            naive.overall_max_err < TOLERANCE && naive.argmax_matches == naive.n &&
            tiled.overall_max_err < TOLERANCE && tiled.argmax_matches == tiled.n;
        std::printf("----\n%s (tolerance %.1e)\n",
                    ok ? "ALL 10000 IMAGES PASS" : "FAILURES", TOLERANCE);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "hint: run `python3 tools/dump_fixtures.py --bulk "
                             "--count 10000` first to generate fixtures/mnist10k_*.bin\n");
        return 2;
    }
}
