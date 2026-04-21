// Correctness test for the naive CUDA linear / fully-connected kernel.
// For each of a handful of shape configurations (the two LeNet FC
// layers plus an off-model larger in_features case), generates
// reproducible pseudo-random tensors, runs the CPU reference and the
// GPU kernel, and asserts max abs error < 1e-4.

#include "host/reference.h"
#include "kernels/linear_naive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N, in_features, out_features;
};

float run_case(const Shape& s, std::uint32_t seed) {
    const std::size_t in_n  = static_cast<std::size_t>(s.N) * s.in_features;
    const std::size_t w_n   = static_cast<std::size_t>(s.out_features) * s.in_features;
    const std::size_t b_n   = static_cast<std::size_t>(s.out_features);
    const std::size_t out_n = static_cast<std::size_t>(s.N) * s.out_features;

    std::vector<float> input(in_n);
    std::vector<float> weight(w_n);
    std::vector<float> bias(b_n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input)  x = dist(rng);
    for (auto& x : weight) x = dist(rng);
    for (auto& x : bias)   x = dist(rng);

    std::vector<float> out_ref(out_n);
    std::vector<float> out_gpu(out_n);
    cnn::linear_ref(input.data(), weight.data(), bias.data(), out_ref.data(),
                    s.N, s.in_features, s.out_features);
    cnn::cuda::linear_naive(input.data(), weight.data(), bias.data(), out_gpu.data(),
                            s.N, s.in_features, s.out_features);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < out_n; ++i) {
        max_err = std::max(max_err, std::fabs(out_ref[i] - out_gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    const Shape cases[] = {
        {"LeNet fc1 (320 -> 50)", 2, 320, 50},
        {"LeNet fc2 ( 50 -> 10)", 8,  50, 10},
        {"off-model (1024 -> 17)", 3, 1024, 17},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i], /*seed=*/0xBEEF + static_cast<std::uint32_t>(i));
        const bool ok = err < TOLERANCE;
        std::printf("%s  max_abs_err=%.3e  %s\n",
                    ok ? "PASS" : "FAIL", err, cases[i].name);
        if (!ok) ++failures;
    }

    std::printf("----\n%d/%zu shape(s) passed (tolerance %.1e)\n",
                static_cast<int>(sizeof(cases) / sizeof(cases[0])) - failures,
                sizeof(cases) / sizeof(cases[0]),
                TOLERANCE);
    return failures == 0 ? 0 : 1;
}
