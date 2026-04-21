// Correctness test for the naive CUDA conv2d kernel. For each of a
// handful of shape configurations (covering the two LeNet conv layers
// plus an off-LeNet odd-dim case to stress boundary handling in the
// grid cover), this generates reproducible pseudo-random input,
// weight, and bias tensors, runs the CPU reference and the GPU
// kernel, and asserts the maximum per-element absolute error is
// below the project-wide 1e-4 parity tolerance.
//
// Exit 0 on success, 1 on any config failing.

#include "host/reference.h"
#include "kernels/conv2d_naive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N, C_in, H_in, W_in, C_out, Kh, Kw;
};

float run_case(const Shape& s, std::uint32_t seed) {
    const int H_out = s.H_in - s.Kh + 1;
    const int W_out = s.W_in - s.Kw + 1;

    const std::size_t in_n  = static_cast<std::size_t>(s.N) * s.C_in * s.H_in * s.W_in;
    const std::size_t w_n   = static_cast<std::size_t>(s.C_out) * s.C_in * s.Kh * s.Kw;
    const std::size_t b_n   = static_cast<std::size_t>(s.C_out);
    const std::size_t out_n = static_cast<std::size_t>(s.N) * s.C_out * H_out * W_out;

    std::vector<float> input(in_n);
    std::vector<float> weight(w_n);
    std::vector<float> bias(b_n);

    // Uniform [-1, 1]. Matches the rough scale of post-normalization
    // MNIST inputs and small initialized weights, keeping accumulated
    // sums in a regime where 1e-4 absolute tolerance is meaningful.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input)  x = dist(rng);
    for (auto& x : weight) x = dist(rng);
    for (auto& x : bias)   x = dist(rng);

    std::vector<float> out_ref(out_n);
    std::vector<float> out_gpu(out_n);

    cnn::conv2d_ref(input.data(), weight.data(), bias.data(), out_ref.data(),
                    s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);
    cnn::cuda::conv2d_naive(input.data(), weight.data(), bias.data(), out_gpu.data(),
                            s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);

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
        {"LeNet conv1 (1->10, 5x5 @ 28x28)",  2,  1, 28, 28, 10, 5, 5},
        {"LeNet conv2 (10->20, 5x5 @ 12x12)", 2, 10, 12, 12, 20, 5, 5},
        {"odd-dim boundary (3->5, 3x3 @ 13x17)", 3,  3, 13, 17,  5, 3, 3},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i], /*seed=*/0x5A5A + static_cast<std::uint32_t>(i));
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
