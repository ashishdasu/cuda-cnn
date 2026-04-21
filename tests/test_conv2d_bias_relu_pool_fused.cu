// Correctness test for the fused conv+bias+ReLU+pool kernel. For each
// shape, runs the CPU reference three-op sequence (conv -> relu ->
// max_pool2d) and the single GPU fused kernel, then diffs at the
// project-wide 1e-4 tolerance.
//
// Exit 0 on success, 1 on any config failing.

#include "host/reference.h"
#include "kernels/conv2d_bias_relu_pool_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N, C_in, H_in, W_in, C_out, Kh, Kw;
};

float run_case(const Shape& s, std::uint32_t seed) {
    const int H_conv = s.H_in - s.Kh + 1;
    const int W_conv = s.W_in - s.Kw + 1;
    const int H_pool = H_conv / 2;
    const int W_pool = W_conv / 2;

    const std::size_t in_n   = static_cast<std::size_t>(s.N) * s.C_in * s.H_in * s.W_in;
    const std::size_t w_n    = static_cast<std::size_t>(s.C_out) * s.C_in * s.Kh * s.Kw;
    const std::size_t b_n    = static_cast<std::size_t>(s.C_out);
    const std::size_t conv_n = static_cast<std::size_t>(s.N) * s.C_out * H_conv * W_conv;
    const std::size_t pool_n = static_cast<std::size_t>(s.N) * s.C_out * H_pool * W_pool;

    std::vector<float> input(in_n);
    std::vector<float> weight(w_n);
    std::vector<float> bias(b_n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input)  x = dist(rng);
    for (auto& x : weight) x = dist(rng);
    for (auto& x : bias)   x = dist(rng);

    // Reference: conv -> relu -> max_pool2d, using a scratch buffer
    // for the conv output. This is the exact three-call sequence the
    // fused kernel collapses.
    std::vector<float> conv_ref(conv_n);
    std::vector<float> pool_ref(pool_n);
    cnn::conv2d_ref(input.data(), weight.data(), bias.data(), conv_ref.data(),
                    s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);
    cnn::relu_ref(conv_ref.data(), conv_n);
    cnn::max_pool2d_ref(conv_ref.data(), pool_ref.data(),
                        s.N, s.C_out, H_conv, W_conv);

    std::vector<float> pool_gpu(pool_n);
    cnn::cuda::conv2d_bias_relu_pool_fused(
        input.data(), weight.data(), bias.data(), pool_gpu.data(),
        s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < pool_n; ++i) {
        max_err = std::max(max_err, std::fabs(pool_ref[i] - pool_gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    const Shape cases[] = {
        {"LeNet conv1 (N=2, 1->10, 5x5 @ 28x28)",   2,  1, 28, 28, 10, 5, 5},
        {"LeNet conv2 (N=2, 10->20, 5x5 @ 12x12)",  2, 10, 12, 12, 20, 5, 5},
        {"batch-64 conv1 (N=64, 1->10, 5x5 @ 28x28)",64, 1, 28, 28, 10, 5, 5},
        {"non-aligned pool (N=1, 3->5, 3x3 @ 15x15)", 1, 3, 15, 15,  5, 3, 3},
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
