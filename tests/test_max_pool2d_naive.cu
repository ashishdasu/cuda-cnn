// Correctness test for the naive CUDA 2x2 max-pool kernel. Diffs
// GPU output against cnn::max_pool2d_ref on a handful of shape
// configurations under the project's 1e-4 tolerance.

#include "host/reference.h"
#include "kernels/max_pool2d_naive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N, C, H_in, W_in;
};

float run_case(const Shape& s, std::uint32_t seed) {
    const int H_out = s.H_in / 2;
    const int W_out = s.W_in / 2;
    const std::size_t in_n  = static_cast<std::size_t>(s.N) * s.C * s.H_in * s.W_in;
    const std::size_t out_n = static_cast<std::size_t>(s.N) * s.C * H_out * W_out;

    std::vector<float> input(in_n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input) x = dist(rng);

    std::vector<float> out_ref(out_n);
    std::vector<float> out_gpu(out_n);
    cnn::max_pool2d_ref(input.data(), out_ref.data(), s.N, s.C, s.H_in, s.W_in);
    cnn::cuda::max_pool2d_naive(input.data(), out_gpu.data(), s.N, s.C, s.H_in, s.W_in);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < out_n; ++i) {
        max_err = std::max(max_err, std::fabs(out_ref[i] - out_gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    // LeNet uses pool after conv1 (24x24 -> 12x12) and after conv2
    // (8x8 -> 4x4). The 5x7 case exercises odd dims (drops last
    // row/col, matching PyTorch integer-div semantics).
    const Shape cases[] = {
        {"after-conv1 (N=2, C=10, 24x24)", 2, 10, 24, 24},
        {"after-conv2 (N=2, C=20,  8x8)",  2, 20,  8,  8},
        {"odd-dim (N=1, C=3, 5x7 -> 2x3)", 1,  3,  5,  7},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i], /*seed=*/0xC0DE + static_cast<std::uint32_t>(i));
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
