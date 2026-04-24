// Correctness test for the cuDNN-backed conv baseline. Diffs vs the
// CPU reference on LeNet's two conv shapes. cuDNN uses different
// summation orders and potentially different algorithms per shape,
// but the result must still agree with the CPU reference at the
// 1e-4 project tolerance — that's the contract the bench numbers
// lean on later (any speedup headline is worthless if the reference
// output is different).

#include "host/reference.h"
#include "kernels/conv2d_cudnn.h"

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
    const int H_out = s.H_in - s.Kh + 1;
    const int W_out = s.W_in - s.Kw + 1;

    const std::size_t in_elems  = static_cast<std::size_t>(s.N) * s.C_in * s.H_in * s.W_in;
    const std::size_t w_elems   = static_cast<std::size_t>(s.C_out) * s.C_in * s.Kh * s.Kw;
    const std::size_t b_elems   = static_cast<std::size_t>(s.C_out);
    const std::size_t out_elems = static_cast<std::size_t>(s.N) * s.C_out * H_out * W_out;

    std::vector<float> input(in_elems), weight(w_elems), bias(b_elems);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input)  x = dist(rng);
    for (auto& x : weight) x = dist(rng);
    for (auto& x : bias)   x = dist(rng);

    std::vector<float> out_ref(out_elems);
    std::vector<float> out_gpu(out_elems);
    cnn::conv2d_ref(input.data(), weight.data(), bias.data(), out_ref.data(),
                    s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);
    cnn::cuda::conv2d_cudnn(input.data(), weight.data(), bias.data(), out_gpu.data(),
                            s.N, s.C_in, s.H_in, s.W_in, s.C_out, s.Kh, s.Kw);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < out_elems; ++i) {
        max_err = std::max(max_err, std::fabs(out_ref[i] - out_gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    const Shape cases[] = {
        {"LeNet conv1 (N=2, 1->10, 5x5 @ 28x28)",   2,  1, 28, 28, 10, 5, 5},
        {"LeNet conv2 (N=2, 10->20, 5x5 @ 12x12)",  2, 10, 12, 12, 20, 5, 5},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i], /*seed=*/0xDEAD + static_cast<std::uint32_t>(i));
        const bool ok = err < TOLERANCE;
        std::printf("%s  max_abs_err=%.3e  %s\n",
                    ok ? "PASS" : "FAIL", err, cases[i].name);
        if (!ok) ++failures;
    }

    std::printf("----\n%d/%zu case(s) passed (tolerance %.1e)\n",
                static_cast<int>(sizeof(cases) / sizeof(cases[0])) - failures,
                sizeof(cases) / sizeof(cases[0]),
                TOLERANCE);
    return failures == 0 ? 0 : 1;
}
