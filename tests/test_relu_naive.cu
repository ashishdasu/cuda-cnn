// Correctness test for the naive CUDA ReLU kernel. Generates a mixed
// sign random buffer, runs the CPU reference and the GPU kernel on
// independent copies, diffs element-wise. Includes a pathological
// zero-and-negative-only buffer to make sure nothing weird happens
// to exact zeros on the boundary.

#include "host/reference.h"
#include "kernels/relu_naive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Case {
    const char* name;
    std::size_t count;
    std::uint32_t seed;
};

float run_case(const Case& cs) {
    std::vector<float> data(cs.count);
    std::mt19937 rng(cs.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : data) x = dist(rng);

    // Salt a few exact zeros to exercise the > 0 boundary.
    if (cs.count >= 4) {
        data[0] = 0.0f;
        data[cs.count / 2] = -0.0f;
        data[cs.count - 1] = 0.0f;
    }

    std::vector<float> ref(data);
    std::vector<float> gpu(data);
    cnn::relu_ref(ref.data(), ref.size());
    cnn::cuda::relu_naive(gpu.data(), gpu.size());

    float max_err = 0.0f;
    for (std::size_t i = 0; i < cs.count; ++i) {
        max_err = std::max(max_err, std::fabs(ref[i] - gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    const Case cases[] = {
        // LeNet post-conv1 pool activation volume: 10*12*12 = 1440
        {"post-conv1 pool (10*12*12)", 10 * 12 * 12, 0xA1},
        // LeNet post-conv2 pool activation volume: 20*4*4 = 320
        {"post-conv2 pool (20*4*4)", 20 * 4 * 4, 0xA2},
        // fc1 output: 50
        {"fc1 output (50)", 50, 0xA3},
        // non-multiple-of-block-size to exercise bounds check
        {"odd size (1023)", 1023, 0xA4},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i]);
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
