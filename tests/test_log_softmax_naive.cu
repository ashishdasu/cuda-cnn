// Correctness test for the naive CUDA log-softmax kernel. Beyond
// normal random input, this explicitly exercises the numerical-
// stability path by injecting large positive logits (~80) into one
// of the rows: a naive implementation without the subtract-max
// trick would overflow exp() and produce NaN/Inf, so this test
// catches that regression directly.

#include "host/reference.h"
#include "kernels/log_softmax_naive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N, C;
    bool stability_stress;   // if true, inject large logits in one row
};

float run_case(const Shape& s, std::uint32_t seed) {
    const std::size_t n_elems = static_cast<std::size_t>(s.N) * s.C;

    std::vector<float> input(n_elems);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input) x = dist(rng);

    if (s.stability_stress && s.N >= 1) {
        // Row 0: huge logits to force the max-subtraction code path.
        // exp(80) is already ~5e34; without subtraction the sum and
        // the subsequent log both overflow.
        for (int c = 0; c < s.C; ++c) {
            input[c] = 80.0f + static_cast<float>(c);
        }
    }

    std::vector<float> out_ref(n_elems);
    std::vector<float> out_gpu(n_elems);
    cnn::log_softmax_ref(input.data(), out_ref.data(), s.N, s.C);
    cnn::cuda::log_softmax_naive(input.data(), out_gpu.data(), s.N, s.C);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < n_elems; ++i) {
        // Also flag NaN/Inf explicitly so a broken kernel gets called out
        // rather than silently passing on comparison.
        if (!std::isfinite(out_gpu[i])) return std::numeric_limits<float>::infinity();
        max_err = std::max(max_err, std::fabs(out_ref[i] - out_gpu[i]));
    }
    return max_err;
}

}  // namespace

int main() {
    constexpr float TOLERANCE = 1e-4f;

    const Shape cases[] = {
        {"LeNet final (N=1, C=10)",   1,  10, false},
        {"batched (N=8, C=10)",       8,  10, false},
        {"wider-C stress (N=2, C=128)", 2, 128, false},
        {"numerical-stability (N=2, C=10, row0 logits ~80)", 2, 10, true},
    };

    int failures = 0;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float err = run_case(cases[i], /*seed=*/0xF00D + static_cast<std::uint32_t>(i));
        const bool ok = std::isfinite(err) && err < TOLERANCE;
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
