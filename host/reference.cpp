// CPU reference implementations. Correctness-first, speed-later.
// See reference.h for the contract.

#include "host/reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cnn {

void conv2d_ref(const float* input,
                const float* weight,
                const float* bias,
                float* output,
                int N,
                int C_in, int H_in, int W_in,
                int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < Kh; ++kh) {
                            for (int kw = 0; kw < Kw; ++kw) {
                                const int ih = oh + kh;
                                const int iw = ow + kw;
                                const float x = input[((n * C_in + ic) * H_in + ih) * W_in + iw];
                                const float w = weight[((oc * C_in + ic) * Kh + kh) * Kw + kw];
                                sum += x * w;
                            }
                        }
                    }
                    output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }
}

void max_pool2d_ref(const float* input,
                    float* output,
                    int N, int C, int H_in, int W_in) {
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    const int ih = oh * 2;
                    const int iw = ow * 2;
                    const float a = input[((n * C + c) * H_in + ih    ) * W_in + iw];
                    const float b = input[((n * C + c) * H_in + ih    ) * W_in + iw + 1];
                    const float cc = input[((n * C + c) * H_in + ih + 1) * W_in + iw];
                    const float d = input[((n * C + c) * H_in + ih + 1) * W_in + iw + 1];
                    output[((n * C + c) * H_out + oh) * W_out + ow] =
                        std::max(std::max(a, b), std::max(cc, d));
                }
            }
        }
    }
}

void relu_ref(float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

void linear_ref(const float* input,
                const float* weight,
                const float* bias,
                float* output,
                int N, int in_features, int out_features) {
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_features; ++o) {
            float sum = bias[o];
            for (int i = 0; i < in_features; ++i) {
                sum += input[n * in_features + i] * weight[o * in_features + i];
            }
            output[n * out_features + o] = sum;
        }
    }
}

void log_softmax_ref(const float* input,
                     float* output,
                     int N, int C) {
    for (int n = 0; n < N; ++n) {
        const float* row_in = input + n * C;
        float* row_out = output + n * C;

        // Subtract row max for numerical stability (the trap called
        // out in the proposal — naive exp overflows on large logits).
        float m = row_in[0];
        for (int c = 1; c < C; ++c) {
            if (row_in[c] > m) m = row_in[c];
        }

        float s = 0.0f;
        for (int c = 0; c < C; ++c) {
            s += std::exp(row_in[c] - m);
        }
        const float logs = std::log(s);

        for (int c = 0; c < C; ++c) {
            row_out[c] = row_in[c] - m - logs;
        }
    }
}

void forward_ref(const Weights& w,
                 const float* input,
                 float* output,
                 int N) {
    // Layer 1: conv1 (1 -> 10, 5x5, no pad, stride 1): 28x28 -> 24x24
    std::vector<float> c1(static_cast<std::size_t>(N) * 10 * 24 * 24);
    conv2d_ref(input, w.conv1_weight.data(), w.conv1_bias.data(), c1.data(),
               N, 1, 28, 28, 10, 5, 5);

    // max_pool(2) -> 12x12, then relu in place
    std::vector<float> p1(static_cast<std::size_t>(N) * 10 * 12 * 12);
    max_pool2d_ref(c1.data(), p1.data(), N, 10, 24, 24);
    relu_ref(p1.data(), p1.size());

    // Layer 2: conv2 (10 -> 20, 5x5) [Dropout2d skipped at inference]
    // 12x12 -> 8x8
    std::vector<float> c2(static_cast<std::size_t>(N) * 20 * 8 * 8);
    conv2d_ref(p1.data(), w.conv2_weight.data(), w.conv2_bias.data(), c2.data(),
               N, 10, 12, 12, 20, 5, 5);

    // max_pool(2) -> 4x4, then relu in place
    std::vector<float> p2(static_cast<std::size_t>(N) * 20 * 4 * 4);
    max_pool2d_ref(c2.data(), p2.data(), N, 20, 8, 8);
    relu_ref(p2.data(), p2.size());

    // Flatten: p2 is already contiguous as [N, 320] in memory.

    // fc1 (320 -> 50) + relu
    std::vector<float> h1(static_cast<std::size_t>(N) * 50);
    linear_ref(p2.data(), w.fc1_weight.data(), w.fc1_bias.data(), h1.data(),
               N, 320, 50);
    relu_ref(h1.data(), h1.size());

    // fc2 (50 -> 10) + log_softmax
    std::vector<float> h2(static_cast<std::size_t>(N) * 10);
    linear_ref(h1.data(), w.fc2_weight.data(), w.fc2_bias.data(), h2.data(),
               N, 50, 10);
    log_softmax_ref(h2.data(), output, N, 10);
}

}  // namespace cnn
