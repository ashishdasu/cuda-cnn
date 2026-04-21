// CPU reference implementations of the LeNet-MNIST forward-pass
// primitives. Semantics match PyTorch exactly (for the operators and
// shapes used by Project 5's MyNetwork); correctness is the only
// requirement here — these functions exist to be the numerical oracle
// that every CUDA kernel is validated against.
//
// All buffers are plain float*, row-major (C order). Shape conventions
// match PyTorch's native layout so tensors loaded from the exported
// .bin files can be passed in directly:
//   activations: [N, C, H, W]
//   conv weights: [C_out, C_in, Kh, Kw]
//   linear weights: [out_features, in_features]
//   biases: [channels] or [out_features]
//
// Nothing here is tuned for speed; loops are straight triple/quadruple
// nests for readability. Runtime on one MNIST image is ~1 ms, plenty
// fast enough for unit tests.

#pragma once

#include "host/weights.h"

#include <cstddef>

namespace cnn {

// 2D convolution. Stride 1, no padding. Matches
// torch.nn.functional.conv2d(input, weight, bias) with the default
// stride/padding/dilation arguments.
//   output[n, oc, oh, ow] = bias[oc]
//       + sum over (ic, kh, kw) of
//         input[n, ic, oh+kh, ow+kw] * weight[oc, ic, kh, kw]
// Output spatial dims: H_out = H_in - Kh + 1, W_out = W_in - Kw + 1.
void conv2d_ref(const float* input,
                const float* weight,
                const float* bias,
                float* output,
                int N,
                int C_in, int H_in, int W_in,
                int C_out, int Kh, int Kw);

// 2x2 max pooling with stride 2, no padding. Matches
// torch.nn.functional.max_pool2d(input, 2). Output spatial dims are
// H_out = H_in / 2, W_out = W_in / 2 (integer division; odd inputs
// would drop a row/column, same as PyTorch).
void max_pool2d_ref(const float* input,
                    float* output,
                    int N, int C, int H_in, int W_in);

// In-place ReLU over `count` elements. Matches torch.relu_(x).
void relu_ref(float* data, std::size_t count);

// Linear / fully-connected layer. Matches torch.nn.functional.linear.
//   output[n, o] = bias[o] + sum over i of input[n, i] * weight[o, i]
// `weight` is row-major [out_features, in_features] — PyTorch's native
// layout, same as what export_weights.py writes.
void linear_ref(const float* input,
                const float* weight,
                const float* bias,
                float* output,
                int N, int in_features, int out_features);

// Numerically stable log-softmax over the last dimension. Subtracts
// the per-row max before exp to avoid overflow (the trap called out
// explicitly in the proposal).
//   output[n, c] = input[n, c] - m - log(sum_c' exp(input[n, c'] - m))
//   where m = max_c input[n, c]
// Matches torch.nn.functional.log_softmax(x, dim=1) for 2D input.
void log_softmax_ref(const float* input,
                     float* output,
                     int N, int C);

// Full LeNet-MNIST forward pass composed from the primitives above.
// Input shape [N, 1, 28, 28]; output shape [N, 10] (log-probabilities).
// Dropout2d is skipped (inference mode, matching model.eval()).
void forward_ref(const Weights& w,
                 const float* input,
                 float* output,
                 int N);

}  // namespace cnn
