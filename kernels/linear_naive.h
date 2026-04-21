// Naive CUDA implementation of a linear / fully-connected layer.
// Semantics are identical to cnn::linear_ref (matches
// torch.nn.functional.linear):
//   output[n, o] = bias[o] + sum_i input[n, i] * weight[o, i]
//
// Tensor layout (row-major, PyTorch-native):
//   input:  [N, in_features]
//   weight: [out_features, in_features]
//   bias:   [out_features]
//   output: [N, out_features]
//
// Thread/block layout:
//   blockDim = 256
//   gridDim  = ceil(N * out_features / 256)
// Output elements are linearized as (n * out_features + o). One thread
// per output element, straight in_features-long dot product, no shared
// memory. This is the naive GEMM baseline that the tiled variant
// (target tier) will be diffed and speedup-compared against.

#pragma once

namespace cnn {
namespace cuda {

// Host launcher: self-contained device alloc / H2D / launch / D2H / free.
// Throws std::runtime_error on any CUDA error.
void linear_naive(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N, int in_features, int out_features);

}  // namespace cuda
}  // namespace cnn
