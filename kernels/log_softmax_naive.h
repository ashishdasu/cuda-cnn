// Naive CUDA implementation of numerically stable row-wise log-softmax.
// Semantics are identical to cnn::log_softmax_ref (matches
// torch.nn.functional.log_softmax(x, dim=1) for 2D input):
//   m        = max_c input[n, c]
//   output[n, c] = input[n, c] - m - log(sum_c' exp(input[n, c'] - m))
//
// Tensor layout (row-major):
//   input:  [N, C]
//   output: [N, C]
//
// Per-row max subtraction is required for numerical stability: naive
// exp on raw logits overflows. Subtracting the max bounds the
// exponent at 0.
//
// Thread/block layout:
//   One block per row; blockDim.x = min(next pow2 >= C, 1024).
//   Each thread handles one column (C is small here -- LeNet's final
//   layer is C = 10, fits in a single warp). Two shared-memory
//   reductions per row:
//     pass 1: tree-reduce to find row max
//     pass 2: tree-reduce sum of exp(x - max)
//     pass 3: write output[n, c] = x - max - log(sum)
// Correct for C <= 1024; beyond that the kernel throws rather than
// silently producing wrong results (a grid-stride reduction would be needed).

#pragma once

namespace cnn {
namespace cuda {

// Host launcher: self-contained device alloc / H2D / launch / D2H / free.
// Throws std::runtime_error on any CUDA error or if C > 1024.
void log_softmax_naive(const float* input,
                       float* output,
                       int N, int C);

// Device-pointer variant for the end-to-end forward pass. Pointers are
// device memory. Throws on C > 1024 (same constraint as the host
// variant — the single-block-per-row layout caps at blockDim.x = 1024).
void log_softmax_naive_device(const float* d_input,
                              float* d_output,
                              int N, int C);

}  // namespace cuda
}  // namespace cnn
