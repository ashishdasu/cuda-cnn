// Tiled shared-memory GEMM for the linear / fully-connected layer.
// Same semantics as linear_naive (and torch.nn.functional.linear):
//   output[n, o] = bias[o] + sum_i input[n, i] * weight[o, i]
// Classic 2D output-tile GEMM: each block computes a BM x BN patch of
// the output matrix, iterating over BK-wide panels of the shared
// reduction dimension. Every element of input and weight is loaded
// from global memory once per K-panel and reused BM (or BN) times
// from shared memory, reducing global-memory traffic by BM on the
// input side and BN on the weight side.
//
// Tensor layout (row-major, matches PyTorch and linear_naive):
//   input:  [N, in_features]
//   weight: [out_features, in_features]
//   bias:   [out_features]
//   output: [N, out_features]
// Access pattern treats the problem as  C = A @ B^T + bias  where
//   A = input   (N x K),
//   B = weight  (M x K) with M = out_features, K = in_features,
// so the kernel loads a B^T tile by reading the weight row-major and
// storing it transposed into shared memory.
//
// Thread/block layout:
//   BM = BN = BK = 16
//   blockDim = (BN, BM, 1)   (256 threads)
//   gridDim.x = ceil(out_features / BN)
//   gridDim.y = ceil(N           / BM)
// Each thread computes exactly one output element. Two BM x BK and
// BK x BN shared-memory tiles are reused across BM (resp. BN) thread
// rows per K-step. Threads whose global (row, col) fall outside the
// [N, out_features] rectangle still participate in the cooperative
// loads (zero-filling OOB reads) but skip the final store — the
// standard "guard at the edges, load-zero everywhere else" pattern
// for a bounds-tolerant tiled GEMM.

#pragma once

namespace cnn {
namespace cuda {

// Host-facing launcher: alloc / H2D / launch / sync / D2H / free.
// Throws std::runtime_error on any CUDA error.
void linear_tiled(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N, int in_features, int out_features);

// Device-pointer variant for the end-to-end forward pass.
void linear_tiled_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N, int in_features, int out_features);

}  // namespace cuda
}  // namespace cnn
