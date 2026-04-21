// Naive CUDA implementation of 2x2 stride-2 max pooling. Semantics are
// identical to cnn::max_pool2d_ref (matches torch.nn.functional.max_pool2d
// with kernel=2, stride=2, no padding).
//
// Tensor layout (row-major):
//   input:  [N, C, H_in, W_in]
//   output: [N, C, H_in/2, W_in/2]
//
// Thread/block layout:
//   blockDim  = (TILE_W, TILE_H, 1)  with TILE_W = TILE_H = 16
//   gridDim.x = ceil(W_out / TILE_W)
//   gridDim.y = ceil(H_out / TILE_H)
//   gridDim.z = N * C                  (blockIdx.z indexes (n, c))
// One thread per output element reads the 2x2 input window and writes
// the max — the exact pattern the proposal calls out. No shared memory;
// every thread reads 4 distinct input values from global, no re-use to
// exploit, so a naive version is already close to optimal here.

#pragma once

namespace cnn {
namespace cuda {

// Host launcher: self-contained device alloc / H2D / launch / D2H / free.
// Throws std::runtime_error on any CUDA error.
void max_pool2d_naive(const float* input,
                      float* output,
                      int N, int C, int H_in, int W_in);

}  // namespace cuda
}  // namespace cnn
