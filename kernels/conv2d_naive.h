// Naive CUDA implementation of 2D convolution. Semantics are identical
// to cnn::conv2d_ref (stride 1, no padding, PyTorch layout); this is
// the correctness baseline against which every tiled/optimized conv
// kernel that follows will be diffed.
//
// Tensor layout (row-major, matches PyTorch and the host reference):
//   input:  [N, C_in, H_in, W_in]
//   weight: [C_out, C_in, Kh, Kw]
//   bias:   [C_out]
//   output: [N, C_out, H_out, W_out],  H_out = H_in - Kh + 1,
//                                      W_out = W_in - Kw + 1
//
// Thread/block layout (documented here, asserted in the .cu):
//   blockDim  = (TILE_W, TILE_H, 1)         with TILE_W = TILE_H = 16
//   gridDim.x = ceil(W_out / TILE_W)
//   gridDim.y = ceil(H_out / TILE_H)
//   gridDim.z = N * C_out                     (blockIdx.z indexes (n, oc))
// One thread computes exactly one output element via the straight
// triple-nested reduction over (ic, kh, kw), starting from bias[oc].
// No shared memory, no re-use — every thread re-reads input and
// weight values from global memory. This is deliberately the dumbest
// correct version; the tiled variant is a separate kernel.
//
// The launcher takes host pointers and manages its own device
// allocations + H2D/D2H copies. That makes correctness tests
// self-contained — the kernel can be dropped in for the CPU reference
// in any test without touching surrounding code. The device-pointer
// variant below skips the round-trip for use in the end-to-end pipeline.

#pragma once

namespace cnn {
namespace cuda {

// Host-facing launcher. Allocates device buffers, H2D-copies inputs,
// launches the naive kernel, D2H-copies the result, and frees. Any
// CUDA error aborts via std::runtime_error.
void conv2d_naive(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N,
                  int C_in, int H_in, int W_in,
                  int C_out, int Kh, int Kw);

// Device-pointer variant used by the end-to-end forward pass. All
// pointers must refer to device memory. No allocation, no copies, no
// per-call synchronize — the caller owns lifetime and streaming. Errors
// from the launch surface via cudaGetLastError at the next check.
void conv2d_naive_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw);

}  // namespace cuda
}  // namespace cnn
