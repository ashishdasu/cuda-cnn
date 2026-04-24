// Tiled shared-memory CUDA implementation of 2D convolution. Same
// semantics as conv2d_naive (stride 1, no padding, PyTorch layout) —
// Each block cooperatively
// stages an input patch into shared memory once per input channel,
// then every thread in the block reuses those loads to produce one
// output element. Global-memory traffic on the input drops by
// (Kh * Kw) relative to the naive kernel.
//
// Tensor layout (identical to the naive variant):
//   input:  [N, C_in, H_in, W_in]
//   weight: [C_out, C_in, Kh, Kw]
//   bias:   [C_out]
//   output: [N, C_out, H_out, W_out],  H_out = H_in - Kh + 1,
//                                      W_out = W_in - Kw + 1
//
// Thread/block layout:
//   blockDim  = (TILE_W, TILE_H, 1)   with TILE_W = TILE_H = 16
//   gridDim.x = ceil(W_out / TILE_W)
//   gridDim.y = ceil(H_out / TILE_H)
//   gridDim.z = N * C_out
// Each block produces a TILE_H x TILE_W patch of the output for a
// single (n, oc). Shared-memory working set per block is
//   (TILE_H + Kh - 1) * (TILE_W + Kw - 1) floats
// holding the input patch for one input channel at a time. The
// kernel loops over ic, reloading shmem each iteration and
// accumulating into a per-thread register `sum`. Weights are read
// directly from global memory per (kh, kw) — at LeNet sizes the
// weight tensor is tiny (at most 5000 floats for conv2) and hot in
// L1/L2 after the first few threads, so staging it into shmem would
// complicate the code without a measurable win. The tiled GEMM
// addresses weight re-use; that's a different kernel.
//
// Correctness invariants:
//   * __syncthreads() after the cooperative load AND before the
//     next ic iteration overwrites shmem.
//   * Out-of-bounds input loads produce 0.0f; those slots are only
//     read by threads that are themselves OOB for the output tile,
//     so the zero never leaks into a real output element.
//   * Boundary threads (ow >= W_out || oh >= H_out) still
//     participate in the shmem load but skip the accumulate and the
//     final write.

#pragma once

namespace cnn {
namespace cuda {

// Host-facing launcher: alloc / H2D / launch / sync / D2H / free.
// Throws std::runtime_error on any CUDA error.
void conv2d_tiled(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N,
                  int C_in, int H_in, int W_in,
                  int C_out, int Kh, int Kw);

// Device-pointer variant. See conv2d_naive_device for the shared
// rationale (end-to-end pipeline keeps activations on-device). The
// tiled forward pass substitutes this call in for conv2d_naive_device.
void conv2d_tiled_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw);

}  // namespace cuda
}  // namespace cnn
