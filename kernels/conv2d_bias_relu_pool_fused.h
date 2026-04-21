// Fused conv2d + bias + ReLU + 2x2 max-pool (stride 2) kernel.
//
// Semantics are identical to the naive four-call sequence:
//     conv2d_ref(input, weight, bias, conv_out, ...)
//     relu_ref(conv_out, ...)
//     max_pool2d_ref(conv_out, pool_out, ...)
// but produces pool_out directly, without materializing the intermediate
// conv_out or relu_out tensors in global memory. This is the end-to-end
// lever named in the report: fusing the four ops cuts three kernel
// launches and two DRAM round-trips on the two LeNet conv stages.
//
// Tensor layout (row-major, PyTorch-compatible):
//   input:  [N, C_in, H_in, W_in]
//   weight: [C_out, C_in, Kh, Kw]
//   bias:   [C_out]
//   output: [N, C_out, H_pool, W_pool]
// Output spatial dims:
//   H_conv = H_in - Kh + 1,        W_conv = W_in - Kw + 1,
//   H_pool = H_conv / 2,           W_pool = W_conv / 2.
// Odd conv output dims drop the trailing row/column, matching
// torch.nn.functional.max_pool2d(..., 2) with default padding.
//
// Thread/block layout (POOL_TILE is a compile-time constant in the .cu):
//   blockDim  = (POOL_TILE, POOL_TILE, 1)
//   gridDim.x = ceil(W_pool / POOL_TILE)
//   gridDim.y = ceil(H_pool / POOL_TILE)
//   gridDim.z = N * C_out             (blockIdx.z indexes (n, oc))
// One thread produces one pool output = ReLU-max over a 2x2 window of
// conv outputs. Each thread therefore computes four conv accumulations
// in registers. The input tile required by a whole block for one input
// channel is (2*POOL_TILE + Kh - 1) x (2*POOL_TILE + Kw - 1); the ic
// loop reloads that dynamic shared-memory tile and folds its
// contribution into each thread's four accumulators.
//
// Numerical contract: bit-for-bit equivalent to the four-call sequence
// up to FP32 summation-reorder noise. Validated at the project-wide
// 1e-4 tolerance by tests/test_conv2d_bias_relu_pool_fused.cu.

#pragma once

namespace cnn {
namespace cuda {

// Host-facing launcher. Allocates device buffers, H2D/D2H, frees.
void conv2d_bias_relu_pool_fused(const float* input,
                                 const float* weight,
                                 const float* bias,
                                 float* output,
                                 int N,
                                 int C_in, int H_in, int W_in,
                                 int C_out, int Kh, int Kw);

// Device-pointer variant for composition into forward pipelines.
// The caller owns allocation/lifetime; this just launches the kernel.
void conv2d_bias_relu_pool_fused_device(const float* d_input,
                                        const float* d_weight,
                                        const float* d_bias,
                                        float* d_output,
                                        int N,
                                        int C_in, int H_in, int W_in,
                                        int C_out, int Kh, int Kw);

}  // namespace cuda
}  // namespace cnn
