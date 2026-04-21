// cuDNN-backed 2D convolution. Exists only to provide the stretch-tier
// baseline — the proposal's goal is to "close the gap to cuDNN on
// convolution throughput," so we need cuDNN numbers on the same shapes
// and the same RTX 4090 to make the comparison honest.
//
// Same mathematical contract as conv2d_naive / conv2d_tiled (stride 1,
// no padding, PyTorch-native tensor layout). This is thin plumbing:
// build a handle, configure tensor/filter/convolution/activation
// descriptors, ask cuDNN to pick a forward algorithm, and call
// cudnnConvolutionForward. Workspace memory is allocated on demand.
//
// Only compiled when cuDNN is found at configure time (see CMakeLists).

#pragma once

#include <cstddef>

namespace cnn {
namespace cuda {

// Host-facing launcher: alloc / H2D / conv / D2H / free.
void conv2d_cudnn(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N,
                  int C_in, int H_in, int W_in,
                  int C_out, int Kh, int Kw);

// Device-pointer variant. Internally manages the cuDNN handle lifetime
// per call; for a real throughput benchmark that would be hoisted out,
// but we bench cold-handle cost as part of the honest comparison.
void conv2d_cudnn_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw);

// Optional: reusable handle variant so the bench harness doesn't pay
// handle-creation cost on every call. The caller owns the handle and
// workspace buffer. If d_workspace is nullptr (or workspace_bytes is
// 0), the function allocates/frees its own workspace internally.
void conv2d_cudnn_device_with_handle(void* cudnn_handle,
                                     const float* d_input,
                                     const float* d_weight,
                                     const float* d_bias,
                                     float* d_output,
                                     int N,
                                     int C_in, int H_in, int W_in,
                                     int C_out, int Kh, int Kw);

}  // namespace cuda
}  // namespace cnn
