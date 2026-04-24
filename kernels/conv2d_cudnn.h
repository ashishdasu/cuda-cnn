// cuDNN-backed 2D convolution, used as the throughput baseline for the
// hand-written kernels. Runs the same shapes on the same hardware so
// the comparison is apples-to-apples.
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
// per call, so this includes handle-creation overhead in the timing.
// Use conv2d_cudnn_device_with_handle to separate that cost.
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
