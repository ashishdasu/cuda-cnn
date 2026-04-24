// End-to-end CUDA forward pass for the LeNet-MNIST network. Semantics
// must match cnn::forward_ref exactly (same layer sequence, same
// skipped Dropout2d at inference, same PyTorch layout) and therefore
// match torch.nn.functional.log_softmax outputs to within the 1e-4
// project tolerance.
//
// Runtime contract: weights are cudaMemcpy'd once to the GPU,
// activations stay on-device between layers, and the only host traffic
// per inference call is the input H2D and the log-probability D2H.
// No per-layer round-trips.
//
// Layer sequence and intermediate device-buffer shapes for input
// [N, 1, 28, 28]:
//   conv1  (1  -> 10, 5x5, stride 1, no pad) -> [N, 10, 24, 24]
//   pool   (2x2, stride 2)                    -> [N, 10, 12, 12]
//   relu   (in place)
//   conv2  (10 -> 20, 5x5)                    -> [N, 20,  8,  8]
//   pool   (2x2, stride 2)                    -> [N, 20,  4,  4] = [N, 320]
//   relu   (in place)
//   fc1    (320 -> 50)                        -> [N, 50]
//   relu   (in place)
//   fc2    (50  -> 10)                        -> [N, 10]
//   log_softmax (row-wise, C = 10)            -> [N, 10]
//
// All device buffers are allocated up front and freed at the end.
// Error handling is consistent with the naive kernels: any CUDA error
// throws std::runtime_error.

#pragma once

#include "host/weights.h"

namespace cnn {
namespace cuda {

// End-to-end CUDA forward pass. `h_input` is a host pointer to
// [N, 1, 28, 28] float data; `h_output` is a host pointer that will
// receive [N, 10] log-probabilities. The function handles every device
// allocation and copy internally — the caller only ever sees host
// memory. Matches cnn::forward_ref numerically to within 1e-4.
void forward_naive_gpu(const Weights& w,
                       const float* h_input,
                       float* h_output,
                       int N);

// Same end-to-end contract as forward_naive_gpu, but substitutes the
// tiled shared-memory conv2d and linear kernels for their naive
// counterparts. Pool, ReLU, and log_softmax remain the naive kernels
// (pool, ReLU, and log_softmax aren't tile-bound at LeNet scale;
// the optimization scope is conv + GEMM).
void forward_tiled_gpu(const Weights& w,
                       const float* h_input,
                       float* h_output,
                       int N);

}  // namespace cuda
}  // namespace cnn
