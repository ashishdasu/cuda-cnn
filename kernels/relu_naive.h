// Naive CUDA in-place ReLU over a flat float buffer. Semantics are
// identical to cnn::relu_ref (and torch.relu_): x <- max(x, 0).
//
// Thread/block layout:
//   blockDim = 256
//   gridDim  = ceil(count / 256)
// One thread per element, grid-stride loop omitted because count is
// known up front and launches are cheap. Bounds-checked.

#pragma once

#include <cstddef>

namespace cnn {
namespace cuda {

// Host launcher: self-contained device alloc / H2D / launch / D2H / free.
// Throws std::runtime_error on any CUDA error.
void relu_naive(float* data, std::size_t count);

// Device-pointer variant for the end-to-end forward pass. `d_data` is
// a device pointer; kernel runs in-place. No alloc/copy/sync.
void relu_naive_device(float* d_data, std::size_t count);

}  // namespace cuda
}  // namespace cnn
