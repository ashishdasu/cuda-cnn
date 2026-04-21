// See relu_naive.h for the contract.

#include "kernels/relu_naive.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace cnn {
namespace cuda {

namespace {

constexpr int BLOCK = 256;

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

__global__ void relu_naive_kernel(float* __restrict__ data, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = data[i];
    data[i] = v > 0.0f ? v : 0.0f;
}

}  // namespace

void relu_naive_device(float* d_data, std::size_t count) {
    if (count == 0) return;
    const std::size_t grid = (count + BLOCK - 1) / BLOCK;
    relu_naive_kernel<<<static_cast<unsigned>(grid), BLOCK>>>(d_data, count);
    check(cudaGetLastError(), "relu_naive_kernel launch");
}

void relu_naive(float* data, std::size_t count) {
    if (count == 0) return;

    float* d_buf = nullptr;
    check(cudaMalloc(&d_buf, count * sizeof(float)), "cudaMalloc(data)");
    check(cudaMemcpy(d_buf, data, count * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(data)");

    relu_naive_device(d_buf, count);
    check(cudaDeviceSynchronize(), "relu_naive_kernel execution");

    check(cudaMemcpy(data, d_buf, count * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(data)");
    cudaFree(d_buf);
}

}  // namespace cuda
}  // namespace cnn
