// See linear_naive.h for the contract, layout, and thread/block map.

#include "kernels/linear_naive.h"

#include <cuda_runtime.h>

#include <cstddef>
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

__global__ void linear_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N, int in_features, int out_features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * out_features;
    if (idx >= total) return;

    const int n = idx / out_features;
    const int o = idx % out_features;

    float sum = bias[o];
    const float* x_row = input  + static_cast<std::size_t>(n) * in_features;
    const float* w_row = weight + static_cast<std::size_t>(o) * in_features;
    for (int i = 0; i < in_features; ++i) {
        sum += x_row[i] * w_row[i];
    }
    output[idx] = sum;
}

}  // namespace

void linear_naive(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N, int in_features, int out_features) {
    const std::size_t in_elems  = static_cast<std::size_t>(N) * in_features;
    const std::size_t w_elems   = static_cast<std::size_t>(out_features) * in_features;
    const std::size_t b_elems   = static_cast<std::size_t>(out_features);
    const std::size_t out_elems = static_cast<std::size_t>(N) * out_features;

    float *d_in = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in,  in_elems  * sizeof(float)), "cudaMalloc(input)");
    check(cudaMalloc(&d_w,   w_elems   * sizeof(float)), "cudaMalloc(weight)");
    check(cudaMalloc(&d_b,   b_elems   * sizeof(float)), "cudaMalloc(bias)");
    check(cudaMalloc(&d_out, out_elems * sizeof(float)), "cudaMalloc(output)");

    check(cudaMemcpy(d_in,  input,  in_elems * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(input)");
    check(cudaMemcpy(d_w,   weight, w_elems  * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(weight)");
    check(cudaMemcpy(d_b,   bias,   b_elems  * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(bias)");

    const int total = N * out_features;
    const int grid  = (total + BLOCK - 1) / BLOCK;
    linear_naive_kernel<<<grid, BLOCK>>>(d_in, d_w, d_b, d_out,
                                         N, in_features, out_features);
    check(cudaGetLastError(), "linear_naive_kernel launch");
    check(cudaDeviceSynchronize(), "linear_naive_kernel execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
