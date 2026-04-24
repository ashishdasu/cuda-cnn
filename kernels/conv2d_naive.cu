// See conv2d_naive.h for the contract, layout, and thread/block map.

#include "kernels/conv2d_naive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cnn {
namespace cuda {

namespace {

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

// One thread per output element. blockIdx.z encodes (n, oc) so the
// divide/modulo is paid once per block, not per thread.
__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int C_in, int H_in, int W_in,
                                    int C_out, int Kh, int Kw,
                                    int H_out, int W_out) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int noc = blockIdx.z;
    const int n  = noc / C_out;
    const int oc = noc % C_out;

    if (ow >= W_out || oh >= H_out) return;

    float sum = bias[oc];
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                const int ih = oh + kh;
                const int iw = ow + kw;
                const float x = input[((n * C_in + ic) * H_in + ih) * W_in + iw];
                const float w = weight[((oc * C_in + ic) * Kh + kh) * Kw + kw];
                sum += x * w;
            }
        }
    }
    output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
}

}  // namespace

void conv2d_naive_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;

    const dim3 block(TILE_W, TILE_H, 1);
    const dim3 grid((W_out + TILE_W - 1) / TILE_W,
                    (H_out + TILE_H - 1) / TILE_H,
                    N * C_out);
    conv2d_naive_kernel<<<grid, block>>>(d_input, d_weight, d_bias, d_output,
                                         C_in, H_in, W_in,
                                         C_out, Kh, Kw,
                                         H_out, W_out);
    check(cudaGetLastError(), "conv2d_naive_kernel launch");
}

void conv2d_naive(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N,
                  int C_in, int H_in, int W_in,
                  int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;

    const std::size_t in_elems  = static_cast<std::size_t>(N) * C_in * H_in * W_in;
    const std::size_t w_elems   = static_cast<std::size_t>(C_out) * C_in * Kh * Kw;
    const std::size_t b_elems   = static_cast<std::size_t>(C_out);
    const std::size_t out_elems = static_cast<std::size_t>(N) * C_out * H_out * W_out;

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

    conv2d_naive_device(d_in, d_w, d_b, d_out,
                        N, C_in, H_in, W_in, C_out, Kh, Kw);
    check(cudaDeviceSynchronize(), "conv2d_naive_kernel execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
