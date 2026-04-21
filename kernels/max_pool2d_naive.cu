// See max_pool2d_naive.h for the contract, layout, and thread/block map.

#include "kernels/max_pool2d_naive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

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

__global__ void max_pool2d_naive_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int C, int H_in, int W_in,
                                        int H_out, int W_out) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z;
    const int n  = nc / C;
    const int c  = nc % C;

    if (ow >= W_out || oh >= H_out) return;

    const int ih = oh * 2;
    const int iw = ow * 2;
    const std::size_t base = (static_cast<std::size_t>(n) * C + c) * H_in * W_in;

    const float a = input[base + (ih    ) * W_in + iw    ];
    const float b = input[base + (ih    ) * W_in + iw + 1];
    const float c2 = input[base + (ih + 1) * W_in + iw    ];
    const float d = input[base + (ih + 1) * W_in + iw + 1];

    float m = a > b ? a : b;
    m = m > c2 ? m : c2;
    m = m > d  ? m : d;

    output[((static_cast<std::size_t>(n) * C + c) * H_out + oh) * W_out + ow] = m;
}

}  // namespace

void max_pool2d_naive(const float* input,
                      float* output,
                      int N, int C, int H_in, int W_in) {
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;

    const std::size_t in_elems  = static_cast<std::size_t>(N) * C * H_in * W_in;
    const std::size_t out_elems = static_cast<std::size_t>(N) * C * H_out * W_out;

    float *d_in = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in,  in_elems  * sizeof(float)), "cudaMalloc(input)");
    check(cudaMalloc(&d_out, out_elems * sizeof(float)), "cudaMalloc(output)");

    check(cudaMemcpy(d_in, input, in_elems * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(input)");

    const dim3 block(TILE_W, TILE_H, 1);
    const dim3 grid((W_out + TILE_W - 1) / TILE_W,
                    (H_out + TILE_H - 1) / TILE_H,
                    N * C);
    max_pool2d_naive_kernel<<<grid, block>>>(d_in, d_out, C, H_in, W_in, H_out, W_out);
    check(cudaGetLastError(), "max_pool2d_naive_kernel launch");
    check(cudaDeviceSynchronize(), "max_pool2d_naive_kernel execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
