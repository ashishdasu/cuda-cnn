// See linear_tiled.h for the contract, layout, and tile strategy.

#include "kernels/linear_tiled.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace cnn {
namespace cuda {

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 16;

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

// Each block computes a BM x BN tile of the output matrix. Threads
// cooperatively load a BM x BK panel of the input (As) and a BK x BN
// panel of the weight^T (Bs) into shared memory, accumulate BK partial
// products into a register `sum`, then advance to the next K-panel.
// Loading weight[col, t*BK+ty] into Bs[ty][tx] stores the transpose
// implicitly, so the inner loop reads As[ty][k] * Bs[k][tx] without
// an extra index reversal.
__global__ void linear_tiled_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N, int in_features, int out_features) {
    __shared__ float As[BM][BK];   // input tile
    __shared__ float Bs[BK][BN];   // weight^T tile

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * BM + ty;   // which n (output row)
    const int col = blockIdx.x * BN + tx;   // which o (output col)

    float sum = 0.0f;
    const int num_tiles = (in_features + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * BK + tx;
        As[ty][tx] = (row < N && a_col < in_features)
                     ? input[row * in_features + a_col]
                     : 0.0f;

        // weight is stored [out_features, in_features] — loading
        // weight[col, t*BK + ty] into Bs[ty][tx] effectively stores
        // the transpose, so the inner k loop below reads Bs[k][tx]
        // (one output col per thread).
        const int b_row = t * BK + ty;
        Bs[ty][tx] = (col < out_features && b_row < in_features)
                     ? weight[col * in_features + b_row]
                     : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < out_features) {
        output[row * out_features + col] = bias[col] + sum;
    }
}

}  // namespace

void linear_tiled_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N, int in_features, int out_features) {
    const dim3 block(BN, BM, 1);
    const dim3 grid((out_features + BN - 1) / BN,
                    (N            + BM - 1) / BM,
                    1);
    linear_tiled_kernel<<<grid, block>>>(d_input, d_weight, d_bias, d_output,
                                         N, in_features, out_features);
    check(cudaGetLastError(), "linear_tiled_kernel launch");
}

void linear_tiled(const float* input,
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

    linear_tiled_device(d_in, d_w, d_b, d_out, N, in_features, out_features);
    check(cudaDeviceSynchronize(), "linear_tiled_kernel execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
