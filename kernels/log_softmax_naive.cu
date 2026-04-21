// See log_softmax_naive.h for the contract, layout, and thread/block map.

#include "kernels/log_softmax_naive.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace cnn {
namespace cuda {

namespace {

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// One block per row. blockDim.x is the next power of two >= C, so the
// tree reduction is clean; threads with tid >= C are idle sentinels
// (filled with -inf for the max pass, 0 for the sum pass). Shared
// memory holds exactly blockDim.x floats and is reused for both the
// max and the sum reductions.
__global__ void log_softmax_naive_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int C) {
    extern __shared__ float sdata[];
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in  = input  + static_cast<std::size_t>(n) * C;
    float*       row_out = output + static_cast<std::size_t>(n) * C;

    // ---- pass 1: row max (subtract-max trick for numerical stability) ----
    sdata[tid] = (tid < C) ? row_in[tid] : -FLT_MAX;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float a = sdata[tid];
            const float b = sdata[tid + stride];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];
    __syncthreads();

    // ---- pass 2: sum of exp(x - max) ----
    sdata[tid] = (tid < C) ? __expf(row_in[tid] - m) : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    const float logs = __logf(sdata[0]);

    // ---- pass 3: write output ----
    if (tid < C) {
        row_out[tid] = row_in[tid] - m - logs;
    }
}

}  // namespace

void log_softmax_naive(const float* input,
                       float* output,
                       int N, int C) {
    if (C <= 0 || N <= 0) return;
    if (C > 1024) {
        throw std::runtime_error("log_softmax_naive: C > 1024 not supported "
                                 "by this kernel; use a two-pass row reduction");
    }

    const std::size_t n_elems = static_cast<std::size_t>(N) * C;
    float *d_in = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in,  n_elems * sizeof(float)), "cudaMalloc(input)");
    check(cudaMalloc(&d_out, n_elems * sizeof(float)), "cudaMalloc(output)");

    check(cudaMemcpy(d_in, input, n_elems * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(input)");

    const int block = next_pow2(C);
    const std::size_t shmem = block * sizeof(float);
    log_softmax_naive_kernel<<<N, block, shmem>>>(d_in, d_out, C);
    check(cudaGetLastError(), "log_softmax_naive_kernel launch");
    check(cudaDeviceSynchronize(), "log_softmax_naive_kernel execution");

    check(cudaMemcpy(output, d_out, n_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
