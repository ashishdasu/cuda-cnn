// See conv2d_tiled.h for the contract, layout, and tile strategy.

#include "kernels/conv2d_tiled.h"

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

// Tiled convolution. Each block produces a TILE_H x TILE_W output
// patch for one (n, oc). Dynamic shared memory holds a
// (TILE_H + Kh - 1) x (TILE_W + Kw - 1) input patch for one input
// channel at a time; the ic loop reloads shmem and accumulates into
// `sum` in registers.
__global__ void conv2d_tiled_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int C_in, int H_in, int W_in,
                                    int C_out, int Kh, int Kw,
                                    int H_out, int W_out,
                                    int tile_in_h, int tile_in_w) {
    extern __shared__ float smem[];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ow  = blockIdx.x * TILE_W + tx;
    const int oh  = blockIdx.y * TILE_H + ty;
    const int noc = blockIdx.z;
    const int n   = noc / C_out;
    const int oc  = noc % C_out;

    // Top-left corner of this block's input region (in input coords).
    const int in_row0 = blockIdx.y * TILE_H;
    const int in_col0 = blockIdx.x * TILE_W;

    const bool out_in_bounds = (ow < W_out) && (oh < H_out);
    float sum = out_in_bounds ? bias[oc] : 0.0f;

    const int tid       = ty * TILE_W + tx;
    const int nthreads  = TILE_H * TILE_W;
    const int tile_size = tile_in_h * tile_in_w;

    for (int ic = 0; ic < C_in; ++ic) {
        // Cooperative load: stride the block's threads across the
        // (tile_in_h x tile_in_w) shmem patch. Out-of-input reads
        // are zeroed; those slots are only read by OOB output
        // threads (guarded below).
        for (int t = tid; t < tile_size; t += nthreads) {
            const int r  = t / tile_in_w;
            const int c  = t - r * tile_in_w;
            const int ih = in_row0 + r;
            const int iw = in_col0 + c;
            float v = 0.0f;
            if (ih < H_in && iw < W_in) {
                v = input[((n * C_in + ic) * H_in + ih) * W_in + iw];
            }
            smem[r * tile_in_w + c] = v;
        }
        __syncthreads();

        if (out_in_bounds) {
            // Accumulate this ic's contribution from shmem.
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    const float x = smem[(ty + kh) * tile_in_w + (tx + kw)];
                    const float w = weight[((oc * C_in + ic) * Kh + kh) * Kw + kw];
                    sum += x * w;
                }
            }
        }
        __syncthreads();  // guard shmem before the next ic overwrites it
    }

    if (out_in_bounds) {
        output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
    }
}

}  // namespace

void conv2d_tiled_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;
    const int tile_in_h = TILE_H + Kh - 1;
    const int tile_in_w = TILE_W + Kw - 1;

    const dim3 block(TILE_W, TILE_H, 1);
    const dim3 grid((W_out + TILE_W - 1) / TILE_W,
                    (H_out + TILE_H - 1) / TILE_H,
                    N * C_out);
    const std::size_t shmem = static_cast<std::size_t>(tile_in_h) *
                              tile_in_w * sizeof(float);

    conv2d_tiled_kernel<<<grid, block, shmem>>>(d_input, d_weight, d_bias, d_output,
                                                C_in, H_in, W_in,
                                                C_out, Kh, Kw,
                                                H_out, W_out,
                                                tile_in_h, tile_in_w);
    check(cudaGetLastError(), "conv2d_tiled_kernel launch");
}

void conv2d_tiled(const float* input,
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

    conv2d_tiled_device(d_in, d_w, d_b, d_out,
                        N, C_in, H_in, W_in, C_out, Kh, Kw);
    check(cudaDeviceSynchronize(), "conv2d_tiled_kernel execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
