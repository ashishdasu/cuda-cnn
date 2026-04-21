// See conv2d_bias_relu_pool_fused.h for the contract, layout, and
// thread/block map.

#include "kernels/conv2d_bias_relu_pool_fused.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cnn {
namespace cuda {

namespace {

// One thread produces one POOL output. A POOL_TILE x POOL_TILE block
// therefore produces a POOL_TILE x POOL_TILE pool tile = a
// (2*POOL_TILE) x (2*POOL_TILE) conv tile. 4 is the smallest tile that
// exactly fits LeNet conv2's 4x4 pool output (no wasted threads) and
// tiles LeNet conv1's 12x12 pool output with a 3x3 block grid.
constexpr int POOL_TILE = 4;

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

__global__ void conv2d_bias_relu_pool_fused_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int C_in, int H_in, int W_in,
        int C_out, int Kh, int Kw,
        int H_conv, int W_conv,
        int H_pool, int W_pool) {
    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int pool_ow = blockIdx.x * POOL_TILE + tx;
    const int pool_oh = blockIdx.y * POOL_TILE + ty;
    const int noc = blockIdx.z;
    const int n  = noc / C_out;
    const int oc = noc % C_out;

    // Conv output coordinates of this thread's 2x2 window.
    const int conv_ow0 = 2 * pool_ow;
    const int conv_oh0 = 2 * pool_oh;
    const bool in_pool_bounds = (pool_ow < W_pool) && (pool_oh < H_pool);

    // Origin (in conv-output coords) of the 2*POOL_TILE x 2*POOL_TILE
    // conv tile this block produces. The corresponding input tile
    // starts at the same (row, col) because stride=1, pad=0.
    const int conv_row0 = 2 * blockIdx.y * POOL_TILE;
    const int conv_col0 = 2 * blockIdx.x * POOL_TILE;

    const int tile_in_h = 2 * POOL_TILE + Kh - 1;
    const int tile_in_w = 2 * POOL_TILE + Kw - 1;
    const int tile_sz   = tile_in_h * tile_in_w;

    const int tid      = ty * POOL_TILE + tx;
    const int nthreads = POOL_TILE * POOL_TILE;

    // Accumulators for the four conv outputs that feed this thread's
    // single pool output. Start from bias[oc] so the inner loop is a
    // pure multiply-add.
    const float b = bias[oc];
    float acc00 = b, acc01 = b, acc10 = b, acc11 = b;

    for (int ic = 0; ic < C_in; ++ic) {
        // Cooperative load of this ic's input tile into shmem. One
        // thread may load multiple elements when tile_sz > nthreads,
        // which is always the case for the LeNet shapes
        // (tile_sz = 144 > nthreads = 16 for POOL_TILE=4, Kh=Kw=5).
        for (int idx = tid; idx < tile_sz; idx += nthreads) {
            const int trow = idx / tile_in_w;
            const int tcol = idx % tile_in_w;
            const int irow = conv_row0 + trow;
            const int icol = conv_col0 + tcol;
            float v = 0.0f;
            if (irow < H_in && icol < W_in) {
                v = input[((n * C_in + ic) * H_in + irow) * W_in + icol];
            }
            smem[idx] = v;
        }
        __syncthreads();

        // Accumulate this ic's contribution into all four conv outputs
        // of this thread's 2x2 window. The shmem offset for conv coord
        // (r, c) is at smem[(r - conv_row0) * tile_in_w + (c - conv_col0)].
        // Because the tile is sized to exactly cover the 2x2 output
        // block's receptive field, we can reuse the tile for both
        // output rows and both output cols without a re-sync.
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                const int o_row = conv_oh0 + dy;
                const int o_col = conv_ow0 + dx;
                // If our conv output is past the end, don't accumulate;
                // the pool-bounds check below also gates the write.
                if (o_row >= H_conv || o_col >= W_conv) continue;
                const int base_r = o_row - conv_row0;
                const int base_c = o_col - conv_col0;
                float s = 0.0f;
                for (int kh = 0; kh < Kh; ++kh) {
                    for (int kw = 0; kw < Kw; ++kw) {
                        const float x = smem[(base_r + kh) * tile_in_w + (base_c + kw)];
                        const float w = weight[((oc * C_in + ic) * Kh + kh) * Kw + kw];
                        s += x * w;
                    }
                }
                if (dy == 0 && dx == 0) acc00 += s;
                else if (dy == 0 && dx == 1) acc01 += s;
                else if (dy == 1 && dx == 0) acc10 += s;
                else                         acc11 += s;
            }
        }
        __syncthreads();  // guard smem before the next ic overwrites it
    }

    // ReLU each of the four conv outputs, then take the max. Matches
    // torch.relu followed by torch.nn.functional.max_pool2d(x, 2).
    if (!in_pool_bounds) return;
    const float r00 = acc00 > 0.0f ? acc00 : 0.0f;
    const float r01 = acc01 > 0.0f ? acc01 : 0.0f;
    const float r10 = acc10 > 0.0f ? acc10 : 0.0f;
    const float r11 = acc11 > 0.0f ? acc11 : 0.0f;
    const float m0 = r00 > r01 ? r00 : r01;
    const float m1 = r10 > r11 ? r10 : r11;
    const float out_val = m0 > m1 ? m0 : m1;

    output[((n * C_out + oc) * H_pool + pool_oh) * W_pool + pool_ow] = out_val;
}

}  // namespace

void conv2d_bias_relu_pool_fused_device(const float* d_input,
                                        const float* d_weight,
                                        const float* d_bias,
                                        float* d_output,
                                        int N,
                                        int C_in, int H_in, int W_in,
                                        int C_out, int Kh, int Kw) {
    const int H_conv = H_in - Kh + 1;
    const int W_conv = W_in - Kw + 1;
    const int H_pool = H_conv / 2;
    const int W_pool = W_conv / 2;

    if (H_pool <= 0 || W_pool <= 0) {
        throw std::runtime_error(
            "conv2d_bias_relu_pool_fused: degenerate output shape "
            "(H_pool <= 0 or W_pool <= 0)");
    }

    const int tile_in_h = 2 * POOL_TILE + Kh - 1;
    const int tile_in_w = 2 * POOL_TILE + Kw - 1;
    const std::size_t shmem = static_cast<std::size_t>(tile_in_h) *
                              static_cast<std::size_t>(tile_in_w) *
                              sizeof(float);

    const dim3 block(POOL_TILE, POOL_TILE, 1);
    const dim3 grid((W_pool + POOL_TILE - 1) / POOL_TILE,
                    (H_pool + POOL_TILE - 1) / POOL_TILE,
                    N * C_out);
    conv2d_bias_relu_pool_fused_kernel<<<grid, block, shmem>>>(
        d_input, d_weight, d_bias, d_output,
        C_in, H_in, W_in,
        C_out, Kh, Kw,
        H_conv, W_conv,
        H_pool, W_pool);
    check(cudaGetLastError(), "conv2d_bias_relu_pool_fused_kernel launch");
}

void conv2d_bias_relu_pool_fused(const float* input,
                                 const float* weight,
                                 const float* bias,
                                 float* output,
                                 int N,
                                 int C_in, int H_in, int W_in,
                                 int C_out, int Kh, int Kw) {
    const int H_conv = H_in - Kh + 1;
    const int W_conv = W_in - Kw + 1;
    const int H_pool = H_conv / 2;
    const int W_pool = W_conv / 2;

    const std::size_t in_elems  = static_cast<std::size_t>(N) * C_in * H_in * W_in;
    const std::size_t w_elems   = static_cast<std::size_t>(C_out) * C_in * Kh * Kw;
    const std::size_t b_elems   = static_cast<std::size_t>(C_out);
    const std::size_t out_elems = static_cast<std::size_t>(N) * C_out * H_pool * W_pool;

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

    conv2d_bias_relu_pool_fused_device(d_in, d_w, d_b, d_out,
                                       N, C_in, H_in, W_in, C_out, Kh, Kw);
    check(cudaDeviceSynchronize(), "conv2d_bias_relu_pool_fused execution");

    check(cudaMemcpy(output, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
