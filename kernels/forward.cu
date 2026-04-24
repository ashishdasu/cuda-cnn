// See forward.h for the contract and layer sequence.
//
// Implementation notes:
// - Weights are copied H2D once, into dedicated device buffers held for
//   the lifetime of this call. A longer-lived inference driver would
//   hoist these above the per-image call; for the per-fixture test
//   harness that's unnecessary and would complicate teardown.
// - Two activation buffers (ping-pong) would suffice for minimizing
//   peak memory, but at LeNet scale (largest intermediate is
//   N * 10 * 24 * 24 * 4 B = 23 KB per batch item) a distinct buffer
//   per layer is trivially cheap and keeps the code linear and easy
//   to audit against forward_ref.
// - No intermediate cudaDeviceSynchronize between layers — launches
//   queue on the default stream and execute in order. One sync at the
//   end, right before the D2H copy.

#include "kernels/forward.h"

#include "kernels/conv2d_naive.h"
#include "kernels/conv2d_tiled.h"
#include "kernels/linear_naive.h"
#include "kernels/linear_tiled.h"
#include "kernels/log_softmax_naive.h"
#include "kernels/max_pool2d_naive.h"
#include "kernels/relu_naive.h"

#include <cuda_runtime.h>

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

// RAII-lite device buffer. The forward pass holds a lot of them;
// writing the alloc/free pairs out by hand is noisy and leaks on
// exception. This is still just a handle over cudaMalloc/cudaFree.
struct DevBuf {
    float* ptr = nullptr;
    std::size_t elems = 0;

    DevBuf() = default;
    explicit DevBuf(std::size_t n) : elems(n) {
        check(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
    }
    ~DevBuf() { if (ptr) cudaFree(ptr); }

    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), elems(o.elems) { o.ptr = nullptr; o.elems = 0; }
    DevBuf& operator=(DevBuf&&) = delete;
};

void h2d(float* dst, const float* src, std::size_t elems, const char* what) {
    check(cudaMemcpy(dst, src, elems * sizeof(float), cudaMemcpyHostToDevice), what);
}

}  // namespace

void forward_naive_gpu(const Weights& w,
                       const float* h_input,
                       float* h_output,
                       int N) {
    if (N <= 0) return;

    // --- weight buffers --------------------------------------------------
    DevBuf d_conv1_w(w.conv1_weight.size());
    DevBuf d_conv1_b(w.conv1_bias.size());
    DevBuf d_conv2_w(w.conv2_weight.size());
    DevBuf d_conv2_b(w.conv2_bias.size());
    DevBuf d_fc1_w(w.fc1_weight.size());
    DevBuf d_fc1_b(w.fc1_bias.size());
    DevBuf d_fc2_w(w.fc2_weight.size());
    DevBuf d_fc2_b(w.fc2_bias.size());

    h2d(d_conv1_w.ptr, w.conv1_weight.data(), w.conv1_weight.size(), "H2D conv1_w");
    h2d(d_conv1_b.ptr, w.conv1_bias.data(),   w.conv1_bias.size(),   "H2D conv1_b");
    h2d(d_conv2_w.ptr, w.conv2_weight.data(), w.conv2_weight.size(), "H2D conv2_w");
    h2d(d_conv2_b.ptr, w.conv2_bias.data(),   w.conv2_bias.size(),   "H2D conv2_b");
    h2d(d_fc1_w.ptr,   w.fc1_weight.data(),   w.fc1_weight.size(),   "H2D fc1_w");
    h2d(d_fc1_b.ptr,   w.fc1_bias.data(),     w.fc1_bias.size(),     "H2D fc1_b");
    h2d(d_fc2_w.ptr,   w.fc2_weight.data(),   w.fc2_weight.size(),   "H2D fc2_w");
    h2d(d_fc2_b.ptr,   w.fc2_bias.data(),     w.fc2_bias.size(),     "H2D fc2_b");

    // --- activation buffers (one per layer output, sized for batch N) ----
    const std::size_t sN = static_cast<std::size_t>(N);
    DevBuf d_input(sN * 1 * 28 * 28);
    DevBuf d_c1   (sN * 10 * 24 * 24);
    DevBuf d_p1   (sN * 10 * 12 * 12);
    DevBuf d_c2   (sN * 20 *  8 *  8);
    DevBuf d_p2   (sN * 20 *  4 *  4);   // logically [N, 320] after this
    DevBuf d_h1   (sN * 50);
    DevBuf d_h2   (sN * 10);
    DevBuf d_out  (sN * 10);

    h2d(d_input.ptr, h_input, d_input.elems, "H2D input");

    // --- layer pipeline (mirrors forward_ref exactly) --------------------
    conv2d_naive_device(d_input.ptr, d_conv1_w.ptr, d_conv1_b.ptr, d_c1.ptr,
                        N, 1, 28, 28, 10, 5, 5);
    max_pool2d_naive_device(d_c1.ptr, d_p1.ptr, N, 10, 24, 24);
    relu_naive_device(d_p1.ptr, d_p1.elems);

    conv2d_naive_device(d_p1.ptr, d_conv2_w.ptr, d_conv2_b.ptr, d_c2.ptr,
                        N, 10, 12, 12, 20, 5, 5);
    max_pool2d_naive_device(d_c2.ptr, d_p2.ptr, N, 20, 8, 8);
    relu_naive_device(d_p2.ptr, d_p2.elems);

    // Flatten is a no-op: d_p2 is already [N, 320] contiguous.
    linear_naive_device(d_p2.ptr, d_fc1_w.ptr, d_fc1_b.ptr, d_h1.ptr,
                        N, 320, 50);
    relu_naive_device(d_h1.ptr, d_h1.elems);

    linear_naive_device(d_h1.ptr, d_fc2_w.ptr, d_fc2_b.ptr, d_h2.ptr,
                        N, 50, 10);
    log_softmax_naive_device(d_h2.ptr, d_out.ptr, N, 10);

    // Single sync at the end of the pipeline.
    check(cudaDeviceSynchronize(), "forward_naive_gpu pipeline");

    check(cudaMemcpy(h_output, d_out.ptr, d_out.elems * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");
}

void forward_tiled_gpu(const Weights& w,
                       const float* h_input,
                       float* h_output,
                       int N) {
    if (N <= 0) return;

    // Buffer setup is identical to forward_naive_gpu; only the conv
    // and linear kernel calls differ. Kept as a separate function
    // (rather than a boolean parameter) so each variant reads
    // top-to-bottom as a straight pipeline.
    DevBuf d_conv1_w(w.conv1_weight.size());
    DevBuf d_conv1_b(w.conv1_bias.size());
    DevBuf d_conv2_w(w.conv2_weight.size());
    DevBuf d_conv2_b(w.conv2_bias.size());
    DevBuf d_fc1_w(w.fc1_weight.size());
    DevBuf d_fc1_b(w.fc1_bias.size());
    DevBuf d_fc2_w(w.fc2_weight.size());
    DevBuf d_fc2_b(w.fc2_bias.size());

    h2d(d_conv1_w.ptr, w.conv1_weight.data(), w.conv1_weight.size(), "H2D conv1_w");
    h2d(d_conv1_b.ptr, w.conv1_bias.data(),   w.conv1_bias.size(),   "H2D conv1_b");
    h2d(d_conv2_w.ptr, w.conv2_weight.data(), w.conv2_weight.size(), "H2D conv2_w");
    h2d(d_conv2_b.ptr, w.conv2_bias.data(),   w.conv2_bias.size(),   "H2D conv2_b");
    h2d(d_fc1_w.ptr,   w.fc1_weight.data(),   w.fc1_weight.size(),   "H2D fc1_w");
    h2d(d_fc1_b.ptr,   w.fc1_bias.data(),     w.fc1_bias.size(),     "H2D fc1_b");
    h2d(d_fc2_w.ptr,   w.fc2_weight.data(),   w.fc2_weight.size(),   "H2D fc2_w");
    h2d(d_fc2_b.ptr,   w.fc2_bias.data(),     w.fc2_bias.size(),     "H2D fc2_b");

    const std::size_t sN = static_cast<std::size_t>(N);
    DevBuf d_input(sN * 1 * 28 * 28);
    DevBuf d_c1   (sN * 10 * 24 * 24);
    DevBuf d_p1   (sN * 10 * 12 * 12);
    DevBuf d_c2   (sN * 20 *  8 *  8);
    DevBuf d_p2   (sN * 20 *  4 *  4);
    DevBuf d_h1   (sN * 50);
    DevBuf d_h2   (sN * 10);
    DevBuf d_out  (sN * 10);

    h2d(d_input.ptr, h_input, d_input.elems, "H2D input");

    conv2d_tiled_device(d_input.ptr, d_conv1_w.ptr, d_conv1_b.ptr, d_c1.ptr,
                        N, 1, 28, 28, 10, 5, 5);
    max_pool2d_naive_device(d_c1.ptr, d_p1.ptr, N, 10, 24, 24);
    relu_naive_device(d_p1.ptr, d_p1.elems);

    conv2d_tiled_device(d_p1.ptr, d_conv2_w.ptr, d_conv2_b.ptr, d_c2.ptr,
                        N, 10, 12, 12, 20, 5, 5);
    max_pool2d_naive_device(d_c2.ptr, d_p2.ptr, N, 20, 8, 8);
    relu_naive_device(d_p2.ptr, d_p2.elems);

    linear_tiled_device(d_p2.ptr, d_fc1_w.ptr, d_fc1_b.ptr, d_h1.ptr,
                        N, 320, 50);
    relu_naive_device(d_h1.ptr, d_h1.elems);

    linear_tiled_device(d_h1.ptr, d_fc2_w.ptr, d_fc2_b.ptr, d_h2.ptr,
                        N, 50, 10);
    log_softmax_naive_device(d_h2.ptr, d_out.ptr, N, 10);

    check(cudaDeviceSynchronize(), "forward_tiled_gpu pipeline");

    check(cudaMemcpy(h_output, d_out.ptr, d_out.elems * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(output)");
}

}  // namespace cuda
}  // namespace cnn
