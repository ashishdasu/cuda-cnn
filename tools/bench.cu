// Throughput benchmark for the naive, tiled, and (when available) cuDNN
// kernels plus the composed forward pass. Times kernels with cudaEvent
// amortized over many launches after a warmup. Pretty-prints to stdout
// by default; pass --csv to emit a CSV-only stream suitable for piping
// into a plotting script.
//
// CSV columns (header emitted as the first row):
//   category,shape,variant,time_ms,speedup_vs_naive
//
// Categories:
//   conv  — single 2D convolution forward
//   lin   — single linear / GEMM forward
//   fwd   — composed LeNet forward pass (H2D + all 10 kernels + D2H)
//   sweep — GEMM scaling sweep over N at fixed 1024x1024 problem
//
// Variants: naive, tiled, cudnn.

#include "host/weights.h"
#include "kernels/conv2d_naive.h"
#include "kernels/conv2d_tiled.h"
#include "kernels/forward.h"
#include "kernels/linear_naive.h"
#include "kernels/linear_tiled.h"

#ifdef CNN_HAS_CUDNN
#include "kernels/conv2d_cudnn.h"
#include <cudnn.h>
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool g_csv = false;

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#ifdef CNN_HAS_CUDNN
inline void check_cudnn(cudnnStatus_t s, const char* what) {
    if (s != CUDNN_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuDNN " << what << " failed: " << cudnnGetErrorString(s);
        throw std::runtime_error(oss.str());
    }
}
#endif

template <typename F>
double time_ms(F&& launch, int reps, int warmup = 5) {
    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "eventCreate(start)");
    check(cudaEventCreate(&stop),  "eventCreate(stop)");
    check(cudaEventRecord(start), "eventRecord(start)");
    for (int i = 0; i < reps; ++i) launch();
    check(cudaEventRecord(stop), "eventRecord(stop)");
    check(cudaEventSynchronize(stop), "eventSynchronize(stop)");

    float total_ms = 0.0f;
    check(cudaEventElapsedTime(&total_ms, start, stop), "eventElapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(total_ms) / reps;
}

void fill_rand(float* d_buf, std::size_t n, std::uint32_t seed) {
    std::vector<float> h(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h) x = dist(rng);
    check(cudaMemcpy(d_buf, h.data(), n * sizeof(float), cudaMemcpyHostToDevice),
          "H2D rand fill");
}

void emit_row(const char* category, const char* shape, const char* variant,
              double time_ms_val, double speedup) {
    if (g_csv) {
        // Shape strings contain commas (e.g., "N=1, 1->10, 5x5 @ 28x28"),
        // so they must be quoted for round-trip CSV parsing.
        std::printf("%s,\"%s\",%s,%.6f,%.4f\n", category, shape, variant,
                    time_ms_val, speedup);
    } else {
        std::printf("%-6s | %-42s | %-6s %8.4f ms | speedup %5.2fx\n",
                    category, shape, variant, time_ms_val, speedup);
    }
}

struct ConvShape {
    const char* name;
    int N, C_in, H_in, W_in, C_out, Kh, Kw;
};

struct LinShape {
    const char* name;
    int N, in_f, out_f;
};

#ifdef CNN_HAS_CUDNN
void bench_conv(const ConvShape& s, int reps, cudnnHandle_t cudnn_handle) {
#else
void bench_conv(const ConvShape& s, int reps) {
#endif
    const int H_out = s.H_in - s.Kh + 1;
    const int W_out = s.W_in - s.Kw + 1;
    const std::size_t in_n  = static_cast<std::size_t>(s.N) * s.C_in * s.H_in * s.W_in;
    const std::size_t w_n   = static_cast<std::size_t>(s.C_out) * s.C_in * s.Kh * s.Kw;
    const std::size_t b_n   = static_cast<std::size_t>(s.C_out);
    const std::size_t out_n = static_cast<std::size_t>(s.N) * s.C_out * H_out * W_out;

    float *d_in = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in,  in_n  * sizeof(float)), "malloc in");
    check(cudaMalloc(&d_w,   w_n   * sizeof(float)), "malloc w");
    check(cudaMalloc(&d_b,   b_n   * sizeof(float)), "malloc b");
    check(cudaMalloc(&d_out, out_n * sizeof(float)), "malloc out");
    fill_rand(d_in, in_n, 1);
    fill_rand(d_w,  w_n,  2);
    fill_rand(d_b,  b_n,  3);

    const double t_naive = time_ms([&]() {
        cnn::cuda::conv2d_naive_device(d_in, d_w, d_b, d_out,
                                       s.N, s.C_in, s.H_in, s.W_in,
                                       s.C_out, s.Kh, s.Kw);
    }, reps);
    const double t_tiled = time_ms([&]() {
        cnn::cuda::conv2d_tiled_device(d_in, d_w, d_b, d_out,
                                       s.N, s.C_in, s.H_in, s.W_in,
                                       s.C_out, s.Kh, s.Kw);
    }, reps);

    emit_row("conv", s.name, "naive", t_naive, 1.0);
    emit_row("conv", s.name, "tiled", t_tiled, t_naive / t_tiled);

#ifdef CNN_HAS_CUDNN
    const double t_cudnn = time_ms([&]() {
        cnn::cuda::conv2d_cudnn_device_with_handle(
            static_cast<void*>(cudnn_handle),
            d_in, d_w, d_b, d_out,
            s.N, s.C_in, s.H_in, s.W_in,
            s.C_out, s.Kh, s.Kw);
    }, reps);
    emit_row("conv", s.name, "cudnn", t_cudnn, t_naive / t_cudnn);
#endif

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}

void bench_linear(const LinShape& s, int reps, const char* category = "lin") {
    const std::size_t in_n  = static_cast<std::size_t>(s.N) * s.in_f;
    const std::size_t w_n   = static_cast<std::size_t>(s.out_f) * s.in_f;
    const std::size_t b_n   = static_cast<std::size_t>(s.out_f);
    const std::size_t out_n = static_cast<std::size_t>(s.N) * s.out_f;

    float *d_in = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in,  in_n  * sizeof(float)), "malloc in");
    check(cudaMalloc(&d_w,   w_n   * sizeof(float)), "malloc w");
    check(cudaMalloc(&d_b,   b_n   * sizeof(float)), "malloc b");
    check(cudaMalloc(&d_out, out_n * sizeof(float)), "malloc out");
    fill_rand(d_in, in_n, 4);
    fill_rand(d_w,  w_n,  5);
    fill_rand(d_b,  b_n,  6);

    const double t_naive = time_ms([&]() {
        cnn::cuda::linear_naive_device(d_in, d_w, d_b, d_out,
                                       s.N, s.in_f, s.out_f);
    }, reps);
    const double t_tiled = time_ms([&]() {
        cnn::cuda::linear_tiled_device(d_in, d_w, d_b, d_out,
                                       s.N, s.in_f, s.out_f);
    }, reps);

    emit_row(category, s.name, "naive", t_naive, 1.0);
    emit_row(category, s.name, "tiled", t_tiled, t_naive / t_tiled);

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}

void bench_forward(const cnn::Weights& w, int N, int reps) {
    std::vector<float> h_in(static_cast<std::size_t>(N) * 1 * 28 * 28);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_in) x = dist(rng);
    std::vector<float> h_out(static_cast<std::size_t>(N) * 10);

    const double t_naive = time_ms([&]() {
        cnn::cuda::forward_naive_gpu(w, h_in.data(), h_out.data(), N);
    }, reps, /*warmup=*/2);
    const double t_tiled = time_ms([&]() {
        cnn::cuda::forward_tiled_gpu(w, h_in.data(), h_out.data(), N);
    }, reps, /*warmup=*/2);

    char label[64];
    std::snprintf(label, sizeof(label), "end-to-end forward (N=%d)", N);
    emit_row("fwd", label, "naive", t_naive, 1.0);
    emit_row("fwd", label, "tiled", t_tiled, t_naive / t_tiled);
}

// GEMM scaling sweep: fix in_f=out_f=1024, vary batch N. Produces the
// data for the speedup-vs-problem-size curve in the report/slides.
// At N=1 the tile mostly idles; at N=1024 the tiled kernel should be
// ~6.5x faster than naive.
void bench_gemm_sweep(int reps) {
    const int Ns[] = {1, 4, 16, 64, 256, 1024};
    const int in_f = 1024, out_f = 1024;
    for (int n : Ns) {
        char label[64];
        std::snprintf(label, sizeof(label),
                      "sweep N=%d (1024->1024)", n);
        LinShape s{label, n, in_f, out_f};
        bench_linear(s, reps, /*category=*/"sweep");
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0) g_csv = true;
    }

    try {
        int device = 0;
        check(cudaGetDevice(&device), "getDevice");
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, device), "getDeviceProperties");
        if (!g_csv) {
            std::printf("Device: %s (sm_%d%d, %d SMs, %zu MB global)\n",
                        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                        prop.totalGlobalMem / (1024 * 1024));
#ifdef CNN_HAS_CUDNN
            std::printf("cuDNN: enabled (CUDNN_FMA_MATH, IMPLICIT_PRECOMP_GEMM)\n");
#else
            std::printf("cuDNN: disabled (not found at build time)\n");
#endif
            std::printf("------------------------------------------------------------------------------------------------\n");
        } else {
            std::printf("category,shape,variant,time_ms,speedup_vs_naive\n");
        }

#ifdef CNN_HAS_CUDNN
        cudnnHandle_t cudnn_handle;
        check_cudnn(cudnnCreate(&cudnn_handle), "cudnnCreate");
#endif

        const ConvShape convs[] = {
            {"LeNet conv1 (N=1, 1->10, 5x5 @ 28x28)",     1,  1, 28, 28, 10, 5, 5},
            {"LeNet conv2 (N=1, 10->20, 5x5 @ 12x12)",    1, 10, 12, 12, 20, 5, 5},
            {"batch-64 conv1 (N=64, 1->10, 5x5 @ 28x28)", 64, 1, 28, 28, 10, 5, 5},
            {"batch-64 conv2 (N=64, 10->20, 5x5 @ 12x12)",64,10, 12, 12, 20, 5, 5},
            {"large synth (N=1, 32->64, 5x5 @ 64x64)",    1, 32, 64, 64, 64, 5, 5},
        };
        for (const auto& s : convs) {
#ifdef CNN_HAS_CUDNN
            bench_conv(s, /*reps=*/200, cudnn_handle);
#else
            bench_conv(s, /*reps=*/200);
#endif
        }

        const LinShape lins[] = {
            {"LeNet fc1 (N=1, 320->50)",    1,  320,   50},
            {"LeNet fc2 (N=1, 50->10)",     1,   50,   10},
            {"batch-64 fc1 (N=64, 320->50)", 64, 320,  50},
            {"large synth (N=1024, 1024->1024)", 1024, 1024, 1024},
        };
        for (const auto& s : lins) bench_linear(s, /*reps=*/200);

        bench_gemm_sweep(/*reps=*/100);

        const cnn::Weights w = cnn::load_weights("weights/");
        bench_forward(w, /*N=*/1,  /*reps=*/200);
        bench_forward(w, /*N=*/64, /*reps=*/50);

#ifdef CNN_HAS_CUDNN
        cudnnDestroy(cudnn_handle);
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
