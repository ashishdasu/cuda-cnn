// See conv2d_cudnn.h for the contract. cuDNN owns all of the algorithm
// selection, workspace sizing, and actual kernel dispatch.

#include "kernels/conv2d_cudnn.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace cnn {
namespace cuda {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void check_cudnn(cudnnStatus_t s, const char* what) {
    if (s != CUDNN_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuDNN " << what << " failed: " << cudnnGetErrorString(s);
        throw std::runtime_error(oss.str());
    }
}

void launch(cudnnHandle_t handle,
            const float* d_input, const float* d_weight, const float* d_bias,
            float* d_output,
            int N, int C_in, int H_in, int W_in,
            int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;

    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc), "createTensorDesc(x)");
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc), "createTensorDesc(y)");
    check_cudnn(cudnnCreateTensorDescriptor(&b_desc), "createTensorDesc(b)");
    check_cudnn(cudnnCreateFilterDescriptor(&w_desc), "createFilterDesc");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "createConvDesc");

    // PyTorch-native layout: NCHW, float32.
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C_in, H_in, W_in), "setTensor x");
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C_out, H_out, W_out), "setTensor y");
    check_cudnn(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, C_out, 1, 1), "setTensor b");
    check_cudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           C_out, C_in, Kh, Kw), "setFilter");
    // Stride 1, no padding, dilation 1 — matches the naive/tiled kernels.
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                0, 0,   // pad
                                                1, 1,   // stride
                                                1, 1,   // dilation
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT), "setConv2d");
    // Force real FP32 FMA, not TF32. On Ampere/Ada GPUs cuDNN defaults
    // to TF32 for FP32 convolutions, which uses a 10-bit mantissa and
    // introduces ~1e-3 absolute error — above the 1e-4 project tolerance
    // and inconsistent with the CPU reference and the naive/tiled kernels.
    check_cudnn(cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH),
                "setConvMathType(FMA)");

    // Pin to IMPLICIT_PRECOMP_GEMM: direct convolution reformulated as an
    // implicit GEMM, same numerics as the naive/tiled direct kernels. Without
    // this, cuDNN auto-selects WINOGRAD_NONFUSED for the larger conv2 shape,
    // which has ~1e-3 absolute error relative to direct convolution and
    // exceeds the 1e-4 tolerance. Matching algorithm = matching numerics.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    std::size_t ws_bytes = 0;
    check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
                    handle, x_desc, w_desc, conv_desc, y_desc,
                    algo, &ws_bytes), "getFwdWorkspaceSize");

    void* d_ws = nullptr;
    if (ws_bytes > 0) {
        check_cuda(cudaMalloc(&d_ws, ws_bytes), "cudaMalloc(workspace)");
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    check_cudnn(cudnnConvolutionForward(handle,
                                        &alpha, x_desc, d_input,
                                        w_desc, d_weight,
                                        conv_desc, algo,
                                        d_ws, ws_bytes,
                                        &beta, y_desc, d_output),
                "convolutionForward");
    // Bias add: y = y + b, broadcast over N/H/W.
    const float one = 1.0f;
    check_cudnn(cudnnAddTensor(handle, &one, b_desc, d_bias,
                               &one, y_desc, d_output), "addTensor(bias)");

    if (d_ws) cudaFree(d_ws);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(x_desc);
}

}  // namespace

void conv2d_cudnn_device_with_handle(void* handle_raw,
                                     const float* d_input,
                                     const float* d_weight,
                                     const float* d_bias,
                                     float* d_output,
                                     int N,
                                     int C_in, int H_in, int W_in,
                                     int C_out, int Kh, int Kw) {
    auto handle = static_cast<cudnnHandle_t>(handle_raw);
    launch(handle, d_input, d_weight, d_bias, d_output,
           N, C_in, H_in, W_in, C_out, Kh, Kw);
}

void conv2d_cudnn_device(const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         int N,
                         int C_in, int H_in, int W_in,
                         int C_out, int Kh, int Kw) {
    cudnnHandle_t handle;
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    launch(handle, d_input, d_weight, d_bias, d_output,
           N, C_in, H_in, W_in, C_out, Kh, Kw);
    cudnnDestroy(handle);
}

void conv2d_cudnn(const float* input,
                  const float* weight,
                  const float* bias,
                  float* output,
                  int N,
                  int C_in, int H_in, int W_in,
                  int C_out, int Kh, int Kw) {
    const int H_out = H_in - Kh + 1;
    const int W_out = W_in - Kw + 1;

    const std::size_t in_n  = static_cast<std::size_t>(N) * C_in * H_in * W_in;
    const std::size_t w_n   = static_cast<std::size_t>(C_out) * C_in * Kh * Kw;
    const std::size_t b_n   = static_cast<std::size_t>(C_out);
    const std::size_t out_n = static_cast<std::size_t>(N) * C_out * H_out * W_out;

    float *d_in = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    check_cuda(cudaMalloc(&d_in,  in_n  * sizeof(float)), "cudaMalloc(in)");
    check_cuda(cudaMalloc(&d_w,   w_n   * sizeof(float)), "cudaMalloc(w)");
    check_cuda(cudaMalloc(&d_b,   b_n   * sizeof(float)), "cudaMalloc(b)");
    check_cuda(cudaMalloc(&d_out, out_n * sizeof(float)), "cudaMalloc(out)");
    check_cuda(cudaMemcpy(d_in, input,  in_n*sizeof(float), cudaMemcpyHostToDevice), "H2D(in)");
    check_cuda(cudaMemcpy(d_w,  weight, w_n *sizeof(float), cudaMemcpyHostToDevice), "H2D(w)");
    check_cuda(cudaMemcpy(d_b,  bias,   b_n *sizeof(float), cudaMemcpyHostToDevice), "H2D(b)");

    conv2d_cudnn_device(d_in, d_w, d_b, d_out,
                        N, C_in, H_in, W_in, C_out, Kh, Kw);
    check_cuda(cudaDeviceSynchronize(), "cuDNN conv sync");

    check_cuda(cudaMemcpy(output, d_out, out_n*sizeof(float), cudaMemcpyDeviceToHost), "D2H(out)");
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}

}  // namespace cuda
}  // namespace cnn
