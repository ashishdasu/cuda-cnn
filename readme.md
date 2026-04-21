# cuda-cnn

A from-scratch CUDA forward-pass inference engine for a LeNet-style MNIST CNN.
Every layer — 2D convolution, max pooling, ReLU, fully-connected (GEMM), and
log-softmax — is implemented as a hand-written CUDA kernel. No PyTorch,
no cuDNN, no deep learning framework at runtime.

CS5330 Pattern Recognition & Computer Vision, final project — Ashish Dasu.

## Project description

I use PyTorch throughout the course without a concrete sense of what
`model.to("cuda")` actually does. This project answers that question by
replacing every layer operation in the Project 5 LeNet-MNIST network with a
hand-written CUDA kernel. Weights are exported once from the trained PyTorch
model as raw float32 binaries; at inference time the engine links only against
the CUDA runtime. Correctness is validated against PyTorch's log-softmax
outputs to within 1e-4 absolute tolerance on the full MNIST test set.
Performance is measured with Nsight Compute and compared to cuDNN as an
upper-bound reference.

## Build & run

This project builds on Linux with the NVIDIA CUDA Toolkit installed.
It does **not** build on macOS (no `nvcc`).

```bash
cmake -B build -S .
cmake --build build -j
./build/infer                # runs forward pass on a sample input
ctest --test-dir build       # runs per-kernel unit tests
```

## Repository layout

```
kernels/      CUDA kernels (conv, gemm, pool, activations)
host/         Host-side driver, weight loader, CPU reference implementations
tests/        Per-kernel unit tests — each CUDA kernel's output is compared
              against a CPU reference implementation on small fixtures
tools/        Python tool for exporting weights from Project 5's .pth checkpoint
weights/      Exported weight binaries (float32, little-endian)
fixtures/     Small test-input / expected-output binaries for unit tests
report/       Final IEEE-format report
```

## Demo video

TODO — URL will appear here once uploaded.

## Group members

Solo project: Ashish Dasu.
