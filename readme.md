# cuda-cnn

A from-scratch CUDA forward-pass inference engine for a LeNet-style MNIST CNN.
Every layer — 2D convolution, max pooling, ReLU, fully-connected (GEMM), and
log-softmax — is implemented as a hand-written CUDA kernel. No PyTorch,
no cuDNN, no deep learning framework at runtime.

CS5330 Pattern Recognition & Computer Vision, final project — Ashish Dasu.

## PRESENTATION VIDEO LINK: https://drive.google.com/file/d/1DaHe_1kZR2vl7LqqHog4ZXsbRiAndUjT/view?usp=sharing

## Project description

I use PyTorch throughout the course without a concrete sense of what
`model.to("cuda")` actually does. This project answers that question by
replacing every layer operation in the Project 5 LeNet-MNIST network with a
hand-written CUDA kernel. Weights are exported once from the trained PyTorch
model as raw float32 binaries; at inference time the engine links only against
the CUDA runtime. The pipeline is implemented in two variants — naive kernels
for the minimum tier and tiled shared-memory kernels for the target tier — and
compared to a cuDNN baseline for the stretch tier. Correctness is validated
against PyTorch log-softmax on all 10,000 MNIST test images with max absolute
error 1.91e-5, 100% argmax agreement, and accuracy (97.93%) identical to
PyTorch's eval-mode accuracy on the same weights.

## Build & run

Requires an NVIDIA GPU with CUDA Toolkit ≥ 12.0. All development and
benchmarking was done on a RunPod RTX 4090 instance (CUDA 12.4, driver
550.127.05). cuDNN is optional — the cuDNN conv wrapper is only built when
it is detected by CMake.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build       # per-kernel + end-to-end correctness tests
build/bench                  # throughput numbers (add --csv to pipe to plot_bench.py)
build/predict --fixture 0    # single-image inference
```

## Reproducing the report numbers and figures

Weights in `weights/` are checked in (float32 binaries exported from the
Project 5 PyTorch checkpoint via `tools/export_weights.py`; regeneration
requires the original `.pth`). Fixtures in `fixtures/` (the five small
ones used by the unit tests) are also checked in.

Python side (weight export, fixture dump, figure plotting) needs
`torch`, `torchvision`, `numpy`, `matplotlib`. CUDA side needs CUDA
Toolkit ≥ 12.0 and, for the stretch tier, `cuDNN`
(`sudo apt install libcudnn9-dev-cuda-12` on the RunPod instance used
for all numbers in the report).

```bash
build/bench --csv > bench.csv          # produces the data in report/figs/
python3 tools/plot_bench.py bench.csv report/figs/
cd report && pdflatex report.tex && pdflatex report.tex
```

## Repository layout

```
kernels/      CUDA kernels — naive + tiled conv2d / linear, pool, ReLU,
              log_softmax, and the cuDNN wrapper. Per-kernel .h/.cu pairs.
host/         Host-side driver, weight loader, CPU reference implementations
tests/        Per-kernel unit tests (diffed against CPU reference at 1e-4)
              and the end-to-end forward-pass parity harness
tools/        Python: weight export, fixture dump, figure plotting.
              CUDA:   throughput microbenchmark (tools/bench.cu).
weights/      Exported weight binaries (float32, little-endian, PyTorch layout)
fixtures/     MNIST inputs + PyTorch log-softmax oracles for correctness tests
report/       Final IEEE-format report (report.pdf is the submitted artifact)
```

## Report

The final report is at `report/report.pdf` (IEEE 2-column conference format,
built from `report/report.tex`). 

## Author

Ashish Dasu (`dasu.a@northeastern.edu`) — solo project.

Huge thanks to Prof. Bruce Maxwell for designing and teaching CS5330 Pattern
Recognition and Computer Vision — the course that prompted this project and provided all the educational background.

## Reflection

The biggest lesson from this project was that "the optimization" and
"the speedup" are separate things. Shared-memory tiling and kernel
fusion are both textbook-correct ideas, and at LeNet's shapes both
empirically made the network slower — tiled conv regressed against
naive, and the fused conv+ReLU+pool kernel was 2.5× slower on conv2
because the 4×4 output grid under-subscribed the GPU. Writing a CPU
reference for every kernel first and keeping end-to-end PyTorch parity
green on every commit meant I could chase throughput without ever
debugging "the logits drifted" — the correctness oracle chain paid for
itself many times over.

## AI assistance disclosure

Anthropic's Claude Code CLI was used as a development aid throughout this
project. Specific uses include: code scaffolding and repository structure,
kernel comment formatting, debugging assistance, rapid design prototyping,
DevOps configuration for the RunPod compute environment, figure plotting with
matplotlib, Beamer slide templating, and report prose editing for academic
style and clarity. All architectural decisions, kernel designs, correctness
methodology, benchmark measurements, and the analysis in the report are my
own; every line of CUDA was read, understood, and edited by me before it was
committed.
