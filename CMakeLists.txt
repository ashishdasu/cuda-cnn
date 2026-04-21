# Build configuration for the from-scratch CUDA CNN inference engine.
#
# Layout:
#   kernels/   CUDA kernels (conv, gemm, pool, activations)
#   host/      Host-side driver, weight loader, CPU reference implementations
#   tests/     Per-kernel unit tests (compare CUDA output vs CPU reference)
#
# This project is developed on macOS (no CUDA) and compiled/run on a remote
# Linux + NVIDIA GPU host. Local `cmake` configuration is not expected to
# succeed; run cmake on the GPU host only.

cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_cnn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate cubin for common recent NVIDIA architectures. T4 is sm_75,
# RTX 30-series is sm_86, RTX 40-series is sm_89, A100 is sm_80.
# If the target device isn't listed, append its compute capability here.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Release-by-default: these are throughput experiments, not debugging sessions.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Warn aggressively on host code. Kernels get --expt-relaxed-constexpr so that
# std::numeric_limits helpers work inside __device__ functions.
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

include_directories(${CMAKE_SOURCE_DIR})

# --- Core library: kernels + host-side glue ---------------------------------
add_library(cnn_core STATIC
    kernels/conv.cu
    kernels/gemm.cu
    kernels/pool.cu
    kernels/activations.cu
    host/weights.cpp
    host/reference.cpp
    host/model.cpp
)
set_target_properties(cnn_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# --- Inference driver -------------------------------------------------------
add_executable(infer host/main.cpp)
target_link_libraries(infer PRIVATE cnn_core)

# --- Unit tests -------------------------------------------------------------
enable_testing()

foreach(test_name IN ITEMS test_conv test_gemm test_pool test_activations test_forward)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE cnn_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
