# Build configuration for the from-scratch CUDA CNN inference engine.
#
# Structure:
#   host/      Host-side code: weight loader, CPU reference implementations.
#              Always built; does not depend on CUDA.
#   kernels/   CUDA kernels. Added target-by-target in later commits.
#   tests/     Unit tests. CPU-only tests always built; CUDA tests
#              conditionally built when nvcc is available.
#
# CUDA is detected automatically. If nvcc is not found (e.g. on macOS
# development hosts), the CPU reference library and its tests still
# build; CUDA targets are skipped. This lets us iterate on correctness
# locally and only move to a GPU box when kernels are ready to compile.

cmake_minimum_required(VERSION 3.18)
project(cuda_cnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # T4=75, A100=80, RTX30=86, RTX40/L4=89. Covers both the proposal's
        # original T4 target and the RunPod RTX 4090 used for final runs.
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
    message(STATUS "CUDA available (${CMAKE_CUDA_COMPILER}); CUDA targets will be built")
else()
    message(STATUS "CUDA not available; building CPU-only targets")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)
if(CMAKE_CUDA_COMPILER)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

include_directories(${CMAKE_SOURCE_DIR})

# --- CPU reference library ------------------------------------------------
# Host-only. The oracle every CUDA kernel will be validated against.
add_library(cnn_host STATIC
    host/weights.cpp
    host/reference.cpp
)
set_target_properties(cnn_host PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Unit tests -----------------------------------------------------------
enable_testing()

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE cnn_host)
# Working directory is the repo root so relative paths to weights/ and
# fixtures/ resolve regardless of where the build lives.
add_test(NAME test_reference
         COMMAND test_reference
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- CUDA kernels ---------------------------------------------------------
# Everything below is compiled only when nvcc is available. On macOS
# dev hosts this block is skipped entirely so the CPU reference still
# builds and tests cleanly.
if(CMAKE_CUDA_COMPILER)
    add_library(cnn_kernels STATIC
        kernels/conv2d_naive.cu
        kernels/max_pool2d_naive.cu
        kernels/relu_naive.cu
        kernels/linear_naive.cu
        kernels/log_softmax_naive.cu
        kernels/forward.cu
        kernels/conv2d_tiled.cu
        kernels/linear_tiled.cu
    )
    set_target_properties(cnn_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON)

    # Per-kernel correctness test: diffs the GPU output against the
    # host reference (cnn_host) under the project's 1e-4 tolerance.
    add_executable(test_conv2d_naive tests/test_conv2d_naive.cu)
    target_link_libraries(test_conv2d_naive PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_conv2d_naive
             COMMAND test_conv2d_naive
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_max_pool2d_naive tests/test_max_pool2d_naive.cu)
    target_link_libraries(test_max_pool2d_naive PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_max_pool2d_naive
             COMMAND test_max_pool2d_naive
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_relu_naive tests/test_relu_naive.cu)
    target_link_libraries(test_relu_naive PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_relu_naive
             COMMAND test_relu_naive
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_linear_naive tests/test_linear_naive.cu)
    target_link_libraries(test_linear_naive PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_linear_naive
             COMMAND test_linear_naive
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_log_softmax_naive tests/test_log_softmax_naive.cu)
    target_link_libraries(test_log_softmax_naive PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_log_softmax_naive
             COMMAND test_log_softmax_naive
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_conv2d_tiled tests/test_conv2d_tiled.cu)
    target_link_libraries(test_conv2d_tiled PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_conv2d_tiled
             COMMAND test_conv2d_tiled
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(test_linear_tiled tests/test_linear_tiled.cu)
    target_link_libraries(test_linear_tiled PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_linear_tiled
             COMMAND test_linear_tiled
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # End-to-end: compose all five naive kernels through device-pointer
    # launchers and validate the full pipeline vs PyTorch fixtures.
    # Passing this is the minimum-tier correctness milestone.
    add_executable(test_forward_gpu tests/test_forward_gpu.cu)
    target_link_libraries(test_forward_gpu PRIVATE cnn_kernels cnn_host)
    add_test(NAME test_forward_gpu
             COMMAND test_forward_gpu
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
