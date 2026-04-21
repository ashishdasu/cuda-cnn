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
