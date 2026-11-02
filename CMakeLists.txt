cmake_minimum_required(VERSION 3.24)

# Compile device code for the GPU in this machine unless the caller picks architectures
# (CI has no GPU, so it passes them explicitly).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpu_spectral_stack LANGUAGES CXX CUDA)

# Timing is only meaningful in Release, so make it the default.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# Flags shared by every target: strict warnings for host C++, and line information in device
# code so Nsight Compute can map each stall back to a source line (no effect on speed).
set(SPECTRAL_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# The kernels and host helpers. The FFT kernels (csrc/fft/) join this library in Phase 1;
# the PyTorch extension (0.2) compiles the same sources.
add_library(spectral_core STATIC
  csrc/common/bench.cpp
  csrc/ops/scale.cu
)
target_include_directories(spectral_core PUBLIC csrc)
target_link_libraries(spectral_core PUBLIC CUDA::cudart CUDA::cufft)
target_compile_options(spectral_core PRIVATE ${SPECTRAL_COMPILE_OPTIONS})

# Toolchain smoke test: prints the GPU and checks that a kernel runs and its result returns.
add_executable(spectral_smoke csrc/tools/smoke.cu)
target_link_libraries(spectral_smoke PRIVATE spectral_core)
target_compile_options(spectral_smoke PRIVATE ${SPECTRAL_COMPILE_OPTIONS})