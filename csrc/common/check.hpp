#pragma once

// Error checking for every CUDA runtime and cuFFT call. Both macros throw std::runtime_error
// with the failing call, the library's error text, and the source location. main() catches
// it, prints it, and exits with status 1.
//
//   CUDA_CHECK(cudaMalloc(&ptr, bytes));
//   CUFFT_CHECK(cufftExecC2C(plan, in, out, CUFFT_FORWARD));
//
// A kernel launch returns nothing, so check it right after the launch:
//
//   my_kernel<<<grid, block>>>(args);
//   CUDA_CHECK(cudaGetLastError());

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <string>

inline const char* cufft_error_string(cufftResult result)
{
    switch (result) {
    case CUFFT_SUCCESS:         return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:    return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:    return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:    return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:   return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:  return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:     return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:    return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:    return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:  return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE:  return "CUFFT_INVALID_DEVICE";
    case CUFFT_NO_WORKSPACE:    return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:   return "CUFFT_NOT_SUPPORTED";
    default:                    return "unknown cuFFT error";
    }
}

[[noreturn]] inline void throw_gpu_error(const char* library, const char* text, int code,
                                         const char* call, const char* file, int line)
{
    throw std::runtime_error(std::string(library) + " error: " + text + " (code "
                             + std::to_string(code) + ")\n  in: " + call + "\n  at: " + file
                             + ":" + std::to_string(line));
}

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t cuda_check_err_ = (call);                                           \
        if (cuda_check_err_ != cudaSuccess) {                                                 \
            throw_gpu_error("CUDA", cudaGetErrorString(cuda_check_err_),                      \
                            static_cast<int>(cuda_check_err_), #call, __FILE__, __LINE__);    \
        }                                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                                     \
    do {                                                                                      \
        const cufftResult cufft_check_err_ = (call);                                          \
        if (cufft_check_err_ != CUFFT_SUCCESS) {                                              \
            throw_gpu_error("cuFFT", cufft_error_string(cufft_check_err_),                    \
                            static_cast<int>(cufft_check_err_), #call, __FILE__, __LINE__);   \
        }                                                                                     \
    } while (0)
