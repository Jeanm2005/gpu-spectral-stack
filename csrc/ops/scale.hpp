#pragma once
#include "common/types.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace spectral {
    void scale_c2c(const cf32* x, cf32* y, std::size_t n, float alpha, cudaStream_t stream);
}