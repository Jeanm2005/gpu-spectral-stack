#pragma once
#include "common/types.hpp"
#include <cuda_runtime.h>

namespace spectral::fft {
    enum class Direction { Forward, Inverse };

    void c2c_naive(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);
}