#include "ops/scale.hpp"
#include "common/check.hpp"
#include <algorithm>

namespace spectral{
    namespace{
        __global__ void scale_kernel(const cf32* x, cf32* y, std::size_t n, float alpha)
        {
            const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
                const cf32 v = x[i];
                y[i] = cf32{alpha * v.re, alpha * v.im};
            }
        } 
    } // namespace

    void scale_c2c(const cf32* x, cf32* y, std::size_t n, float alpha, cudaStream_t stream)
    {
        if (n == 0) {
            return;
        }
        constexpr int block = 256;
        constexpr std::size_t max_blocks = 65535;
        const auto blocks = static_cast<unsigned>(std::min((n + block - 1) / block, max_blocks));
        scale_kernel<<<blocks, block, 0, stream>>>(x, y, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }
} // namespace spectral