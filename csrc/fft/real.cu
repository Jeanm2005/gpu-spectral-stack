// Real transforms around any complex implementation (Task 1.5): an n/2-point complex transform
// plus one pass that combines or prepares the half spectrum (the arithmetic is in real.cuh).
// Two passes over global memory instead of one: implementations with a fused kernel (the Stockham
// family, stockham.cuh) do the same steps in shared memory instead.

#include "fft/real.cuh"

#include "fft/fft.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace spectral::fft {
namespace real {
namespace {

constexpr int kBlock = 256;
constexpr std::size_t kMaxBlocks = 65535;  // the grid-stride loops cover any remainder

unsigned blocks_for(std::size_t threads)
{
    return static_cast<unsigned>(std::min((threads + kBlock - 1) / kBlock, kMaxBlocks));
}

// One thread per output: t = row, k = 0 .. keep-1.
__global__ void r2c_post_kernel(const cf32* __restrict__ z, cf32* __restrict__ y, int m,
                                int keep, std::size_t total)
{
    const auto row_len = static_cast<std::size_t>(keep);
    const float inv_m = 1.0f / static_cast<float>(m);  // exact: m = 2^p
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += stride) {
        const std::size_t t = e / row_len;
        const int k = static_cast<int>(e - t * row_len);
        const cf32* row = z + t * static_cast<std::size_t>(m);
        y[e] = r2c_combine(row[k & (m - 1)], row[(m - k) & (m - 1)],
                           static_cast<float>(k) * inv_m);
    }
}

// One thread per prepared value: t = row, k = 0 .. m-1. Rows of x hold `keep` inputs.
__global__ void c2r_pre_kernel(const cf32* __restrict__ x, cf32* __restrict__ z, int m,
                               int keep, std::size_t total)
{
    const auto row_len = static_cast<std::size_t>(keep);
    const float inv_m = 1.0f / static_cast<float>(m);
    const cf32 zero{0.0f, 0.0f};
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += stride) {
        const std::size_t t = e / static_cast<std::size_t>(m);
        const int k = static_cast<int>(e - t * static_cast<std::size_t>(m));
        const cf32* row = x + t * row_len;
        const cf32 xk = k < keep ? row[k] : zero;
        const cf32 xmk = m - k < keep ? row[m - k] : zero;
        z[e] = c2r_prepare(xk, xmk, k, static_cast<float>(k) * inv_m);
    }
}

}  // namespace

void r2c_postprocess(const cf32* z, cf32* y, int m, int batch, int keep, cudaStream_t stream)
{
    const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(keep);
    r2c_post_kernel<<<blocks_for(total), kBlock, 0, stream>>>(z, y, m, keep, total);
    CUDA_CHECK(cudaGetLastError());
}

void c2r_preprocess(const cf32* x, cf32* z, int m, int batch, int keep, cudaStream_t stream)
{
    const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(m);
    c2r_pre_kernel<<<blocks_for(total), kBlock, 0, stream>>>(x, z, m, keep, total);
    CUDA_CHECK(cudaGetLastError());
}

void check_arguments(const void* real_side, const void* complex_side, int n, int batch,
                     int max_n, const char* who)
{
    const std::string name(who);
    if (n < 2 || n > max_n || (n & (n - 1)) != 0) {
        throw std::invalid_argument(name + ": n must be a power of two from 2 to "
                                    + std::to_string(max_n) + ", got " + std::to_string(n));
    }
    if (batch < 1) {
        throw std::invalid_argument(name + ": batch must be at least 1, got "
                                    + std::to_string(batch));
    }
    const auto rows = static_cast<std::size_t>(batch);
    const auto* r = static_cast<const char*>(real_side);
    const auto* c = static_cast<const char*>(complex_side);
    const std::size_t real_bytes = rows * static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t complex_bytes = rows * (static_cast<std::size_t>(n) / 2 + 1) * sizeof(cf32);
    if (r < c + complex_bytes && c < r + real_bytes) {
        throw std::invalid_argument(name + ": input and output must not overlap");
    }
}

}  // namespace real

void r2c_via_c2c(C2cFn c2c, const float* x, cf32* y, int n, int batch, cf32* work,
                 cudaStream_t stream)
{
    real::check_arguments(x, y, n, batch, 1 << 30, "r2c");
    if (work == nullptr) {
        throw std::invalid_argument("r2c: this implementation needs a workspace");
    }
    const int m = n / 2;
    c2c(reinterpret_cast<const cf32*>(x), work, m, batch, Direction::Forward, stream);
    real::r2c_postprocess(work, y, m, batch, m + 1, stream);
}

void c2r_via_c2c(C2cFn c2c, const cf32* x, float* y, int n, int batch, cf32* work,
                 cudaStream_t stream)
{
    real::check_arguments(y, x, n, batch, 1 << 30, "c2r");
    if (work == nullptr) {
        throw std::invalid_argument("c2r: this implementation needs a workspace");
    }
    const int m = n / 2;
    real::c2r_preprocess(x, work, m, batch, m + 1, stream);
    c2c(work, reinterpret_cast<cf32*>(y), m, batch, Direction::Inverse, stream);
}

void r2c_naive(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream)
{
    r2c_via_c2c(&c2c_naive, x, y, n, batch, work, stream);
}

void c2r_naive(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream)
{
    c2r_via_c2c(&c2c_naive, x, y, n, batch, work, stream);
}

}  // namespace spectral::fft
