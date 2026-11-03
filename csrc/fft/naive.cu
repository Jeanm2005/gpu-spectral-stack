#include "fft/fft.hpp"

#include "common/check.hpp"
#include "ops/scale.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace spectral::fft {
namespace {

constexpr int kBlock = 256;
constexpr std::size_t kMaxBlocks = 65535;  // the grid-stride loops cover any remainder

unsigned blocks_for(std::size_t threads)
{
    return static_cast<unsigned>(std::min((threads + kBlock - 1) / kBlock, kMaxBlocks));
}

// y[t] = x[base + reverse(i)] for every element t = base + i, where base is the start of t's
// transform and reverse() mirrors the low log2n bits of i. A gather: the writes are coalesced
// (neighbouring threads write neighbouring elements), the reads are scattered.
__global__ void bit_reverse_kernel(const cf32* __restrict__ x, cf32* __restrict__ y,
                                   std::size_t total, int log2n)
{
    const std::size_t mask = (std::size_t{1} << log2n) - 1;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        const std::size_t base = t & ~mask;  // n is a power of two, so this rounds down
        const auto i = static_cast<unsigned>(t & mask);
        const unsigned r = __brev(i) >> (32 - log2n);  // __brev mirrors all 32 bits
        y[t] = x[base + r];
    }
}

// One radix-2 stage over every transform in the batch, in place. Blocks of length `half` are
// combined into blocks of length 2 * half; one thread per butterfly. Butterfly t belongs to
// group g = t / half (which covers elements [2 * half * g, 2 * half * (g + 1))) at position
// k = t % half within it. Groups never straddle two transforms, because 2 * half divides n.
__global__ void radix2_stage_kernel(cf32* y, std::size_t butterflies, int log2half, float sign)
{
    const std::size_t half = std::size_t{1} << log2half;
    const float inv_half = 1.0f / static_cast<float>(half);  // exact: half is a power of two
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < butterflies; t += stride) {
        const std::size_t k = t & (half - 1);
        const std::size_t i0 = ((t >> log2half) << (log2half + 1)) | k;
        const std::size_t i1 = i0 + half;

        // w^k = exp(sign * pi * i * k / half). sincospif(a) computes sin(pi a) and cos(pi a)
        // without first rounding pi, and k / half is exact, so the twiddle is accurate to
        // about one ulp.
        float s = 0.0f;
        float c = 0.0f;
        sincospif(sign * static_cast<float>(k) * inv_half, &s, &c);

        const cf32 u = y[i0];
        const cf32 b = y[i1];
        const cf32 v{b.re * c - b.im * s, b.re * s + b.im * c};  // (c + i s)(b.re + i b.im)
        y[i0] = cf32{u.re + v.re, u.im + v.im};
        y[i1] = cf32{u.re - v.re, u.im - v.im};
    }
}

int log2_exact(int n)
{
    int log2n = 0;
    while ((1 << log2n) < n) {
        ++log2n;
    }
    return log2n;
}

void check_arguments(const cf32* x, const cf32* y, int n, int batch)
{
    if (n < 1 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("fft: n must be a power of two, got " + std::to_string(n));
    }
    if (batch < 1) {
        throw std::invalid_argument("fft: batch must be at least 1, got "
                                    + std::to_string(batch));
    }
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    if (x < y + total && y < x + total) {
        throw std::invalid_argument("fft: x and y must not overlap (out-of-place only)");
    }
}

}  // namespace

void c2c_naive(const cf32* x, cf32* y, int n, int batch, Direction direction,
               cudaStream_t stream)
{
    check_arguments(x, y, n, batch);
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);

    if (n == 1) {  // a 1-point DFT is the identity, in both directions
        CUDA_CHECK(cudaMemcpyAsync(y, x, total * sizeof(cf32), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const int log2n = log2_exact(n);
    bit_reverse_kernel<<<blocks_for(total), kBlock, 0, stream>>>(x, y, total, log2n);
    CUDA_CHECK(cudaGetLastError());

    const float sign = direction == Direction::Forward ? -1.0f : 1.0f;
    for (int s = 0; s < log2n; ++s) {
        radix2_stage_kernel<<<blocks_for(total / 2), kBlock, 0, stream>>>(y, total / 2, s, sign);
        CUDA_CHECK(cudaGetLastError());
    }

    if (direction == Direction::Inverse) {
        // 1/n is a power of two, so this scaling is exact: it adds no rounding error.
        scale_c2c(y, y, total, 1.0f / static_cast<float>(n), stream);
    }
}

}  // namespace spectral::fft