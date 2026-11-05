// Pruned transforms (Task 1.7); see pruned.hpp.

#include "fft/pruned.hpp"

#include "fft/fft2d.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <climits>
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

// Index in `full` (arrays of n1 x w) of mode e in `modes` (arrays of rows x m2, rows = 2 m1, or
// n1 when m1 = 0). One thread per mode, in the order of `modes`, so its accesses are coalesced.
__device__ __forceinline__ std::size_t full_index(std::size_t e, int n1, int w, int m1, int m2)
{
    const int rows = m1 > 0 ? 2 * m1 : n1;
    const std::size_t per_array = static_cast<std::size_t>(rows) * static_cast<std::size_t>(m2);
    const std::size_t b = e / per_array;
    const auto rem = static_cast<int>(e - b * per_array);
    const int r = rem / m2;
    const int c = rem - r * m2;
    const int row = (m1 > 0 && r >= m1) ? r + (n1 - rows) : r;  // negative frequencies
    return (b * static_cast<std::size_t>(n1) + static_cast<std::size_t>(row))
               * static_cast<std::size_t>(w)
           + static_cast<std::size_t>(c);
}

__global__ void gather_kernel(const cf32* __restrict__ full, cf32* __restrict__ modes, int n1,
                              int w, int m1, int m2, std::size_t total)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += stride) {
        modes[e] = full[full_index(e, n1, w, m1, m2)];
    }
}

__global__ void scatter_kernel(const cf32* __restrict__ modes, cf32* __restrict__ full, int n1,
                               int w, int m1, int m2, std::size_t total)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += stride) {
        full[full_index(e, n1, w, m1, m2)] = modes[e];
    }
}

std::size_t mode_count(int n1, int m1, int m2, int batch)
{
    const int rows = m1 > 0 ? 2 * m1 : n1;
    return static_cast<std::size_t>(batch) * static_cast<std::size_t>(rows)
           * static_cast<std::size_t>(m2);
}

std::size_t count(int n1, int n2, int batch)
{
    return static_cast<std::size_t>(batch) * static_cast<std::size_t>(n1)
           * static_cast<std::size_t>(n2);
}

bool is_pow2(int n) { return n >= 1 && (n & (n - 1)) == 0; }

void check_1d(const FftImpl& impl, int n, int keep, int batch)
{
    if (!is_pow2(n) || n > impl.max_n) {
        throw std::invalid_argument("fft_c2c_pruned: " + std::string(impl.name)
                                    + " needs n to be a power of two up to "
                                    + std::to_string(impl.max_n) + ", got " + std::to_string(n));
    }
    if (keep < 1 || keep > n) {
        throw std::invalid_argument("fft_c2c_pruned: keep must be in [1, " + std::to_string(n)
                                    + "], got " + std::to_string(keep));
    }
    if (batch < 1) {
        throw std::invalid_argument("fft_c2c_pruned: batch must be at least 1");
    }
}

void check_2d(const FftImpl& impl, int n1, int n2, int m1, int m2, int batch)
{
    const std::string name = impl.name;
    if (!is_pow2(n1) || n1 > impl.max_n) {
        throw std::invalid_argument("rfft2 modes: " + name + " needs n1 to be a power of two up to "
                                    + std::to_string(impl.max_n) + ", got "
                                    + std::to_string(n1));
    }
    if (!is_pow2(n2) || n2 < 2 || n2 > impl.max_real_n) {
        throw std::invalid_argument("rfft2 modes: " + name
                                    + " needs n2 to be a power of two from 2 up to "
                                    + std::to_string(impl.max_real_n) + ", got "
                                    + std::to_string(n2));
    }
    if (m1 < 1 || 2 * m1 > n1 || m2 < 1 || m2 > n2 / 2 + 1) {
        throw std::invalid_argument("rfft2 modes: need 1 <= m1 <= n1/2 and 1 <= m2 <= n2/2 + 1, "
                                    "got m1 = " + std::to_string(m1) + ", m2 = "
                                    + std::to_string(m2) + " for " + std::to_string(n1) + " x "
                                    + std::to_string(n2));
    }
    if (batch < 1 || static_cast<long long>(batch) * std::max(n1, n2) > INT_MAX) {
        throw std::invalid_argument("rfft2 modes: batch out of range");
    }
}

}  // namespace

void gather_modes(const cf32* full, cf32* modes, int n1, int w, int m1, int m2, int batch,
                  cudaStream_t stream)
{
    const std::size_t total = mode_count(n1, m1, m2, batch);
    gather_kernel<<<blocks_for(total), kBlock, 0, stream>>>(full, modes, n1, w, m1, m2, total);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_modes(const cf32* modes, cf32* full, int n1, int w, int m1, int m2, int batch,
                   cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(full, 0, count(n1, w, batch) * sizeof(cf32), stream));
    const std::size_t total = mode_count(n1, m1, m2, batch);
    scatter_kernel<<<blocks_for(total), kBlock, 0, stream>>>(modes, full, n1, w, m1, m2, total);
    CUDA_CHECK(cudaGetLastError());
}

// --- 1D complex --------------------------------------------------------------------------------

std::size_t fft_c2c_pruned_workspace(const FftImpl& impl, int n, int, int batch)
{
    return impl.pruned != nullptr ? 0 : count(1, n, batch);
}

void fft_c2c_pruned(const FftImpl& impl, const cf32* x, cf32* y, int n, int keep, int batch,
                    Direction direction, cf32* work, cudaStream_t stream)
{
    check_1d(impl, n, keep, batch);
    if (impl.pruned != nullptr) {
        impl.pruned->c2c(x, y, n, batch, keep, direction, stream);
        return;
    }
    // Each transform is an array of one row of n values, of which the first keep are modes.
    if (direction == Direction::Forward) {
        impl.c2c(x, work, n, batch, direction, stream);
        gather_modes(work, y, 1, n, 0, keep, batch, stream);
    } else {
        scatter_modes(x, work, 1, n, 0, keep, batch, stream);
        impl.c2c(work, y, n, batch, direction, stream);
    }
}

// --- 2D real -----------------------------------------------------------------------------------

std::size_t rfft2_modes_workspace(const FftImpl& impl, int n1, int n2, int, int m2, int batch)
{
    if (impl.pruned != nullptr) {
        return count(n1, m2, batch);  // the row transforms' m2 columns
    }
    return count(n1, n2 / 2 + 1, batch)
           + fft2_r2c_workspace(impl, Columns::Transpose, n1, n2, batch);
}

void rfft2_modes(const FftImpl& impl, const float* x, cf32* modes, int n1, int n2, int m1, int m2,
                 int batch, cf32* work, cudaStream_t stream)
{
    check_2d(impl, n1, n2, m1, m2, batch);
    if (impl.pruned != nullptr) {
        impl.pruned->r2c(x, work, n2, batch * n1, m2, stream);  // n1 x m2
        impl.pruned->columns(work, modes, n1, m2, batch, 0, m1, Direction::Forward, stream);
        return;
    }
    cf32* full = work;
    cf32* rest = work + count(n1, n2 / 2 + 1, batch);
    fft2_r2c(impl, Columns::Transpose, x, full, n1, n2, batch, rest, stream);
    gather_modes(full, modes, n1, n2 / 2 + 1, m1, m2, batch, stream);
}

std::size_t irfft2_modes_workspace(const FftImpl& impl, int n1, int n2, int, int m2, int batch)
{
    if (impl.pruned != nullptr) {
        return count(n1, m2, batch);
    }
    return count(n1, n2 / 2 + 1, batch)
           + fft2_c2r_workspace(impl, Columns::Transpose, n1, n2, batch);
}

void irfft2_modes(const FftImpl& impl, const cf32* modes, float* y, int n1, int n2, int m1, int m2,
                  int batch, cf32* work, cudaStream_t stream)
{
    check_2d(impl, n1, n2, m1, m2, batch);
    if (impl.pruned != nullptr) {
        impl.pruned->columns(modes, work, n1, m2, batch, m1, 0, Direction::Inverse, stream);
        impl.pruned->c2r(work, y, n2, batch * n1, m2, stream);  // columns m2.. are zero
        return;
    }
    cf32* full = work;
    cf32* rest = work + count(n1, n2 / 2 + 1, batch);
    scatter_modes(modes, full, n1, n2 / 2 + 1, m1, m2, batch, stream);
    fft2_c2r(impl, Columns::Transpose, full, y, n1, n2, batch, rest, stream);
}

}  // namespace spectral::fft
