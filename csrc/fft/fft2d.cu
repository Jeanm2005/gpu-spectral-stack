// Two-dimensional FFTs from 1D transforms (Task 1.6); see fft2d.hpp.

#include "fft/fft2d.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace spectral::fft {
namespace {

constexpr int kTile = 32;
constexpr int kTileRows = 8;  // a 32 x 8 block moves a 32 x 32 tile in four steps

// Classic shared-memory transpose: a block reads a 32 x 32 tile with coalesced row reads and
// writes it back as coalesced rows of the transposed array. The extra column of padding keeps a
// warp's column reads of the tile out of a single bank.
__global__ void transpose_kernel(const cf32* __restrict__ x, cf32* __restrict__ y, int rows,
                                 int cols, int batch)
{
    __shared__ cf32 tile[kTile][kTile + 1];
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    for (int b = blockIdx.z; b < batch; b += gridDim.z) {
        const cf32* xs = x + static_cast<std::size_t>(b) * size;
        cf32* ys = y + static_cast<std::size_t>(b) * size;

        const int c = blockIdx.x * kTile + threadIdx.x;
        for (int k = 0; k < kTile; k += kTileRows) {
            const int r = blockIdx.y * kTile + threadIdx.y + k;
            if (r < rows && c < cols) {
                tile[threadIdx.y + k][threadIdx.x] = xs[static_cast<std::size_t>(r) * cols + c];
            }
        }
        __syncthreads();

        const int r_out = blockIdx.y * kTile + threadIdx.x;  // a column of y is a row of x
        for (int k = 0; k < kTile; k += kTileRows) {
            const int c_out = blockIdx.x * kTile + threadIdx.y + k;
            if (r_out < rows && c_out < cols) {
                ys[static_cast<std::size_t>(c_out) * rows + r_out] =
                    tile[threadIdx.x][threadIdx.y + k];
            }
        }
        __syncthreads();  // the next array reuses the tile
    }
}

bool is_pow2(int n) { return n >= 1 && (n & (n - 1)) == 0; }

std::size_t count(int n1, int n2, int batch)
{
    return static_cast<std::size_t>(batch) * static_cast<std::size_t>(n1)
           * static_cast<std::size_t>(n2);
}

// Throws unless the shape suits the implementation, the method is available, and every
// intermediate transform count fits an int.
void check(const FftImpl& impl, Columns columns, int n1, int n2, int batch, bool real)
{
    const std::string name = impl.name;
    const int max_n2 = real ? impl.max_real_n : impl.max_n;
    if (!is_pow2(n1) || n1 > impl.max_n) {
        throw std::invalid_argument("fft2: " + name + " needs n1 to be a power of two up to "
                                    + std::to_string(impl.max_n) + ", got "
                                    + std::to_string(n1));
    }
    if (!is_pow2(n2) || n2 > max_n2 || (real && n2 < 2)) {
        throw std::invalid_argument("fft2: " + name + " needs n2 to be a power of two"
                                    + (real ? std::string(" from 2") : std::string())
                                    + " up to " + std::to_string(max_n2) + ", got "
                                    + std::to_string(n2));
    }
    if (batch < 1) {
        throw std::invalid_argument("fft2: batch must be at least 1, got "
                                    + std::to_string(batch));
    }
    if (static_cast<long long>(batch) * std::max(n1, n2) > INT_MAX) {
        throw std::invalid_argument("fft2: too many transforms");
    }
    if (columns == Columns::Strided && impl.columns == nullptr) {
        throw std::invalid_argument("fft2: " + name
                                    + " has no strided column kernel; use the transpose method");
    }
}

std::size_t real_work(const FftImpl& impl, int n1, int n2, int batch)
{
    return impl.real_needs_work ? count(n1, n2 / 2, batch) : 0;
}

}  // namespace

void transpose(const cf32* x, cf32* y, int rows, int cols, int batch, cudaStream_t stream)
{
    const dim3 block(kTile, kTileRows);
    const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile),
                    static_cast<unsigned>((rows + kTile - 1) / kTile),
                    static_cast<unsigned>(std::min(batch, 65535)));
    transpose_kernel<<<grid, block, 0, stream>>>(x, y, rows, cols, batch);
    CUDA_CHECK(cudaGetLastError());
}

// --- Complex to complex ------------------------------------------------------------------------

std::size_t fft2_c2c_workspace(const FftImpl&, Columns columns, int n1, int n2, int batch)
{
    return columns == Columns::Strided ? 0 : 2 * count(n1, n2, batch);
}

void fft2_c2c(const FftImpl& impl, Columns columns, const cf32* x, cf32* y, int n1, int n2,
              int batch, Direction direction, cf32* work, cudaStream_t stream)
{
    check(impl, columns, n1, n2, batch, false);
    if (columns == Columns::Strided) {
        impl.c2c(x, y, n2, batch * n1, direction, stream);  // rows
        impl.columns(y, y, n1, n2, batch, direction, stream);
        return;
    }
    cf32* a = work;
    cf32* b = work + count(n1, n2, batch);
    impl.c2c(x, a, n2, batch * n1, direction, stream);  // rows
    transpose(a, b, n1, n2, batch, stream);              // n2 x n1
    impl.c2c(b, a, n1, batch * n2, direction, stream);  // former columns
    transpose(a, y, n2, n1, batch, stream);
}

// --- Real to complex ---------------------------------------------------------------------------

std::size_t fft2_r2c_workspace(const FftImpl& impl, Columns columns, int n1, int n2, int batch)
{
    const std::size_t half = columns == Columns::Strided ? 0 : 2 * count(n1, n2 / 2 + 1, batch);
    return half + real_work(impl, n1, n2, batch);
}

void fft2_r2c(const FftImpl& impl, Columns columns, const float* x, cf32* y, int n1, int n2,
              int batch, cf32* work, cudaStream_t stream)
{
    check(impl, columns, n1, n2, batch, true);
    const int w = n2 / 2 + 1;  // columns on the complex side
    if (columns == Columns::Strided) {
        cf32* rwork = impl.real_needs_work ? work : nullptr;
        impl.r2c(x, y, n2, batch * n1, rwork, stream);  // rows: n1 x w
        impl.columns(y, y, n1, w, batch, Direction::Forward, stream);
        return;
    }
    cf32* a = work;
    cf32* b = work + count(n1, w, batch);
    cf32* rwork = impl.real_needs_work ? work + 2 * count(n1, w, batch) : nullptr;
    impl.r2c(x, a, n2, batch * n1, rwork, stream);
    transpose(a, b, n1, w, batch, stream);  // w x n1
    impl.c2c(b, a, n1, batch * w, Direction::Forward, stream);
    transpose(a, y, w, n1, batch, stream);
}

// --- Complex to real ---------------------------------------------------------------------------

std::size_t fft2_c2r_workspace(const FftImpl& impl, Columns columns, int n1, int n2, int batch)
{
    const std::size_t q = count(n1, n2 / 2 + 1, batch);
    return (columns == Columns::Strided ? q : 2 * q) + real_work(impl, n1, n2, batch);
}

void fft2_c2r(const FftImpl& impl, Columns columns, const cf32* x, float* y, int n1, int n2,
              int batch, cf32* work, cudaStream_t stream)
{
    check(impl, columns, n1, n2, batch, true);
    const int w = n2 / 2 + 1;
    const std::size_t q = count(n1, w, batch);
    if (columns == Columns::Strided) {
        // Columns first, out of place (the input is not ours to overwrite), then rows.
        cf32* a = work;
        cf32* rwork = impl.real_needs_work ? work + q : nullptr;
        impl.columns(x, a, n1, w, batch, Direction::Inverse, stream);
        impl.c2r(a, y, n2, batch * n1, rwork, stream);
        return;
    }
    cf32* a = work;
    cf32* b = work + q;
    cf32* rwork = impl.real_needs_work ? work + 2 * q : nullptr;
    transpose(x, a, n1, w, batch, stream);  // w x n1
    impl.c2c(a, b, n1, batch * w, Direction::Inverse, stream);
    transpose(b, a, w, n1, batch, stream);  // n1 x w
    impl.c2r(a, y, n2, batch * n1, rwork, stream);
}

}  // namespace spectral::fft
