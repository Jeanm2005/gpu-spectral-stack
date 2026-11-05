// Stockham FFT, radix 2, twiddles computed with sincospif.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r2(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham::c2c<2, false>(x, y, n, batch, direction, stream);
}

void r2c_stockham_r2(const float* x, cf32* y, int n, int batch, cf32* /*work*/, cudaStream_t stream)
{
    stockham::r2c<2, false>(x, y, n, batch, stream);
}

void c2r_stockham_r2(const cf32* x, float* y, int n, int batch, cf32* /*work*/, cudaStream_t stream)
{
    stockham::c2r<2, false>(x, y, n, batch, stream);
}

}  // namespace spectral::fft
