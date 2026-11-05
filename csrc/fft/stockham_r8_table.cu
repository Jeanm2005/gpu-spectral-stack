// Stockham FFT, radix 8 like stockham_r8, twiddles read from the table in twiddles.cu.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r8_table(const cf32* x, cf32* y, int n, int batch, Direction direction,
                           cudaStream_t stream)
{
    stockham::c2c<8, true>(x, y, n, batch, direction, stream);
}

}  // namespace spectral::fft
