// Stockham FFT, radix 8 like stockham_r8, twiddles read from the table in
// twiddles.cu.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r8_table(const cf32* x, cf32* y, int n, int batch, Direction direction,
                           cudaStream_t stream)
{
    stockham::c2c<8, true>(x, y, n, batch, direction, stream);
}

void r2c_stockham_r8_table(const float* x, cf32* y, int n, int batch, cf32* /*work*/,
                           cudaStream_t stream)
{
    stockham::r2c<8, true>(x, y, n, batch, stream);
}

void c2r_stockham_r8_table(const cf32* x, float* y, int n, int batch, cf32* /*work*/,
                           cudaStream_t stream)
{
    stockham::c2r<8, true>(x, y, n, batch, stream);
}

void columns_stockham_r8_table(const cf32* x, cf32* y, int n, int cols, int slabs,
                               Direction direction, cudaStream_t stream)
{
    stockham::columns<8, true>(x, y, n, cols, slabs, direction, stream);
}

const PrunedFns pruned_stockham_r8_table = stockham::pruned_fns<8, true>();

}  // namespace spectral::fft
