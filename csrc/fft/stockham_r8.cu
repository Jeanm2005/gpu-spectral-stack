// Stockham FFT, radix 8 (a final radix-2 or radix-4 stage when 3 does not divide
// log2 N), twiddles computed with sincospif.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r8(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham::c2c<8, false>(x, y, n, batch, direction, stream);
}

void r2c_stockham_r8(const float* x, cf32* y, int n, int batch, cf32* /*work*/, cudaStream_t stream)
{
    stockham::r2c<8, false>(x, y, n, batch, stream);
}

void c2r_stockham_r8(const cf32* x, float* y, int n, int batch, cf32* /*work*/, cudaStream_t stream)
{
    stockham::c2r<8, false>(x, y, n, batch, stream);
}

void columns_stockham_r8(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction,
                         cudaStream_t stream)
{
    stockham::columns<8, false>(x, y, n, cols, slabs, direction, stream);
}

const PrunedFns pruned_stockham_r8 = stockham::pruned_fns<8, false>();

}  // namespace spectral::fft
