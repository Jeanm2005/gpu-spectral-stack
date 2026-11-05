// Stockham FFT, radix 8 (a final radix-2 or radix-4 stage when 3 does not divide log2 N), twiddles computed with sincospif.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r8(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham::c2c<8, false>(x, y, n, batch, direction, stream);
}

}  // namespace spectral::fft
