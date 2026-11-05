// Stockham FFT, radix 4 (a final radix-2 stage when log2 N is odd), twiddles computed with sincospif.
// The kernels themselves are in stockham.cuh.

#include "fft/stockham.cuh"

namespace spectral::fft {

void c2c_stockham_r4(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham::c2c<4, false>(x, y, n, batch, direction, stream);
}

}  // namespace spectral::fft
