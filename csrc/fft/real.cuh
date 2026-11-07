#pragma once

// Real transforms through an n/2-point complex transform (Task 1.5): the arithmetic shared by the
// generic path (real.cu, around any complex implementation) and the fused Stockham kernels
// (stockham.cuh), which apply it in shared memory.
//
// Forward (R2C). Let m = n / 2. Reading the n real inputs as m complex values
// z[j] = x[2j] + i x[2j + 1] costs nothing: the memory layout is the same. With Z = DFT_m(z),
// the transforms of the even and odd samples are
//
//   E[k] = (Z[k] + conj(Z[m - k])) / 2        O[k] = (Z[k] - conj(Z[m - k])) / (2i)
//
// (indices mod m), and the n-point transform of x is X[k] = E[k] + w^k O[k] with
// w = exp(-2 pi i / n), for the m + 1 outputs k = 0 .. m.
//
// Inverse (C2R) runs the same steps backwards. From the m + 1 inputs X[0 .. m],
//
//   E[k] = (X[k] + conj(X[m - k])) / 2        O[k] = (X[k] - conj(X[m - k])) w^-k / 2
//
// for k = 0 .. m-1, then Z[k] = E[k] + i O[k], z = IDFT_m(Z) with the 1/m scaling, and z read as
// n reals is x. As in numpy and torch, the imaginary parts of X[0] and X[m] are ignored: a real
// signal's spectrum has none there, so only the real parts carry information.

#include "common/types.hpp"

#include <cuda_runtime.h>

namespace spectral::fft::real {

// X[k] from zk = Z[k mod m] and zmk = Z[(m - k) mod m]; k_over_m = k / m, exact for m = 2^p.
__device__ __forceinline__ cf32 r2c_combine(cf32 zk, cf32 zmk, float k_over_m)
{
    const cf32 e{0.5f * (zk.re + zmk.re), 0.5f * (zk.im - zmk.im)};  // (Z[k] + conj Z[m-k]) / 2
    const cf32 d{0.5f * (zk.re - zmk.re), 0.5f * (zk.im + zmk.im)};  // (Z[k] - conj Z[m-k]) / 2
    // O = d / i = -i d.  w^k = exp(-pi i k / m).
    float s = 0.0f;
    float c = 0.0f;
    sincospif(-k_over_m, &s, &c);
    const float o_re = d.im;
    const float o_im = -d.re;
    return {e.re + (c * o_re - s * o_im), e.im + (c * o_im + s * o_re)};
}

// Z[k] from xk = X[k] and xmk = X[m - k], k = 0 .. m-1. At k = 0 these are X[0] and X[m], whose
// imaginary parts are dropped.
__device__ __forceinline__ cf32 c2r_prepare(cf32 xk, cf32 xmk, int k, float k_over_m)
{
    if (k == 0) {
        xk.im = 0.0f;
        xmk.im = 0.0f;
    }
    const cf32 e{0.5f * (xk.re + xmk.re), 0.5f * (xk.im - xmk.im)};  // (X[k] + conj X[m-k]) / 2
    const cf32 d{0.5f * (xk.re - xmk.re), 0.5f * (xk.im + xmk.im)};  // (X[k] - conj X[m-k]) / 2
    // O = d w^-k with w^-k = exp(pi i k / m); Z = E + i O.
    float s = 0.0f;
    float c = 0.0f;
    sincospif(k_over_m, &s, &c);
    const float o_re = d.re * c - d.im * s;
    const float o_im = d.re * s + d.im * c;
    return {e.re - o_im, e.im + o_re};
}

// --- Host side (real.cu) -----------------------------------------------------------------------

// y[t * keep + k] = X[k], k < keep <= m + 1, for each of `batch` rows, from z = DFT_m of the
// row, m per row.
void r2c_postprocess(const cf32* z, cf32* y, int m, int batch, int keep, cudaStream_t stream);

// z[t * m + k] = Z[k] for each of `batch` rows, from the first keep <= m + 1 complex inputs of
// each row of x (keep per row; the inputs beyond them are zero).
void c2r_preprocess(const cf32* x, cf32* z, int m, int batch, int keep, cudaStream_t stream);

// Throws std::invalid_argument unless n is a power of two in [2, max_n], batch >= 1, and the
// real side (n floats per row) and complex side (complex_row values per row: n/2 + 1, or fewer
// for a pruned transform) do not overlap.
void check_arguments(const void* real_side, const void* complex_side, int n, int batch,
                     int complex_row, int max_n, const char* who);

}  // namespace spectral::fft::real
