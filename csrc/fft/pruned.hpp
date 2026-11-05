#pragma once

// Pruned transforms (Task 1.7): only the low-frequency modes a Fourier Neural Operator keeps.
//
//   1D complex   fft_c2c_pruned: forward writes the first `keep` of n outputs per transform;
//                inverse reads `keep` inputs per transform, treats the rest as zero, and writes
//                all n outputs (divided by n). The layout of spectral.reference.fft_truncate.
//   2D real      rfft2_modes: of rfft2's n1 x (n2/2 + 1) spectrum, the 2 m1 x m2 block of rows
//                0 .. m1-1 and n1-m1 .. n1-1 (positive, then negative frequencies) and columns
//                0 .. m2-1, the layout of spectral.reference.rfft2_truncate. irfft2_modes is its
//                inverse: every other mode zero, then irfft2 (spectral.reference.irfft2_from_modes).
//
// Implementations with FftImpl::pruned prune inside their kernels: the 2D forward runs real row
// transforms that write only m2 columns, then column transforms over those m2 columns that write
// only the 2 m1 kept rows. The inverse mirrors it. Global memory then carries little more than
// the modes themselves. Other implementations run the full transform and gather the modes from
// it (or scatter them into zeros and run the full inverse), so they produce the same results.

#include "fft/registry.hpp"

#include <cstddef>

namespace spectral::fft {

// Workspace each call needs, in complex values (0 means `work` may be nullptr).
std::size_t fft_c2c_pruned_workspace(const FftImpl& impl, int n, int keep, int batch);
std::size_t rfft2_modes_workspace(const FftImpl& impl, int n1, int n2, int m1, int m2, int batch);
std::size_t irfft2_modes_workspace(const FftImpl& impl, int n1, int n2, int m1, int m2,
                                   int batch);

void fft_c2c_pruned(const FftImpl& impl, const cf32* x, cf32* y, int n, int keep, int batch,
                    Direction direction, cf32* work, cudaStream_t stream);
void rfft2_modes(const FftImpl& impl, const float* x, cf32* modes, int n1, int n2, int m1, int m2,
                 int batch, cf32* work, cudaStream_t stream);
void irfft2_modes(const FftImpl& impl, const cf32* modes, float* y, int n1, int n2, int m1, int m2,
                  int batch, cf32* work, cudaStream_t stream);

// The modes of `batch` arrays of n1 x w values (see the top of this file; m1 = 0 keeps all n1
// rows): gather copies them out of `full`, scatter writes them into `full`, which it zeroes first.
void gather_modes(const cf32* full, cf32* modes, int n1, int w, int m1, int m2, int batch,
                  cudaStream_t stream);
void scatter_modes(const cf32* modes, cf32* full, int n1, int w, int m1, int m2, int batch,
                   cudaStream_t stream);

}  // namespace spectral::fft
