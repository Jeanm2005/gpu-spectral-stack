#pragma once

// Two-dimensional FFTs (Task 1.6): `batch` row-major arrays of n1 rows by n2 columns, back to
// back, transformed along both axes. Rows go first, through the implementation's 1D transform;
// columns go one of two ways:
//
//   Transpose  transpose each array, run the 1D transform over the former columns (now
//              contiguous rows), transpose back. Works with every implementation; costs two
//              extra passes over global memory and a workspace.
//   Strided    transform the columns where they are, with a kernel that loads them directly
//              (FftImpl::columns; the Stockham family has one). No extra passes, but each warp
//              reads short runs of neighbouring columns instead of long contiguous rows.
//
// Real transforms keep n2/2 + 1 columns on the complex side (the last axis is the real one), as
// numpy.fft.rfft2 and torch.fft.rfft2 do. Normalization follows numpy: forward unnormalized,
// inverse divided by n1 * n2.

#include "fft/registry.hpp"

#include <cstddef>

namespace spectral::fft {

enum class Columns { Transpose, Strided };

// Workspace each call needs, in complex values (0 means `work` may be nullptr).
std::size_t fft2_c2c_workspace(const FftImpl& impl, Columns columns, int n1, int n2, int batch);
std::size_t fft2_r2c_workspace(const FftImpl& impl, Columns columns, int n1, int n2, int batch);
std::size_t fft2_c2r_workspace(const FftImpl& impl, Columns columns, int n1, int n2, int batch);

void fft2_c2c(const FftImpl& impl, Columns columns, const cf32* x, cf32* y, int n1, int n2,
              int batch, Direction direction, cf32* work, cudaStream_t stream);
void fft2_r2c(const FftImpl& impl, Columns columns, const float* x, cf32* y, int n1, int n2,
              int batch, cf32* work, cudaStream_t stream);
void fft2_c2r(const FftImpl& impl, Columns columns, const cf32* x, float* y, int n1, int n2,
              int batch, cf32* work, cudaStream_t stream);

// y = the transpose of x for `batch` row-major arrays of rows x cols, back to back.
void transpose(const cf32* x, cf32* y, int rows, int cols, int batch, cudaStream_t stream);

}  // namespace spectral::fft
