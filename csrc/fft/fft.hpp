#pragma once

// Every FFT entry point. Conventions are numpy's and torch's: forward transforms are
// unnormalized, inverse ones divide by n; all of them are out of place and batched, with the
// transforms of a batch stored back to back. registry.hpp lists them by name.

#include "common/types.hpp"
#include <cuda_runtime.h>

namespace spectral::fft {
    enum class Direction { Forward, Inverse };

    // Complex to complex: `batch` transforms of n points, n a power of two.
    using C2cFn = void (*)(const cf32* x, cf32* y, int n, int batch, Direction direction,
                           cudaStream_t stream);

    void c2c_naive(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);
    constexpr int kStockhamMaxN = 4096;
    void c2c_stockham_r2(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);
    void c2c_stockham_r4(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);
    void c2c_stockham_r8(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);
    void c2c_stockham_r8_table(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream);

    // Real to complex and back (Task 1.5): `batch` transforms of n real points, n a power of two
    // from 2, each with n/2 + 1 complex values on the complex side. `work` is scratch space for
    // batch * n/2 complex values; only implementations without a fused kernel use it (see
    // FftImpl::real_needs_work in registry.hpp), and the others accept nullptr.
    using R2cFn = void (*)(const float* x, cf32* y, int n, int batch, cf32* work,
                           cudaStream_t stream);
    using C2rFn = void (*)(const cf32* x, float* y, int n, int batch, cf32* work,
                           cudaStream_t stream);

    // Any complex implementation, plus one combine pass (real.cu).
    void r2c_via_c2c(C2cFn c2c, const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_via_c2c(C2cFn c2c, const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);

    void r2c_naive(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_naive(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);
    void r2c_stockham_r2(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_stockham_r2(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);
    void r2c_stockham_r4(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_stockham_r4(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);
    void r2c_stockham_r8(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_stockham_r8(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);
    void r2c_stockham_r8_table(const float* x, cf32* y, int n, int batch, cf32* work, cudaStream_t stream);
    void c2r_stockham_r8_table(const cf32* x, float* y, int n, int batch, cf32* work, cudaStream_t stream);

    // Column transforms (Task 1.6): `slabs` row-major arrays of n rows by `cols` columns, back
    // to back; transforms every column (n points, stride cols) straight from that layout, in
    // place when x == y. Only implementations with a strided kernel provide one.
    using ColumnsFn = void (*)(const cf32* x, cf32* y, int n, int cols, int slabs,
                               Direction direction, cudaStream_t stream);

    void columns_stockham_r2(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction, cudaStream_t stream);
    void columns_stockham_r4(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction, cudaStream_t stream);
    void columns_stockham_r8(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction, cudaStream_t stream);
    void columns_stockham_r8_table(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction, cudaStream_t stream);

    // Pruned transforms (Task 1.7), for implementations that prune inside their kernels:
    //   c2c      forward: only the first `keep` outputs per transform; inverse: only the first
    //            `keep` inputs (the rest are zero). Rows hold keep values on the pruned side.
    //   r2c      only the first `keep` of the n/2 + 1 outputs per transform
    //   c2r      only the first `keep` of the n/2 + 1 inputs (the rest are zero)
    //   columns  only the 2 m1 lowest-frequency rows (0 .. m1-1, then n-m1 .. n-1) on the input
    //            (m1_in) or output (m1_out) side; 0 means all n rows. Not in place.
    // Other implementations prune by gathering from, or scattering into, a full transform
    // (pruned.cu).
    struct PrunedFns {
        void (*c2c)(const cf32* x, cf32* y, int n, int batch, int keep, Direction direction,
                    cudaStream_t stream);
        void (*r2c)(const float* x, cf32* y, int n, int batch, int keep, cudaStream_t stream);
        void (*c2r)(const cf32* x, float* y, int n, int batch, int keep, cudaStream_t stream);
        void (*columns)(const cf32* x, cf32* y, int n, int cols, int slabs, int m1_in,
                        int m1_out, Direction direction, cudaStream_t stream);
    };

    extern const PrunedFns pruned_stockham_r2;
    extern const PrunedFns pruned_stockham_r4;
    extern const PrunedFns pruned_stockham_r8;
    extern const PrunedFns pruned_stockham_r8_table;
}
