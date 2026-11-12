#pragma once

// Complex FFTs larger than one block's shared memory (Task 1.4): the four-step algorithm
// (Bailey, "FFTs in external or hierarchical memory", 1990), built from the Stockham kernels.
//
// Write n = n1 * n2, index the input as an n1 x n2 row-major matrix, x[n1_idx * n2 + n2_idx],
// and the output as X[k1 + n1 * k2]. Then
//
//   X[k1 + n1 k2] = sum_{n2} W_{n2}^(n2_idx k2) * [ W_n^(k1 n2_idx) * sum_{n1} x[...] W_{n1}^(n1_idx k1) ]
//                     (3) row transforms           (2) twiddle       (1) column transforms
//
// with W_m = exp(-2 pi i / m). In global-memory passes:
//
//   1. columns     n1-point transforms down each of the n2 columns     x    -> y      (strided)
//   2+3. rows      multiply by the twiddle on the way in, then n2-point
//                  transforms along each of the n1 rows                 y    -> work   (fused)
//   4. transpose   row k1, column k2 holds X[k1 + n1 k2]: transposing
//                  each n1 x n2 matrix puts it at k2 * n1 + k1          work -> y
//
// Three passes over global memory against the ideal two (one read and one write), so the
// effective bandwidth is at best about two thirds of a single-pass FFT's. n2 is fixed at 4096,
// the largest single-kernel size: the row transforms are then as long as possible, and the
// column transforms as short as possible (n1 = n / 4096), which keeps the strided column reads
// coalesced in long runs (see StridedIo). Sizes from 8192 to 4096^2 = 2^24.
//
// For n up to 4096, c2c() below is exactly stockham::c2c.

#include "fft/fft2d.hpp"
#include "fft/stockham.cuh"
#include <climits>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace spectral::fft::fourstep {
    constexpr int kMaxN = kFourStepMaxN; // 2^24

    // Makes freed stream-ordered allocations stay in the device's memory pool instead of returning
    // to the system at every synchronization, so repeated calls reuse the same memory (fourstep.cu).
    // Returns false where the device has no memory pools.
    bool retain_pool_memory();

    // Scratch memory for one call, allocated and freed in stream order: the free is queued behind
    // the kernels that use it, so it never needs a synchronization. Falls back to cudaMalloc and
    // cudaFree (which synchronize the device) on devices without memory pools
    class Workspace {
        public:
            Workspace(std::size_t count, cudaStream_t stream) : stream_(stream)
            {
                async_ = retain_pool_memory();
                void* raw = nullptr;
                if (async_) {
                    CUDA_CHECK(cudaMallocAsync(&raw, count * sizeof(cf32), stream));
                } else {
                    CUDA_CHECK(cudaMalloc(&raw, count * sizeof(cf32)));
                }
                ptr_ = static_cast<cf32*>(raw);
            }
            ~Workspace()
            {
                const cudaError_t err = async_ ? cudaFreeAsync(ptr_, stream_) : cudaFree(ptr_);
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "warning: freeing the four-step workspace failed: %s\n", cudaGetErrorString(err));
                }
            }
            Workspace(const Workspace&) = delete;
            Workspace& operator=(const Workspace&) = delete;

            cf32* get() const { return ptr_; }

        private:
            cf32* ptr_ = nullptr;
            cudaStream_t stream_;
            bool async_ = false;
    };

    // Steps 2 and 3: `batch` arrays of n1 rows by N columns, each row one N-point transform. Before
    // the transform, element n2_idx of row k1 is multiplied by exp(sign 2 pi i k1 n2_idx / (n1 N)).
    // Rows are contiguous, so the loads and stores are C2cIo's: coalesced.    
    struct TwiddleRowsIo {
        const cf32* x;
        cf32* y;
        float scale;
        float sign;
        int n1;
        float two_over_total;

        template <int N, int T>
        __device__ __forceinline__ void load(cf32* smem, int first, int count) const
        {
            const std::size_t base = static_cast<std::size_t>(first) * N;
            for (int e = threadIdx.x; e < count * N; e += stockham::kThreads) {
                const int k1 = (first + e / N) & (n1 - 1);
                const int col = e & (N - 1);
                const float a = static_cast<float>(k1 * col) * two_over_total;
                smem[e] = stockham::mul(x[base + e], stockham::twiddle(sign, a));
            }
        }

        template <int N, int T>
        __device__ __forceinline__ void store(const cf32* smem, int first, int count) const
        {
            stockham::C2cIo{nullptr, y, scale, N, N}.store<N, T>(smem, first, count);
        }
    };

    // Complex transforms of any power-of-two size up to kMaxN: one Stockham kernel up to 4096, the
    // four steps above beyond.
    template <int R, bool Table>
    void c2c(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream)
    {
        if (n <= kStockhamMaxN) {
            stockham::c2c<R, Table>(x, y, n, batch, direction, stream);
            return;
        }
        if ((n & (n - 1)) != 0 || n > kMaxN) {
            throw std::invalid_argument("fourstep: n must be a power of two up to " + std::to_string(kMaxN) + ", got " + std::to_string(n));
        }
        if (batch < 1) {
            throw std::invalid_argument("fourstep: batch must be at least 1, got " + std::to_string(batch));
        }
        const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        if (x < y + total && y < x + total) {
            throw std::invalid_argument("fourstep: x and y must not overlap (out-of-place only)");
        }
        constexpr int n2 = kStockhamMaxN;
        const int n1 = n / n2;
        if (static_cast<long long>(batch) * n1 > INT_MAX || static_cast<long long>(batch) * n2 > INT_MAX) {
            throw std::invalid_argument("fourstep: too many transforms");
        }

        const bool inverse = direction == Direction::Inverse;
        const float sign = inverse ? 1.0f : -1.0f;
        Workspace work(total, stream);

        // 1. n1-point transforms down every column; the inverse scales by 1/n1 here.
        stockham::columns<R, Table>(x, y, n1, n2, batch, direction, stream);

        // 2 + 3. Twiddle, then n2-point transforms along every row; the inverse scales by 1/n2.
        const float scale = inverse ? 1.0f / static_cast<float>(n2) : 1.0f;
        const TwiddleRowsIo rows{y, work.get(), scale, sign, n1, 2.0f / static_cast<float>(n)};
        stockham::dispatch<R, Table>(n2, rows, batch * n1, sign, stream);

         // 4. Row k1, column k2 holds X[k1 + n1 k2]; its natural position is k2 * n1 + k1.
        transpose(work.get(), y, n1, n2, batch, stream);
    }
}