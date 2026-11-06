#pragma once

// The Stockham FFT core shared by every shared-memory implementation (Tasks 1.2 and 1.3).
//
// One kernel launch per call. Each block loads its transforms from global memory once, runs every
// stage in shared memory, and writes the results once, so the data crosses global memory exactly
// twice, as in an ideal FFT. The Stockham formulation writes each stage's outputs to their sorted
// positions as it goes, so the result comes out in natural order without a bit-reversal pass.
//
// Every size from 2 to 4096 has its own compiled kernel (N is a template parameter): all loop
// counts are then compile-time constants, the loops unroll, and each thread's values stay in
// registers.
//
// What a kernel reads and writes is a policy, `Io`: an object with two device functions,
//
//   load<N, T>(smem, first, count)   fill smem[t * N + i] for the block's transforms t < count
//   store<N, T>(smem, first, count)  write them out
//
// where `first` is the index of the block's first transform. Contiguous complex transforms are
// one policy (C2cIo below); later tasks add real input and output, strided columns, and pruning
// as further policies around the same stages.
//
// Each source file stockham_<variant>.cu instantiates this header for one radix and twiddle
// source, so the variants compile in parallel.

#include "fft/fft.hpp"
#include "fft/real.cuh"

#include "common/check.hpp"

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace spectral::fft::stockham {

constexpr int kThreads = 256;  // threads per block, for every size and radix

__host__ __device__ constexpr int ilog2(int n)
{
    int r = 0;
    while ((1 << r) < n) {
        ++r;
    }
    return r;
}

__host__ __device__ constexpr int min_int(int a, int b) { return a < b ? a : b; }

// The launch shape of the size-N, radix-R kernel. Each transform is handled by kLanes threads,
// and a block holds kTransformsPerBlock transforms side by side, so that small transforms still
// fill a block of 256 threads.
//
// Mixed radix: log2 N = kStages * log2 R + leftover. The kStages radix-R stages run first, then
// one stage of radix kFinalRadix = 2^leftover (none when leftover is 0). With R = 8, N = 2048
// runs three radix-8 stages and one radix-4 stage.
template <int N, int R>
struct Shape {
    static_assert(N >= R && (N & (N - 1)) == 0, "N must be a power of two, at least R");
    static constexpr int kButterflies = N / R;  // per radix-R stage
    static constexpr int kLanes = kButterflies < kThreads ? kButterflies : kThreads;
    static constexpr int kTransformsPerBlock = kThreads / kLanes;  // at most 4096 / N
    static constexpr int kStages = ilog2(N) / ilog2(R);              // radix-R stages
    static constexpr int kFinalRadix = 1 << (ilog2(N) - kStages * ilog2(R));
};

__device__ __forceinline__ cf32 add(cf32 a, cf32 b) { return {a.re + b.re, a.im + b.im}; }
__device__ __forceinline__ cf32 sub(cf32 a, cf32 b) { return {a.re - b.re, a.im - b.im}; }
__device__ __forceinline__ cf32 mul(cf32 a, cf32 w)
{
    return {a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
}
__device__ __forceinline__ cf32 scaled(cf32 a, float s) { return {a.re * s, a.im * s}; }
__device__ __forceinline__ cf32 conj(cf32 a) { return {a.re, -a.im}; }

// exp(sign * pi * i * a). sincospif rounds neither pi nor pi * a.
__device__ __forceinline__ cf32 twiddle(float sign, float a)
{
    float s = 0.0f;
    float c = 0.0f;
    sincospif(sign * a, &s, &c);
    return {c, s};
}

// The table of twiddle factors (see twiddles.cu): table[m] = exp(-2 pi i m / kTwiddleTableSize),
// rounded once from double precision.
constexpr int kTwiddleTableSize = kStockhamMaxN;

// Device memory holding the table for the current device, built on first use.
const cf32* twiddle_table();

// The R-point DFT of v, in place; sign is -1 for the forward transform, +1 for the inverse.
template <int R>
__device__ __forceinline__ void dft(cf32 (&v)[R], float sign);

template <>
__device__ __forceinline__ void dft<2>(cf32 (&v)[2], float)
{
    const cf32 a = v[0];
    v[0] = add(a, v[1]);
    v[1] = sub(a, v[1]);
}

// X0 = (v0 + v2) + (v1 + v3)           X2 = (v0 + v2) - (v1 + v3)
// X1 = (v0 - v2) + sign i (v1 - v3)    X3 = (v0 - v2) - sign i (v1 - v3)
// Multiplying by +-i swaps the real and imaginary parts and flips one sign: no multiplication.
template <>
__device__ __forceinline__ void dft<4>(cf32 (&v)[4], float sign)
{
    const cf32 a = add(v[0], v[2]);
    const cf32 b = sub(v[0], v[2]);
    const cf32 c = add(v[1], v[3]);
    const cf32 t = sub(v[1], v[3]);
    const cf32 d{-sign * t.im, sign * t.re};
    v[0] = add(a, c);
    v[1] = add(b, d);
    v[2] = sub(a, c);
    v[3] = sub(b, d);
}

// Radix 8 as one radix-2 step over two 4-point DFTs: E = DFT4(even inputs), O = DFT4(odd
// inputs), X[k] = E[k] + w^k O[k] and X[k + 4] = E[k] - w^k O[k], with w = exp(sign 2 pi i / 8).
// w^1 = h (1 + sign i) and w^3 = h (-1 + sign i) with h = sqrt(1/2) cost two multiplications
// each; w^2 = sign i costs none.
template <>
__device__ __forceinline__ void dft<8>(cf32 (&v)[8], float sign)
{
    cf32 e[4] = {v[0], v[2], v[4], v[6]};
    cf32 o[4] = {v[1], v[3], v[5], v[7]};
    dft<4>(e, sign);
    dft<4>(o, sign);

    constexpr float h = 0.70710678118654752440f;
    const cf32 o1 = o[1];
    const cf32 o2 = o[2];
    const cf32 o3 = o[3];
    o[1] = cf32{h * (o1.re - sign * o1.im), h * (o1.im + sign * o1.re)};
    o[2] = cf32{-sign * o2.im, sign * o2.re};
    o[3] = cf32{-h * (o3.re + sign * o3.im), h * (sign * o3.re - o3.im)};

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = add(e[k], o[k]);
        v[k + 4] = sub(e[k], o[k]);
    }
}

// exp(sign * 2 pi i * k * r / span) for k < span / R, r < R, span = ns * R <= N. Either computed
// with sincospif, or read from the table: k r / span = m / kTwiddleTableSize with
// m = k r (kTwiddleTableSize / span) < kTwiddleTableSize, an exact integer because span divides
// the table size. The table holds the forward factors; the inverse ones are their conjugates.
template <bool Table>
__device__ __forceinline__ cf32 stage_twiddle(const cf32* __restrict__ table, float sign, int k,
                                              int r, int span)
{
    if constexpr (Table) {
        // __ldg has no overload for cf32; float2 has the same layout. cudaMalloc aligns the table.
        const float2 w = __ldg(reinterpret_cast<const float2*>(table)
                               + k * r * (kTwiddleTableSize / span));
        return {w.x, -sign * w.y};
    } else {
        // 2k / span and its multiples by r < R are exact in float32.
        const float step = 2.0f * static_cast<float>(k) / static_cast<float>(span);
        return twiddle(sign, step * static_cast<float>(r));
    }
}

// One radix-R Stockham stage over an N-point transform in shared memory `s`: sub-transforms of
// length ns become sub-transforms of length ns * R. Butterfly j (with k = j % ns) reads
// s[j + r * N/R] for r < R, multiplies input r by exp(sign * 2 pi i * k * r / (ns * R)), takes an
// R-point DFT, and writes output r to s[(j - k) * R + k + r * ns].
//
// Each lane handles butterflies lane, lane + Lanes, ... Every thread reads all of its inputs
// before any thread writes, so a single buffer is enough: the first barrier separates this
// stage's reads from its writes, the second its writes from the next stage's reads.
template <int N, int R, int Lanes, bool Table>
__device__ __forceinline__ void stage(cf32* s, const cf32* __restrict__ table, int ns, int lane,
                                      float sign)
{
    constexpr int kButterflies = N / R;
    constexpr int kPerLane = kButterflies / Lanes;
    cf32 v[kPerLane][R];

#pragma unroll
    for (int b = 0; b < kPerLane; ++b) {
        const int j = lane + b * Lanes;
        const int k = j % ns;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const cf32 x = s[j + r * kButterflies];
            v[b][r] = r == 0 ? x : mul(x, stage_twiddle<Table>(table, sign, k, r, ns * R));
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < kPerLane; ++b) {
        dft<R>(v[b], sign);
        const int j = lane + b * Lanes;
        const int k = j % ns;
        const int out = (j - k) * R + k;  // (j / ns) * ns * R + k
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[out + r * ns] = v[b][r];
        }
    }
    __syncthreads();
}

// --- I/O policies ------------------------------------------------------------------------------
//
// Pruning (Task 1.7) lives here too: a policy can read fewer inputs than the transform has
// (the rest are zero) or write fewer outputs, which saves the global-memory traffic of the modes
// nobody needs. The stages always run at full size. Each policy keeps a fast path for the
// unpruned case, where every row length is a compile-time constant.

// `batch` contiguous complex transforms of N points: transform t is x[t * N, (t + 1) * N).
// Consecutive threads read and write consecutive elements, so every access is coalesced.
//
// Pruned: rows of x hold only the first keep_in inputs (the others are zero), and rows of y only
// the first keep_out outputs.
struct C2cIo {
    const cf32* x;
    cf32* y;
    float scale;  // applied on the way out: 1/N for the inverse transform, else 1
    int keep_in;
    int keep_out;

    template <int N, int T>
    __device__ __forceinline__ void load(cf32* smem, int first, int count) const
    {
        if (keep_in == N) {
            const std::size_t base = static_cast<std::size_t>(first) * N;
            for (int e = threadIdx.x; e < count * N; e += kThreads) {
                smem[e] = x[base + e];
            }
            return;
        }
        const cf32* rows = x + static_cast<std::size_t>(first) * keep_in;
        for (int e = threadIdx.x; e < count * N; e += kThreads) {
            const int t = e / N;
            const int i = e & (N - 1);
            smem[e] = i < keep_in ? rows[t * keep_in + i] : cf32{0.0f, 0.0f};
        }
    }

    template <int N, int T>
    __device__ __forceinline__ void store(const cf32* smem, int first, int count) const
    {
        if (keep_out == N) {
            const std::size_t base = static_cast<std::size_t>(first) * N;
            for (int e = threadIdx.x; e < count * N; e += kThreads) {
                y[base + e] = scaled(smem[e], scale);
            }
            return;
        }
        cf32* rows = y + static_cast<std::size_t>(first) * keep_out;
        for (int e = threadIdx.x; e < count * keep_out; e += kThreads) {
            const int t = e / keep_out;
            const int i = e - t * keep_out;
            rows[e] = scaled(smem[t * N + i], scale);
        }
    }
};

// Real input (Task 1.5): `batch` rows of 2N reals, read as N complex values each (see real.cuh),
// so the load is C2cIo's. After the forward N-point transform, the store combines Z[k] and
// Z[N - k] from shared memory into the outputs of the 2N-point real transform. The writes stay
// coalesced: consecutive threads, consecutive outputs.
//
// keep is the number of outputs per row of y: N + 1 for the full half spectrum, fewer to prune.
struct R2cIo {
    const cf32* x;
    cf32* y;
    int keep;

    template <int N, int T>
    __device__ __forceinline__ void load(cf32* smem, int first, int count) const
    {
        C2cIo{x, nullptr, 1.0f, N, N}.load<N, T>(smem, first, count);
    }

    template <int N>
    __device__ __forceinline__ void store_rows(const cf32* smem, int first, int count,
                                               int row) const
    {
        cf32* rows = y + static_cast<std::size_t>(first) * row;
        for (int e = threadIdx.x; e < count * row; e += kThreads) {
            const int t = e / row;
            const int k = e - t * row;
            const cf32* z = smem + t * N;
            const float k_over_n = static_cast<float>(k) / static_cast<float>(N);  // exact
            rows[e] = real::r2c_combine(z[k & (N - 1)], z[(N - k) & (N - 1)], k_over_n);
        }
    }

    template <int N, int T>
    __device__ __forceinline__ void store(const cf32* smem, int first, int count) const
    {
        if (keep == N + 1) {
            store_rows<N>(smem, first, count, N + 1);  // a constant: the division is cheap
        } else {
            store_rows<N>(smem, first, count, keep);
        }
    }
};

// Real output (Task 1.5): rows of complex inputs X[0 .. N]; the load prepares Z[k] from X[k] and
// X[N - k] (both reads are contiguous runs, one ascending, one descending), the kernel runs the
// inverse N-point transform, and the store writes z, scaled by 1/N, as 2N reals.
//
// keep is the number of inputs per row of x: N + 1, or fewer when the rest are zero.
struct C2rIo {
    const cf32* x;
    cf32* y;
    float scale;  // 1/N
    int keep;

    template <int N, int T>
    __device__ __forceinline__ void load(cf32* smem, int first, int count) const
    {
        const cf32 zero{0.0f, 0.0f};
        for (int e = threadIdx.x; e < count * N; e += kThreads) {
            const int t = e / N;
            const int k = e & (N - 1);
            const cf32* row = x + static_cast<std::size_t>(first + t) * keep;
            const cf32 xk = k < keep ? row[k] : zero;
            const cf32 xmk = N - k < keep ? row[N - k] : zero;
            const float k_over_n = static_cast<float>(k) / static_cast<float>(N);  // exact
            smem[e] = real::c2r_prepare(xk, xmk, k, k_over_n);
        }
    }

    template <int N, int T>
    __device__ __forceinline__ void store(const cf32* smem, int first, int count) const
    {
        C2cIo{nullptr, y, scale, N, N}.store<N, T>(smem, first, count);
    }
};

// Columns of a row-major array (Task 1.6): `slabs` arrays of N rows by `cols` columns, stored
// back to back, each column one transform; transform t is column t % cols of slab t / cols. A
// block's T transforms are T neighbouring columns, and consecutive threads take consecutive
// columns of one row, so each warp reads and writes runs of T adjacent elements: coalesced in
// runs of T * 8 bytes, which is short when N is large and T small (T = 1 at N = 4096). Measuring
// that cost against transposing is the point of having both.
//
// The load writes column j's element i to smem[j * (N + kPad) + i]. Without padding, the threads
// of a warp (consecutive j, one i) would write addresses N apart, all in one bank; one extra
// element per transform spreads them over the banks.
//
// Pruned (Task 1.7), as an FNO keeps the lowest frequencies of each column: with m1_in > 0 the
// arrays of x hold only 2 m1_in rows, frequencies 0 .. m1_in - 1 then N - m1_in .. N - 1 (the
// rest are zero); with m1_out > 0 the arrays of y keep only those 2 m1_out rows. 0 means all N.
//
// x and y may be the same array when neither side is pruned: every block reads all of its
// columns before writing any, and no two blocks share a column.
struct StridedIo {
    static constexpr int kPad = 1;
    const cf32* x;
    cf32* y;
    int cols;
    float scale;
    int m1_in;
    int m1_out;

    // Element (row, column t) of arrays with `rows` rows.
    __device__ __forceinline__ std::size_t offset(int t, int row, int rows) const
    {
        const int slab = t / cols;
        const int col = t - slab * cols;
        return (static_cast<std::size_t>(slab) * rows + row) * static_cast<std::size_t>(cols)
               + col;
    }

    template <int N, int T>
    __device__ __forceinline__ void load(cf32* smem, int first, int count) const
    {
        const int rows = m1_in > 0 ? 2 * m1_in : N;
        for (int e = threadIdx.x; e < T * N; e += kThreads) {
            const int i = e / T;  // T is a power of two: a shift
            const int j = e % T;
            if (j >= count) {
                continue;
            }
            int row = i;  // where frequency i is stored, or -1 if it is not
            if (m1_in > 0) {
                row = i < m1_in ? i : (i >= N - m1_in ? i - (N - rows) : -1);
            }
            smem[j * (N + kPad) + i] =
                row >= 0 ? x[offset(first + j, row, rows)] : cf32{0.0f, 0.0f};
        }
    }

    template <int N, int T>
    __device__ __forceinline__ void store(const cf32* smem, int first, int count) const
    {
        const int rows = m1_out > 0 ? 2 * m1_out : N;
        for (int e = threadIdx.x; e < T * rows; e += kThreads) {
            const int row = e / T;
            const int j = e % T;
            if (j >= count) {
                continue;
            }
            const int i = (m1_out > 0 && row >= m1_out) ? row + (N - rows) : row;  // frequency
            y[offset(first + j, row, rows)] = scaled(smem[j * (N + kPad) + i], scale);
        }
    }
};

// --- The kernel and its dispatch ---------------------------------------------------------------

// Shared-memory padding after each transform: a policy may ask for some (StridedIo does).
template <class Io, class = void>
struct PadOf {
    static constexpr int value = 0;
};
template <class Io>
struct PadOf<Io, decltype(void(Io::kPad))> {
    static constexpr int value = Io::kPad;
};

// The last block may hold fewer than kTransformsPerBlock transforms; its threads still take part
// in every barrier, the policy just loads and stores fewer elements.
template <int N, int R, bool Table, class Io>
__global__ void __launch_bounds__(kThreads)
    stockham_kernel(Io io, const cf32* __restrict__ table, int batch, float sign)
{
    using S = Shape<N, R>;
    constexpr int kStride = N + PadOf<Io>::value;  // shared-memory distance between transforms
    __shared__ cf32 smem[S::kTransformsPerBlock * kStride];

    const int first = blockIdx.x * S::kTransformsPerBlock;
    const int count = min_int(S::kTransformsPerBlock, batch - first);
    io.template load<N, S::kTransformsPerBlock>(smem, first, count);
    __syncthreads();

    cf32* s = smem + (threadIdx.x / S::kLanes) * kStride;
    const int lane = threadIdx.x % S::kLanes;
    int ns = 1;
#pragma unroll
    for (int i = 0; i < S::kStages; ++i) {
        stage<N, R, S::kLanes, Table>(s, table, ns, lane, sign);
        ns *= R;
    }
    if constexpr (S::kFinalRadix > 1) {
        stage<N, S::kFinalRadix, S::kLanes, Table>(s, table, ns, lane, sign);
    }

    io.template store<N, S::kTransformsPerBlock>(smem, first, count);
}

template <int N, int R, bool Table, class Io>
void run(const Io& io, int batch, float sign, cudaStream_t stream)
{
    using S = Shape<N, R>;
    const long long per_block = S::kTransformsPerBlock;
    const auto blocks = static_cast<unsigned>((batch + per_block - 1) / per_block);
    const cf32* table = Table ? twiddle_table() : nullptr;
    stockham_kernel<N, R, Table, Io><<<blocks, kThreads, 0, stream>>>(io, table, batch, sign);
    CUDA_CHECK(cudaGetLastError());
}

// One case per size, 2 to kStockhamMaxN. Transforms shorter than the radix use their own length
// as the radix (a 4-point radix-8 kernel is a single radix-4 butterfly).
template <int R, bool Table, class Io>
void dispatch(int n, const Io& io, int batch, float sign, cudaStream_t stream)
{
    switch (n) {
    case 2: run<2, 2, Table>(io, batch, sign, stream); break;
    case 4: run<4, min_int(4, R), Table>(io, batch, sign, stream); break;
    case 8: run<8, R, Table>(io, batch, sign, stream); break;
    case 16: run<16, R, Table>(io, batch, sign, stream); break;
    case 32: run<32, R, Table>(io, batch, sign, stream); break;
    case 64: run<64, R, Table>(io, batch, sign, stream); break;
    case 128: run<128, R, Table>(io, batch, sign, stream); break;
    case 256: run<256, R, Table>(io, batch, sign, stream); break;
    case 512: run<512, R, Table>(io, batch, sign, stream); break;
    case 1024: run<1024, R, Table>(io, batch, sign, stream); break;
    case 2048: run<2048, R, Table>(io, batch, sign, stream); break;
    case 4096: run<4096, R, Table>(io, batch, sign, stream); break;
    default: throw std::invalid_argument("stockham: unsupported size " + std::to_string(n));
    }
}

// --- Entry points ------------------------------------------------------------------------------

inline void check_size(int n, const char* who)
{
    if (n < 1 || n > kStockhamMaxN || (n & (n - 1)) != 0) {
        throw std::invalid_argument(std::string(who) + ": n must be a power of two from 1 to "
                                    + std::to_string(kStockhamMaxN) + ", got "
                                    + std::to_string(n));
    }
}

// Complex transforms of n points. keep (1 .. n) prunes the side that is in the frequency domain:
// a forward transform writes only the first keep outputs of each transform, an inverse one reads
// only the first keep inputs (the rest are zero). keep = n is the plain transform.
template <int R, bool Table>
void c2c(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream,
         int keep)
{
    check_size(n, "stockham");
    if (batch < 1) {
        throw std::invalid_argument("stockham: batch must be at least 1, got "
                                    + std::to_string(batch));
    }
    if (keep < 1 || keep > n) {
        throw std::invalid_argument("stockham: keep must be in [1, " + std::to_string(n)
                                    + "], got " + std::to_string(keep));
    }
    const bool inverse = direction == Direction::Inverse;
    const int row_in = inverse ? keep : n;
    const int row_out = inverse ? n : keep;
    const auto rows = static_cast<std::size_t>(batch);
    if (x < y + rows * row_out && y < x + rows * row_in) {
        throw std::invalid_argument("stockham: x and y must not overlap (out-of-place only)");
    }
    if (n == 1) {  // a 1-point DFT is the identity, in both directions
        CUDA_CHECK(cudaMemcpyAsync(y, x, rows * sizeof(cf32), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    const float sign = inverse ? 1.0f : -1.0f;
    const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;  // exact: n = 2^k
    dispatch<R, Table>(n, C2cIo{x, y, scale, row_in, row_out}, batch, sign, stream);
}

template <int R, bool Table>
void c2c(const cf32* x, cf32* y, int n, int batch, Direction direction, cudaStream_t stream)
{
    c2c<R, Table>(x, y, n, batch, direction, stream, n);
}

// Column transforms; m1_in / m1_out > 0 prune the rows as StridedIo describes.
template <int R, bool Table>
void columns(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction,
             cudaStream_t stream, int m1_in, int m1_out)
{
    check_size(n, "stockham columns");
    if (cols < 1 || slabs < 1 || static_cast<long long>(cols) * slabs > INT_MAX) {
        throw std::invalid_argument("stockham columns: need 1 <= cols * slabs <= INT_MAX");
    }
    if (m1_in < 0 || m1_out < 0 || 2 * m1_in > n || 2 * m1_out > n) {
        throw std::invalid_argument("stockham columns: need 0 <= 2 m1 <= n = "
                                    + std::to_string(n));
    }
    if ((m1_in > 0 || m1_out > 0) && x == y) {
        throw std::invalid_argument("stockham columns: a pruned transform cannot run in place");
    }
    const int batch = cols * slabs;
    if (n == 1) {  // m1_in = m1_out = 0
        if (x != y) {
            const std::size_t bytes = static_cast<std::size_t>(batch) * sizeof(cf32);
            CUDA_CHECK(cudaMemcpyAsync(y, x, bytes, cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    const bool inverse = direction == Direction::Inverse;
    const float sign = inverse ? 1.0f : -1.0f;
    const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;
    dispatch<R, Table>(n, StridedIo{x, y, cols, scale, m1_in, m1_out}, batch, sign, stream);
}

template <int R, bool Table>
void columns(const cf32* x, cf32* y, int n, int cols, int slabs, Direction direction,
             cudaStream_t stream)
{
    columns<R, Table>(x, y, n, cols, slabs, direction, stream, 0, 0);
}

// Real transforms of n = 2m points run one m-point kernel each, so n goes up to 2 * 4096.
constexpr int kRealMaxN = 2 * kStockhamMaxN;

inline void check_keep(int keep, int n, const char* who)
{
    if (keep < 1 || keep > n / 2 + 1) {
        throw std::invalid_argument(std::string(who) + ": keep must be in [1, n/2 + 1 = "
                                    + std::to_string(n / 2 + 1) + "], got "
                                    + std::to_string(keep));
    }
}

// keep (1 .. n/2 + 1): the number of complex values per row of y.
template <int R, bool Table>
void r2c(const float* x, cf32* y, int n, int batch, cudaStream_t stream, int keep)
{
    check_keep(keep, n, "stockham r2c");
    real::check_arguments(x, y, n, batch, keep, kRealMaxN, "stockham r2c");
    const int m = n / 2;
    const auto* z = reinterpret_cast<const cf32*>(x);  // n reals are m complex values
    if (m == 1) {  // a 1-point DFT is the identity: only the combine step is left
        real::r2c_postprocess(z, y, 1, batch, keep, stream);
        return;
    }
    dispatch<R, Table>(m, R2cIo{z, y, keep}, batch, -1.0f, stream);
}

template <int R, bool Table>
void r2c(const float* x, cf32* y, int n, int batch, cudaStream_t stream)
{
    r2c<R, Table>(x, y, n, batch, stream, n / 2 + 1);
}

// keep (1 .. n/2 + 1): the number of complex values per row of x; the rest are zero.
template <int R, bool Table>
void c2r(const cf32* x, float* y, int n, int batch, cudaStream_t stream, int keep)
{
    check_keep(keep, n, "stockham c2r");
    real::check_arguments(y, x, n, batch, keep, kRealMaxN, "stockham c2r");
    const int m = n / 2;
    auto* z = reinterpret_cast<cf32*>(y);
    if (m == 1) {
        real::c2r_preprocess(x, z, 1, batch, keep, stream);
        return;
    }
    const float scale = 1.0f / static_cast<float>(m);  // exact: m = 2^k
    dispatch<R, Table>(m, C2rIo{x, z, scale, keep}, batch, 1.0f, stream);
}

template <int R, bool Table>
void c2r(const cf32* x, float* y, int n, int batch, cudaStream_t stream)
{
    c2r<R, Table>(x, y, n, batch, stream, n / 2 + 1);
}

// The pruned entry points of one variant, for the registry (registry.hpp).
template <int R, bool Table>
constexpr PrunedFns pruned_fns()
{
    return {
        [](const cf32* x, cf32* y, int n, int batch, int keep, Direction direction,
           cudaStream_t stream) { c2c<R, Table>(x, y, n, batch, direction, stream, keep); },
        [](const float* x, cf32* y, int n, int batch, int keep, cudaStream_t stream) {
            r2c<R, Table>(x, y, n, batch, stream, keep);
        },
        [](const cf32* x, float* y, int n, int batch, int keep, cudaStream_t stream) {
            c2r<R, Table>(x, y, n, batch, stream, keep);
        },
        [](const cf32* x, cf32* y, int n, int cols, int slabs, int m1_in, int m1_out,
           Direction direction, cudaStream_t stream) {
            columns<R, Table>(x, y, n, cols, slabs, direction, stream, m1_in, m1_out);
        },
    };
}

}  // namespace spectral::fft::stockham
