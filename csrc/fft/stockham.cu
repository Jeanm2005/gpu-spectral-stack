// Stockham FFTs that keep each transform in shared memory (Tasks 1.2 and 1.3).
//
// One kernel launch per call. Each block loads its transforms from global memory once
// (coalesced), runs every stage in shared memory, and writes the results once, so the data
// crosses global memory exactly twice, as in an ideal FFT. The Stockham formulation writes each
// stage's outputs to their sorted positions as it goes, so the result comes out in natural order
// without a bit-reversal pass.
//
// Every size from 2 to 4096 has its own compiled kernel (N is a template parameter): all loop
// counts are then compile-time constants, the loops unroll, and each thread's values stay in
// registers.

#include "fft/fft.hpp"

#include "common/check.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace spectral::fft {
namespace {

constexpr int kThreads = 256;  // threads per block, for every size and radix

__host__ __device__ constexpr int ilog2(int n)
{
    int r = 0;
    while ((1 << r) < n) {
        ++r;
    }
    return r;
}

// The launch shape of the size-N, radix-R kernel. Each transform is handled by kLanes threads,
// and a block holds kTransformsPerBlock transforms side by side, so that small transforms still
// fill a block of 256 threads.
template <int N, int R>
struct Shape {
    static_assert(N >= R && (N & (N - 1)) == 0, "N must be a power of two, at least R");
    static constexpr int kButterflies = N / R;  // per radix-R stage
    static constexpr int kLanes = kButterflies < kThreads ? kButterflies : kThreads;
    static constexpr int kTransformsPerBlock = kThreads / kLanes;
    static constexpr int kSharedElements = kTransformsPerBlock * N;  // at most 4096: 32 KiB
    static constexpr int kStages = ilog2(N) / ilog2(R);              // radix-R stages
    static constexpr bool kFinalRadix2 = kStages * ilog2(R) < ilog2(N);
};

__device__ __forceinline__ cf32 add(cf32 a, cf32 b) { return {a.re + b.re, a.im + b.im}; }
__device__ __forceinline__ cf32 sub(cf32 a, cf32 b) { return {a.re - b.re, a.im - b.im}; }
__device__ __forceinline__ cf32 mul(cf32 a, cf32 w)
{
    return {a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
}

// exp(sign * pi * i * a). sincospif rounds neither pi nor pi * a.
__device__ __forceinline__ cf32 twiddle(float sign, float a)
{
    float s = 0.0f;
    float c = 0.0f;
    sincospif(sign * a, &s, &c);
    return {c, s};
}

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

// One radix-R Stockham stage over an N-point transform in shared memory `s`: sub-transforms of
// length ns become sub-transforms of length ns * R. Butterfly j (with k = j % ns) reads
// s[j + r * N/R] for r < R, multiplies input r by exp(sign * 2 pi i * k * r / (ns * R)), takes an
// R-point DFT, and writes output r to s[(j - k) * R + k + r * ns].
//
// Each lane handles butterflies lane, lane + Lanes, ... Every thread reads all of its inputs
// before any thread writes, so a single buffer is enough: the first barrier separates this
// stage's reads from its writes, the second its writes from the next stage's reads.
template <int N, int R, int Lanes>
__device__ __forceinline__ void stage(cf32* s, int ns, int lane, float sign)
{
    constexpr int kButterflies = N / R;
    constexpr int kPerLane = kButterflies / Lanes;
    cf32 v[kPerLane][R];

#pragma unroll
    for (int b = 0; b < kPerLane; ++b) {
        const int j = lane + b * Lanes;
        const int k = j % ns;
        // 2k / (ns R) and its multiples by r < R are exact in float32.
        const float step = 2.0f * static_cast<float>(k) / static_cast<float>(ns * R);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const cf32 x = s[j + r * kButterflies];
            v[b][r] = r == 0 ? x : mul(x, twiddle(sign, step * static_cast<float>(r)));
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

template <int N, int R>
__global__ void __launch_bounds__(kThreads)
    stockham_kernel(const cf32* __restrict__ x, cf32* __restrict__ y, int batch, float sign,
                    float scale)
{
    using S = Shape<N, R>;
    __shared__ cf32 smem[S::kSharedElements];

    // This block's transforms are contiguous in global memory: load them with consecutive
    // threads reading consecutive elements. The last block may hold fewer transforms; its
    // threads still take part in every barrier, they just skip the loads and stores.
    const int first = blockIdx.x * S::kTransformsPerBlock;
    const int count = min(S::kTransformsPerBlock, batch - first) * N;
    const std::size_t base = static_cast<std::size_t>(first) * N;
    for (int e = threadIdx.x; e < count; e += kThreads) {
        smem[e] = x[base + e];
    }
    __syncthreads();

    cf32* s = smem + (threadIdx.x / S::kLanes) * N;
    const int lane = threadIdx.x % S::kLanes;
    int ns = 1;
#pragma unroll
    for (int i = 0; i < S::kStages; ++i) {
        stage<N, R, S::kLanes>(s, ns, lane, sign);
        ns *= R;
    }
    if constexpr (S::kFinalRadix2) {
        stage<N, 2, S::kLanes>(s, ns, lane, sign);
    }

    for (int e = threadIdx.x; e < count; e += kThreads) {
        const cf32 v = smem[e];
        y[base + e] = cf32{v.re * scale, v.im * scale};
    }
}

template <int N, int R>
void run(const cf32* x, cf32* y, int batch, float sign, float scale, cudaStream_t stream)
{
    using S = Shape<N, R>;
    const long long per_block = S::kTransformsPerBlock;
    const auto blocks = static_cast<unsigned>((batch + per_block - 1) / per_block);
    stockham_kernel<N, R><<<blocks, kThreads, 0, stream>>>(x, y, batch, sign, scale);
    CUDA_CHECK(cudaGetLastError());
}

// One case per size. A 2-point transform is a single radix-2 butterfly for either radix.
template <int R>
void dispatch(const cf32* x, cf32* y, int n, int batch, float sign, float scale,
              cudaStream_t stream)
{
    switch (n) {
    case 2: run<2, 2>(x, y, batch, sign, scale, stream); break;
    case 4: run<4, R>(x, y, batch, sign, scale, stream); break;
    case 8: run<8, R>(x, y, batch, sign, scale, stream); break;
    case 16: run<16, R>(x, y, batch, sign, scale, stream); break;
    case 32: run<32, R>(x, y, batch, sign, scale, stream); break;
    case 64: run<64, R>(x, y, batch, sign, scale, stream); break;
    case 128: run<128, R>(x, y, batch, sign, scale, stream); break;
    case 256: run<256, R>(x, y, batch, sign, scale, stream); break;
    case 512: run<512, R>(x, y, batch, sign, scale, stream); break;
    case 1024: run<1024, R>(x, y, batch, sign, scale, stream); break;
    case 2048: run<2048, R>(x, y, batch, sign, scale, stream); break;
    case 4096: run<4096, R>(x, y, batch, sign, scale, stream); break;
    default: throw std::invalid_argument("stockham: unsupported size " + std::to_string(n));
    }
}

void stockham(const cf32* x, cf32* y, int n, int batch, Direction direction,
              cudaStream_t stream, int radix)
{
    if (n < 1 || n > kStockhamMaxN || (n & (n - 1)) != 0) {
        throw std::invalid_argument("stockham: n must be a power of two from 1 to "
                                    + std::to_string(kStockhamMaxN) + ", got "
                                    + std::to_string(n));
    }
    if (batch < 1) {
        throw std::invalid_argument("stockham: batch must be at least 1, got "
                                    + std::to_string(batch));
    }
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    if (x < y + total && y < x + total) {
        throw std::invalid_argument("stockham: x and y must not overlap (out-of-place only)");
    }

    if (n == 1) {  // a 1-point DFT is the identity, in both directions
        CUDA_CHECK(cudaMemcpyAsync(y, x, total * sizeof(cf32), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const bool inverse = direction == Direction::Inverse;
    const float sign = inverse ? 1.0f : -1.0f;
    const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;  // exact: n = 2^k
    if (radix == 2) {
        dispatch<2>(x, y, n, batch, sign, scale, stream);
    } else {
        dispatch<4>(x, y, n, batch, sign, scale, stream);
    }
}

}  // namespace

void c2c_stockham_r2(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham(x, y, n, batch, direction, stream, 2);
}

void c2c_stockham_r4(const cf32* x, cf32* y, int n, int batch, Direction direction,
                     cudaStream_t stream)
{
    stockham(x, y, n, batch, direction, stream, 4);
}

}  // namespace spectral::fft