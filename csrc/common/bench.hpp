#pragma once

// Timing harness for GPU kernels.
//
// time_kernel() measures how long one launch of `launch` takes, with the data already on the
// GPU. A single small kernel (a batch of 64-point FFTs can take a few microseconds) is close to
// the resolution of a pair of CUDA events, so each measurement times a loop of `iters`
// back-to-back launches and divides by `iters`. After `warmup` untimed launches, the measurement
// is repeated `rounds` times and the median is reported: it ignores the occasional round
// disturbed by clocks ramping up or another process using the GPU. The minimum is the best the
// kernel managed.
//
// One-time setup (allocation, cuFFT plan creation, uploading inputs) happens before the call and
// is never timed.

#include <cuda_runtime.h>

#include <functional>

struct TimingConfig {
    int warmup = 3;  // untimed launches before measuring
    int rounds = 5;  // timed measurements; the median is reported
    int iters = 20;  // launches per measurement
};

struct Timing {
    double median_ms = 0.0;  // all three are per launch
    double min_ms = 0.0;
    double max_ms = 0.0;
};

// Enqueues one launch of the kernel or library call being measured on `stream`.
using LaunchFn = std::function<void(cudaStream_t stream)>;

Timing time_kernel(const LaunchFn& launch, cudaStream_t stream, const TimingConfig& config);

// The work one launch does, for turning a time into rates.
struct Work {
    double flops = 0.0;  // floating-point operations, by the operation's counting convention
    double bytes = 0.0;  // minimum global-memory traffic: each input read once, each output
                         // written once
};

double gflops_per_s(const Work& work, double ms);
double gbytes_per_s(const Work& work, double ms);

// The convention every FFT implementation is measured by, so all of them share one scale:
// 5 N log2 N flops per complex transform of N points (the operation count of radix-2
// Cooley-Tukey, used by FFTW's and cuFFT's own benchmarks), and one complex64 read plus one
// write per point. spectral.benchmark.fft_c2c_work in Python uses the same convention.
Work fft_c2c_work(long long n, long long batch);

// Real transforms of n points (either direction) by the same convention: half the flops of a
// complex transform, 2.5 N log2 N, and n float32 values on the real side plus n/2 + 1 complex64
// values on the complex side. spectral.benchmark.fft_real_work matches it.
Work fft_real_work(long long n, long long batch);