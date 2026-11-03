// spectral_peaks: measures the two ceilings of the roofline model on this GPU and writes them as
// JSON.
//
//   peak_bandwidth_gbps  a copy kernel streaming 1 GiB through global memory (512 MiB read and
//                        512 MiB written), far more than the L2 cache holds, so every byte comes
//                        from DRAM. Counted like spectral_bench: bytes read plus bytes written.
//   peak_fp32_gflops     independent fused multiply-adds (2 flops each) with no memory traffic,
//                        with enough warps and independent chains to keep every FP32 unit busy.
//
// These are achievable peaks, measured on this machine at its current clocks and power limit,
// not the datasheet numbers: the best a well-written kernel can actually reach here.
//
//   spectral_peaks [--out FILE]

#include "common/bench.hpp"
#include "common/check.hpp"
#include "common/device_buffer.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// 16 bytes per load and per store, the widest single memory instructions, so instruction issue
// never becomes the limit before the memory system does.
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            std::size_t n)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = in[i];
    }
}

constexpr int kChains = 8;   // independent FMA chains per thread, to hide FMA latency
constexpr int kUnroll = 16;  // FMAs per chain per loop iteration

// Each chain repeats a <- a * b + c. With 0 < b < 1 the values converge (to c / (1 - b)) instead
// of overflowing. The final store is guarded by a condition the compiler cannot prove false, so
// it cannot delete the arithmetic as unused.
__global__ void fma_kernel(float* out, int iters, float b, float c)
{
    float a[kChains];
#pragma unroll
    for (int k = 0; k < kChains; ++k) {
        a[k] = static_cast<float>(threadIdx.x + k);
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
#pragma unroll
            for (int k = 0; k < kChains; ++k) {
                a[k] = fmaf(a[k], b, c);
            }
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kChains; ++k) {
        sum += a[k];
    }
    if (sum == -1.0f) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    }
}

int sm_count()
{
    int device = 0;
    int sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms;
}

double measure_bandwidth(cudaStream_t stream, const TimingConfig& config)
{
    constexpr std::size_t bytes = std::size_t{512} << 20;  // per buffer
    constexpr std::size_t n = bytes / sizeof(float4);
    DeviceBuffer<float4> in(n);
    DeviceBuffer<float4> out(n);
    in.zero();

    constexpr int block = 256;
    const auto blocks = static_cast<unsigned>(sm_count() * 32);  // many waves; the loop covers n
    const Timing t = time_kernel(
        [&](cudaStream_t s) { copy_kernel<<<blocks, block, 0, s>>>(in.get(), out.get(), n); },
        stream, config);
    Work w;
    w.bytes = 2.0 * static_cast<double>(bytes);
    return gbytes_per_s(w, t.median_ms);
}

double measure_fp32(cudaStream_t stream, const TimingConfig& config)
{
    constexpr int block = 256;
    constexpr int iters = 4096;
    // Exactly one full wave: as many blocks as fit on every SM at once, so no SM sits idle
    // waiting for a last, partial wave.
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fma_kernel, block, 0));
    const int blocks = sm_count() * per_sm;
    DeviceBuffer<float> out(static_cast<std::size_t>(blocks) * block);

    const Timing t = time_kernel(
        [&](cudaStream_t s) {
            fma_kernel<<<blocks, block, 0, s>>>(out.get(), iters, 0.999f, 0.001f);
        },
        stream, config);
    Work w;
    w.flops = 2.0 * kChains * kUnroll * iters * static_cast<double>(blocks) * block;
    return gflops_per_s(w, t.median_ms);
}

void write_json(std::ostream& os, double bandwidth, double fp32)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    char text[512];
    std::snprintf(text, sizeof text,
                  "{\n"
                  "  \"device\": \"%s\",\n"
                  "  \"peak_bandwidth_gbps\": %.1f,\n"
                  "  \"peak_fp32_gflops\": %.1f,\n"
                  "  \"ridge_flop_per_byte\": %.2f\n"
                  "}\n",
                  prop.name, bandwidth, fp32, fp32 / bandwidth);
    os << text;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::string out;
        if (argc == 3 && std::string(argv[1]) == "--out") {
            out = argv[2];
        } else if (argc != 1) {
            std::fputs("usage: spectral_peaks [--out FILE]\n", stderr);
            return 1;
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        TimingConfig config;
        config.iters = 10;
        const double bandwidth = measure_bandwidth(stream, config);
        const double fp32 = measure_fp32(stream, config);
        CUDA_CHECK(cudaStreamDestroy(stream));

        if (out.empty()) {
            write_json(std::cout, bandwidth, fp32);
        } else {
            std::ofstream file(out);
            if (!file) {
                throw std::runtime_error("cannot open " + out + " for writing");
            }
            write_json(file, bandwidth, fp32);
            std::fprintf(stderr, "bandwidth %.1f GB/s, FP32 %.1f GFLOP/s -> %s\n", bandwidth,
                         fp32, out.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}