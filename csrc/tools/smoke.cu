#include "common/check.hpp"
#include "common/device_buffer.hpp"
#include "common/types.hpp"
#include <cstdio>
#include <exception>
#include <vector>

namespace {
    __global__ void fill_kernel(cf32* out, int n)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i].re = static_cast<float>(i);
            out[i].im = -static_cast<float>(i);
        }
    }

    void print_version(const char* label, int v)
    {
        std::printf("%-20s %d.%d\n", label, v / 1000, (v % 1000) / 10);
    }

    void print_device()
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        int runtime = 0;
        int driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));

        std::printf("%-20s %s\n", "device", prop.name);
        std::printf("%-20s %d.%d\n", "compute capability", prop.major, prop.minor);
        std::printf("%-20s %d\n", "SMs", prop.multiProcessorCount);
        std::printf("%-20s %.1f GiB\n", "global memory",
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("%-20s %zu KiB\n", "shared mem / block",
                prop.sharedMemPerBlockOptin / 1024);
        print_version("CUDA runtime", runtime);
        print_version("CUDA driver", driver);
    }
} // namespace

int main()
{
    try {
        print_device();

        constexpr int n = 1 << 20;
        constexpr int block = 256;
        DeviceBuffer<cf32> buffer(n);

        fill_kernel<<<(n + block - 1) / block, block>>>(buffer.get(), n);
        CUDA_CHECK(cudaGetLastError());

        std::vector<cf32> host(n);
        buffer.download(host.data(), n);

        for (int i = 0; i < n; ++i) {
            if (host[i].re != static_cast<float>(i) || host[i].im != -static_cast<float>(i)) {
                std::fprintf(stderr, "kernel check FAILED at element %d: (%g, %g)\n", i, host[i].re, host[i].im);
                return 1;
            }
        }
        std::printf("%-20s OK (%d elements)\n", "kernel check", n);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}