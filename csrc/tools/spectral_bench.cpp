// spectral_bench: times each implementation of each operation on the GPU and writes one CSV row
// per (operation, implementation, size, batch).
//
// Inputs are random complex64 values generated once on the host and uploaded once. Each size
// runs batch = total / n transforms, so every row processes the same number of points (2^24, or
// 128 MiB of complex64, by default): enough to fill the GPU at every size, and the rows are
// comparable with each other. Lines starting with '#' record the device and the settings.
//
// A row whose slowest round is more than 5 % slower than its fastest gets a warning on stderr:
// on this laptop that almost always means the GPU clock is not locked (docs/DEVELOPMENT.md).

#include "common/bench.hpp"
#include "common/check.hpp"
#include "common/device_buffer.hpp"
#include "common/fft_plan.hpp"
#include "common/types.hpp"
#include "fft/registry.hpp"
#include "ops/scale.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kUnstableSpread = 0.05;  // max_ms / min_ms - 1 above this gets a warning

const char* const kUsage =
    "usage: spectral_bench [options]\n"
    "  --op NAME       all (default), fft_c2c, or scale\n"
    "  --impl NAME     all (default), cufft, ours (the scale kernel), or one of our FFTs\n"
    "  --sizes LIST    comma-separated transform sizes (default: 2,4,...,4096)\n"
    "  --total N       points per row; batch = total / n (default: 16777216)\n"
    "  --warmup N      untimed launches (default: 3)\n"
    "  --rounds N      timed measurements; the median is reported (default: 5)\n"
    "  --iters N       launches per measurement (default: 20)\n"
    "  --out FILE      write the CSV to FILE instead of standard output\n";

struct Options {
    std::string op = "all";
    std::string impl = "all";
    std::vector<long long> sizes;
    long long total = 1LL << 24;
    TimingConfig timing;
    std::string out;

    bool wants(const std::string& name) const { return impl == "all" || impl == name; }
};

[[noreturn]] void usage_error(const std::string& message)
{
    throw std::invalid_argument(message + "\n" + kUsage);
}

long long parse_int(const std::string& text, const std::string& flag, long long min_value)
{
    std::size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || value < min_value) {
        usage_error(flag + " needs an integer of at least " + std::to_string(min_value)
                    + ", got '" + text + "'");
    }
    return value;
}

std::vector<long long> parse_sizes(const std::string& text)
{
    std::vector<long long> sizes;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        sizes.push_back(parse_int(item, "--sizes", 1));
    }
    if (sizes.empty()) {
        usage_error("--sizes needs at least one size");
    }
    return sizes;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::fputs(kUsage, stdout);
            std::printf("our FFTs: %s\n", spectral::fft::c2c_names().c_str());
            std::exit(0);
        }
        if (i + 1 >= argc) {
            usage_error("missing value after " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "--op") {
            options.op = value;
        } else if (flag == "--impl") {
            options.impl = value;
        } else if (flag == "--sizes") {
            options.sizes = parse_sizes(value);
        } else if (flag == "--total") {
            options.total = parse_int(value, flag, 1);
        } else if (flag == "--warmup") {
            options.timing.warmup = static_cast<int>(parse_int(value, flag, 0));
        } else if (flag == "--rounds") {
            options.timing.rounds = static_cast<int>(parse_int(value, flag, 1));
        } else if (flag == "--iters") {
            options.timing.iters = static_cast<int>(parse_int(value, flag, 1));
        } else if (flag == "--out") {
            options.out = value;
        } else {
            usage_error("unknown option " + flag);
        }
    }
    if (options.op != "all" && options.op != "fft_c2c" && options.op != "scale") {
        usage_error("unknown --op " + options.op);
    }
    if (options.impl != "all" && options.impl != "cufft" && options.impl != "ours"
        && spectral::fft::find_c2c(options.impl) == nullptr) {
        usage_error("unknown --impl " + options.impl + " (our FFTs: "
                    + spectral::fft::c2c_names() + ")");
    }
    if (options.sizes.empty()) {
        for (long long n = 2; n <= 4096; n *= 2) {
            options.sizes.push_back(n);
        }
    }
    return options;
}

void write_header(std::ostream& os, const Options& options)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int runtime = 0;
    int driver = 0;
    int cufft = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUFFT_CHECK(cufftGetVersion(&cufft));

    os << "# device: " << prop.name << ", compute capability " << prop.major << '.'
       << prop.minor << ", " << prop.multiProcessorCount << " SMs\n"
       << "# CUDA runtime " << runtime << ", driver " << driver << ", cuFFT " << cufft << '\n'
       << "# timing: " << options.timing.warmup << " warm-up launches, then the median over "
       << options.timing.rounds << " rounds of " << options.timing.iters
       << " launches; times are per launch\n"
       << "op,impl,n,batch,time_ms,min_ms,max_ms,gflops,gbps\n";
}

struct Row {
    const char* op;
    const char* impl;
    long long n;
    long long batch;
    Timing timing;
    Work work;
};

int unstable_rows = 0;

void write_row(std::ostream& os, const Row& r, bool progress)
{
    const double gflops = gflops_per_s(r.work, r.timing.median_ms);
    const double gbps = gbytes_per_s(r.work, r.timing.median_ms);
    char line[256];
    std::snprintf(line, sizeof line, "%s,%s,%lld,%lld,%.6f,%.6f,%.6f,%.1f,%.1f\n", r.op, r.impl,
                  r.n, r.batch, r.timing.median_ms, r.timing.min_ms, r.timing.max_ms, gflops,
                  gbps);
    os << line << std::flush;
    if (progress) {
        std::fprintf(stderr, "%-8s %-12s n=%-6lld batch=%-9lld %9.4f ms %8.1f GB/s\n", r.op,
                     r.impl, r.n, r.batch, r.timing.median_ms, gbps);
    }
    const double spread = r.timing.max_ms / r.timing.min_ms - 1.0;
    if (spread > kUnstableSpread) {
        ++unstable_rows;
        std::fprintf(stderr, "warning: %s %s n=%lld: rounds differ by %.0f %%\n", r.op, r.impl,
                     r.n, 100.0 * spread);
    }
}

std::vector<cf32> random_input(std::size_t count)
{
    std::mt19937 gen(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<cf32> values(count);
    for (cf32& z : values) {
        z = cf32{normal(gen), normal(gen)};
    }
    return values;
}

void bench_fft_c2c(const Options& options, const cf32* x, cf32* y, cudaStream_t stream,
                   std::ostream& os, bool progress)
{
    for (const long long n : options.sizes) {
        const long long batch = std::max(1LL, options.total / n);
        const Work work = fft_c2c_work(n, batch);

        if (options.wants("cufft")) {
            FftPlan plan(static_cast<int>(n), static_cast<int>(batch));
            CUFFT_CHECK(cufftSetStream(plan.get(), stream));
            // cuFFT's signature takes a non-const input even out of place, but a C2C transform
            // out of place leaves its input untouched.
            auto* in = const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(x));
            auto* out = reinterpret_cast<cufftComplex*>(y);
            const Timing t = time_kernel(
                [&](cudaStream_t) {
                    CUFFT_CHECK(cufftExecC2C(plan.get(), in, out, CUFFT_FORWARD));
                },
                stream, options.timing);
            write_row(os, {"fft_c2c", "cufft", n, batch, t, work}, progress);
        }

        for (const spectral::fft::C2cImpl& impl : spectral::fft::kC2cImpls) {
            if (!options.wants(impl.name) || n > impl.max_n) {
                continue;
            }
            const Timing t = time_kernel(
                [&](cudaStream_t s) {
                    impl.run(x, y, static_cast<int>(n), static_cast<int>(batch),
                             spectral::fft::Direction::Forward, s);
                },
                stream, options.timing);
            write_row(os, {"fft_c2c", impl.name, n, batch, t, work}, progress);
        }
    }
}

// A pure bandwidth reference: one read and one write per point, two multiplies.
void bench_scale(const Options& options, const cf32* x, cf32* y, cudaStream_t stream,
                 std::ostream& os, bool progress)
{
    const auto n = static_cast<std::size_t>(options.total);
    const Timing t = time_kernel(
        [&](cudaStream_t s) { spectral::scale_c2c(x, y, n, 0.5f, s); }, stream, options.timing);
    Work w;
    w.flops = 2.0 * static_cast<double>(n);
    w.bytes = 16.0 * static_cast<double>(n);
    write_row(os, {"scale", "ours", options.total, 1, t, w}, progress);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);

        long long capacity = options.total;
        for (const long long n : options.sizes) {
            capacity = std::max(capacity, n);
        }
        const auto count = static_cast<std::size_t>(capacity);
        DeviceBuffer<cf32> x(count);
        DeviceBuffer<cf32> y(count);
        {
            const std::vector<cf32> host = random_input(count);
            x.upload(host.data(), count);
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::ofstream file;
        if (!options.out.empty()) {
            file.open(options.out);
            if (!file) {
                throw std::runtime_error("cannot open " + options.out + " for writing");
            }
        }
        std::ostream& os = options.out.empty() ? std::cout : file;
        const bool progress = !options.out.empty();  // the CSV already goes to the terminal

        write_header(os, options);
        if (options.op == "all" || options.op == "fft_c2c") {
            bench_fft_c2c(options, x.get(), y.get(), stream, os, progress);
        }
        if ((options.op == "all" || options.op == "scale") && options.wants("ours")) {
            bench_scale(options, x.get(), y.get(), stream, os, progress);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        if (unstable_rows > 0) {
            std::fprintf(stderr,
                         "warning: %d unstable row(s); lock the GPU clock before trusting these "
                         "numbers (docs/DEVELOPMENT.md, Benchmarks)\n",
                         unstable_rows);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}