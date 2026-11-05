// spectral_bench: times each implementation of each operation on the GPU and writes one CSV row
// per (operation, implementation, size, batch).
//
// Inputs are random complex64 values generated once on the host and uploaded once. Each size
// runs batch = total / n transforms, so every row processes the same number of points (2^24, or
// 128 MiB of complex64, by default): enough to fill the GPU at every size, and the rows are
// comparable with each other. Lines starting with '#' record the device and the settings.
//
// 2D rows (fft2_*) transform square n x n arrays, batch = total / n^2 of them (sizes whose array
// exceeds `total` are skipped). Our implementations appear once per column method, as
// "<impl>/strided" and "<impl>/transpose" (fft/fft2d.hpp).
//
// Pruned 2D rows (rfft2_modes, irfft2_modes) use the same arrays and keep m1 = m2 = n/8 modes
// along each axis (at least 1), an FNO-like fraction. cuFFT has no pruned transform, so its row
// is the full transform plus our gather (or scatter) of the modes: "cufft+gather".
//
// A row whose slowest round is more than 5 % slower than its fastest gets a warning on stderr:
// on this laptop that almost always means the GPU clock is not locked (docs/DEVELOPMENT.md).

#include "common/bench.hpp"
#include "common/check.hpp"
#include "common/device_buffer.hpp"
#include "common/fft_plan.hpp"
#include "common/types.hpp"
#include "fft/fft2d.hpp"
#include "fft/pruned.hpp"
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
    "  --op NAME       all (default), fft_c2c, fft_r2c, fft_c2r, fft2_c2c, fft2_r2c,\n"
    "                  fft2_c2r, rfft2_modes, irfft2_modes, or scale\n"
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
            std::printf("our FFTs: %s\n", spectral::fft::impl_names().c_str());
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
    const bool known_op =
        options.op == "all" || options.op == "fft_c2c" || options.op == "fft_r2c"
        || options.op == "fft_c2r" || options.op == "fft2_c2c" || options.op == "fft2_r2c"
        || options.op == "fft2_c2r" || options.op == "rfft2_modes"
        || options.op == "irfft2_modes" || options.op == "scale";
    if (!known_op) {
        usage_error("unknown --op " + options.op);
    }
    if (options.impl != "all" && options.impl != "cufft" && options.impl != "ours"
        && spectral::fft::find_impl(options.impl) == nullptr) {
        usage_error("unknown --impl " + options.impl + " (our FFTs: "
                    + spectral::fft::impl_names() + ")");
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

        for (const spectral::fft::FftImpl& impl : spectral::fft::kFftImpls) {
            if (!options.wants(impl.name) || n > impl.max_n) {
                continue;
            }
            const Timing t = time_kernel(
                [&](cudaStream_t s) {
                    impl.c2c(x, y, static_cast<int>(n), static_cast<int>(batch),
                             spectral::fft::Direction::Forward, s);
                },
                stream, options.timing);
            write_row(os, {"fft_c2c", impl.name, n, batch, t, work}, progress);
        }
    }
}

// Real transforms of n points, batch = total / n: forward (fft_r2c) reads n floats and writes
// n/2 + 1 complex values per transform; fft_c2r the reverse. The buffers hold `total` complex
// values, more than either side needs. The C2R input is random complex data: only the timing
// matters here, and cuFFT may overwrite a C2R input, which it would also do to real data.
void bench_fft_real(const Options& options, const char* op, const cf32* x, cf32* y, cf32* work,
                    cudaStream_t stream, std::ostream& os, bool progress)
{
    const bool forward = std::string(op) == "fft_r2c";
    for (const long long n : options.sizes) {
        if (n < 2) {
            continue;
        }
        const long long batch = std::max(1LL, options.total / n);
        const Work w = fft_real_work(n, batch);
        const auto* x_real = reinterpret_cast<const float*>(x);
        auto* y_real = reinterpret_cast<float*>(y);

        if (options.wants("cufft")) {
            FftPlan plan(static_cast<int>(n), static_cast<int>(batch),
                         forward ? CUFFT_R2C : CUFFT_C2R);
            CUFFT_CHECK(cufftSetStream(plan.get(), stream));
            const Timing t = time_kernel(
                [&](cudaStream_t) {
                    if (forward) {
                        CUFFT_CHECK(cufftExecR2C(plan.get(), const_cast<float*>(x_real),
                                                 reinterpret_cast<cufftComplex*>(y)));
                    } else {
                        CUFFT_CHECK(cufftExecC2R(
                            plan.get(), const_cast<cufftComplex*>(
                                            reinterpret_cast<const cufftComplex*>(x)),
                            y_real));
                    }
                },
                stream, options.timing);
            write_row(os, {op, "cufft", n, batch, t, w}, progress);
        }

        for (const spectral::fft::FftImpl& impl : spectral::fft::kFftImpls) {
            if (!options.wants(impl.name) || n > impl.max_real_n) {
                continue;
            }
            const Timing t = time_kernel(
                [&](cudaStream_t s) {
                    if (forward) {
                        impl.r2c(x_real, y, static_cast<int>(n), static_cast<int>(batch), work, s);
                    } else {
                        impl.c2r(x, y_real, static_cast<int>(n), static_cast<int>(batch), work, s);
                    }
                },
                stream, options.timing);
            write_row(os, {op, impl.name, n, batch, t, w}, progress);
        }
    }
}

// 2D transforms of square n x n arrays (see the top of this file). `work` must hold the largest
// workspace any implementation asks for; main() allocates 3 * total complex values.
void bench_fft2(const Options& options, const std::string& op, const cf32* x, cf32* y,
                cf32* work, cudaStream_t stream, std::ostream& os, bool progress)
{
    using spectral::fft::Columns;
    const bool c2c = op == "fft2_c2c";
    const bool r2c = op == "fft2_r2c";
    for (const long long n : options.sizes) {
        if (n * n > options.total || (!c2c && n < 2)) {
            continue;
        }
        const long long batch = options.total / (n * n);
        const int ni = static_cast<int>(n);
        const int bi = static_cast<int>(batch);
        const Work w = c2c ? fft2_c2c_work(n, n, batch) : fft2_real_work(n, n, batch);
        const auto* x_real = reinterpret_cast<const float*>(x);
        auto* y_real = reinterpret_cast<float*>(y);
        auto* x_mut = const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(x));

        if (options.wants("cufft")) {
            FftPlan plan(ni, ni, bi, c2c ? CUFFT_C2C : (r2c ? CUFFT_R2C : CUFFT_C2R));
            CUFFT_CHECK(cufftSetStream(plan.get(), stream));
            const Timing t = time_kernel(
                [&](cudaStream_t) {
                    if (c2c) {
                        CUFFT_CHECK(cufftExecC2C(plan.get(), x_mut,
                                                 reinterpret_cast<cufftComplex*>(y),
                                                 CUFFT_FORWARD));
                    } else if (r2c) {
                        CUFFT_CHECK(cufftExecR2C(plan.get(), const_cast<float*>(x_real),
                                                 reinterpret_cast<cufftComplex*>(y)));
                    } else {
                        CUFFT_CHECK(cufftExecC2R(plan.get(), x_mut, y_real));
                    }
                },
                stream, options.timing);
            write_row(os, {op.c_str(), "cufft", n, batch, t, w}, progress);
        }

        for (const spectral::fft::FftImpl& impl : spectral::fft::kFftImpls) {
            const int max_n2 = c2c ? impl.max_n : impl.max_real_n;
            if (!options.wants(impl.name) || n > impl.max_n || n > max_n2) {
                continue;
            }
            for (const Columns method : {Columns::Strided, Columns::Transpose}) {
                if (method == Columns::Strided && impl.columns == nullptr) {
                    continue;
                }
                const Timing t = time_kernel(
                    [&](cudaStream_t s) {
                        if (c2c) {
                            spectral::fft::fft2_c2c(impl, method, x, y, ni, ni, bi,
                                                    spectral::fft::Direction::Forward, work, s);
                        } else if (r2c) {
                            spectral::fft::fft2_r2c(impl, method, x_real, y, ni, ni, bi, work, s);
                        } else {
                            spectral::fft::fft2_c2r(impl, method, x, y_real, ni, ni, bi, work, s);
                        }
                    },
                    stream, options.timing);
                const std::string label = std::string(impl.name)
                                          + (method == Columns::Strided ? "/strided"
                                                                        : "/transpose");
                write_row(os, {op.c_str(), label.c_str(), n, batch, t, w}, progress);
            }
        }
    }
}

// Pruned 2D real transforms of square n x n arrays (see the top of this file).
void bench_modes(const Options& options, const std::string& op, const cf32* x, cf32* y,
                 cf32* work, cudaStream_t stream, std::ostream& os, bool progress)
{
    const bool forward = op == "rfft2_modes";
    for (const long long n : options.sizes) {
        if (n < 2 || n * n > options.total) {
            continue;
        }
        const long long batch = options.total / (n * n);
        const long long m = std::max(1LL, n / 8);
        const int ni = static_cast<int>(n);
        const int mi = static_cast<int>(m);
        const int bi = static_cast<int>(batch);
        const Work w = fft2_modes_work(n, n, m, m, batch);
        const auto* x_real = reinterpret_cast<const float*>(x);
        auto* y_real = reinterpret_cast<float*>(y);

        if (options.wants("cufft")) {
            FftPlan plan(ni, ni, bi, forward ? CUFFT_R2C : CUFFT_C2R);
            CUFFT_CHECK(cufftSetStream(plan.get(), stream));
            auto* full = reinterpret_cast<cufftComplex*>(work);
            const Timing t = time_kernel(
                [&](cudaStream_t s) {
                    if (forward) {
                        CUFFT_CHECK(cufftExecR2C(plan.get(), const_cast<float*>(x_real), full));
                        spectral::fft::gather_modes(work, y, ni, ni / 2 + 1, mi, mi, bi, s);
                    } else {
                        spectral::fft::scatter_modes(x, work, ni, ni / 2 + 1, mi, mi, bi, s);
                        CUFFT_CHECK(cufftExecC2R(plan.get(), full, y_real));
                    }
                },
                stream, options.timing);
            write_row(os, {op.c_str(), forward ? "cufft+gather" : "cufft+scatter", n, batch, t, w},
                      progress);
        }

        for (const spectral::fft::FftImpl& impl : spectral::fft::kFftImpls) {
            if (!options.wants(impl.name) || n > impl.max_n) {
                continue;
            }
            const Timing t = time_kernel(
                [&](cudaStream_t s) {
                    if (forward) {
                        spectral::fft::rfft2_modes(impl, x_real, y, ni, ni, mi, mi, bi, work, s);
                    } else {
                        spectral::fft::irfft2_modes(impl, x, y_real, ni, ni, mi, mi, bi, work, s);
                    }
                },
                stream, options.timing);
            write_row(os, {op.c_str(), impl.name, n, batch, t, w}, progress);
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
        DeviceBuffer<cf32> work(3 * count);  // the largest workspace: 2D, transposed, naive
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
        for (const char* op : {"fft_r2c", "fft_c2r"}) {
            if (options.op == "all" || options.op == op) {
                bench_fft_real(options, op, x.get(), y.get(), work.get(), stream, os, progress);
            }
        }
        for (const char* op : {"fft2_c2c", "fft2_r2c", "fft2_c2r"}) {
            if (options.op == "all" || options.op == op) {
                bench_fft2(options, op, x.get(), y.get(), work.get(), stream, os, progress);
            }
        }
        for (const char* op : {"rfft2_modes", "irfft2_modes"}) {
            if (options.op == "all" || options.op == op) {
                bench_modes(options, op, x.get(), y.get(), work.get(), stream, os, progress);
            }
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