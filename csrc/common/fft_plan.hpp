#pragma once

// Owning wrapper around a cuFFT plan, shared by spectrum.cu and xcorr.cu. Creating a plan is
// one-time setup: build it before Bench::run() so it stays out of every timing.

#include "check.hpp"

#include <cstdio>

// A cuFFT plan, destroyed when it goes out of scope.
class FftPlan {
public:
    // `batch` one-dimensional transforms of `nfft` points each, stored back to back in one
    // buffer: complex-to-complex by default, or CUFFT_R2C / CUFFT_C2R, whose complex side holds
    // nfft/2 + 1 values per transform.
    FftPlan(int nfft, int batch, cufftType type = CUFFT_C2C)
    {
        // The two nullptr "embed" arguments select the simple layout: frame f occupies
        // elements [f * len, (f + 1) * len), where len is nfft on the real or complex-to-complex
        // side and nfft/2 + 1 on the complex side of a real transform. The stride and distance
        // values are then ignored by cuFFT.
        CUFFT_CHECK(cufftPlanMany(&plan_, 1, &nfft, nullptr, 1, nfft, nullptr, 1, nfft, type,
                                  batch));
    }
    ~FftPlan()
    {
        const cufftResult err = cufftDestroy(plan_);
        if (err != CUFFT_SUCCESS) {
            std::fprintf(stderr, "warning: cufftDestroy failed: %s\n", cufft_error_string(err));
        }
    }
    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;

    cufftHandle get() const { return plan_; }

private:
    cufftHandle plan_{};
};
