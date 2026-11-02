#pragma once

// Owning wrapper around a cuFFT plan, shared by spectrum.cu and xcorr.cu. Creating a plan is
// one-time setup: build it before Bench::run() so it stays out of every timing.

#include "check.hpp"

#include <cstdio>

// A cuFFT plan, destroyed when it goes out of scope.
class FftPlan {
public:
    // `batch` one-dimensional complex-to-complex transforms of `nfft` points each, stored
    // back to back in one buffer.
    FftPlan(int nfft, int batch)
    {
        // The two nullptr "embed" arguments select the simple layout: frame f occupies
        // elements [f * nfft, (f + 1) * nfft). The stride and distance values are then
        // ignored by cuFFT, but are filled in with what that layout means.
        CUFFT_CHECK(cufftPlanMany(&plan_, 1, &nfft, nullptr, 1, nfft, nullptr, 1, nfft,
                                  CUFFT_C2C, batch));
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
