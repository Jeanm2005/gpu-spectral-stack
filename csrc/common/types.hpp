#pragma once

// One complex float32 sample, laid out exactly as it is stored in a .cf32 file: I then Q.
// An array of cf32 is therefore the interleaved [I0, Q0, I1, Q1, ...] buffer, and it has the
// same layout as cufftComplex, so buffers can be handed to cuFFT without copying.
// Plain struct with no constructors so that it can be used in device code.
struct cf32 {
    float re;
    float im;
};

static_assert(sizeof(cf32) == 2 * sizeof(float), "cf32 must be two packed floats");
