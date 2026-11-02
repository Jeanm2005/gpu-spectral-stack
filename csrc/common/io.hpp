#pragma once

// File I/O for the two raw formats the pipeline uses, and for SigMF recordings. Both raw formats
// are little-endian float32 with no header, read and written directly, so this code assumes a
// little-endian host (x86-64 and the ARM cores in Jetson boards both are). Every function
// throws std::runtime_error with the file path on failure.

#include "types.hpp"

#include <string>
#include <vector>

// .cf32: interleaved [I0, Q0, I1, Q1, ...]. Fails if the file holds an odd number of floats.
std::vector<cf32> read_cf32(const std::string& path);
void write_cf32(const std::string& path, const std::vector<cf32>& samples);

// .f32: plain float32 array (filter taps, or a magnitude result).
std::vector<float> read_f32(const std::string& path);
void write_f32(const std::string& path, const std::vector<float>& values);

// SigMF recordings (https://sigmf.org): PATH.sigmf-meta (JSON) next to PATH.sigmf-data (raw
// samples). read_cf32 accepts either file of a pair, reads "core:datatype" from the metadata,
// and converts cf32_le, ci16_le, and ci8 to cf32, scaling integers by 1/32768 and 1/128 so
// full scale is about 1.0. Big-endian, real-valued, and multi-channel recordings are rejected.
bool is_sigmf(const std::string& path);

// Sample rate in Hz from a file's metadata: "core:sample_rate" for a SigMF recording,
// "sample_rate" from the PATH.json sidecar tools/gen_fixtures.py writes next to a .cf32 file,
// or 0 when there is no metadata.
double sample_rate(const std::string& path);
