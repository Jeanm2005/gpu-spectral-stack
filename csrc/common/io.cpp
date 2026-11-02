#include "io.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Reads a whole file as an array of T. `what` names the element in error messages.
template <class T>
std::vector<T> read_array(const std::string& path, const char* what)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamoff bytes = file.tellg();
    if (bytes < 0) {
        throw std::runtime_error("cannot read the size of " + path);
    }
    if (static_cast<std::uintmax_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error(path + ": size " + std::to_string(bytes)
                                 + " bytes is not a whole number of " + what + " ("
                                 + std::to_string(sizeof(T)) + " bytes each)");
    }
    std::vector<T> data(static_cast<std::size_t>(bytes) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!file) {
        throw std::runtime_error("failed while reading " + path);
    }
    return data;
}

// Writes `data`, `count` elements of T, to `path`, creating the parent directory if needed.
template <class T>
void write_array(const std::string& path, const T* data, std::size_t count)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("cannot create directory " + parent.string() + ": "
                                     + ec.message());
        }
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open " + path + " for writing");
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    file.close();
    if (!file) {
        throw std::runtime_error("failed while writing " + path);
    }
}

bool ends_with(const std::string& text, const std::string& suffix)
{
    return text.size() >= suffix.size()
           && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string read_text(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

// The value of the first "key": VALUE pair in a JSON text, as raw text: a string without its
// quotes, or a number. Empty when the key is missing. Enough for the flat metadata fields read
// here; it is not a JSON parser (escapes inside strings are not decoded).
std::string json_field(const std::string& json, const std::string& key)
{
    const std::size_t at = json.find("\"" + key + "\"");
    if (at == std::string::npos) {
        return "";
    }
    std::size_t i = json.find(':', at + key.size() + 2);
    if (i == std::string::npos) {
        return "";
    }
    i = json.find_first_not_of(" \t\r\n", i + 1);
    if (i == std::string::npos) {
        return "";
    }
    if (json[i] == '"') {
        const std::size_t end = json.find('"', i + 1);
        return end == std::string::npos ? "" : json.substr(i + 1, end - i - 1);
    }
    const std::size_t end = json.find_first_of(",}] \t\r\n", i);
    return json.substr(i, end == std::string::npos ? std::string::npos : end - i);
}

// PATH.sigmf-meta and PATH.sigmf-data for either file of a SigMF pair.
std::string sigmf_base(const std::string& path)
{
    return path.substr(0, path.size() - std::string(".sigmf-meta").size());
}

// Integer samples, interleaved I and Q, scaled by `scale` into cf32.
template <class T>
std::vector<cf32> int_to_cf32(const std::string& path, float scale)
{
    const std::vector<T> raw = read_array<T>(path, "integer components");
    if (raw.size() % 2 != 0) {
        throw std::runtime_error(path + ": odd number of components, not I/Q pairs");
    }
    std::vector<cf32> out(raw.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].re = static_cast<float>(raw[2 * i]) * scale;
        out[i].im = static_cast<float>(raw[2 * i + 1]) * scale;
    }
    return out;
}

std::vector<cf32> read_sigmf(const std::string& path)
{
    const std::string base = sigmf_base(path);
    const std::string meta = read_text(base + ".sigmf-meta");
    const std::string data = base + ".sigmf-data";
    const std::string datatype = json_field(meta, "core:datatype");
    const std::string channels = json_field(meta, "core:num_channels");
    if (!channels.empty() && std::atoi(channels.c_str()) != 1) {
        throw std::runtime_error(base + ".sigmf-meta: " + channels
                                 + " channels; only single-channel recordings are supported");
    }
    if (datatype == "cf32_le") {
        return read_array<cf32>(data, "complex float32 samples");
    }
    if (datatype == "ci16_le") {
        return int_to_cf32<std::int16_t>(data, 1.0f / 32768.0f);
    }
    if (datatype == "ci8" || datatype == "ci8_le") {
        return int_to_cf32<std::int8_t>(data, 1.0f / 128.0f);
    }
    throw std::runtime_error(base + ".sigmf-meta: core:datatype '" + datatype
                             + "' is not supported (cf32_le, ci16_le, ci8)");
}

}  // namespace

bool is_sigmf(const std::string& path)
{
    return ends_with(path, ".sigmf-meta") || ends_with(path, ".sigmf-data");
}

double sample_rate(const std::string& path)
{
    std::string meta_path;
    std::string key;
    if (is_sigmf(path)) {
        meta_path = sigmf_base(path) + ".sigmf-meta";
        key = "core:sample_rate";
    } else {
        meta_path = std::filesystem::path(path).replace_extension(".json").string();
        key = "sample_rate";
        if (!std::filesystem::exists(meta_path)) {
            return 0.0;
        }
    }
    const std::string value = json_field(read_text(meta_path), key);
    return value.empty() ? 0.0 : std::strtod(value.c_str(), nullptr);
}

std::vector<cf32> read_cf32(const std::string& path)
{
    if (is_sigmf(path)) {
        return read_sigmf(path);
    }
    return read_array<cf32>(path, "complex float32 samples");
}

std::vector<float> read_f32(const std::string& path)
{
    return read_array<float>(path, "float32 values");
}

void write_cf32(const std::string& path, const std::vector<cf32>& samples)
{
    write_array(path, samples.data(), samples.size());
}

void write_f32(const std::string& path, const std::vector<float>& values)
{
    write_array(path, values.data(), values.size());
}
