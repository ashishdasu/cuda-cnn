// Weight loader implementation.
//
// Each `.bin` file is raw little-endian float32 with no header. This
// loader uses hardcoded element counts (matching the network
// architecture) to size its buffers and verifies every file's byte
// count exactly — a shape-drift bug surfaces as a load-time
// std::runtime_error, not a silent misread.

#include "host/weights.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace cnn {

namespace {

// Read exactly `expected_count` float32s from `path`. Throws if the
// file is missing, unreadable, or has a size that does not match
// `expected_count * sizeof(float)`.
std::vector<float> load_bin(const std::string& path, std::size_t expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("weights: failed to open " + path);
    }
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    const std::size_t expected_bytes = expected_count * sizeof(float);
    if (bytes != expected_bytes) {
        throw std::runtime_error(
            "weights: " + path + " size mismatch: expected " +
            std::to_string(expected_bytes) + " bytes, got " +
            std::to_string(bytes));
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(bytes));
    if (!f) {
        throw std::runtime_error("weights: " + path + " read failed");
    }
    return data;
}

}  // namespace

Weights load_weights(const std::string& weights_dir) {
    const std::string dir =
        (weights_dir.empty() || weights_dir.back() == '/')
            ? weights_dir
            : weights_dir + "/";

    Weights w;
    w.conv1_weight = load_bin(dir + "conv1.weight.bin", 10 * 1 * 5 * 5);
    w.conv1_bias   = load_bin(dir + "conv1.bias.bin",   10);
    w.conv2_weight = load_bin(dir + "conv2.weight.bin", 20 * 10 * 5 * 5);
    w.conv2_bias   = load_bin(dir + "conv2.bias.bin",   20);
    w.fc1_weight   = load_bin(dir + "fc1.weight.bin",   50 * 320);
    w.fc1_bias     = load_bin(dir + "fc1.bias.bin",     50);
    w.fc2_weight   = load_bin(dir + "fc2.weight.bin",   10 * 50);
    w.fc2_bias     = load_bin(dir + "fc2.bias.bin",     10);
    return w;
}

}  // namespace cnn
