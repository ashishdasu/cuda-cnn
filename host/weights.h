// Weight loader for the LeNet-MNIST CUDA inference engine.
//
// Loads the eight float32 binaries produced by tools/export_weights.py
// into a plain struct of std::vector<float>. Shapes are not stored — they
// are implicit from the network architecture and verified at load time
// against weights/shapes.txt.
//
// Memory layout matches PyTorch's native storage:
//   conv weights: [out_channels, in_channels, kH, kW] row-major
//   linear weights: [out_features, in_features]      row-major
//   biases: [out_channels] or [out_features]         row-major
//
// This is pure host code; no CUDA dependency. Device copies happen in
// the inference driver, not here.

#pragma once

#include <string>
#include <vector>

namespace cnn {

struct Weights {
    std::vector<float> conv1_weight;  // 10*1*5*5  = 250
    std::vector<float> conv1_bias;    // 10
    std::vector<float> conv2_weight;  // 20*10*5*5 = 5000
    std::vector<float> conv2_bias;    // 20
    std::vector<float> fc1_weight;    // 50*320    = 16000
    std::vector<float> fc1_bias;      // 50
    std::vector<float> fc2_weight;    // 10*50     = 500
    std::vector<float> fc2_bias;      // 10
};

// Reads every tensor from `<weights_dir>/<name>.bin` and returns them
// in a Weights struct. Verifies per-file byte counts match the expected
// element counts above; throws std::runtime_error on any mismatch or
// missing file. `weights_dir` should contain `shapes.txt` and all eight
// `.bin` files, e.g. "weights/".
Weights load_weights(const std::string& weights_dir);

}  // namespace cnn
