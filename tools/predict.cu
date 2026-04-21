// Single-image inference CLI for the LeNet-MNIST engine. Loads weights
// once, reads one 1x28x28 float32 input (from a named fixture, an
// index into the bulk 10k MNIST test set, or an arbitrary raw file),
// runs the selected forward variant, and prints top-K log-probabilities
// plus optional timing.
//
// This is the user-facing inference tool, complementing:
//   * tests/test_forward_mnist10k.cu  — full 10k parity harness (pass/fail)
//   * tools/bench.cu                  — throughput microbenchmark (CSV)
// Where those are testing/benchmarking rigs, predict is a thin driver
// that treats the engine as a black box: image in, prediction out.
//
// Usage:
//   build/predict --fixture 0
//   build/predict --bulk 42 --variant naive
//   build/predict --input my_digit.bin --reps 100 --top-k 5
//
// Input file format: raw float32, exactly 784 floats (1*28*28) after
// MNIST-standard normalization (x - 0.1307) / 0.3081, matching what
// tools/dump_fixtures.py writes.

#include "host/weights.h"
#include "kernels/forward.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int INPUT_FLOATS = 1 * 28 * 28;
constexpr int NUM_CLASSES  = 10;

void die(const std::string& msg) {
    std::fprintf(stderr, "predict: %s\n", msg.c_str());
    std::exit(1);
}

std::vector<float> load_raw_floats(const std::string& path, std::size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("failed to open " + path);
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    if (bytes != n * sizeof(float)) {
        die(path + ": expected " + std::to_string(n * sizeof(float))
            + " bytes (" + std::to_string(n) + " floats), got "
            + std::to_string(bytes));
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(bytes));
    return data;
}

std::vector<float> load_bulk_row(const std::string& path, int idx) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("failed to open " + path
                + " (run `python3 tools/dump_fixtures.py --bulk --count 10000`)");
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    const std::size_t n_images = bytes / (INPUT_FLOATS * sizeof(float));
    if (idx < 0 || static_cast<std::size_t>(idx) >= n_images) {
        die("bulk index " + std::to_string(idx) + " out of range [0, "
            + std::to_string(n_images) + ")");
    }
    f.seekg(static_cast<std::streamoff>(idx) * INPUT_FLOATS * sizeof(float),
            std::ios::beg);
    std::vector<float> data(INPUT_FLOATS);
    f.read(reinterpret_cast<char*>(data.data()),
           INPUT_FLOATS * sizeof(float));
    return data;
}

using ForwardFn = void(*)(const cnn::Weights&, const float*, float*, int);

// Wall-clock per-call timing, amortized over reps with cudaEvent to
// stay consistent with bench.cu's methodology. Reps=1 still calls
// once and reports the single-run time.
float time_forward(ForwardFn fwd, const cnn::Weights& w,
                   const float* h_in, float* h_out, int reps) {
    // Warmup once so the first-call allocation / JIT doesn't skew the mean.
    fwd(w, h_in, h_out, /*N=*/1);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) fwd(w, h_in, h_out, /*N=*/1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return total_ms / reps;
}

}  // namespace

int main(int argc, char** argv) {
    std::string weights_dir = "weights/";
    std::string variant     = "tiled";
    std::string input_path;
    int  fixture_idx = -1;
    int  bulk_idx    = -1;
    int  reps        = 1;
    int  top_k       = 3;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) die(std::string("missing value for ") + flag);
            return std::string(argv[++i]);
        };
        if      (a == "--weights")  weights_dir  = need("--weights");
        else if (a == "--variant")  variant      = need("--variant");
        else if (a == "--input")    input_path   = need("--input");
        else if (a == "--fixture")  fixture_idx  = std::atoi(need("--fixture").c_str());
        else if (a == "--bulk")     bulk_idx     = std::atoi(need("--bulk").c_str());
        else if (a == "--reps")     reps         = std::atoi(need("--reps").c_str());
        else if (a == "--top-k")    top_k        = std::atoi(need("--top-k").c_str());
        else if (a == "-h" || a == "--help") {
            std::printf(
                "usage: predict [options]\n"
                "  --weights DIR     weight directory (default: weights/)\n"
                "  --variant V       naive | tiled (default: tiled)\n"
                "  --fixture N       use fixtures/input_N.bin\n"
                "  --bulk N          use row N of fixtures/mnist10k_inputs.bin\n"
                "  --input PATH      read raw float32 1x28x28 from PATH\n"
                "  --reps N          repeat inference N times for timing (default: 1)\n"
                "  --top-k K         print top K classes (default: 3)\n"
                "exactly one of --fixture / --bulk / --input must be given.\n");
            return 0;
        }
        else die("unknown flag: " + a);
    }

    const int sources = (fixture_idx >= 0) + (bulk_idx >= 0) + (!input_path.empty());
    if (sources != 1) die("exactly one of --fixture / --bulk / --input required");
    if (variant != "naive" && variant != "tiled") {
        die("--variant must be naive or tiled (got " + variant + ")");
    }
    if (reps < 1)  die("--reps must be >= 1");
    if (top_k < 1 || top_k > NUM_CLASSES) {
        die("--top-k must be in [1, " + std::to_string(NUM_CLASSES) + "]");
    }

    std::vector<float> input;
    std::string source_desc;
    if (fixture_idx >= 0) {
        const std::string path = "fixtures/input_" + std::to_string(fixture_idx) + ".bin";
        input = load_raw_floats(path, INPUT_FLOATS);
        source_desc = "fixture " + std::to_string(fixture_idx) + " (" + path + ")";
    } else if (bulk_idx >= 0) {
        input = load_bulk_row("fixtures/mnist10k_inputs.bin", bulk_idx);
        source_desc = "bulk row " + std::to_string(bulk_idx);
    } else {
        input = load_raw_floats(input_path, INPUT_FLOATS);
        source_desc = input_path;
    }

    const cnn::Weights w = cnn::load_weights(weights_dir);

    ForwardFn fwd = (variant == "naive")
        ? cnn::cuda::forward_naive_gpu
        : cnn::cuda::forward_tiled_gpu;

    std::vector<float> logp(NUM_CLASSES);
    const float ms_per_call = time_forward(fwd, w, input.data(), logp.data(), reps);

    // Argsort descending by log-probability for the top-K readout.
    std::vector<int> order(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c) order[c] = c;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return logp[a] > logp[b]; });

    std::printf("input:    %s\n", source_desc.c_str());
    std::printf("variant:  %s  (weights from %s)\n",
                variant.c_str(), weights_dir.c_str());
    std::printf("predicted: %d  (log-prob %.4f, prob %.4f)\n",
                order[0], logp[order[0]], std::exp(logp[order[0]]));
    std::printf("top-%d:\n", top_k);
    for (int k = 0; k < top_k; ++k) {
        const int c = order[k];
        std::printf("  %d) class %d  log-prob %8.4f  prob %.4f\n",
                    k + 1, c, logp[c], std::exp(logp[c]));
    }
    if (reps == 1) {
        std::printf("time:     %.3f ms (single run, after warmup)\n",
                    ms_per_call);
    } else {
        std::printf("time:     %.3f ms/call (mean of %d reps after warmup)\n",
                    ms_per_call, reps);
    }
    return 0;
}
