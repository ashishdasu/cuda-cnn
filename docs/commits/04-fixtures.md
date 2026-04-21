# 04 — Fixture dumper and ground-truth outputs

A second one-shot Python tool, plus the five input/output fixture pairs
it produced. These files are the **numerical oracle** for the whole
project: every CPU reference function and every CUDA kernel will be
validated by comparing its output against the PyTorch log-softmax
saved here.

The 1e-4 absolute tolerance promised in the proposal is measured
against these specific files.

## What's in this commit

| Path | Purpose |
|---|---|
| `tools/dump_fixtures.py` | Runs `mnist_model.pth` in `eval()` mode on the first 5 MNIST test images, saves inputs (post-normalization) and log-softmax outputs as raw float32 binaries. |
| `fixtures/input_0.bin` … `input_4.bin` | Five normalized MNIST images, shape `(1, 1, 28, 28)`, 784 float32s = 3,136 bytes each. **Already normalized** with `(x − 0.1307)/0.3081`. |
| `fixtures/output_0.bin` … `output_4.bin` | Matching PyTorch log-softmax outputs, shape `(1, 10)`, 10 float32s = 40 bytes each. |
| `fixtures/manifest.txt` | Human-readable table: index, true label, predicted class, max log-prob. Quick eyeball check. |
| `docs/commits/04-fixtures.md` | This walkthrough. |

**Total fixture payload: ~15.5 KB.** Committed directly alongside the
weights for the same reproducibility reason — the exact bytes behind
every parity claim in the report.

## What the manifest tells us

```
# index  label  predicted  max_logprob
    0      7          7      -0.0000
    1      2          2      -0.0011
    2      1          1      -0.0003
    3      0          0      -0.0000
    4      4          4      -0.0006
```

All five classified correctly with max log-prob very close to 0 (i.e.
softmax probability ≈ 1.0). The model is confident; these are not
borderline examples. Good property for a regression fixture — a
correct CUDA kernel should reproduce these near-saturated outputs
within 1e-4.

## Design decisions

- **Inputs saved post-normalization.** The C++ inference engine never
  needs to know about `MNIST_MEAN = 0.1307` or `MNIST_STD = 0.3081`.
  It loads 784 floats, `cudaMemcpy`s them to the device, and runs
  forward. Normalization is a preprocessing concern, not an inference
  concern.
- **`model.eval()` is mandatory.** PyTorch's `Dropout2d` is active by
  default and stochastic; forgetting `.eval()` would produce fixtures
  that randomly zero channels, making the 1e-4 tolerance uncheckable.
  The script's `load_model` function calls `.eval()` unconditionally.
- **5 fixtures, not 10,000.** The full MNIST test set is 10k images.
  We only need enough to catch bugs. Five diverse labels (7, 2, 1, 0,
  4) gives coverage of four odd digits and one zero. Full-test-set
  accuracy will be measured later by the inference driver, not here.
- **MyNetwork is duplicated.** Same tensor-name contract as
  `tools/export_weights.py`, so if Project 5's architecture ever
  changes we update two dicts in two files. Intentional: keeping these
  tools independently runnable trumps DRY.

## How to regenerate

```
python tools/dump_fixtures.py \
    --checkpoint ../p5_DNNrecog_cs5330_adasu/mnist_model.pth \
    --data-dir   ../p5_DNNrecog_cs5330_adasu/data \
    --out-dir    fixtures/ \
    --count      5
```

Defaults match that invocation, so `python tools/dump_fixtures.py`
from the repo root works. First run downloads the MNIST test set
(~9.9 MB) into `../p5_DNNrecog_cs5330_adasu/data/` if it's not already
there.

## How to verify

1. **Byte counts** — every `input_*.bin` is exactly 3,136 bytes
   (784 floats × 4). Every `output_*.bin` is exactly 40 bytes
   (10 floats × 4).
2. **Sanity roundtrip** in Python:
   ```python
   import numpy as np
   x = np.fromfile("fixtures/input_0.bin", dtype=np.float32).reshape(1, 1, 28, 28)
   y = np.fromfile("fixtures/output_0.bin", dtype=np.float32)
   print(x.min(), x.max(), y.argmax())
   # x range should be roughly [-0.42, 2.82] (the post-normalization extremes)
   # y.argmax() should be 7 (matches the manifest)
   ```
3. **Manifest matches reality** — labels and predictions line up with
   the raw `output_*.bin` argmax.

## What this commit unblocks

- **Commit 05 (CPU reference forward pass)** can now implement conv,
  pool, relu, linear, log-softmax in plain C++17 and assert its
  outputs match `output_{i}.bin` to 1e-4. That assertion is the gate
  on "our forward-pass math is right."
- **Every subsequent CUDA kernel** can run against the same fixtures
  via per-layer intermediate comparisons — we'll dump
  `fixtures/intermediate_{layer}_{i}.bin` in a later commit to make
  this granular.

## What this commit does **not** do

- Does not run any C++ code. Does not touch CUDA.
- Does not measure model accuracy — that's what the full MNIST test
  run in the inference driver will do at the end. These five fixtures
  are for bug-catching, not performance reporting.
