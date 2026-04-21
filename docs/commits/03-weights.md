# 03 — Trained weights (data artifact)

The eight exported weight binaries plus their shape manifest. Produced by
running `tools/export_weights.py` against Project 5's trained
`mnist_model.pth`. Committing them directly (not gitignoring) so the
repo is self-contained — a reviewer or future-you can build and run
`infer` without cloning Project 5.

## What's in this commit

| Path | Bytes | Tensor shape | Notes |
|---|---:|---|---|
| `weights/conv1.weight.bin` | 1,000 | `(10, 1, 5, 5)` | First conv: 10 out-channels, 1 in-channel, 5×5 kernel |
| `weights/conv1.bias.bin` | 40 | `(10,)` | One bias per output channel |
| `weights/conv2.weight.bin` | 20,000 | `(20, 10, 5, 5)` | Second conv: 20 out-channels, 10 in-channels |
| `weights/conv2.bias.bin` | 80 | `(20,)` | |
| `weights/fc1.weight.bin` | 64,000 | `(50, 320)` | Row-major; 50 out-features × 320 in-features |
| `weights/fc1.bias.bin` | 200 | `(50,)` | |
| `weights/fc2.weight.bin` | 2,000 | `(10, 50)` | Final classifier |
| `weights/fc2.bias.bin` | 40 | `(10,)` | |
| `weights/shapes.txt` | 132 | — | Manifest: one line per tensor, `name d1,d2,...` |

**Total payload: 87,492 bytes** (~85 KB). Every file is raw little-endian
float32, no header, `cudaMemcpy`-ready.

## Why commit data artifacts

Normally trained weights don't belong in git (they're regenerable, large,
and bloat history). Three reasons they're here anyway:

1. **Size.** 85 KB total. Below any threshold that would matter.
2. **Reproducibility.** The graded rubric requires the code to be
   runnable from what's submitted. Without the weights in the repo, a
   grader needs Project 5 and a working PyTorch install just to run
   `./build/infer` on one image.
3. **These are the exact weights behind every number in the report.**
   Regenerating them by re-running Project 5's training is not
   bit-identical (random seeds, cudnn nondeterminism). Pinning them
   kills a class of "why don't these logits match the report" questions.

## How to regenerate

If Project 5's `mnist_model.pth` ever changes (e.g. the network is
retrained), regenerate by rerunning:

```
python tools/export_weights.py \
    --checkpoint ../p5_DNNrecog_cs5330_adasu/mnist_model.pth \
    --out-dir weights/
```

The script prints the shape and byte count of every file it writes, and
fails loudly if any expected tensor is missing or has the wrong shape.

## How to verify this commit

1. **Byte counts** — each file's size equals `product(shape) * 4`:
   ```
   stat -f '%z' weights/conv1.weight.bin   # expect 1000
   stat -f '%z' weights/fc1.weight.bin     # expect 64000
   ```
   (Linux equivalent: `stat -c '%s' <file>`.)

2. **Roundtrip** — spot-check a tensor's statistics in Python:
   ```python
   import numpy as np
   w = np.fromfile("weights/conv1.weight.bin", dtype=np.float32).reshape(10, 1, 5, 5)
   print(w.shape, w.mean(), w.std())
   ```
   Non-zero std in the 1e-1 range means a real trained layer (not a
   zeroed or uniformly-random file).

3. **Manifest parses** — `cat weights/shapes.txt` should show eight
   lines, no blanks, every tensor name matching a `.bin` sibling.

## What this commit unblocks

- **Commit 04 (CPU reference forward pass)** can now load the actual
  weights and run a real forward pass in plain C++ — the source of
  truth that every CUDA kernel will validate against.
- **Commit 05 (PyTorch fixture dump)** will use the same checkpoint to
  generate a handful of known input/output pairs for regression tests.

## What this commit does **not** do

- Does not run the network. Does not contain any code.
- Does not pin the MNIST dataset. The test set is downloaded by
  torchvision on demand; we'll record the specific subset we use for
  regression fixtures in the next commit.
