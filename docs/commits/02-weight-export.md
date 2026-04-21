# 02 — Weight export tool

One-shot Python script that reads the trained PyTorch checkpoint from
Project 5 and writes each learnable tensor as a raw little-endian float32
binary. After this runs, the CUDA inference engine never needs to touch
PyTorch again — weights are loaded from flat `.bin` files via
`cudaMemcpy`.

## What's in this commit

| Path | Purpose |
|---|---|
| `tools/export_weights.py` | PyTorch → raw-bytes exporter. Hardcodes the eight expected tensor names (`conv1.weight/bias`, `conv2.weight/bias`, `fc1.weight/bias`, `fc2.weight/bias`) and their expected shapes, fails loudly if any are missing or mis-shaped. |
| `docs/commits/02-weight-export.md` | This walkthrough. |

The script defines a local copy of Project 5's `MyNetwork` so that
`torch.load` works whether the checkpoint is a `state_dict` or a pickled
module instance. The tensor-name contract with the C++ loader is the
`EXPECTED_SHAPES` dict — if Project 5's model ever changes, this dict is
the one place to update.

## Why a manifest file instead of a header

Each `.bin` is raw bytes, no header, no magic number. Shape information
lives in a sibling `shapes.txt` manifest. This keeps the binary files
literally `cudaMemcpy`-ready — the host loader computes
`num_elements * sizeof(float)` from the manifest and does a single
`cudaMemcpy` with no offset math.

## How to run it

From the repo root on any machine with Python + PyTorch installed (does
not need CUDA):

```
python tools/export_weights.py \
    --checkpoint ../p5_DNNrecog_cs5330_adasu/mnist_model.pth \
    --out-dir weights/
```

Defaults match that invocation, so `python tools/export_weights.py` with
no flags works from the repo root.

Expected output:

```
wrote weights/conv1.weight.bin  shape=(10, 1, 5, 5)  bytes=1000
wrote weights/conv1.bias.bin    shape=(10,)           bytes=40
wrote weights/conv2.weight.bin  shape=(20, 10, 5, 5)  bytes=20000
wrote weights/conv2.bias.bin    shape=(20,)           bytes=80
wrote weights/fc1.weight.bin    shape=(50, 320)       bytes=64000
wrote weights/fc1.bias.bin      shape=(50,)           bytes=200
wrote weights/fc2.weight.bin    shape=(10, 50)        bytes=2000
wrote weights/fc2.bias.bin      shape=(10,)           bytes=40
wrote weights/shapes.txt
```

Total payload: **87,360 bytes** of weights plus a few-line text manifest.

## How to verify

1. **Byte counts match** — each file's size equals
   `product(shape) * 4`. Conv and Linear weights are in PyTorch's native
   memory layout (row-major, outer-to-inner), so no transposes were done.
2. **Manifest is legible** — `cat weights/shapes.txt` lists eight entries,
   one per line, each `name d1,d2,...`.
3. **Roundtrip sanity** — a quick spot-check:
   ```python
   import numpy as np
   w = np.fromfile("weights/conv1.weight.bin", dtype=np.float32).reshape(10, 1, 5, 5)
   print(w.shape, w.mean(), w.std())   # should look like a trained conv layer
   ```
   You should see non-zero std and small absolute values (~1e-1). A
   zeroed or uniformly-random result means the script loaded the wrong
   file.

## Design notes

- **Why float32 not float16?** The proposal commits to matching PyTorch
  logits to 1e-4 absolute tolerance. float16 costs ~3 decimal digits of
  precision; no room to spend. Quantization is an explicit non-goal.
- **Why raw bytes not NumPy `.npy`?** `.npy` has a version-prefixed
  header; the C++ loader would need to parse it. Raw bytes +
  external-manifest is simpler and the disk cost is identical.
- **Why skip `conv2_drop`?** `nn.Dropout2d` has no learnable parameters —
  it's a regularizer with no state dict entry at inference. The forward
  pass simply does not call it when `model.eval()` is set.

## What this commit unblocks

- **Commit 03 (CPU reference forward pass)** can now load real weights
  and produce real expected outputs.
- **Commit 04 (fixture dump)** will use this same PyTorch environment to
  run a handful of MNIST inputs through the reference model and save
  both inputs and log-softmax outputs as `.bin` fixtures for
  kernel-level regression tests.

## What this commit does **not** do

- Does not run the network. Does not produce fixtures. Does not touch
  CUDA. Those are separate steps.
- Does not validate the checkpoint against MNIST test accuracy — we
  trust Project 5's training. A 1e-4 logit mismatch between our CUDA
  engine and PyTorch would reveal a bug in the engine, not the weights.
