"""Export weights from the Project 5 trained PyTorch MNIST model as raw
float32 binaries usable directly by the CUDA inference engine.

PyTorch is a build-time dependency only: this script runs once per trained
checkpoint to produce files under weights/. The CUDA runtime never links
against PyTorch.

Output layout under `weights/`:
  conv1.weight.bin   [10, 1, 5, 5]   250   floats
  conv1.bias.bin     [10]             10   floats
  conv2.weight.bin   [20, 10, 5, 5]  5000  floats
  conv2.bias.bin     [20]             20   floats
  fc1.weight.bin     [50, 320]      16000  floats  (row-major: out, in)
  fc1.bias.bin       [50]             50   floats
  fc2.weight.bin     [10, 50]         500  floats
  fc2.bias.bin       [10]             10   floats
  shapes.txt                              per-tensor shape manifest

All binaries are raw little-endian float32, no header. Host C++ loader reads
the shape manifest, allocates on device, and cudaMemcpys each file's bytes
directly. Layouts match PyTorch's storage convention: conv weights are
[out_channels, in_channels, kH, kW] in row-major; linear weights are
[out_features, in_features] in row-major.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# MyNetwork is defined locally so torch.load works whether the checkpoint is a
# state_dict or a full pickled module. Must match Project 5's definition in
# p5_DNNrecog_cs5330_adasu/train_network.py exactly — tensor names are the
# contract with the C++ loader.
class MyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


EXPECTED_SHAPES = {
    "conv1.weight": (10, 1, 5, 5),
    "conv1.bias":   (10,),
    "conv2.weight": (20, 10, 5, 5),
    "conv2.bias":   (20,),
    "fc1.weight":   (50, 320),
    "fc1.bias":     (50,),
    "fc2.weight":   (10, 50),
    "fc2.bias":     (10,),
}


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Accept either a state_dict or a pickled MyNetwork instance."""
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../p5_DNNrecog_cs5330_adasu/mnist_model.pth"),
        help="Path to the trained .pth file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to write raw float32 binaries into",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1

    state = load_state_dict(args.checkpoint)

    missing = set(EXPECTED_SHAPES) - set(state)
    if missing:
        print(f"error: checkpoint missing tensors: {sorted(missing)}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for name, expected_shape in EXPECTED_SHAPES.items():
        tensor = state[name].detach().cpu().contiguous().to(torch.float32)
        if tuple(tensor.shape) != expected_shape:
            print(
                f"error: {name} has shape {tuple(tensor.shape)}, expected {expected_shape}",
                file=sys.stderr,
            )
            return 1

        out_path = args.out_dir / f"{name}.bin"
        out_path.write_bytes(tensor.numpy().tobytes())
        shape_str = ",".join(str(d) for d in expected_shape)
        manifest_lines.append(f"{name} {shape_str}")
        print(f"wrote {out_path}  shape={expected_shape}  bytes={out_path.stat().st_size}")

    manifest_path = args.out_dir / "shapes.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
