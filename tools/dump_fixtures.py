"""Dump MNIST test-set fixtures and their PyTorch log-softmax outputs.

For each of N test images, this script writes two raw float32 binaries
under `fixtures/`:
  input_{i}.bin   shape (1, 1, 28, 28)  784 floats  post-normalization
  output_{i}.bin  shape (1, 10)         10 floats   PyTorch log-softmax

Inputs are saved AFTER the standard MNIST normalization
`(x - 0.1307) / 0.3081` so the C++ inference engine can load them
directly via cudaMemcpy without needing the constants.

Outputs are the raw PyTorch forward-pass results in eval mode (Dropout2d
disabled, no gradient tracking). They are the numerical oracle that
every CPU reference function and every CUDA kernel is validated
against — the 1e-4 absolute tolerance promised in the proposal is
measured against these files.

A human-readable `fixtures/manifest.txt` is written alongside with
columns: index | label | predicted_class | max_logprob. Handy for
eyeballing correctness before running any tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# MyNetwork is a local mirror of Project 5's model. Must stay in sync
# with p5_DNNrecog_cs5330_adasu/train_network.py. Duplicated (rather
# than imported) so this script has no runtime dependency on Project 5
# beyond the trained checkpoint.
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


# Standard MNIST normalization constants used by Project 5.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_model(checkpoint_path: Path) -> MyNetwork:
    model = MyNetwork()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model.load_state_dict(state)
    model.eval()  # critical: disables Dropout2d
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../p5_DNNrecog_cs5330_adasu/mnist_model.pth"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../p5_DNNrecog_cs5330_adasu/data"),
        help="Where torchvision looks for the MNIST test set (downloads if missing)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fixtures"),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many test images to dump",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1

    model = load_model(args.checkpoint)

    # Load the MNIST test set with the same normalization Project 5 trained on.
    # `download=True` fetches once and caches; subsequent runs are offline.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_set = torchvision.datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["# index  label  predicted  max_logprob"]
    with torch.no_grad():
        for i in range(args.count):
            image, label = test_set[i]        # image: (1, 28, 28) float32
            x = image.unsqueeze(0)             # → (1, 1, 28, 28)
            logp = model(x)                    # → (1, 10), log-softmax
            predicted = int(logp.argmax(dim=1).item())
            max_lp = float(logp.max().item())

            input_path = args.out_dir / f"input_{i}.bin"
            output_path = args.out_dir / f"output_{i}.bin"
            input_path.write_bytes(x.contiguous().to(torch.float32).numpy().tobytes())
            output_path.write_bytes(logp.contiguous().to(torch.float32).numpy().tobytes())

            print(
                f"wrote {input_path} ({input_path.stat().st_size}B)  "
                f"{output_path} ({output_path.stat().st_size}B)  "
                f"label={label}  predicted={predicted}  max_logprob={max_lp:.4f}"
            )
            manifest_lines.append(
                f"{i:5d}  {label:5d}  {predicted:9d}  {max_lp:11.4f}"
            )

    manifest_path = args.out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
