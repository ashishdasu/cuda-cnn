"""Generate the report/slide figures from tools/bench.cu --csv output.

Usage:
    python3 tools/plot_bench.py <bench.csv> <out_dir>

Produces six PNGs:
    1. conv_bars.png     — tiled and cuDNN speedup vs naive per conv shape.
    2. linear_bars.png   — tiled speedup vs naive per linear shape.
    3. gemm_sweep.png    — tiled-GEMM speedup vs batch N at a 1024x1024 problem.
    4. e2e_breakdown.png — per-kernel contribution to the N=1 forward pass.
    5. e2e_bars.png      — naive vs tiled end-to-end forward pass ms + speedup.
    6. fused_bars.png    — three-kernel (naive / tiled) vs single fused kernel
                           for the conv+bias+ReLU+pool stage, raw-us bars with
                           speedup annotated.

No seaborn, no pandas — matplotlib only. Speedup plots (not raw-ms bars)
because ms values span two orders of magnitude across shapes and log-scale
bar charts are painful to read.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _short_shape(s: str) -> str:
    head = s.split("(")[0].strip()
    tail = "(" + s.split("(", 1)[1] if "(" in s else ""
    return f"{head}\n{tail}"


def _annotate(ax, bars, values) -> None:
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.2f}x",
                    (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=7)


def plot_conv_bars(rows: list[dict[str, str]], out_path: Path) -> None:
    conv_rows = [r for r in rows if r["category"] == "conv"]
    shapes: list[str] = []
    for r in conv_rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])

    variants = ["tiled", "cudnn"]
    width = 0.35
    x = list(range(len(shapes)))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, variant in enumerate(variants):
        speedups = []
        for shape in shapes:
            match = [r for r in conv_rows
                     if r["shape"] == shape and r["variant"] == variant]
            speedups.append(float(match[0]["speedup_vs_naive"]) if match else 0.0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, speedups, width, label=variant)
        _annotate(ax, bars, speedups)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
               label="naive baseline")
    ax.set_ylabel("Speedup vs. naive ($\\times$)")
    ax.set_title("Conv2d: tiled and cuDNN speedup over naive (RTX 4090)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_shape(s) for s in shapes], fontsize=7)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_linear_bars(rows: list[dict[str, str]], out_path: Path) -> None:
    lin_rows = [r for r in rows if r["category"] == "lin"]
    shapes: list[str] = []
    for r in lin_rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])

    width = 0.5
    x = list(range(len(shapes)))

    speedups = []
    for shape in shapes:
        match = [r for r in lin_rows
                 if r["shape"] == shape and r["variant"] == "tiled"]
        speedups.append(float(match[0]["speedup_vs_naive"]) if match else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(x, speedups, width, color="tab:orange")
    _annotate(ax, bars, speedups)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
               label="naive baseline")
    ax.set_ylabel("Tiled speedup over naive ($\\times$)")
    ax.set_title("Linear / GEMM: tiled speedup over naive (RTX 4090)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_shape(s) for s in shapes], fontsize=7)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gemm_sweep(rows: list[dict[str, str]], out_path: Path) -> None:
    sweep_rows = [r for r in rows if r["category"] == "sweep"]
    ns: list[int] = []
    for r in sweep_rows:
        n = int(r["shape"].split("=", 1)[1].split(" ", 1)[0])
        if n not in ns:
            ns.append(n)
    ns.sort()

    speedups: list[float] = []
    for n in ns:
        match = [r for r in sweep_rows
                 if r["variant"] == "tiled" and f"N={n} " in r["shape"]]
        speedups.append(float(match[0]["speedup_vs_naive"]))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, speedups, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Batch N (K = M = 1024 fixed)")
    ax.set_ylabel("Tiled speedup over naive ($\\times$)")
    ax.set_title("Tiled GEMM speedup vs batch size (1024 $\\times$ 1024 problem)")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    for n, s in zip(ns, speedups):
        ax.annotate(f"{s:.2f}x", (n, s),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# E2E LeNet forward-pass per-kernel timings captured in a one-shot
# microbenchmark on the same RTX 4090 (reps=400, warmup, cudaEvent).
# Pool, ReLU, log_softmax times come from a standalone measurement
# because the main bench.cu shape sweep doesn't cover them at their
# LeNet-native sizes. All values are ms per launch at batch N=1.
_E2E_KERNELS_N1 = [
    ("conv1",        0.002688),
    ("pool1",        0.001736),
    ("relu1",        0.001751),
    ("conv2",        0.012261),
    ("pool2",        0.001733),
    ("relu2",        0.001754),
    ("fc1",          0.010731),
    ("relu3",        0.001748),
    ("fc2",          0.002488),
    ("log_softmax",  0.001828),
]
_E2E_TOTAL_N1_MS = 0.164780  # end-to-end forward at N=1, same run


def plot_e2e_breakdown(out_path: Path) -> None:
    """Stacked-bar-ish horizontal chart showing per-kernel contribution
    to the N=1 LeNet forward pass, plus the residual (host overhead,
    launch-queue stall, synchronization) that makes up the gap between
    summed kernel time and end-to-end wall time."""
    names = [n for n, _ in _E2E_KERNELS_N1]
    times_us = [t * 1000.0 for _, t in _E2E_KERNELS_N1]
    kernel_sum_us = sum(times_us)
    residual_us = _E2E_TOTAL_N1_MS * 1000.0 - kernel_sum_us
    names.append("host / launch / sync")
    times_us.append(residual_us)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["tab:blue" if n.startswith(("conv", "fc")) else
              "tab:orange" if n in {"pool1", "pool2"} else
              "tab:green" if n.startswith("relu") else
              "tab:purple" if n == "log_softmax" else
              "tab:gray" for n in names]
    bars = ax.barh(range(len(names)), times_us, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Per-launch time ($\\mu$s)")
    ax.set_title(f"LeNet forward pass, N=1: per-kernel breakdown on RTX 4090\n"
                 f"(kernels $\\Sigma$ = {kernel_sum_us:.1f}$\\mu$s, "
                 f"end-to-end = {_E2E_TOTAL_N1_MS*1000:.1f}$\\mu$s)")
    for bar, v in zip(bars, times_us):
        ax.annotate(f"{v:.1f}",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(3, 0),
                    va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_e2e_bars(rows: list[dict[str, str]], out_path: Path) -> None:
    """End-to-end forward-pass naive vs tiled raw-ms bars at each N,
    with speedup annotated above each tiled bar. Raw ms is fine here
    because the two forward-pass shapes sit within 3x of each other."""
    fwd_rows = [r for r in rows if r["category"] == "fwd"]
    shapes: list[str] = []
    for r in fwd_rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])

    variants = ["naive", "tiled"]
    width = 0.35
    x = list(range(len(shapes)))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    variant_bars: dict[str, list[float]] = {}
    for i, variant in enumerate(variants):
        times = []
        for shape in shapes:
            match = [r for r in fwd_rows
                     if r["shape"] == shape and r["variant"] == variant]
            times.append(float(match[0]["time_ms"]) if match else 0.0)
        variant_bars[variant] = times
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, times, width, label=variant)
        for bar, v in zip(bars, times):
            ax.annotate(f"{v:.3f}",
                        (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)

    for i, shape in enumerate(shapes):
        naive_t = variant_bars["naive"][i]
        tiled_t = variant_bars["tiled"][i]
        if tiled_t > 0:
            speedup = naive_t / tiled_t
            ax.annotate(f"{speedup:.2f}x",
                        (i + 0.5 * width, tiled_t),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=8, color="tab:green",
                        fontweight="bold")

    ax.set_ylabel("Per-invocation time (ms)")
    ax.set_title("End-to-end LeNet forward pass: naive vs tiled (RTX 4090)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("end-to-end forward ", "") for s in shapes],
                       fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fused_bars(rows: list[dict[str, str]], out_path: Path) -> None:
    """Grouped bar chart: for each shape, show the naive three-kernel time,
    the tiled three-kernel time, and the single fused-kernel time in
    microseconds. Speedup vs naive three-call is annotated in green above
    the fused bar so the reader can read the win/loss per shape at a
    glance. The story the chart tells is shape-dependent: fusion wins
    when launch overhead dominates (small, batch-1), breaks even or
    loses when the SMs are starved by a too-small fused kernel."""
    fused_rows = [r for r in rows if r["category"] == "fused"]
    shapes: list[str] = []
    for r in fused_rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])

    variants = ["naive", "tiled", "fused"]
    colors = {"naive": "tab:blue", "tiled": "tab:orange", "fused": "tab:green"}
    width = 0.27
    x = list(range(len(shapes)))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    time_by_variant: dict[str, list[float]] = {}
    for i, variant in enumerate(variants):
        times_us = []
        for shape in shapes:
            match = [r for r in fused_rows
                     if r["shape"] == shape and r["variant"] == variant]
            times_us.append(float(match[0]["time_ms"]) * 1000.0 if match else 0.0)
        time_by_variant[variant] = times_us
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, times_us, width,
                      label=("naive conv+ReLU+pool" if variant == "naive"
                             else "tiled conv+ReLU+pool" if variant == "tiled"
                             else "fused kernel"),
                      color=colors[variant])
        for bar, v in zip(bars, times_us):
            ax.annotate(f"{v:.1f}",
                        (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)

    for i, shape in enumerate(shapes):
        naive_t = time_by_variant["naive"][i]
        fused_t = time_by_variant["fused"][i]
        if fused_t > 0:
            speedup = naive_t / fused_t
            color = "tab:green" if speedup >= 1.0 else "tab:red"
            ax.annotate(f"{speedup:.2f}x",
                        (i + width, fused_t),
                        textcoords="offset points", xytext=(0, 16),
                        ha="center", fontsize=9, color=color,
                        fontweight="bold")

    ax.set_ylabel("Stage time ($\\mu$s)")
    ax.set_title("Conv + bias + ReLU + pool: three launches vs one fused kernel"
                 " (RTX 4090)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_shape(s).replace(" stage", "") for s in shapes],
                       fontsize=7)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: plot_bench.py <bench.csv> <out_dir>", file=sys.stderr)
        return 1
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    plot_conv_bars(rows, out_dir / "conv_bars.png")
    plot_linear_bars(rows, out_dir / "linear_bars.png")
    plot_gemm_sweep(rows, out_dir / "gemm_sweep.png")
    plot_e2e_breakdown(out_dir / "e2e_breakdown.png")
    plot_e2e_bars(rows, out_dir / "e2e_bars.png")
    plot_fused_bars(rows, out_dir / "fused_bars.png")
    print("wrote:", *sorted(out_dir.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
