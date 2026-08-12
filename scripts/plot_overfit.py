#!/usr/bin/env python3
"""Plot overfit / training diagnostics from metrics.jsonl into local PNGs.

train_decoder.py appends one JSON object per line to <output_dir>/metrics.jsonl:
  {"kind":"train","step":..,"epoch":..,"loss":..,"lr":..}
  {"kind":"val","step":..,"epoch":..,"val/seq_recovery":.., "val/t05/seq_recovery":.., ...}

This reads that file and writes PNGs next to it (no wandb needed):
  - train_loss.png       : training flow-matching loss vs step
  - val_recovery.png     : seq_recovery vs epoch, one line per decoding temperature
  - val_kl.png           : aa_kl_div  vs epoch, one line per decoding temperature

Usage:
  python scripts/plot_overfit.py <output_dir>        # dir holding metrics.jsonl
  python scripts/plot_overfit.py <path/to/metrics.jsonl>
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# decoding temperatures logged by train_decoder: flat key = t0.0, t05 = 0.5, t10 = 1.0
TEMPS = [("0.0", ""), ("0.5", "t05/"), ("1.0", "t10/")]


def load(path: Path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (train if r.get("kind") == "train" else val).append(r)
    return train, val


def _series(val, metric, prefix):
    """(epochs, values) for val/{prefix}{metric}, skipping missing points."""
    xs, ys = [], []
    for r in val:
        key = f"val/{prefix}{metric}"
        if key in r and r[key] == r[key]:  # not NaN
            xs.append(r.get("epoch", r.get("step")))
            ys.append(r[key])
    return xs, ys


def plot_train_loss(train, out):
    if not train:
        return
    steps = [r["step"] for r in train]
    loss = [r["loss"] for r in train]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, loss, lw=1.0)
    ax.set_xlabel("step"); ax.set_ylabel("train flow-matching loss")
    ax.set_yscale("log" if min(loss) > 0 else "linear")
    ax.set_title("Training loss"); ax.grid(alpha=0.3)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot_val_metric(val, metric, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_data = False
    for temp, prefix in TEMPS:
        xs, ys = _series(val, metric, prefix)
        if xs:
            any_data = True
            ax.plot(xs, ys, "o-", ms=3, label=f"temp={temp}")
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    arg = Path(sys.argv[1])
    mpath = arg if arg.is_file() else arg / "metrics.jsonl"
    if not mpath.is_file():
        print(f"ERROR: metrics file not found: {mpath}"); sys.exit(1)
    outdir = mpath.parent

    train, val = load(mpath)
    print(f"loaded {len(train)} train rows, {len(val)} val rows from {mpath}")

    plot_train_loss(train, outdir / "train_loss.png")
    plot_val_metric(val, "seq_recovery", "seq recovery",
                    "Validation sequence recovery (per decoding temperature)",
                    outdir / "val_recovery.png")
    plot_val_metric(val, "aa_kl_div", "AA KL divergence",
                    "Validation AA KL divergence (per decoding temperature)",
                    outdir / "val_kl.png")


if __name__ == "__main__":
    main()
