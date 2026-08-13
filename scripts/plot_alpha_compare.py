#!/usr/bin/env python3
"""Compare overfit runs across alpha-geometry (α = -0.5, 0, 0.5) in one figure.

Each `run_overfit.sh` run writes <out_dir>/metrics.jsonl with val rows holding
generation metrics at three decoding temperatures:
    val/seq_recovery       val/aa_kl_div        (temp = 0.0, flat keys)
    val/t05/seq_recovery   val/t05/aa_kl_div    (temp = 0.5)
    val/t10/seq_recovery   val/t10/aa_kl_div    (temp = 1.0)

For each α this script reduces over the temperature axis by taking the BEST
temperature (lowest KL, highest recovery), then compares the α runs:

  top row  — best-ever scalar per α (bars): best KL (lower=better),
             best seq_recovery (higher=better), annotated with (temp, epoch).
  bottom   — best-over-temperature vs epoch (lines, one per α): shows the
             trajectory, not just the single best point.

Output: <work_dir>/alpha_compare_<target>.png

Usage:
  # auto-discover decoder_<target>{,_a<val>} dirs under a work dir
  python scripts/plot_alpha_compare.py --target 7tlh
  python scripts/plot_alpha_compare.py --target 7tlh --work_dir runs/overfit

  # or point at the run dirs explicitly (α parsed from the _a<val> suffix,
  # bare dir = α 0.0; override with --alphas)
  python scripts/plot_alpha_compare.py --dirs runs/overfit/decoder_7tlh_a-0.5 \
      runs/overfit/decoder_7tlh runs/overfit/decoder_7tlh_a0.5
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, key prefix) for each decoding temperature train_decoder logs.
TEMPS = [("0.0", ""), ("0.5", "t05/"), ("1.0", "t10/")]

# metric -> (jsonl base key, direction). "min" = lower is better.
METRICS = {
    "aa_kl_div":    ("aa_kl_div", "min"),
    "seq_recovery": ("seq_recovery", "max"),
}


def load_val(mpath: Path) -> list:
    """Return the val rows of a metrics.jsonl file."""
    rows = []
    with open(mpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "val":
                rows.append(r)
    return rows


def _epoch(r):
    return r.get("epoch", r.get("step"))


def best_scalar(val, base, direction):
    """Best value over ALL (temperature, epoch) points → (value, temp, epoch)."""
    best = None
    for r in val:
        for temp, prefix in TEMPS:
            key = f"val/{prefix}{base}"
            v = r.get(key)
            if v is None or v != v:  # missing or NaN
                continue
            if (best is None
                    or (direction == "min" and v < best[0])
                    or (direction == "max" and v > best[0])):
                best = (v, temp, _epoch(r))
    return best  # None if no data


def best_over_temp_curve(val, base, direction):
    """Per-epoch best across temperatures → (epochs, values), sorted by epoch."""
    pts = []
    for r in val:
        vals = [r[f"val/{p}{base}"] for _, p in TEMPS
                if f"val/{p}{base}" in r and r[f"val/{p}{base}"] == r[f"val/{p}{base}"]]
        if vals:
            pts.append((_epoch(r), min(vals) if direction == "min" else max(vals)))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def parse_alpha(dir_name: str, target: str):
    """decoder_<target>_a<val> → float(val); decoder_<target> → 0.0."""
    m = re.search(r"_a(-?\d+(?:\.\d+)?)$", dir_name)
    if m:
        return float(m.group(1))
    return 0.0


def discover(work_dir: Path, target: str):
    """Find decoder_<target> and decoder_<target>_a* dirs; dedupe by α."""
    found = {}
    for d in sorted(work_dir.glob(f"decoder_{target}*")):
        if not d.is_dir():
            continue
        suffix = d.name[len(f"decoder_{target}"):]
        if suffix and not suffix.startswith("_a"):
            continue  # a different target that merely shares this prefix
        mpath = d / "metrics.jsonl"
        if not mpath.is_file():
            continue
        alpha = parse_alpha(d.name, target)
        val = load_val(mpath)
        # On duplicate α, keep the run with more val rows (the longer one).
        if alpha not in found or len(val) > len(found[alpha][1]):
            found[alpha] = (d, val)
    return found


def collect(dirs, alphas, target):
    """dirs/alphas overrides → {alpha: (dir, val_rows)}."""
    out = {}
    for i, d in enumerate(dirs):
        d = Path(d)
        mpath = d if d.is_file() else d / "metrics.jsonl"
        if not mpath.is_file():
            print(f"WARN: no metrics.jsonl in {d}, skipping")
            continue
        alpha = alphas[i] if alphas else parse_alpha(d.name, target or "")
        out[alpha] = (d, load_val(mpath))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="target id (e.g. 7tlh) for auto-discovery")
    ap.add_argument("--work_dir", default="runs/overfit",
                    help="dir holding the decoder_<target>* run dirs")
    ap.add_argument("--dirs", nargs="+", help="explicit run dirs (overrides discovery)")
    ap.add_argument("--alphas", nargs="+", type=float,
                    help="α per --dirs entry (else parsed from dir name)")
    ap.add_argument("--out", help="output PNG (default: <work_dir>/alpha_compare_<target>.png)")
    args = ap.parse_args()

    if args.dirs:
        runs = collect(args.dirs, args.alphas, args.target)
    elif args.target:
        runs = discover(Path(args.work_dir), args.target)
    else:
        ap.error("give --target (auto-discover) or --dirs")

    runs = {a: v for a, v in runs.items() if v[1]}  # drop empty-val runs
    if not runs:
        raise SystemExit("no runs with val metrics found")

    alphas = sorted(runs)
    colors = {a: c for a, c in zip(alphas, plt.cm.viridis(
        [i / max(len(alphas) - 1, 1) for i in range(len(alphas))]))}
    print(f"comparing α = {alphas}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_kl_bar, ax_rec_bar), (ax_kl_line, ax_rec_line) = axes

    # ── top row: best-ever scalar per α (bars) ────────────────────────────────
    bar_specs = [
        (ax_kl_bar, "aa_kl_div", "best AA KL divergence  (lower = better)", "min"),
        (ax_rec_bar, "seq_recovery", "best seq recovery  (higher = better)", "max"),
    ]
    x = list(range(len(alphas)))
    for ax, base, title, direction in bar_specs:
        heights, labels = [], []
        for a in alphas:
            b = best_scalar(runs[a][1], base, direction)
            heights.append(b[0] if b else 0.0)
            labels.append(f"t={b[1]}\nep {b[2]}" if b else "n/a")
        bars = ax.bar(x, heights, color=[colors[a] for a in alphas], width=0.6)
        for rect, h, lab in zip(bars, heights, labels):
            ax.annotate(f"{h:.4f}\n{lab}", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a:g}" for a in alphas])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.18)

    # ── bottom row: best-over-temperature vs epoch (lines) ────────────────────
    line_specs = [
        (ax_kl_line, "aa_kl_div", "AA KL divergence", "min", "best-over-temp KL vs epoch"),
        (ax_rec_line, "seq_recovery", "seq recovery", "max", "best-over-temp recovery vs epoch"),
    ]
    for ax, base, ylabel, direction, title in line_specs:
        for a in alphas:
            xs, ys = best_over_temp_curve(runs[a][1], base, direction)
            if xs:
                ax.plot(xs, ys, "o-", ms=3, color=colors[a], label=f"α={a:g}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    target = args.target or "runs"
    fig.suptitle(f"α-geometry comparison — {target}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out) if args.out else Path(args.work_dir) / f"alpha_compare_{target}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
