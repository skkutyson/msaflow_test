"""
Scan a MSAFlow ShardStore and report the Neff score distribution.

Neff per entry = weights.sum(), precomputed into the `neff` column of the
store's index.parquet (see compute_sequence_weights in preprocessing.py), so
this reads only the index — no embedding shards are touched.

Outputs
-------
  - Text statistics + ASCII log-scale histogram  →  stdout
  - Raw Neff array saved as  <output_dir>/neff_values.npy
  - Per-bin counts saved as  <output_dir>/neff_hist.txt  (bin_lo, bin_hi, count)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def ascii_histogram(values: np.ndarray, n_bins: int = 40, bar_width: int = 50) -> str:
    """
    Histogram of log10(Neff).

    x-axis: log10(Neff) values (e.g. 0.00 ~ 2.71 for Neff 1~512)
    Each row: [log10_lo – log10_hi)  bar  count  (Neff: lo–hi)
    """
    log_vals = np.log10(values.clip(min=1e-3))
    lo, hi = log_vals.min(), log_vals.max()
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(log_vals, bins=bins)

    max_count = counts.max() if counts.max() > 0 else 1
    header = f"  x-axis: log10(Neff)  [actual Neff shown in parentheses]"
    lines = [
        header,
        f"  {'─' * (bar_width + 36)}",
    ]
    for i in range(n_bins):
        log_lo = bins[i]
        log_hi = bins[i + 1]
        neff_lo = 10 ** log_lo
        neff_hi = 10 ** log_hi
        bar_len = int(bar_width * counts[i] / max_count)
        bar = "█" * bar_len
        lines.append(
            f"  [{log_lo:5.2f} – {log_hi:5.2f})  {bar:<{bar_width}}  {counts[i]:>7,}"
            f"  (Neff {neff_lo:6.1f}–{neff_hi:.1f})"
        )
    lines.append(f"  {'─' * (bar_width + 36)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_path", required=True,
                        help="Path to the ShardStore directory")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Keep every Nth entry (1 = all, 10 = 10%% sample)")
    args = parser.parse_args()

    from msaflow.data.shard_store import ShardStore

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Neff is a precomputed index column — one Parquet read, no shard access.
    store = ShardStore(args.store_path)
    neff = store.all_neff()
    if args.sample_every > 1:
        neff = neff[::args.sample_every]
    n_missing = 0

    if neff.size == 0:
        print("ERROR: no Neff values found.", file=sys.stderr)
        sys.exit(1)

    neff = np.asarray(neff, dtype=np.float32)

    # ── Save raw values ───────────────────────────────────────────────────────
    npy_path = out_dir / "neff_values.npy"
    np.save(npy_path, neff)

    # ── Statistics ────────────────────────────────────────────────────────────
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(neff, pcts)

    stats_lines = [
        "",
        "=" * 72,
        f"  Neff Distribution  —  {args.store_path}",
        "=" * 72,
        f"  Entries read  : {len(neff):,}  (sample_every={args.sample_every})",
        f"  Missing weights: {n_missing}",
        "",
        f"  Mean   : {neff.mean():>10.2f}",
        f"  Std    : {neff.std():>10.2f}",
        f"  Min    : {neff.min():>10.2f}",
        f"  Max    : {neff.max():>10.2f}",
        "",
        "  Percentiles:",
    ]
    for p, v in zip(pcts, pct_vals):
        stats_lines.append(f"    p{p:02d}  = {v:>10.2f}")

    stats_lines += [
        "",
        "  Log10(Neff) histogram:",
        ascii_histogram(neff),
        "",
        f"  Raw Neff values saved → {npy_path}",
        "=" * 72,
        "",
    ]

    report = "\n".join(stats_lines)
    print(report)

    # ── Save histogram counts (log10 기준) ───────────────────────────────────
    log_vals = np.log10(neff.clip(min=1e-3))
    n_bins = 40
    bins = np.linspace(log_vals.min(), log_vals.max(), n_bins + 1)
    counts, edges = np.histogram(log_vals, bins=bins)
    hist_path = out_dir / "neff_hist.txt"
    with open(hist_path, "w") as f:
        f.write("# log10_neff_lo\tlog10_neff_hi\tneff_lo\tneff_hi\tcount\n")
        for i in range(n_bins):
            f.write(
                f"{edges[i]:.4f}\t{edges[i+1]:.4f}\t"
                f"{10**edges[i]:.2f}\t{10**edges[i+1]:.2f}\t{counts[i]}\n"
            )
    print(f"  Histogram counts saved → {hist_path}")

    # ── Neff=10 스파이크 진단 ─────────────────────────────────────────────────
    n_exactly_10 = int(((neff >= 9.9) & (neff <= 10.1)).sum())
    n_lte_10 = int((neff <= 10.0).sum())
    print(f"\n  [진단] Neff ≈ 10 인 항목 : {n_exactly_10:>7,}  ({100*n_exactly_10/len(neff):.1f}%)")
    print(f"  [진단] Neff ≤ 10  인 항목 : {n_lte_10:>7,}  ({100*n_lte_10/len(neff):.1f}%)")
    print(f"  → filter_msa min_seqs=10 로 강제 패딩된 MSA일 가능성이 높음")


if __name__ == "__main__":
    main()
