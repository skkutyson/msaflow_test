#!/usr/bin/env python
"""
Compute Neff (number of effective sequences) for generated MSAs.

Neff follows the same definition as msaflow.inference.quality_eval.compute_neff:
two sequences within <`threshold` Hamming distance (default 0.2 → >80% identity)
are neighbours; each sequence contributes weight 1/(#neighbours), and Neff is the
sum of those weights. CPU-only, numpy only — no torch / GPU needed.

Input : one or more A3M files or globs (e.g. zero-shot 32x32 output {target}.a3m).
Output: per-file  N / L / Neff / Neff-per-sqrt(L)  printed, and optionally a CSV.

Usage:
    python scripts/compute_neff.py /PATH/TO/zeroshot/*.a3m -o neff.csv
    python scripts/compute_neff.py A.a3m B.a3m --skip_query
"""

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np

# Must match msaflow.data.preprocessing exactly: '-' -> 20, 'X' -> 21
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
X_IDX = AA_TO_IDX["X"]


def parse_a3m(path: str) -> list[str]:
    """Read an A3M, dropping lowercase insertion columns so alignment columns
    line up (first record = query). Returns a list of equal-length rows."""
    seqs: list[str] = []
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            elif line:
                # keep uppercase match columns + gaps; drop lowercase insertions
                buf.append("".join(c for c in line if not c.islower()))
    if buf:
        seqs.append("".join(buf))
    return seqs


def seqs_to_tokens(seqs: list[str]) -> np.ndarray:
    if not seqs:
        return np.zeros((0, 0), dtype=np.int32)
    L = max(len(s) for s in seqs)
    arr = np.full((len(seqs), L), X_IDX, dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, aa in enumerate(s):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), X_IDX)
    return arr


def compute_neff(tokens: np.ndarray, threshold: float = 0.2, block: int = 256) -> float:
    """Same metric as quality_eval.compute_neff, chunked to bound memory.

    The reference impl builds an (N, N, L) array; here we count neighbours one
    row-block at a time so peak memory is O(block * N * L) instead of O(N^2 * L).
    """
    N = tokens.shape[0]
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    counts = np.zeros(N, dtype=np.float64)
    for i in range(0, N, block):
        blk = tokens[i:i + block]                          # (b, L)
        hamming = (blk[:, None, :] != tokens[None, :, :]).mean(axis=-1)  # (b, N)
        counts[i:i + block] = (hamming < threshold).sum(axis=1)
    return float((1.0 / np.clip(counts, 1.0, None)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Neff for generated MSAs (A3M)")
    ap.add_argument("inputs", nargs="+",
                    help="A3M files or globs, e.g. /PATH/TO/zeroshot/*.a3m")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="Hamming cutoff for clustering (0.2 = 80%% identity)")
    ap.add_argument("--skip_query", action="store_true",
                    help="Exclude the first (query) record from the Neff count")
    ap.add_argument("--block", type=int, default=256,
                    help="Row-block size for the pairwise pass (memory knob)")
    ap.add_argument("-o", "--out_csv", default=None, help="Write per-file CSV here")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.inputs:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])

    rows = []
    for p in paths:
        seqs = parse_a3m(p)
        if args.skip_query and len(seqs) > 1:
            seqs = seqs[1:]
        toks = seqs_to_tokens(seqs)
        N, L = (toks.shape if toks.size else (len(seqs), 0))
        neff = compute_neff(toks, threshold=args.threshold, block=args.block)
        neff_norm = neff / math.sqrt(L) if L else 0.0
        rows.append({"file": p, "n_seqs": N, "seq_len": L,
                     "neff": round(neff, 3), "neff_per_sqrtL": round(neff_norm, 4)})
        print(f"{Path(p).name:40s}  N={N:5d}  L={L:5d}  "
              f"Neff={neff:8.2f}  Neff/sqrtL={neff_norm:6.3f}")

    if rows:
        mean_neff = sum(r["neff"] for r in rows) / len(rows)
        print(f"\n# files={len(rows)}  mean Neff={mean_neff:.2f}")
        if args.out_csv:
            with open(args.out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
