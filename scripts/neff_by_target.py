#!/usr/bin/env python
"""
Per-target depth + Neff for the original MSA vs each generated seed.

Zero-shot generation writes one {target}.a3m per target whose first record is
the query (>{target}) followed by 1024 generated rows tagged
>{target}_s{seed:03d}_n{idx:03d} (default 32 seeds x 32 seqs). This script splits
each file back into its per-seed blocks, pairs it with the original (shallow) MSA,
and reports depth + Neff for every source.

Output is one tidy CSV row per (target, source), grouped by target:

    target,source,depth,neff,neff_per_sqrtL
    8fjf__A,origin,120,45.231,2.820
    8fjf__A,seed01,32,11.874,0.740
    ...
    8fjf__A,seed32,32,12.015,0.749

Neff uses the same metric as msaflow.inference.quality_eval.compute_neff
(>80% identity clustering, weight 1/neighbours); see scripts/compute_neff.py.

Usage:
    python scripts/neff_by_target.py \
        --gen_dir    /PATH/TO/msa_gen \
        --origin_dir /store/deepfold3/results/msa/a3m \
        -o neff_by_target.csv
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_neff import compute_neff, seqs_to_tokens  # noqa: E402

SEED_RE = re.compile(r"_s(\d+)_n\d+\s*$")


def parse_records(path: str):
    """Yield (header, seq) with lowercase insertion columns removed."""
    header, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header, buf = line[1:], []
            elif line:
                buf.append("".join(c for c in line if not c.islower()))
    if header is not None:
        yield header, "".join(buf)


def neff_stats(seqs, threshold, block, max_seqs, rng):
    """Return (depth, seq_len, neff, neff_per_sqrtL). depth is the true count;
    Neff is computed on up to max_seqs rows (subsampled) when max_seqs > 0."""
    depth = len(seqs)
    use = seqs
    if max_seqs and depth > max_seqs:
        idx = rng.choice(depth, size=max_seqs, replace=False)
        use = [seqs[i] for i in sorted(idx)]
    toks = seqs_to_tokens(use)
    L = int(toks.shape[1]) if toks.size else 0
    ne = compute_neff(toks, threshold=threshold, block=block)
    return depth, L, ne, (ne / math.sqrt(L) if L else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-target depth + Neff: origin MSA vs each generated seed")
    ap.add_argument("--gen_dir", required=True,
                    help="Directory of generated {target}.a3m (zero-shot output)")
    ap.add_argument("--origin_dir", default="/store/deepfold3/results/msa/a3m",
                    help="Directory of original/shallow {target}.a3m (Origin)")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="Hamming cutoff for clustering (0.2 = 80%% identity)")
    ap.add_argument("--block", type=int, default=256,
                    help="Row-block size for the pairwise pass (memory knob)")
    ap.add_argument("--max_seqs", type=int, default=0,
                    help="Subsample Neff to this many rows when a source is deeper "
                         "(0 = no cap; depth column is always the true count)")
    ap.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed")
    ap.add_argument("-o", "--out_csv", default="neff_by_target.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gen_files = sorted(Path(args.gen_dir).glob("*.a3m"))
    if not gen_files:
        sys.exit(f"ERROR: no *.a3m in {args.gen_dir}")

    rows = []
    for gen_path in gen_files:
        target = gen_path.stem

        # ── origin (shallow) MSA ──────────────────────────────────────────────
        origin_path = Path(args.origin_dir) / f"{target}.a3m"
        if origin_path.exists():
            origin_seqs = [s for _, s in parse_records(str(origin_path))]
            d, L, ne, nen = neff_stats(origin_seqs, args.threshold, args.block,
                                       args.max_seqs, rng)
            rows.append({"target": target, "source": "origin", "depth": d,
                         "neff": round(ne, 3), "neff_per_sqrtL": round(nen, 4)})
        else:
            rows.append({"target": target, "source": "origin", "depth": 0,
                         "neff": 0.0, "neff_per_sqrtL": 0.0})

        # ── per-seed generated blocks (skip the leading query record) ─────────
        by_seed: dict[int, list[str]] = {}
        for header, seq in parse_records(str(gen_path)):
            m = SEED_RE.search(header)
            if m:
                by_seed.setdefault(int(m.group(1)), []).append(seq)
        for seed in sorted(by_seed):
            d, L, ne, nen = neff_stats(by_seed[seed], args.threshold, args.block,
                                       args.max_seqs, rng)
            rows.append({"target": target, "source": f"seed{seed + 1:02d}",
                         "depth": d, "neff": round(ne, 3),
                         "neff_per_sqrtL": round(nen, 4)})

        n_seed = len(by_seed)
        print(f"{target:28s}  origin={'Y' if origin_path.exists() else 'N'}  "
              f"seeds={n_seed}", flush=True)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "source", "depth",
                                           "neff", "neff_per_sqrtL"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_csv}  ({len(rows)} rows, {len(gen_files)} targets)")


if __name__ == "__main__":
    main()
