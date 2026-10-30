#!/usr/bin/env python3
"""Record the shallow-MSA depth of each foldbench target → CSV.

depth = number of HOMOLOG sequences in the target's a3m, i.e. records minus the
query (the first entry) — a query-only a3m is depth 0. Targets and a3m dir default to the
fold_benchmark config, so the output joins on `target` (pdb_id__chain_id) with
compare_msa_emb's plot-all metrics CSV and the OST TM/lDDT scores.

Usage
-----
  uv run python scripts/msa_depth_csv.py
  uv run python scripts/msa_depth_csv.py --targets 8ux2-assembly1__A --out depth.csv
"""
import argparse
import csv
from pathlib import Path

DEF_CSV = "/home/yjlee4/MSA_FLOW/monomer_protein.csv"
DEF_A3M_DIR = "/store/deepfold3/results/msa/a3m"
DEF_OUT = "/store/yjlee4/runs/fold_benchmark/msa_depth.csv"


def _targets_from_csv(csv_path):
    """monomer_protein.csv (pdb_id,chain_id) → ['<pdb_id>__<chain_id>', ...]."""
    out = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            pdb, chain = row["pdb_id"].strip(), row["chain_id"].strip()
            if pdb and chain:
                out.append(f"{pdb}__{chain}")
    return out


def _depth_and_len(a3m):
    """(depth, uniq, query length) from an a3m; (0, 0, 0) if empty.

    a3m files are often a CONCATENATION of several MSA searches, each carrying
    its OWN copy of the query as a record — so the query header can appear more
    than once. We identify the query by the FIRST header name and drop EVERY
    record with that name (not just record 0):
        depth = homolog records, all query copies removed
        uniq  = depth after de-duplicating identical (ungapped) sequences
    """
    headers, seqs, first = [], [], None
    cur = None
    with open(a3m) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].strip().split()[0]      # accession / name
                headers.append(cur)
            elif line.strip():
                s = line.strip()
                seqs.append((cur, s))
                if first is None:
                    first = s
    if not headers:
        return 0, 0, 0
    qname = headers[0]                                  # target id == the query
    hits = [s for h, s in seqs if h != qname]           # drop all query copies
    depth = len(hits)
    uniq = len({s.replace("-", "").upper() for s in hits})
    qlen = len(first.replace("-", "")) if first else 0
    return depth, uniq, qlen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEF_CSV, help="target list (pdb_id,chain_id)")
    ap.add_argument("--targets", nargs="*", default=None, help="explicit targets; overrides --csv")
    ap.add_argument("--a3m_dir", default=DEF_A3M_DIR)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    targets = args.targets or _targets_from_csv(args.csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "depth", "uniq", "L"])
        for t in targets:
            a3m = Path(args.a3m_dir) / f"{t}.a3m"
            if a3m.is_file():
                depth, uniq, qlen = _depth_and_len(a3m)
            else:
                depth, uniq, qlen = 0, 0, 0
                n_missing += 1
            w.writerow([t, depth, uniq, qlen])

    print(f"wrote {args.out}  ({len(targets)} targets"
          + (f", {n_missing} with no a3m → depth 0)" if n_missing else ")"))


if __name__ == "__main__":
    main()
