#!/usr/bin/env bash
# Compute Neff for generated MSAs (e.g. zero-shot 32x32 output).
# CPU-only + numpy only — no GPU / SLURM / container needed, runs on any node.
#
# Usage:
#   MSA_DIR=/PATH/TO/zeroshot bash scripts/run_neff.sh
#   # or pass explicit files / globs:
#   bash scripts/run_neff.sh /PATH/TO/zeroshot/*.a3m
#
# Env:
#   MSA_DIR   directory of *.a3m to score (ignored if files are passed as args)
#   OUT_CSV   output CSV path (default: neff.csv in the repo root)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MSA_DIR=${MSA_DIR:-}
OUT_CSV=${OUT_CSV:-neff.csv}

if [ "$#" -gt 0 ]; then
    INPUTS=("$@")
elif [ -n "$MSA_DIR" ]; then
    INPUTS=("$MSA_DIR"/*.a3m)
else
    echo "Usage: MSA_DIR=/PATH/TO/zeroshot bash scripts/run_neff.sh"
    echo "   or: bash scripts/run_neff.sh file1.a3m file2.a3m ..."
    exit 1
fi

uv run python scripts/compute_neff.py "${INPUTS[@]}" --out_csv "$OUT_CSV"
