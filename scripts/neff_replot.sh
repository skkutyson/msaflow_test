#!/bin/bash
#SBATCH --job-name=msaflow-neff-replot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/neff_replot_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/neff_replot_%j.err

REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox
NPY_PATH=${NPY_PATH:-/gpfs/deepfold/users/yjlee4/neff_analysis/neff_values.npy}

mkdir -p "$REPO_DIR/logs"

apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$(dirname "$NPY_PATH")":/mnt/neff \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW
        uv run python3 - <<'PYEOF'
import numpy as np, sys
sys.path.insert(0, '/mnt/MSA_FLOW')
from scripts.neff_distribution import ascii_histogram

neff = np.load('/mnt/neff/$(basename "$NPY_PATH")')
print(f'Loaded {len(neff):,} entries')
print()

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_vals = np.percentile(neff, pcts)
print(f'  Mean : {neff.mean():>8.2f}   Std: {neff.std():>8.2f}')
print(f'  Min  : {neff.min():>8.2f}   Max: {neff.max():>8.2f}')
print()
print('  Percentiles (Neff):')
for p, v in zip(pcts, pct_vals):
    print(f'    p{p:02d} = {v:>8.2f}   log10={np.log10(max(v,1e-3)):.3f}')
print()
print('  log10(Neff) histogram:')
print(ascii_histogram(neff))

# 스파이크 진단
n10  = int(((neff >= 9.9) & (neff <= 10.1)).sum())
nle  = int((neff <= 10.0).sum())
print(f'\n  [진단] Neff ≈ 10 : {n10:>7,}  ({100*n10/len(neff):.1f}%)')
print(f'  [진단] Neff ≤ 10 : {nle:>7,}  ({100*nle/len(neff):.1f}%)')
print(f'  → filter_msa min_seqs=10 패딩으로 Neff=10 고정된 MSA 다수')
PYEOF
    "
