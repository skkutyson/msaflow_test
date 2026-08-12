#!/bin/bash
# Orphan protein TM-score benchmark.
#
# Compares two folding conditions for proteins with <15 MSA sequences:
#   shallow  — fold with the original sparse real MSA (baseline)
#   fewshot  — fold with sparse MSA + MSAFlow-generated sequences (augmented)
#
# Pipeline:
#   1. Read orphan protein list (output of find_low_msa.sh)
#   2. Build FASTA + copy sparse a3m files to a working directory
#   3. Download reference CIF files from RCSB PDB
#   4. Run fold_benchmark.py in shallow mode  → benchmark_shallow.csv
#   5. Run fold_benchmark.py in fewshot mode  → benchmark_fewshot.csv
#   6. Run compare_orphan_bench.py to print delta-TM table
#
# Usage:
#   sbatch scripts/orphan_bench.sh
#   ORPHAN_LIST=logs/my_low_msa.tsv sbatch scripts/orphan_bench.sh
#
#SBATCH --job-name=orphan-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/orphan_bench_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/orphan_bench_%j.err

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

# Input: orphan protein list (TSV with columns: protein_id  n_msa_seqs  seq_len)
# Produced by find_low_msa.sh → logs/find_low_msa_{job_id}.out
# Set ORPHAN_LIST to override.
ORPHAN_LIST=${ORPHAN_LIST:-$(ls -t "$REPO_DIR/logs/find_low_msa_"*.out 2>/dev/null | head -1)}

A3M_DIR=${A3M_DIR:-/store/deepfold3/results/msa/a3m}
LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_fold.lmdb}

DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-/home/yjlee4/data/runs/latent_fm/latent_fm_ema_final.pt}
PROTENIX_CKPT=${PROTENIX_CKPT:-/home/yjlee4/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}
PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}

OUTPUT_ROOT=${OUTPUT_ROOT:-/gpfs/deepfold/users/yjlee4/orphan_bench}
MSA_THRESHOLD=${MSA_THRESHOLD:-15}

N_SYN_SEEDS=${N_SYN_SEEDS:-5}
N_SEQS_PER_SEED=${N_SEQS_PER_SEED:-32}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}

TMSCORE_BIN=${TMSCORE_BIN:-USalign}   # USalign handles CIF directly; fallback: TMscore

mkdir -p "$REPO_DIR/logs" "$OUTPUT_ROOT"

echo "============================================================"
echo " Orphan Protein Benchmark"
echo "============================================================"
echo " Job ID        : $SLURM_JOB_ID"
echo " Node          : $SLURMD_NODENAME"
echo " ORPHAN_LIST   : $ORPHAN_LIST"
echo " A3M_DIR       : $A3M_DIR"
echo " DECODER_CKPT  : $DECODER_CKPT"
echo " LATENT_FM_CKPT: $LATENT_FM_CKPT"
echo " PROTENIX_CKPT : $PROTENIX_CKPT"
echo " OUTPUT_ROOT   : $OUTPUT_ROOT"
echo " MSA_THRESHOLD : < $MSA_THRESHOLD"
echo " Started       : $(date)"
echo "============================================================"

if [ -z "$ORPHAN_LIST" ]; then
    echo "ERROR: No orphan protein list found. Run find_low_msa.sh first."
    echo "       Or set ORPHAN_LIST=/path/to/find_low_msa_XXXXX.out"
    exit 1
fi

# ── Step 1-3: 데이터 준비 (FASTA, sparse a3m, ref CIF) ───────────────────────
PREP_DIR="$OUTPUT_ROOT/prep"
mkdir -p "$PREP_DIR/shallow_msas" "$PREP_DIR/ref_pdbs"

apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$A3M_DIR":/mnt/a3m \
    --bind "$PREP_DIR":/mnt/prep \
    --bind "$(dirname "$LMDB_PATH")":/mnt/lmdb \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run python3 scripts/prep_orphan_data.py \
            --orphan_list    '/mnt/MSA_FLOW/logs/$(basename "$ORPHAN_LIST")' \
            --csv            /mnt/MSA_FLOW/monomer_protein.csv \
            --lmdb_path      /mnt/lmdb/$(basename "$LMDB_PATH") \
            --a3m_dir        /mnt/a3m \
            --output_fasta   /mnt/prep/orphan_proteins.fasta \
            --shallow_msa_dir /mnt/prep/shallow_msas \
            --ref_pdb_dir    /mnt/prep/ref_pdbs \
            --threshold      $MSA_THRESHOLD
    "

echo "Data preparation done: $(date)"

# ── Step 4: Shallow baseline (원본 sparse MSA 그대로 folding) ─────────────────
SHALLOW_DIR="$OUTPUT_ROOT/shallow"
mkdir -p "$SHALLOW_DIR"

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$PREP_DIR":/mnt/prep \
    --bind "$SHALLOW_DIR":/mnt/output \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW
        export PROTENIX_ROOT_DIR=/mnt/MSA_FLOW
        export PYTHONPATH=/mnt/MSA_FLOW/Protenix:\${PYTHONPATH:-}

        TORCH_CUDA_ARCH_LIST='12.0' \
        uv run python3 -m msaflow.inference.fold_benchmark \
            --fasta           /mnt/prep/orphan_proteins.fasta \
            --decoder_ckpt    dummy \
            --latent_fm_ckpt  dummy \
            --output_dir      /mnt/output \
            --mode            shallow \
            --shallow_msa_dir /mnt/prep/shallow_msas \
            --ref_pdb_dir     /mnt/prep/ref_pdbs \
            --protenix_model  $PROTENIX_MODEL \
            --tmscore_bin     $TMSCORE_BIN \
            --device          cuda
    "

echo "Shallow baseline done: $(date)"

# ── Step 5: Fewshot augmented (sparse MSA + MSAFlow 증강 MSA로 folding) ───────
FEWSHOT_DIR="$OUTPUT_ROOT/fewshot"
mkdir -p "$FEWSHOT_DIR"

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$PREP_DIR":/mnt/prep \
    --bind "$FEWSHOT_DIR":/mnt/output \
    --bind "$(dirname "$DECODER_CKPT")":/mnt/decoder_ckpt \
    --bind "$(dirname "$LATENT_FM_CKPT")":/mnt/latent_fm_ckpt \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW
        export PROTENIX_ROOT_DIR=/mnt/MSA_FLOW
        export PYTHONPATH=/mnt/MSA_FLOW/Protenix:\${PYTHONPATH:-}

        TORCH_CUDA_ARCH_LIST='12.0' \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python3 -m msaflow.inference.fold_benchmark \
            --fasta           /mnt/prep/orphan_proteins.fasta \
            --decoder_ckpt    /mnt/decoder_ckpt/$(basename "$DECODER_CKPT") \
            --latent_fm_ckpt  /mnt/latent_fm_ckpt/$(basename "$LATENT_FM_CKPT") \
            --output_dir      /mnt/output \
            --mode            fewshot \
            --shallow_msa_dir /mnt/prep/shallow_msas \
            --ref_pdb_dir     /mnt/prep/ref_pdbs \
            --protenix_ckpt   /mnt/MSA_FLOW/checkpoints/checkpoints/$(basename "$PROTENIX_CKPT") \
            --protenix_model  $PROTENIX_MODEL \
            --tmscore_bin     $TMSCORE_BIN \
            --n_seeds         $N_SYN_SEEDS \
            --n_seqs          $N_SEQS_PER_SEED \
            --n_steps         $N_STEPS \
            --temperature     $TEMPERATURE \
            --device          cuda
    "

echo "Fewshot augmented done: $(date)"

# ── Step 6: 결과 비교 ─────────────────────────────────────────────────────────
apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$OUTPUT_ROOT":/mnt/results \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW
        uv run python3 scripts/compare_orphan_bench.py \
            --shallow_csv  /mnt/results/shallow/shard.csv \
            --fewshot_csv  /mnt/results/fewshot/shard.csv \
            --output_csv   /mnt/results/orphan_comparison.csv
    "

echo "============================================================"
echo " Done: $(date)"
echo " Results: $OUTPUT_ROOT/orphan_comparison.csv"
echo "============================================================"
