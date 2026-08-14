#!/bin/bash
#SBATCH --job-name=msaflow-neff-dist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/neff_dist_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/neff_dist_%j.err

REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

STORE_PATH=${STORE_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_new_fold}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/deepfold/users/yjlee4/neff_analysis}
SAMPLE_EVERY=${SAMPLE_EVERY:-1}   # 1=전체, 10=10% 샘플링

mkdir -p "$REPO_DIR/logs"

echo "============================================================"
echo " MSAFlow Neff Distribution Analysis"
echo "============================================================"
echo " STORE      : $STORE_PATH"
echo " OUTPUT_DIR : $OUTPUT_DIR"
echo " SAMPLE     : every ${SAMPLE_EVERY}th entry"
echo " Started    : $(date)"
echo "============================================================"

apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$(dirname "$STORE_PATH")":/mnt/store \
    --bind "$(dirname "$OUTPUT_DIR")":/mnt/output \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        mkdir -p /mnt/output/$(basename "$OUTPUT_DIR")

        uv run python3 scripts/neff_distribution.py \
            --store_path  /mnt/store/$(basename "$STORE_PATH") \
            --output_dir  /mnt/output/$(basename "$OUTPUT_DIR") \
            --sample_every $SAMPLE_EVERY
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
