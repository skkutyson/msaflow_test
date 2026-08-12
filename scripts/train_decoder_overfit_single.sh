#!/bin/bash
#SBATCH --job-name=msaflow-overfit-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:1        # 1 GPU is plenty for a single sample
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/overfit_single_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/overfit_single_%j.err

# ── 목적 ───────────────────────────────────────────────────────────────────────
# faithful Protenix 임베딩 파이프라인 sanity check: preprocess_single.sh 가 만든
# 단일 타겟 LMDB 하나를 500 epoch 오버핏. loss → ~0 이면 파이프라인 정상.
# train_decoder.sh 의 apptainer 패턴을 따르되 단일 GPU / 단일 샘플로 축소.

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

TARGET_ID=${TARGET_ID:-7pv5-assembly1}
LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_overfit_${TARGET_ID}.lmdb}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/deepfold/users/yjlee4/decoder_overfit_single}

CONFIG=${CONFIG:-/mnt/MSA_FLOW/msaflow/configs/decoder_overfit_single.yaml}
ACCEL_CONFIG=/mnt/MSA_FLOW/msaflow/configs/accelerate_1gpu.yaml

# ── 사전 확인 ──────────────────────────────────────────────────────────────────
if [ ! -e "$LMDB_PATH" ]; then
    echo "ERROR: LMDB not found: $LMDB_PATH"
    echo "       Run scripts/preprocess_single.sh first (TARGET_ID=$TARGET_ID)."
    exit 1
fi
mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"

echo "============================================================"
echo " MSAFlow Decoder OVERFIT — SINGLE TARGET"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " TARGET_ID   : $TARGET_ID"
echo " LMDB        : $LMDB_PATH"
echo " OUTPUT_DIR  : $OUTPUT_DIR"
echo " CONFIG      : $CONFIG"
echo " Started     : $(date)"
echo "============================================================"

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

LMDB_DIR="$(dirname "$LMDB_PATH")"

# ── Apptainer + uv run ────────────────────────────────────────────────────────
#   /mnt/MSA_FLOW → REPO_DIR
#   /mnt/lmdb     → LMDB 디렉터리
#   /mnt/output   → OUTPUT_DIR

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$LMDB_DIR":/mnt/lmdb \
    --bind "$OUTPUT_DIR":/mnt/output \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run accelerate launch \
            --config_file $ACCEL_CONFIG \
            msaflow/training/train_decoder.py \
            --config     $CONFIG \
            --lmdb_path  /mnt/lmdb/$(basename "$LMDB_PATH") \
            --output_dir /mnt/output
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
