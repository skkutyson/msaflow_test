#!/bin/bash
#SBATCH --job-name=latent-fm-foldbench
#SBATCH --nodes=1
#SBATCH --nodelist=ada-004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/latent_fm_foldbench_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/latent_fm_foldbench_%j.err

# Latent FM 학습 — foldbench LMDB (msaflow_fold.lmdb) 사용
#
# decoder 학습과 동일한 LMDB를 사용.
# LatentFMDataset이 msa_emb + esm_emb 둘 다 있는 엔트리만 자동으로 필터링함.
#
# 환경변수로 경로 오버라이드:
#   LMDB_PATH=/other/path.lmdb sbatch scripts/train_latent_fm_foldbench.sh
#   OUTPUT_DIR=/other/dir       sbatch scripts/train_latent_fm_foldbench.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

STORE_PATH=${STORE_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_fold}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/deepfold/users/yjlee4/latent_fm_foldbench}

CONFIG=/mnt/MSA_FLOW/msaflow/configs/latent_fm_foldbench.yaml
ACCEL_CONFIG=/mnt/MSA_FLOW/msaflow/configs/accelerate_4gpu.yaml

# ── 로그 / 출력 디렉터리 ────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"

echo "============================================================"
echo " Latent FM Training — foldbench store"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " STORE       : $STORE_PATH"
echo " OUTPUT_DIR  : $OUTPUT_DIR"
echo " CONFIG      : $CONFIG"
echo " Started     : $(date)"
echo "============================================================"

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-$HOME/.wandb_api_key}
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
    export WANDB_API_KEY=$(cat "$WANDB_API_KEY_FILE")
else
    echo "[WARN] W&B API key not found: $WANDB_API_KEY_FILE  →  W&B disabled"
    export WANDB_MODE=disabled
fi

# ── NCCL ──────────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=8

# ── Apptainer + accelerate launch ─────────────────────────────────────────────
apptainer exec --nv \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    --env NCCL_DEBUG="$NCCL_DEBUG" \
    --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    --env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
    --env NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$(dirname "$STORE_PATH")":/mnt/store \
    --bind "$OUTPUT_DIR":/mnt/output \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run accelerate launch \
            --config_file $ACCEL_CONFIG \
            msaflow/training/train_latent_fm.py \
            --config      $CONFIG \
            --store_path  /mnt/store/$(basename "$STORE_PATH") \
            --output_dir  /mnt/output
    "

echo "============================================================"
echo " Done: $(date)"
echo " Checkpoint: $OUTPUT_DIR/latent_fm_ema_final.pt"
echo "============================================================"
