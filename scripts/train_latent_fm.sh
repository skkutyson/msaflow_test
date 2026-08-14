#!/bin/bash
#SBATCH --job-name=msaflow-latent-fm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/latent_fm_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/latent_fm_%j.err

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

STORE_PATH=${STORE_PATH:-/home/yjlee4/data/msaflow_openfold}
OUTPUT_DIR=${OUTPUT_DIR:-/home/yjlee4/data/runs/latent_fm}

CONFIG=/mnt/MSA_FLOW/msaflow/configs/latent_fm.yaml
ACCEL_CONFIG=/mnt/MSA_FLOW/msaflow/configs/accelerate_2gpu.yaml

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"

echo "============================================================"
echo " MSAFlow Latent FM Training (SLURM + Apptainer + uv)"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " STORE       : $STORE_PATH"
echo " OUTPUT_DIR  : $OUTPUT_DIR"
echo " Started     : $(date)"
echo "============================================================"

# ── W&B 설정 ──────────────────────────────────────────────────────────────────
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-$HOME/.wandb_api_key}
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
    export WANDB_API_KEY=$(cat "$WANDB_API_KEY_FILE")
else
    echo "[WARN] W&B API key file not found: $WANDB_API_KEY_FILE  →  logging disabled"
    export WANDB_MODE=disabled
fi

# ── NCCL 튜닝 ─────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=8

# ── Apptainer + uv run ────────────────────────────────────────────────────────
apptainer exec --nv \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
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
echo "============================================================"
