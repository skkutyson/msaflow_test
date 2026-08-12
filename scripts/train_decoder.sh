#!/bin/bash
#SBATCH --job-name=msaflow-decoder
#SBATCH --nodes=1
#SBATCH --nodelist=ada-004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:4
#SBATCH --cpus-per-task=16
#SBATCH --time=1000:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/decoder_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/decoder_%j.err

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_merged.lmdb}
VAL_LMDB_PATH=${VAL_LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_fold.lmdb}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/deepfold/users/yjlee4/decoder}

# CONFIG is overridable so one script serves both geometries:
#   default (α=0):   sbatch scripts/train_decoder.sh
#   α=0.5:  CONFIG=/mnt/MSA_FLOW/msaflow/configs/decoder_alpha05.yaml \
#           OUTPUT_DIR=/gpfs/deepfold/users/yjlee4/decoder_alpha05 \
#           sbatch --job-name=msaflow-dec-a05 scripts/train_decoder.sh
CONFIG=${CONFIG:-/mnt/MSA_FLOW/msaflow/configs/decoder.yaml}
ACCEL_CONFIG=/mnt/MSA_FLOW/msaflow/configs/accelerate_4gpu.yaml

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"

echo "============================================================"
echo " MSAFlow Decoder Training (SLURM + Apptainer + uv)"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " LMDB        : $LMDB_PATH"
echo " VAL_LMDB    : $VAL_LMDB_PATH"
echo " OUTPUT_DIR  : $OUTPUT_DIR"
echo " CONFIG      : $CONFIG"
echo " Started     : $(date)"
echo "============================================================"

# ── NCCL 튜닝 ─────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1   # GPU 1,2가 다른 PCIe root — P2P 비활성화
export NCCL_SHM_DISABLE=0   # shared memory 통신 사용
export OMP_NUM_THREADS=8

# ── Apptainer + uv run ────────────────────────────────────────────────────────
# val lmdb가 train lmdb와 같은 디렉토리이면 동일 마운트 재사용, 다르면 별도 바인드
LMDB_DIR="$(dirname "$LMDB_PATH")"
VAL_LMDB_DIR="$(dirname "$VAL_LMDB_PATH")"

EXTRA_BIND=""
if [ "$VAL_LMDB_DIR" != "$LMDB_DIR" ]; then
    EXTRA_BIND="--bind ${VAL_LMDB_DIR}:/mnt/val_lmdb"
    VAL_LMDB_MOUNT="/mnt/val_lmdb/$(basename "$VAL_LMDB_PATH")"
else
    VAL_LMDB_MOUNT="/mnt/lmdb/$(basename "$VAL_LMDB_PATH")"
fi

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$LMDB_DIR":/mnt/lmdb \
    ${EXTRA_BIND:+$EXTRA_BIND} \
    --bind "$OUTPUT_DIR":/mnt/output \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run accelerate launch \
            --config_file $ACCEL_CONFIG \
            msaflow/training/train_decoder.py \
            --config        $CONFIG \
            --lmdb_path     /mnt/lmdb/$(basename "$LMDB_PATH") \
            --val_lmdb_path $VAL_LMDB_MOUNT \
            --output_dir    /mnt/output
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
