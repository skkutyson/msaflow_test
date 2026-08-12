#!/bin/bash
#SBATCH --job-name=sfm-reconstruct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/reconstruct_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/reconstruct_%j.err

# ── 사용법 ────────────────────────────────────────────────────────────────────
# sbatch scripts/reconstruct_msa.sh <input.a3m> [output.a3m]
#
# 예시:
#   sbatch scripts/reconstruct_msa.sh /path/to/query.a3m /path/to/out.a3m
#   sbatch scripts/reconstruct_msa.sh /path/to/query.a3m   # output은 자동 생성
# ─────────────────────────────────────────────────────────────────────────────

INPUT_A3M=${1:?"Usage: sbatch reconstruct_msa.sh <input.a3m> [output.a3m]"}

# output 미지정 시 input과 같은 디렉토리에 _reconstructed.a3m 으로 저장
BASENAME=$(basename "$INPUT_A3M" .a3m)
DEFAULT_OUT="$(dirname "$INPUT_A3M")/${BASENAME}_reconstructed.a3m"
OUTPUT_A3M=${2:-$DEFAULT_OUT}

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/decoder_ema_final.pt}
PROTENIX_CKPT=$REPO_DIR/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt

N_SEQS=${N_SEQS:-32}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.0}
SEED=${SEED:-42}
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$REPO_DIR/logs" "$(dirname "$OUTPUT_A3M")"

echo "============================================================"
echo " MSAFlow SFM Reconstruct"
echo "============================================================"
echo " Job ID       : $SLURM_JOB_ID"
echo " Node         : $SLURMD_NODENAME"
echo " Input A3M    : $INPUT_A3M"
echo " Output A3M   : $OUTPUT_A3M"
echo " Decoder ckpt : $DECODER_CKPT"
echo " N_SEQS       : $N_SEQS"
echo " N_STEPS      : $N_STEPS"
echo " Temperature  : $TEMPERATURE"
echo " Seed         : $SEED"
echo " Started      : $(date)"
echo "============================================================"

INPUT_DIR=$(dirname "$INPUT_A3M")
OUTPUT_DIR=$(dirname "$OUTPUT_A3M")

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$INPUT_DIR":/mnt/input \
    --bind "$OUTPUT_DIR":/mnt/output \
    --bind "$(dirname "$DECODER_CKPT")":/mnt/decoder_ckpt \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run python msaflow/inference/generate.py \
            --mode          reconstruct \
            --input_a3m     /mnt/input/$(basename "$INPUT_A3M") \
            --decoder_ckpt  /mnt/decoder_ckpt/$(basename "$DECODER_CKPT") \
            --protenix_ckpt /mnt/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt \
            --output        /mnt/output/$(basename "$OUTPUT_A3M") \
            --n_seqs        $N_SEQS \
            --n_steps       $N_STEPS \
            --temperature   $TEMPERATURE \
            --seed          $SEED \
            --device        cuda
    "

echo "============================================================"
echo " Done: $(date)"
echo " Output: $OUTPUT_A3M"
echo "============================================================"
