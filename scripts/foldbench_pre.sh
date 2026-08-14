#!/bin/bash
#SBATCH --job-name=msaflow-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.err

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

A3M_DIR=${A3M_DIR:-/store/deepfold3/results/msa/a3m}
OUTPUT_STORE=${OUTPUT_STORE:-/gpfs/deepfold/users/yjlee4/msaflow_fold}
# 체크포인트는 repo 안에 있으므로 컨테이너 내부 경로로 바로 지정
PROTENIX_CKPT_CONTAINER=/mnt/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt

MAX_MSA_SEQS=${MAX_MSA_SEQS:-512}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
SHARD_MAX_GB=${SHARD_MAX_GB:-4}
PROTEIN_LIST=${PROTEIN_LIST:-}

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs" "$(dirname "$OUTPUT_STORE")"

echo "============================================================"
echo " MSAFlow Preprocessing (SLURM + Apptainer)"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " REPO_DIR    : $REPO_DIR"
echo " SANDBOX     : $SANDBOX"
echo " A3M_DIR     : $A3M_DIR"
echo " OUTPUT_STORE : $OUTPUT_STORE"
echo " PROTENIX    : ${PROTENIX_CKPT_CONTAINER:-disabled} (inside container)"
echo " MAX_MSA     : $MAX_MSA_SEQS  |  MAX_LEN: $MAX_SEQ_LEN"
echo " Started     : $(date)"
echo "============================================================"

# ── Apptainer 실행 ─────────────────────────────────────────────────────────────
# bind 구성:
#   /mnt/MSA_FLOW  → REPO_DIR (msaflow 패키지 + checkpoint + Protenix/esm 서브모듈)
#   /mnt/data      → A3M 파일 디렉터리
#   /mnt/output    → OUTPUT_STORE 상위 디렉터리
#
# NOTE: --nv 만 사용. 추가 --bind /dev/nvidia* 는 --nv 가 이미 설정한
# NVML 환경을 덮어써서 Can't initialize NVML / device_count=0 을 유발함.
apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$A3M_DIR":/mnt/data \
    --bind "$(dirname "$OUTPUT_STORE")":/mnt/output \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        TORCH_CUDA_ARCH_LIST="12.0" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python3 -m msaflow.data.preprocessing \
            --a3m_dir      /mnt/data \
            --output       /mnt/output/$(basename "$OUTPUT_STORE") \
            --protenix_checkpoint $PROTENIX_CKPT_CONTAINER \
            --max_msa_seqs $MAX_MSA_SEQS \
            --max_seq_len  $MAX_SEQ_LEN \
            --device       cuda \
            --shard_max_gb $SHARD_MAX_GB \
            ${PROTEIN_LIST:+--protein_list $PROTEIN_LIST}
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
