#!/bin/bash
#SBATCH --job-name=msaflow-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.err

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

A3M_DIR=${A3M_DIR:-/gpfs/database/openfold/uniclust30}
OUTPUT_LMDB=${OUTPUT_LMDB:-/gpfs/deepfold/users/yjlee4/msaflow_openfold.lmdb}

# OpenProteinSet 레이아웃({ID}/a3m/uniclust30.a3m)에서는 protein당 db a3m이
# 여러 개일 수 있어 하나만 골라야 함. 빈 문자열이면 모든 *.a3m 처리.
A3M_NAME=${A3M_NAME:-uniclust30.a3m}

# RESUME=1 이면 LMDB에 이미 있는 key는 건너뜀 (48h 안에 못 끝나면 재제출)
RESUME=${RESUME:-1}

# 체크포인트는 repo 안에 있으므로 컨테이너 내부 경로로 바로 지정
PROTENIX_CKPT_CONTAINER=/mnt/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt

# Protenix CCD 파일 디렉터리 ($PROTENIX_ROOT_DIR/common 에서 로드됨)
# faithful featurization(SampleDictToFeatures)에 필수:
#   components.cif, components.cif.rdkit_mol.pkl 등
CCD_DIR=${CCD_DIR:-$HOME/common}

MAX_MSA_SEQS=${MAX_MSA_SEQS:-512}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
MAP_SIZE_GB=${MAP_SIZE_GB:-500}
N_CYCLE=${N_CYCLE:-3}           # Protenix recycling cycles (MSAFlow 기본값)

# ── CCD 파일 사전 검증 ─────────────────────────────────────────────────────────
for f in components.cif components.cif.rdkit_mol.pkl; do
    if [ ! -f "$CCD_DIR/$f" ]; then
        echo "ERROR: missing $CCD_DIR/$f — Protenix CCD files required for faithful featurization."
        echo "       Download from protenix.tos-cn-beijing.volces.com (see probe notes)."
        exit 1
    fi
done

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs" "$(dirname "$OUTPUT_LMDB")"

echo "============================================================"
echo " MSAFlow Preprocessing (SLURM + Apptainer)"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " REPO_DIR    : $REPO_DIR"
echo " SANDBOX     : $SANDBOX"
echo " A3M_DIR     : $A3M_DIR"
echo " OUTPUT_LMDB : $OUTPUT_LMDB"
echo " PROTENIX    : $PROTENIX_CKPT_CONTAINER (inside container)"
echo " CCD_DIR     : $CCD_DIR"
echo " A3M_NAME    : ${A3M_NAME:-<all *.a3m>}  |  RESUME: $RESUME"
echo " MAX_MSA     : $MAX_MSA_SEQS  |  MAX_LEN: $MAX_SEQ_LEN  |  N_CYCLE: $N_CYCLE"
echo " Started     : $(date)"
echo "============================================================"

# ── Apptainer 실행 ─────────────────────────────────────────────────────────────
# bind 구성:
#   /mnt/MSA_FLOW           → REPO_DIR (msaflow 패키지 + checkpoint + Protenix/esm 서브모듈)
#   /mnt/data               → A3M 파일 디렉터리
#   /mnt/output             → OUTPUT_LMDB 상위 디렉터리
#   /mnt/protenix_root/common → CCD_DIR (PROTENIX_ROOT_DIR env로 지정; $HOME bind에
#                             의존하지 않도록 명시적으로 바인드)

EXTRA_ARGS=""
[ -n "$A3M_NAME" ] && EXTRA_ARGS="$EXTRA_ARGS --a3m_name $A3M_NAME"
[ "$RESUME" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --resume"

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$A3M_DIR":/mnt/data \
    --bind "$(dirname "$OUTPUT_LMDB")":/mnt/output \
    --bind "$CCD_DIR":/mnt/protenix_root/common \
    --env PROTENIX_ROOT_DIR=/mnt/protenix_root \
    --env TORCH_CUDA_ARCH_LIST="8.9;12.0" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run python3 -m msaflow.data.preprocessing \
            --a3m_dir      /mnt/data \
            --output       /mnt/output/$(basename "$OUTPUT_LMDB") \
            --protenix_checkpoint $PROTENIX_CKPT_CONTAINER \
            --max_msa_seqs $MAX_MSA_SEQS \
            --max_seq_len  $MAX_SEQ_LEN \
            --device       cuda \
            --map_size_gb  $MAP_SIZE_GB \
            --n_cycle      $N_CYCLE \
            $EXTRA_ARGS
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
