#!/bin/bash
#SBATCH --job-name=msaflow-pre-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/preprocess_single_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/preprocess_single_%j.err

# ── 목적 ───────────────────────────────────────────────────────────────────────
# faithful Protenix 임베딩 파이프라인 sanity check용: 단 하나의 타겟에 대해서만
# LMDB를 만든다. overfit 테스트(train_decoder_overfit_single.sh)의 입력.
# preprocess.sh 를 기반으로 --protein_list 로 타겟 1개만 처리하도록 축소.

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

# 타겟 1개: a3m 파일의 stem(확장자 제외)이 곧 LMDB key.
TARGET_ID=${TARGET_ID:-7pv5-assembly1}
A3M_DIR=${A3M_DIR:-/home/yjlee4/a3m_output}
OUTPUT_LMDB=${OUTPUT_LMDB:-/gpfs/deepfold/users/yjlee4/msaflow_overfit_${TARGET_ID}.lmdb}

# 체크포인트는 repo 안에 있으므로 컨테이너 내부 경로로 바로 지정
PROTENIX_CKPT_CONTAINER=/mnt/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt

# Protenix CCD 파일 디렉터리 ($PROTENIX_ROOT_DIR/common 에서 로드됨)
CCD_DIR=${CCD_DIR:-$HOME/common}

MAX_MSA_SEQS=${MAX_MSA_SEQS:-512}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
MAP_SIZE_GB=${MAP_SIZE_GB:-5}    # 단일 타겟이라 작게
N_CYCLE=${N_CYCLE:-1}            # sanity check: recycling 1회

# ── CCD 파일 사전 검증 ─────────────────────────────────────────────────────────
for f in components.cif components.cif.rdkit_mol.pkl; do
    if [ ! -f "$CCD_DIR/$f" ]; then
        echo "ERROR: missing $CCD_DIR/$f — Protenix CCD files required for faithful featurization."
        echo "       Download from protenix.tos-cn-beijing.volces.com (see probe notes)."
        exit 1
    fi
done

# ── 타겟 a3m 존재 확인 + protein_list 파일 생성 ────────────────────────────────
if [ ! -f "$A3M_DIR/${TARGET_ID}.a3m" ]; then
    echo "ERROR: target a3m not found: $A3M_DIR/${TARGET_ID}.a3m"
    exit 1
fi
mkdir -p "$REPO_DIR/logs" "$(dirname "$OUTPUT_LMDB")"
PROTEIN_LIST_HOST="$REPO_DIR/logs/overfit_single_ids_${TARGET_ID}.txt"
echo "$TARGET_ID" > "$PROTEIN_LIST_HOST"
PROTEIN_LIST_CONTAINER="/mnt/MSA_FLOW/logs/$(basename "$PROTEIN_LIST_HOST")"

echo "============================================================"
echo " MSAFlow Preprocessing — SINGLE TARGET (faithful Protenix)"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPUs        : $CUDA_VISIBLE_DEVICES"
echo " TARGET_ID   : $TARGET_ID"
echo " A3M_DIR     : $A3M_DIR"
echo " OUTPUT_LMDB : $OUTPUT_LMDB"
echo " PROTENIX    : $PROTENIX_CKPT_CONTAINER (inside container)"
echo " CCD_DIR     : $CCD_DIR"
echo " N_CYCLE     : $N_CYCLE  |  MAX_MSA: $MAX_MSA_SEQS  |  MAX_LEN: $MAX_SEQ_LEN"
echo " Started     : $(date)"
echo "============================================================"

# ── Apptainer 실행 ─────────────────────────────────────────────────────────────
#   /mnt/MSA_FLOW             → REPO_DIR
#   /mnt/data                 → A3M_DIR
#   /mnt/output               → OUTPUT_LMDB 상위 디렉터리
#   /mnt/protenix_root/common → CCD_DIR (PROTENIX_ROOT_DIR env로 지정)

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$A3M_DIR":/mnt/data \
    --bind "$(dirname "$OUTPUT_LMDB")":/mnt/output \
    --bind "$CCD_DIR":/mnt/protenix_root/common \
    --env PROTENIX_ROOT_DIR=/mnt/protenix_root \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        TORCH_CUDA_ARCH_LIST="12.0" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python3 -m msaflow.data.preprocessing \
            --a3m_dir      /mnt/data \
            --output       /mnt/output/$(basename "$OUTPUT_LMDB") \
            --protenix_checkpoint $PROTENIX_CKPT_CONTAINER \
            --protein_list $PROTEIN_LIST_CONTAINER \
            --max_msa_seqs $MAX_MSA_SEQS \
            --max_seq_len  $MAX_SEQ_LEN \
            --device       cuda \
            --map_size_gb  $MAP_SIZE_GB \
            --n_cycle      $N_CYCLE
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
