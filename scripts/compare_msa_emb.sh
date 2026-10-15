#!/bin/bash
# ==============================================================================
# compare_msa_emb.sh  —  얇은 GPU 런처 (real vs latent_fm MSA 임베딩 추출)
#
# real(AF3) 임베딩 1개 + zero-shot(latent_fm) seed 4개를 타겟별로 뽑아
#   <OUT_BASE>/<target>/real/<target>.npz        real:     (L,128)
#   <OUT_BASE>/<target>/latent_fm/<target>.npz    zeroshot: (K,L,128)
# 로 저장한다. 플롯은 GPU 없이 CPU 에서 따로:
#   uv run python scripts/compare_msa_emb.py --out_base <OUT_BASE> plot --target <t>
#   uv run python scripts/compare_msa_emb.py --out_base <OUT_BASE> plot-all
#
# 사용법 (반드시 GPU 컴퓨트 노드에서 — 로그인 노드 'ada'엔 GPU/store 없음):
#     srun --partition=normal --gres=gpu:RTX4090:1 --cpus-per-task=8 \
#          --time=4:00:00 --pty bash
#     TARGETS="7tlh 8abc" bash scripts/compare_msa_emb.sh
#
# 환경변수로 제어:
#   TARGETS   공백구분 타겟명 (a3m = <A3M_DIR>/<target>.a3m); 비우면 A3M_DIR 전체 *.a3m
#   A3M_DIR   기본 /store/deepfold3/results/msa/a3m
#   OUT_BASE  기본 /store/yjlee4/runs/compare_emb  (없으면 생성)
#   LATENT_FM_CKPT / PROTENIX_CKPT / N_SEEDS / N_STEPS / TEMPERATURE / N_CYCLE
# CLI 플래그를 뒤에 붙이면 그대로 파이썬으로 전달된다.
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}

A3M_DIR=${A3M_DIR:-/store/deepfold3/results/msa/a3m}
OUT_BASE=${OUT_BASE:-/store/yjlee4/runs/compare_emb}
# 기본은 monomer_protein.csv 의 (pdb_id,chain_id) → pdb_id__chain_id 타겟만.
# a3m 이 없는 타겟은 자동 skip. TARGETS 를 주면 CSV 대신 그 목록을 쓴다.
CSV=${CSV:-$REPO_DIR/monomer_protein.csv}
TARGETS=${TARGETS:-}

LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latest.pt}
PROTENIX_CKPT=${PROTENIX_CKPT:-$REPO_DIR/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}

N_SEEDS=${N_SEEDS:-4}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}
N_CYCLE=${N_CYCLE:-3}

# RESUME=1 (기본): 이미 npz 있는 타겟 skip → GPU 조금씩 날 때마다 끊어 돌려도 이어짐.
# PLOT=1 (기본): 타겟마다 바로 PCA plot 저장. 매 타겟 후 GPU 캐시 비움.
RESUME=${RESUME:-1}
PLOT=${PLOT:-1}
PLOT_METHOD=${PLOT_METHOD:-pca}

# ── GPU 노드 확인 ─────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi가 없습니다. 로그인 노드가 아니라 GPU 컴퓨트 노드에서 실행하세요." >&2
    echo "       예)  srun --gres=gpu:RTX4090:1 --pty bash  후 다시 실행" >&2
    exit 1
fi

# ── 입력 검증 ─────────────────────────────────────────────────────────────────
[ -f "$LATENT_FM_CKPT" ] || { echo "ERROR: latent_fm ckpt 없음: $LATENT_FM_CKPT" >&2; exit 1; }
[ -f "$PROTENIX_CKPT" ]  || { echo "ERROR: protenix ckpt 없음: $PROTENIX_CKPT" >&2; exit 1; }
[ -d "$A3M_DIR" ]        || { echo "ERROR: a3m 디렉터리 없음: $A3M_DIR" >&2; exit 1; }
[ -n "$TARGETS" ] || [ -f "$CSV" ] || { echo "ERROR: csv 없음: $CSV" >&2; exit 1; }

# ── 모듈 / CUDA (있으면 로드) ─────────────────────────────────────────────────
module load cuda/13.0.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/false)")")}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# ── venv 활성화 ───────────────────────────────────────────────────────────────
source "$REPO_DIR/.venv/bin/activate"

# ── Protenix import 경로 (repo root는 esm/ 그림자 문제로 제외) ────────────────
export PYTHONPATH=$REPO_DIR/Protenix:${PYTHONPATH:-}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_BASE"

echo "============================================================"
echo " MSAFlow embedding compare (extract)"
echo " Node    : $(hostname)"
echo " Python  : $(python --version 2>&1)"
echo " A3M_DIR : $A3M_DIR"
echo " OUT_BASE: $OUT_BASE"
echo " TARGETS : ${TARGETS:-<from csv: $CSV>}"
echo " LATENT  : $LATENT_FM_CKPT"
echo " N_SEEDS : $N_SEEDS  N_STEPS: $N_STEPS  TEMP: $TEMPERATURE  N_CYCLE: $N_CYCLE"
echo " RESUME  : $RESUME  PLOT: $PLOT ($PLOT_METHOD)"
echo " Started : $(date)"
echo "============================================================"

EXTRA=""
[ -n "$TARGETS" ] && EXTRA="$EXTRA --targets $TARGETS"
[ "$RESUME" = "1" ] || EXTRA="$EXTRA --no_resume"
[ "$PLOT" = "1" ]   || EXTRA="$EXTRA --no_plot"

python "$REPO_DIR/scripts/compare_msa_emb.py" --out_base "$OUT_BASE" extract \
    --a3m_dir        "$A3M_DIR" \
    --csv            "$CSV" \
    --latent_fm_ckpt "$LATENT_FM_CKPT" \
    --protenix_ckpt  "$PROTENIX_CKPT" \
    --n_seeds "$N_SEEDS" --n_steps "$N_STEPS" \
    --temperature "$TEMPERATURE" --n_cycle "$N_CYCLE" \
    --plot_method "$PLOT_METHOD" \
    $EXTRA "$@"

echo "============================================================"
echo " Done: $(date)   →  $OUT_BASE"
echo " Plot (CPU): uv run python scripts/compare_msa_emb.py --out_base $OUT_BASE plot-all"
echo "============================================================"
