#!/bin/bash
# ==============================================================================
# regression_baseline.sh  —  얇은 GPU 런처 (e→z 회귀 baseline vs latent_fm)
#
# "cosine 이 FM 평가 지표로 유효한가?" 를 판정한다.
#   1) cache-esm (GPU) : real npz의 query 로 ESM2 e 재계산 → esm/<target>.npz
#   2) run       (CPU) : target-단위 K-fold 로 ridge/MLP e→z 학습·평가,
#                        cos_ridge / cos_fm_best / cos_fm_mean / cos_fm_avg /
#                        diversity / cos_null 을 CSV·산점도·verdict 로 출력.
# compare_msa_emb.sh 가 만든 real/latent_fm npz 를 그대로 입력으로 쓴다.
#
# 사용법 (compare_msa_emb.sh 와 동일 — GPU 컴퓨트 노드에서, 로그인노드 X):
#     srun --partition=normal --gres=gpu:RTX4090:1 --cpus-per-task=8 \
#          --time=2:00:00 --pty bash
#     bash scripts/regression_baseline.sh
#
# 환경변수로 제어:
#   OUT_BASE  기본 /store/yjlee4/runs/compare_emb  (compare 와 동일 경로)
#   TARGETS   공백구분 타겟명; 비우면 OUT_BASE 아래 real+latent_fm 있는 전부
#   DEVICE    기본 cuda
#   RESUME    1(기본): 이미 esm npz 있는 타겟은 cache-esm 에서 skip
#   RUN_FIT   1(기본): cache-esm 뒤 CPU fit(run) 까지 이어서 실행. 0이면 캐시만.
#   KFOLD / SEED / MLP / DIV_TOL  (run 단계 파라미터)
# CLI 플래그를 뒤에 붙이면 run 단계 파이썬으로 그대로 전달된다.
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}

OUT_BASE=${OUT_BASE:-/store/yjlee4/runs/compare_emb}
TARGETS=${TARGETS:-}
DEVICE=${DEVICE:-cuda}

RESUME=${RESUME:-1}
RUN_FIT=${RUN_FIT:-1}

KFOLD=${KFOLD:-5}
SEED=${SEED:-0}
MLP=${MLP:-0}
DIV_TOL=${DIV_TOL:-0.01}

# ── GPU 노드 확인 (compare_msa_emb.sh 와 동일) ───────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi가 없습니다. 로그인 노드가 아니라 GPU 컴퓨트 노드에서 실행하세요." >&2
    echo "       예)  srun --gres=gpu:RTX4090:1 --pty bash  후 다시 실행" >&2
    exit 1
fi

# ── 입력 검증 ─────────────────────────────────────────────────────────────────
[ -d "$OUT_BASE" ] || { echo "ERROR: OUT_BASE 없음(먼저 compare_msa_emb.sh 로 추출): $OUT_BASE" >&2; exit 1; }

# ── 모듈 / CUDA / venv / 경로 (compare_msa_emb.sh 와 동일) ────────────────────
module load cuda/13.0.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/false)")")}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

source "$REPO_DIR/.venv/bin/activate"

export PYTHONPATH=$REPO_DIR/Protenix:${PYTHONPATH:-}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PY="$REPO_DIR/scripts/regression_baseline.py"

echo "============================================================"
echo " Regression baseline vs latent_fm"
echo " Node    : $(hostname)"
echo " Python  : $(python --version 2>&1)"
echo " OUT_BASE: $OUT_BASE"
echo " TARGETS : ${TARGETS:-<all real+latent_fm under OUT_BASE>}"
echo " DEVICE  : $DEVICE   RESUME: $RESUME   RUN_FIT: $RUN_FIT"
echo " KFOLD   : $KFOLD   SEED: $SEED   MLP: $MLP   DIV_TOL: $DIV_TOL"
echo " Started : $(date)"
echo "============================================================"

# ── 1) cache-esm (GPU) ────────────────────────────────────────────────────────
CACHE_EXTRA=""
[ -n "$TARGETS" ] && CACHE_EXTRA="$CACHE_EXTRA --targets $TARGETS"
[ "$RESUME" = "1" ] || CACHE_EXTRA="$CACHE_EXTRA --no_resume"

echo "--- [1/2] cache-esm (GPU) ---"
python "$PY" --out_base "$OUT_BASE" cache-esm --device "$DEVICE" $CACHE_EXTRA

# ── 2) run (CPU; GPU 노드에서도 OK) ───────────────────────────────────────────
if [ "$RUN_FIT" = "1" ]; then
    RUN_EXTRA=""
    [ -n "$TARGETS" ] && RUN_EXTRA="$RUN_EXTRA --targets $TARGETS"
    [ "$MLP" = "1" ]  && RUN_EXTRA="$RUN_EXTRA --mlp"

    echo "--- [2/2] run: fit + evaluate (CPU) ---"
    python "$PY" --out_base "$OUT_BASE" run \
        --kfold "$KFOLD" --seed "$SEED" --div_tol "$DIV_TOL" \
        $RUN_EXTRA "$@"
else
    echo "--- RUN_FIT=0 → cache-esm 만 수행. 평가는 따로: ---"
    echo "    python $PY --out_base $OUT_BASE run --kfold $KFOLD"
fi

echo "============================================================"
echo " Done: $(date)   →  $OUT_BASE/regression_baseline.{csv,png}"
echo "============================================================"
