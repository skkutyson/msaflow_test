#!/bin/bash
# ==============================================================================
# run_fold_benchmark_gpu.sh  —  얇은 GPU 런처
#
# 모든 설정(타겟/모드/seed/sample/모델/checkpoint 경로/OST)은 YAML 로 제어한다.
# 이 스크립트는 GPU 노드 환경(venv, CUDA, PYTHONPATH)만 잡고 파이썬으로 넘긴다.
# 실제 파이프라인은 msaflow/inference/fold_benchmark.py 가 end-to-end 로 수행:
#   입력 a3m → MSA 증강 → Protenix 폴딩 → OST GT 채점(monomer_protein_ost.csv).
#
# 사용법 (반드시 GPU 컴퓨트 노드에서 — 로그인 노드 'ada'엔 GPU/gpfs 없음):
#     srun --partition=normal --nodelist=ada-004 --gres=gpu:RTX6000:1 \
#          --cpus-per-task=8 --time=8:00:00 --pty bash
#     bash scripts/run_fold_benchmark_gpu.sh [config.yaml]
#
#   config.yaml 미지정 시 msaflow/configs/fold_benchmark.yaml 사용.
#
# 개별 값을 덮어쓰려면 CLI 플래그를 그대로 뒤에 붙인다:
#     bash scripts/run_fold_benchmark_gpu.sh msaflow/configs/fold_benchmark.yaml \
#          --mode zeroshot --n_sample 3
#
# 전체 타겟 배치 실행은 이 스크립트를 sbatch 로 감싸서 돌린다 (클러스터는 SLURM 전용).
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
DEFAULT_CONFIG=$REPO_DIR/msaflow/configs/fold_benchmark.yaml
# 첫 인자가 '존재하는 파일'이면 config 로 소비하고 나머지는 파이썬 CLI 오버라이드로 전달.
# 첫 인자가 '-'로 시작하는 플래그(예: --mode)면 기본 config 를 쓰고 인자는 그대로 넘긴다.
if [ $# -ge 1 ] && [ -f "$1" ]; then
    CONFIG="$1"; shift
else
    CONFIG="$DEFAULT_CONFIG"
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config 파일 없음: $CONFIG" >&2
    exit 1
fi

# ── GPU 노드 확인 ─────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi가 없습니다. 로그인 노드가 아니라 GPU 컴퓨트 노드에서 실행하세요." >&2
    echo "       예)  srun --gres=gpu:RTX6000:1 --pty bash  후 다시 실행" >&2
    exit 1
fi

# ── 모듈 / CUDA (있으면 로드) ─────────────────────────────────────────────────
module load cuda/13.0.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/false)")")}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# ── venv 활성화 ───────────────────────────────────────────────────────────────
source "$REPO_DIR/.venv/bin/activate"

# ── Protenix import 경로 (repo root는 esm/ 그림자 문제로 제외) ────────────────
export PYTHONPATH=$REPO_DIR/Protenix:${PYTHONPATH:-}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

echo "============================================================"
echo " MSAFlow fold benchmark — config: $CONFIG"
echo " Node   : $(hostname)"
echo " Python : $(python --version 2>&1)"
echo " Started: $(date)"
echo "============================================================"

mkdir -p "$REPO_DIR/runs"
python "$REPO_DIR/msaflow/inference/fold_benchmark.py" --config "$CONFIG" "$@" \
    2>&1 | tee "$REPO_DIR/runs/fold_benchmark_last.log"
# tee 가 파이프 종료코드를 가리지 않도록 python 의 실제 exit code 를 보존해 전파
# (sbatch 잡 상태가 FAILED/COMPLETED 로 정확히 찍히게)
status=${PIPESTATUS[0]}

echo "============================================================"
echo " Done: $(date)  (exit=$status)"
echo "============================================================"
exit "$status"
