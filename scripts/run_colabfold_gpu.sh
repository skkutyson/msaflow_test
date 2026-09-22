#!/bin/bash
# ==============================================================================
# run_colabfold_gpu.sh  —  colabfold_batch 얇은 GPU 런처 (uv run 기반, sbatch 불필요)
#
# 설정(입력 a3m/출력/gpu_id/모델 옵션)은 YAML 로 제어. 이 스크립트는 YAML 을 읽어
# colabfold_batch CLI 로 변환하고, 지정 GPU 하나에 CUDA_VISIBLE_DEVICES 로 묶어
# `uv run colabfold_batch <input> <output>` 를 실행한다.
#
# 사용법 (GPU 가 보이는 노드에서 직접):
#     bash scripts/run_colabfold_gpu.sh [config.yaml] [--target T] [-- <colabfold flags>]
#
#   config 미지정 시 msaflow/configs/colabfold_batch.yaml 사용.
#   '--target T'  : YAML 의 run.target 을 덮어씀.
#   '-- <flags>'  : '--' 뒤 인자는 colabfold_batch 에 그대로 전달(YAML extra_flags 뒤에 추가).
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
DEFAULT_CONFIG=$REPO_DIR/msaflow/configs/colabfold_batch.yaml

# ── 인자 파싱: config / --target / -- passthrough ────────────────────────────
CONFIG=$DEFAULT_CONFIG
TARGET_OVERRIDE=""
PASSTHROUGH=()
if [ $# -ge 1 ] && [ -f "$1" ]; then CONFIG="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET_OVERRIDE="$2"; shift 2 ;;
        --) shift; PASSTHROUGH=("$@"); break ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

[ -f "$CONFIG" ] || { echo "ERROR: config 없음: $CONFIG" >&2; exit 1; }

# ── GPU 노드 확인 ─────────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi 없음. GPU 가 보이는 노드에서 실행하세요." >&2; exit 1; }

cd "$REPO_DIR"

# ── YAML → 쉘 변수 (프로젝트 파이썬+pyyaml 사용) ──────────────────────────────
CFG_DUMP=$(uv run python - "$CONFIG" "$TARGET_OVERRIDE" <<'PY'
import sys, shlex, yaml, os
cfg_path, target_override = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(cfg_path))
run, cf = c.get("run", {}), c.get("colabfold", {})
target = target_override or run.get("target")
inp = run.get("input") or os.path.join(run["a3m_dir"], f"{target}.a3m")
out = os.path.join(run["output_dir"], target) if target else run["output_dir"]
args = []
if cf.get("model_type"):  args += ["--model-type", str(cf["model_type"])]
if cf.get("num_models") is not None:  args += ["--num-models", str(cf["num_models"])]
if cf.get("num_recycle") is not None: args += ["--num-recycle", str(cf["num_recycle"])]
args += [str(x) for x in (cf.get("extra_flags") or [])]
print(f"GPU_ID={shlex.quote(str(run.get('gpu_id', 0)))}")
print(f"INPUT={shlex.quote(inp)}")
print(f"OUTPUT={shlex.quote(out)}")
print(f"CF_ARGS={shlex.quote(' '.join(shlex.quote(a) for a in args))}")
PY
)
eval "$CFG_DUMP"

[ -e "$INPUT" ] || echo "WARN: 입력이 아직 없음(경로만 확인): $INPUT" >&2
mkdir -p "$OUTPUT"

echo "============================================================"
echo " colabfold_batch — config: $CONFIG"
echo " Node   : $(hostname)   GPU: $GPU_ID"
echo " Input  : $INPUT"
echo " Output : $OUTPUT"
echo " Extra  : $CF_ARGS ${PASSTHROUGH[*]:-}"
echo " Started: $(date)"
echo "============================================================"

# CF_ARGS / PASSTHROUGH 는 의도적으로 분할(word-split)되어 colabfold_batch 로 전달.
CUDA_VISIBLE_DEVICES="$GPU_ID" \
    uv run colabfold_batch $CF_ARGS "${PASSTHROUGH[@]}" "$INPUT" "$OUTPUT"

echo "============================================================"
echo " Done: $(date)  →  $OUTPUT"
echo "============================================================"
