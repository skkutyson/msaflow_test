#!/bin/bash
# ==============================================================================
# run_fold_benchmark_4gpu.sh  —  4-GPU 데이터 병렬 fold benchmark 런처
#
# run_fold_benchmark_gpu.sh 의 멀티-GPU 버전. 타겟 목록을 num_shards 로 쪼개
# (proteins[shard_id::num_shards]) GPU 하나당 프로세스 하나로 "따로" 돌린 뒤,
# 각 샤드의 OST CSV 를 하나의 monomer_protein_ost.csv 로 합친다.
#
# train_decoder.sh 와 달리 accelerate(DDP) 가 아니다: 추론은 동기화할 gradient 가
# 없으므로 완전히 독립적인 4개 프로세스로 data-parallel 샤딩하는 게 맞다.
# fold_benchmark.py 는 이미 --shard_id/--num_shards 를 지원하고, 미리 잡힌
# CUDA_VISIBLE_DEVICES 를 존중하며(protenix 서브프로세스까지 상속), 샤드별로 자기
# 타겟만 OST 채점한다. 이 스크립트는 그 위에 (1) 공유 출력 디렉토리에서의 경쟁
# 방지, (2) 결과 병합만 얹는다.
#
# 사용법 (반드시 GPU 4장이 보이는 컴퓨트 노드에서):
#     bash scripts/run_fold_benchmark_4gpu.sh [config.yaml] [--overrides ...]
#
#   config.yaml 미지정 시 msaflow/configs/fold_benchmark.yaml 사용.
#   GPU/샤드 수 조정:  GPUS="0 1 2 3" NUM_SHARDS=4 bash scripts/run_fold_benchmark_4gpu.sh
#   특정 GPU만:        GPUS="1 3" bash scripts/run_fold_benchmark_4gpu.sh   # NUM_SHARDS 자동=2
#
# OST 채점은 config 의 ost.enable=true + ost.gt_dir 이 있어야 돈다(단일 GPU 와 동일).
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
DEFAULT_CONFIG=$REPO_DIR/msaflow/configs/fold_benchmark.yaml
if [ $# -ge 1 ] && [ -f "$1" ]; then
    CONFIG="$1"; shift
else
    CONFIG="$DEFAULT_CONFIG"
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config 파일 없음: $CONFIG" >&2
    exit 1
fi

# ── GPU / 샤드 수 ─────────────────────────────────────────────────────────────
GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS=${NUM_SHARDS:-${#GPU_ARR[@]}}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi가 없습니다. GPU 컴퓨트 노드에서 실행하세요." >&2
    echo "       예)  srun --gres=gpu:RTX6000:4 --cpus-per-task=16 --pty bash" >&2
    exit 1
fi

# ── 모듈 / CUDA / venv (run_fold_benchmark_gpu.sh 와 동일) ────────────────────
module load cuda/13.0.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/false)")")}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
source "$REPO_DIR/.venv/bin/activate"
export PYTHONPATH=$REPO_DIR/Protenix:${PYTHONPATH:-}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

LOG_DIR=${LOG_DIR:-/store/yjlee4/runs}
mkdir -p "$LOG_DIR"
BENCH=$REPO_DIR/msaflow/inference/fold_benchmark.py

echo "============================================================"
echo " MSAFlow fold benchmark (4-GPU 샤딩) — config: $CONFIG"
echo " Node       : $(hostname)"
echo " GPUs       : ${GPU_ARR[*]}   (num_shards=$NUM_SHARDS)"
echo " Python     : $(python --version 2>&1)"
echo " Started    : $(date)"
echo "============================================================"

# ── 유효 출력 경로/모드 해석 (main() 의 mode/output_dir pick 규칙을 재현) ──────
# CLI 오버라이드(--output_dir/--mode/--no_msa)를 존중하고, 없으면 config 에서 읽어
# {output_dir}/{mode} = 실제 산출물 디렉토리를 구한다.
mapfile -t _RESOLVED < <(python - "$CONFIG" "$@" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1]) if sys.argv[1] else OmegaConf.create({})
argv = sys.argv[2:]
def cli(flag):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None
mode = "nomsa" if "--no_msa" in argv else (cli("--mode") or OmegaConf.select(cfg, "run.mode") or "zeroshot")
out = cli("--output_dir") or OmegaConf.select(cfg, "run.output_dir")
if not out:
    sys.exit("ERROR: output_dir 미지정 (--output_dir 또는 config run.output_dir).")
print(str(Path(out) / mode))
print(mode)
PY
)
OUT_MODE_DIR="${_RESOLVED[0]}"
MODE="${_RESOLVED[1]}"
FASTA="$OUT_MODE_DIR/targets_subset.fasta"
echo " 산출물 디렉토리: $OUT_MODE_DIR   (mode=$MODE)"

# ── 0) 사전 빌드: subset FASTA + checkpoint 심링크 (GPU/폴딩 없음) ────────────
# 모든 샤드가 targets_subset.fasta 를 동시에 쓰면 경쟁이 발생하므로 한 번만 만들어
# 둔다. --score_only --skip_ost 는 폴딩/채점 없이 fasta 와 심링크만 준비한다.
echo "[prebuild] subset FASTA / checkpoint 준비 중..."
python "$BENCH" --config "$CONFIG" "$@" --score_only --skip_ost --num_shards 1 \
    > "$LOG_DIR/fold_benchmark_prebuild.log" 2>&1 || {
    echo "ERROR: prebuild 실패 — 로그: $LOG_DIR/fold_benchmark_prebuild.log" >&2
    tail -30 "$LOG_DIR/fold_benchmark_prebuild.log" >&2 || true
    exit 1
}
if [ ! -f "$FASTA" ]; then
    echo "ERROR: prebuild 후에도 FASTA 없음: $FASTA" >&2
    exit 1
fi
echo "[prebuild] FASTA 준비 완료: $FASTA ($(grep -c '^>' "$FASTA") targets)"

# ── 1) 샤드 병렬 실행: GPU 당 프로세스 1개 ────────────────────────────────────
declare -a PIDS
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    shard_log="$LOG_DIR/fold_benchmark_shard${i}.log"
    echo "  shard $i/$NUM_SHARDS → GPU $gpu   (log: $shard_log)"
    CUDA_VISIBLE_DEVICES="$gpu" python "$BENCH" --config "$CONFIG" "$@" \
        --fasta "$FASTA" \
        --shard_id "$i" --num_shards "$NUM_SHARDS" \
        --ost_csv "$OUT_MODE_DIR/ost_shard${i}.csv" \
        > "$shard_log" 2>&1 &
    PIDS[$i]=$!
done

# ── 2) 대기 + 샤드별 종료코드 집계 ────────────────────────────────────────────
rc=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if wait "${PIDS[$i]}"; then
        echo "  shard $i 완료 (OK)"
    else
        echo "  shard $i 실패 (exit≠0) — 로그: $LOG_DIR/fold_benchmark_shard${i}.log" >&2
        rc=1
    fi
done

# ── 3) 결과 병합 ──────────────────────────────────────────────────────────────
# 샤드별 ost_shard{i}.csv → monomer_protein_ost.csv (헤더 1회).
# 샤드별 내부 요약 shard_shard{i}.csv → summary_all.csv.
echo "[merge] 샤드 CSV 병합 중..."
python - "$OUT_MODE_DIR" "$NUM_SHARDS" <<'PY'
import sys, csv, glob, os
out_mode_dir, n = sys.argv[1], int(sys.argv[2])

def concat(paths, final, label):
    header, rows = None, []
    for p in paths:
        with open(p, newline="") as fh:
            r = csv.reader(fh)
            h = next(r, None)
            if header is None:
                header = h
            rows.extend(row for row in r)
    if header is None:
        print(f"[merge] {label}: 병합할 파일 없음")
        return
    with open(final, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[merge] {label}: {len(rows)} rows ({len(paths)} shards) -> {final}")

ost = [os.path.join(out_mode_dir, f"ost_shard{i}.csv") for i in range(n)]
ost = [p for p in ost if os.path.exists(p)]
concat(ost, os.path.join(out_mode_dir, "monomer_protein_ost.csv"), "OST")

summ = sorted(glob.glob(os.path.join(out_mode_dir, "shard_shard*.csv")))
concat(summ, os.path.join(out_mode_dir, "summary_all.csv"), "summary")
PY

echo "============================================================"
echo " Done: $(date)  (rc=$rc)"
echo " 최종 OST CSV: $OUT_MODE_DIR/monomer_protein_ost.csv"
echo "============================================================"
exit "$rc"
