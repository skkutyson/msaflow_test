#!/bin/bash
# ==============================================================================
# run_msa_gen.sh  —  zero-shot MSA(a3m) 대량 생성 런처 (폴딩 없음)
#
# 쿼리 서열 하나당 합성 MSA 를 generate.py 의 zero-shot 경로로 뽑아 타겟별로
# {OUTPUT_DIR}/{target}.a3m 하나씩 저장한다. 폴딩/OST 는 전혀 하지 않는다 —
# 다양성 분석 등 downstream 용 a3m 풀을 만드는 용도.
#
#   총 서열 = N_SEEDS × N_SEQS   (seed 당 a3m 깊이 N_SEQS, 임베딩(seed) N_SEEDS 개)
#   기본 32 × 32 = 1024 seqs/target.
#
# 헤더에 seed/idx 를 태깅(>{target}_s{seed}_n{idx})해 한 파일에 모으므로, 나중에
# within-seed(디코더) vs between-seed(latent) 다양성으로 분해해 볼 수 있다.
#
# 사용법 (GPU 가 보이는 컴퓨트 노드에서):
#     bash scripts/run_msa_gen.sh [config.yaml]
#
#   config.yaml 미지정 시 msaflow/configs/fold_benchmark.yaml 사용
#   (여기서 decoder/latent_fm 체크포인트, shallow_msa_dir, targets.csv 를 읽음).
#
#   파라미터(전부 env 로 덮어쓰기):
#     OUTPUT_DIR  출력 폴더               (기본 /store/yjlee4/runs/fold_benchmark/msa_gen)
#     N_SEEDS     임베딩(=a3m 깊이 블록) 수 (기본 32)
#     N_SEQS      seed 당 서열 수          (기본 32)
#     N_STEPS     ODE 스텝                 (기본 100)
#     TEMPERATURE 디코딩 온도              (기본 0.5)
#     ALPHA       α-geometry override      (기본: 체크포인트 값)
#     REGEN       1 이면 기존 a3m 도 재생성 (기본 0 = skip)
#     GPUS        샤딩할 GPU 목록          (기본 "0";  예: GPUS="0 1 2 3")
#     NUM_SHARDS  샤드 수                  (기본 = GPU 개수)
#     DECODER_CKPT / LATENT_FM_CKPT / SHALLOW_MSA_DIR / TARGETS_CSV  (기본: config)
#
#   예) 4-GPU 로 전체 타겟:
#       GPUS="0 1 2 3" bash scripts/run_msa_gen.sh
# ==============================================================================
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
DEFAULT_CONFIG=$REPO_DIR/msaflow/configs/fold_benchmark.yaml
if [ $# -ge 1 ] && [ -f "$1" ]; then CONFIG="$1"; shift; else CONFIG="$DEFAULT_CONFIG"; fi
[ -f "$CONFIG" ] || { echo "ERROR: config 없음: $CONFIG" >&2; exit 1; }

# ── 파라미터 (env 오버라이드) ─────────────────────────────────────────────────
OUTPUT_DIR=${OUTPUT_DIR:-/store/yjlee4/runs/fold_benchmark/msa_gen}
N_SEEDS=${N_SEEDS:-32}
N_SEQS=${N_SEQS:-32}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}
ALPHA=${ALPHA:-}
REGEN=${REGEN:-0}
GPUS=${GPUS:-"0"}
read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS=${NUM_SHARDS:-${#GPU_ARR[@]}}

# ── GPU 노드 확인 ─────────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi 없음. GPU 컴퓨트 노드에서 실행하세요." >&2
    echo "       예)  srun --gres=gpu:RTX6000:4 --cpus-per-task=16 --pty bash" >&2
    exit 1; }

# ── 모듈 / CUDA / venv (run_fold_benchmark_4gpu.sh 와 동일) ───────────────────
module load cuda/13.0.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/false)")")}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
source "$REPO_DIR/.venv/bin/activate"
export PYTHONPATH=$REPO_DIR:$REPO_DIR/Protenix:${PYTHONPATH:-}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

LOG_DIR=${LOG_DIR:-/store/yjlee4/runs}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "============================================================"
echo " MSAFlow zero-shot MSA 생성 (폴딩 없음) — config: $CONFIG"
echo " Node    : $(hostname)"
echo " GPUs    : ${GPU_ARR[*]}   (num_shards=$NUM_SHARDS)"
echo " Output  : $OUTPUT_DIR"
echo " Per tgt : N_SEEDS=$N_SEEDS × N_SEQS=$N_SEQS = $((N_SEEDS * N_SEQS)) seqs"
echo " Steps   : $N_STEPS   Temp: $TEMPERATURE   Regen: $REGEN"
echo " Started : $(date)"
echo "============================================================"

# ── 샤드 병렬 실행: GPU 당 프로세스 1개 ───────────────────────────────────────
# 각 프로세스는 자기 타겟(targets[shard_id::num_shards])만 생성해 독립 파일에 쓰므로
# 병합이 필요 없다(폴딩 벤치마크와 달리 공유 CSV 경쟁 없음).
declare -a PIDS
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    shard_log="$LOG_DIR/msa_gen_shard${i}.log"
    echo "  shard $i/$NUM_SHARDS → GPU $gpu   (log: $shard_log)"
    CUDA_VISIBLE_DEVICES="$gpu" python - \
        "$CONFIG" "$OUTPUT_DIR" "$N_SEEDS" "$N_SEQS" "$N_STEPS" "$TEMPERATURE" \
        "$i" "$NUM_SHARDS" "$REGEN" "$ALPHA" > "$shard_log" 2>&1 <<'PY' &
import csv, os, sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

(cfg_path, out_dir, n_seeds, n_seqs, n_steps, temperature,
 shard_id, num_shards, regen, alpha) = sys.argv[1:11]
n_seeds, n_seqs, n_steps = int(n_seeds), int(n_seqs), int(n_steps)
temperature = float(temperature)
shard_id, num_shards = int(shard_id), int(num_shards)
regen = regen == "1"
alpha = float(alpha) if alpha else None

cfg = OmegaConf.load(cfg_path)
def pick(env, key, default=None):
    return os.environ.get(env) or OmegaConf.select(cfg, key) or default

decoder_ckpt   = pick("DECODER_CKPT",   "model.decoder_ckpt")
latent_fm_ckpt = pick("LATENT_FM_CKPT", "model.latent_fm_ckpt")
shallow_dir    = pick("SHALLOW_MSA_DIR", "run.shallow_msa_dir")
targets_csv    = pick("TARGETS_CSV",     "targets.csv")
for label, val in [("decoder_ckpt", decoder_ckpt), ("latent_fm_ckpt", latent_fm_ckpt),
                   ("shallow_msa_dir", shallow_dir), ("targets.csv", targets_csv)]:
    if not val:
        sys.exit(f"ERROR: {label} 미지정 (config 또는 env 로 지정하세요).")

# ── 타겟명 (pdb_id__chain), 순서보존 dedup, 샤딩 ──────────────────────────────
names, seen = [], set()
with open(targets_csv) as fh:
    reader = csv.reader(fh)
    next(reader, None)  # header: pdb_id,chain_id
    for row in reader:
        if len(row) < 2 or not row[0].strip():
            continue
        name = f"{row[0].strip()}__{row[1].strip()}"
        if name not in seen:
            seen.add(name)
            names.append(name)
names = names[shard_id::num_shards]
print(f"[shard {shard_id}/{num_shards}] {len(names)} targets", flush=True)

# ── a3m 첫 레코드 = 쿼리 (lowercase=insertion, '-'/'.'=gap 제거) ──────────────
def read_query(path):
    with open(path) as fh:
        header = fh.readline()
        if not header.startswith(">"):
            raise ValueError(f"a3m 형식 아님: {path}")
        seq_lines = []
        for line in fh:
            if line.startswith(">"):
                break
            seq_lines.append(line.strip())
    raw = "".join(seq_lines)
    canon = "".join(c for c in raw if not c.islower() and c not in "-.")
    if not canon:
        raise ValueError(f"a3m query 비어있음: {path}")
    return canon

from msaflow.inference.generate import (
    load_sfm_decoder, load_latent_fm, load_esm2, generate_zeroshot_seeds,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder   = load_sfm_decoder(decoder_ckpt, device, alpha=alpha)
latent_fm = load_latent_fm(latent_fm_ckpt, device)
esm_model, alphabet = load_esm2(device)

out_root = Path(out_dir)
out_root.mkdir(parents=True, exist_ok=True)

n_done = n_skip = n_miss = 0
for idx, name in enumerate(names):
    out_path = out_root / f"{name}.a3m"
    if out_path.exists() and not regen:
        n_skip += 1
        continue
    a3m_in = Path(shallow_dir) / f"{name}.a3m"
    if not a3m_in.exists():
        print(f"  [skip] 입력 a3m 없음: {a3m_in}", flush=True)
        n_miss += 1
        continue
    query = read_query(str(a3m_in))
    print(f"  [{idx+1}/{len(names)}] {name}  L={len(query)}  "
          f"→ {n_seeds}×{n_seqs} seqs", flush=True)

    # 임시 파일에 쓰고 원자적 rename (중단/resume 안전)
    tmp_path = out_path.with_suffix(".a3m.tmp")
    total = 0
    with open(tmp_path, "w") as fh:
        fh.write(f">{name}\n{query}\n")
        for seed, seqs in generate_zeroshot_seeds(
            query_seq=query, decoder=decoder, latent_fm=latent_fm,
            esm_model=esm_model, alphabet=alphabet,
            n_seeds=n_seeds, n_seqs_per_seed=n_seqs,
            n_steps=n_steps, temperature=temperature, device=device,
        ):
            for j, s in enumerate(seqs):
                fh.write(f">{name}_s{seed:03d}_n{j:03d}\n{s}\n")
                total += 1
    os.replace(tmp_path, out_path)
    print(f"      wrote {total} seqs → {out_path}", flush=True)
    n_done += 1

print(f"[shard {shard_id}] done: {n_done} generated, {n_skip} skipped, "
      f"{n_miss} missing-input", flush=True)
PY
    PIDS[$i]=$!
done

# ── 대기 + 샤드별 종료코드 집계 ──────────────────────────────────────────────
rc=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if wait "${PIDS[$i]}"; then
        echo "  shard $i 완료 (OK)"
    else
        echo "  shard $i 실패 (exit≠0) — 로그: $LOG_DIR/msa_gen_shard${i}.log" >&2
        rc=1
    fi
done

echo "============================================================"
echo " Done: $(date)  (rc=$rc)"
echo " 출력: $OUTPUT_DIR/*.a3m"
echo "============================================================"
exit "$rc"
