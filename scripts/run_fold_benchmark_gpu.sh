#!/bin/bash
# ==============================================================================
# run_fold_benchmark_gpu.sh
#
# End-to-end 원트랙:  입력 a3m → MSA augmentation → 폴딩 → OST 로 GT 비교 채점
# foldbench monomer 타겟 3개 대상.  sbatch 아님 — GPU 노드에서 바로 실행.
#
# 파이프라인:
#   1) $SHALLOW_MSA_DIR/{타겟}.a3m 로드                            (fold_benchmark --mode fewshot)
#   2) augment_shallow(): decoder+latent_fm 로 합성/다양성 시퀀스 생성 후 원본에 concat
#      → {OUTPUT}/msas/{타겟}/augmented.a3m
#   3) 그 augmented.a3m 으로 Protenix(AF3) 폴딩 → CIF
#   4) OST compare-structures 로 GT 와 비교 → monomer_protein_ost.csv (+ mode 컬럼)
#      컬럼: pdb_id,chain_id,seed,sample,ranking_score,prediction_path,
#            dockq_score,irmsd,lrmsd,len_dockq,lddt,tm_score,gdt_ts,rmsd,mode
#
# 사용법 (반드시 GPU 컴퓨트 노드에서 — 로그인 노드 'ada'엔 GPU/gpfs 없음):
#     srun --partition=normal --nodelist=ada-004 --gres=gpu:RTX6000:1 \
#          --cpus-per-task=8 --time=8:00:00 --pty bash
#     bash /home/yjlee4/MSA_FLOW/scripts/run_fold_benchmark_gpu.sh
#
# 옵션 (환경변수):
#     GPU_ID=0
#     TARGETS="7pv5-assembly1 7qp5-assembly1 7qsj-assembly1"   폴딩할 타겟(공백 구분)
#     SHALLOW_MSA_DIR=/home/yjlee4/a3m_output                  입력 a3m 디렉토리
#     N_SEQS=32  N_SEEDS=5  N_STEPS=100  TEMPERATURE=0.5
#     PROTENIX_CKPT=...   지정하면 Rec 트랙(Protenix MSA 임베딩) 추가; 없으면 Syn 트랙만
#     DECODER_CKPT=... LATENT_FM_CKPT=... OUTPUT_DIR=...
# ==============================================================================
set -euo pipefail

# ── 경로 ──────────────────────────────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-/gpfs/deepfold/users/yjlee4/latent_fm_foldbench/latent_fm_ema_final.pt}
SHALLOW_MSA_DIR=${SHALLOW_MSA_DIR:-/home/yjlee4/a3m_output}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/fold_benchmark_fewshot_gpu}

# ── 타겟 (기본: monomer 앞 3개) ───────────────────────────────────────────────
TARGETS=${TARGETS:-"7pv5-assembly1 7qp5-assembly1 7qsj-assembly1"}

# ── 실행 파라미터 ─────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-0}
N_SEQS=${N_SEQS:-32}          # sequences per synthetic seed
N_SEEDS=${N_SEEDS:-5}         # synthetic seeds (n_syn_seeds)
N_STEPS=${N_STEPS:-100}       # ODE steps
TEMPERATURE=${TEMPERATURE:-0.5}
PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
PROTENIX_CKPT=${PROTENIX_CKPT:-}   # 있으면 Rec 트랙 활성화
TMSCORE_BIN=${TMSCORE_BIN:-TMscore}
REF_PDB_DIR=${REF_PDB_DIR:-}       # 참조 PDB 있으면 fold_benchmark 자체 TM-score 계산

# ── OST 채점 (folding 이후 GT 비교) ───────────────────────────────────────────
RUN_OST=${RUN_OST:-1}              # 0 이면 OST 채점 건너뜀
GT_DIR=${GT_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
CHAIN_CSV=${CHAIN_CSV:-$REPO_DIR/monomer_protein.csv}
OST_CSV=${OST_CSV:-$OUTPUT_DIR/monomer_protein_ost.csv}
# OST 실행: OST_SIF=/path/ost.sif  또는  OST_CMD='apptainer exec /path/ost.sif ost compare-structures'
# (없으면 ost_score.py 가 PATH 의 ost/compare-structures 자동 탐지)

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

# ── Protenix 체크포인트 경로 보정 ─────────────────────────────────────────────
# Protenix는 $PROTENIX_ROOT_DIR/checkpoint/<model>.pt 를 찾는다.
# 실제 가중치는 $REPO_DIR/checkpoints/checkpoints/ 이므로 없으면 심링크 (재다운로드 방지)
CKPT_EXPECTED="$PROTENIX_ROOT_DIR/checkpoint/${PROTENIX_MODEL}.pt"
CKPT_ACTUAL="$REPO_DIR/checkpoints/checkpoints/${PROTENIX_MODEL}.pt"
if [ ! -f "$CKPT_EXPECTED" ] && [ -f "$CKPT_ACTUAL" ]; then
    mkdir -p "$(dirname "$CKPT_EXPECTED")"
    ln -sf "$CKPT_ACTUAL" "$CKPT_EXPECTED"
    echo "Linked Protenix checkpoint: $CKPT_EXPECTED -> $CKPT_ACTUAL"
fi

# ── 3개 타겟만 담은 fasta 생성 (a3m query = fasta query 로 확인됨) ────────────
mkdir -p "$OUTPUT_DIR"
FASTA_SUBSET="$OUTPUT_DIR/targets_subset.fasta"
python - "$REPO_DIR/foldbench_monomer.fasta" "$FASTA_SUBSET" "$SHALLOW_MSA_DIR" "$TARGETS" << 'PYEOF'
import sys, os
src, dst, msa_dir, targets_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
records, name, seq = [], None, []
with open(src) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None: records.append((name, "".join(seq)))
            name, seq = line[1:].strip().split()[0], []
        elif line: seq.append(line)
    if name is not None: records.append((name, "".join(seq)))
by_name = {n: s for n, s in records}
wanted = targets_str.split()

missing = [t for t in wanted if t not in by_name]
if missing:
    sys.exit(f"ERROR: fasta에 없는 타겟: {missing}")
no_a3m = [t for t in wanted if not os.path.exists(os.path.join(msa_dir, f"{t}.a3m"))]
if no_a3m:
    sys.exit(f"ERROR: 입력 a3m 없음 ({msa_dir}): {no_a3m}")

with open(dst, "w") as out:
    for t in wanted:
        out.write(f">{t}\n{by_name[t]}\n")
print("Targets:", ", ".join(wanted))
PYEOF

# ── 로그 ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " MSAFlow few-shot (a3m 증강 → Protenix/AF3 fold) — 원트랙"
echo "============================================================"
echo " Node          : $(hostname)"
echo " GPU_ID        : $GPU_ID"
nvidia-smi -L | sed -n "$((GPU_ID+1))p" | sed 's/^/ GPU           : /'
echo " Python        : $(python --version 2>&1)"
echo " Decoder ckpt  : $DECODER_CKPT"
echo " Latent FM ckpt: $LATENT_FM_CKPT"
echo " Input a3m dir : $SHALLOW_MSA_DIR"
echo " Targets       : $TARGETS"
echo " Output dir    : $OUTPUT_DIR"
echo " Protenix model: $PROTENIX_MODEL"
echo " Rec track     : $([ -n "$PROTENIX_CKPT" ] && echo "ON ($PROTENIX_CKPT)" || echo "OFF (Syn track only)")"
echo " n_seqs=$N_SEQS  n_syn_seeds=$N_SEEDS  n_steps=$N_STEPS  temperature=$TEMPERATURE"
echo " Started       : $(date)"
echo "============================================================"

REF_ARG=""
[ -n "$REF_PDB_DIR" ]  && REF_ARG="--ref_pdb_dir $REF_PDB_DIR"
PROTENIX_CKPT_ARG=""
[ -n "$PROTENIX_CKPT" ] && PROTENIX_CKPT_ARG="--protenix_ckpt $PROTENIX_CKPT"

# ── 실행 (단일 GPU, fewshot = augment→fold 원트랙) ────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID \
python "$REPO_DIR/msaflow/inference/fold_benchmark.py" \
    --fasta           "$FASTA_SUBSET" \
    --decoder_ckpt    "$DECODER_CKPT" \
    --latent_fm_ckpt  "$LATENT_FM_CKPT" \
    --output_dir      "$OUTPUT_DIR" \
    --mode            fewshot \
    --shallow_msa_dir "$SHALLOW_MSA_DIR" \
    --device          cuda \
    --n_seqs          "$N_SEQS" \
    --n_seeds         "$N_SEEDS" \
    --n_steps         "$N_STEPS" \
    --temperature     "$TEMPERATURE" \
    --protenix_model  "$PROTENIX_MODEL" \
    --tmscore_bin     "$TMSCORE_BIN" \
    $PROTENIX_CKPT_ARG \
    $REF_ARG \
    2>&1 | tee "$OUTPUT_DIR/run.log"

# ── OST 채점: 예측 CIF vs GT → monomer_protein_ost.csv ────────────────────────
if [ "$RUN_OST" = "1" ]; then
    echo "=== OST 채점 (GT: $GT_DIR) ==="
    if [ ! -d "$GT_DIR" ]; then
        echo "WARNING: GT_DIR 없음: $GT_DIR — OST 채점 건너뜀. GT_DIR=... 로 지정하세요." >&2
    else
        python "$REPO_DIR/scripts/ost_score.py" \
            --pred_root "$OUTPUT_DIR/folds" \
            --gt_dir    "$GT_DIR" \
            --mode      fewshot \
            --targets   "$TARGETS" \
            --chain_csv "$CHAIN_CSV" \
            --out_csv   "$OST_CSV" \
            2>&1 | tee -a "$OUTPUT_DIR/run.log"
    fi
fi

echo "============================================================"
echo " Done: $(date)"
echo " 증강 a3m : $OUTPUT_DIR/msas/<타겟>/augmented.a3m"
echo " 폴딩 CIF : $OUTPUT_DIR/folds/<타겟>_fewshot/protenix_out/**/*_sample_*.cif"
echo " fold 요약: $OUTPUT_DIR/shard.csv"
echo " OST 채점 : $OST_CSV"
echo "============================================================"
