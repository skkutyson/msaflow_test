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
DECODER_CKPT=${DECODER_CKPT:-/home/yjlee4/MSA_FLOW/runs/decoder/latest.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-/home/yjlee4/MSA_FLOW/runs/latent_fm/latest.pt}
SHALLOW_MSA_DIR=${SHALLOW_MSA_DIR:-/store/deepfold3/results/msa/a3m}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/fold_benchmark_fewshot_gpu}

# ── 타겟 (기본: monomer 앞 3개) ───────────────────────────────────────────────
TARGETS=${TARGETS:-"8ux2-assembly1__A 8oxk-assembly1__A 8eb9-assembly1__A"}

# ── 실행 파라미터 ─────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-0}
N_SEQS=${N_SEQS:-32}          # sequences per synthetic seed (augmentation)
N_SEEDS=${N_SEEDS:-5}         # synthetic seeds for augmentation (n_syn_seeds)
N_STEPS=${N_STEPS:-100}       # ODE steps
TEMPERATURE=${TEMPERATURE:-0.5}

# ── Protenix 폴딩 샘플링: N_FOLD_SEEDS seed × N_SAMPLE sample (기본 5×5=25) ────
N_FOLD_SEEDS=${N_FOLD_SEEDS:-5}   # Protenix 폴딩 seed 개수 (증강 N_SEEDS 와 별개)
FOLD_SEED_BASE=${FOLD_SEED_BASE:-101}
N_SAMPLE=${N_SAMPLE:-5}           # seed 당 diffusion 샘플 수
FOLD_SEEDS=$(seq -s, "$FOLD_SEED_BASE" $((FOLD_SEED_BASE + N_FOLD_SEEDS - 1)))
PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
PROTENIX_CKPT=${PROTENIX_CKPT:-/home/yjlee4/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}   # 있으면 Rec 트랙 활성화
TMSCORE_BIN=${TMSCORE_BIN:-TMscore}
REF_PDB_DIR=${REF_PDB_DIR:-}        # 참조 PDB 있으면 fold_benchmark 자체 TM-score 계산

# ── OST 채점 (folding 이후 GT 비교) ───────────────────────────────────────────
RUN_OST=${RUN_OST:-1}              # 0 이면 OST 채점 건너뜀
GT_DIR=${GT_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
CHAIN_CSV=${CHAIN_CSV:-$REPO_DIR/monomer_protein.csv}
OST_CSV=${OST_CSV:-$OUTPUT_DIR/monomer_protein_ost.csv}
# 스코어러 선택: ost_score.py(기본, gemmi 로 chain 추출) 또는
#   ost_score_chainmap.py(gemmi 미사용, compare-structures -c A:A 로 chain 매핑)
OST_SCORER=${OST_SCORER:-ost_score.py}
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

# ── 3개 타겟 subset fasta 생성 — query 서열을 각 a3m 첫 서열에서 직접 추출 ────
# (원본 foldbench_monomer.fasta 불필요. a3m 첫 record = query 이므로 그걸로 --fasta 구성)
mkdir -p "$OUTPUT_DIR"
FASTA_SUBSET="$OUTPUT_DIR/targets_subset.fasta"
python - "$FASTA_SUBSET" "$SHALLOW_MSA_DIR" "$TARGETS" << 'PYEOF'
import sys, os
dst, msa_dir, targets_str = sys.argv[1], sys.argv[2], sys.argv[3]
wanted = targets_str.split()

def read_a3m_query(path):
    """a3m 첫 record 의 서열 = query. 삽입(소문자)·gap 제거해 정규 서열 반환."""
    seq = []
    with open(path) as fh:
        first = fh.readline()
        if not first.startswith(">"):
            sys.exit(f"ERROR: a3m 형식 아님(첫 줄 '>' 아님): {path}")
        for line in fh:
            if line.startswith(">"):
                break
            seq.append(line.strip())
    q = "".join(seq).replace("-", "").replace(".", "").upper()
    if not q:
        sys.exit(f"ERROR: a3m query 서열 비어있음: {path}")
    return q

no_a3m = [t for t in wanted if not os.path.exists(os.path.join(msa_dir, f"{t}.a3m"))]
if no_a3m:
    sys.exit(f"ERROR: 입력 a3m 없음 ({msa_dir}): {no_a3m}")

with open(dst, "w") as out:
    for t in wanted:
        q = read_a3m_query(os.path.join(msa_dir, f"{t}.a3m"))
        out.write(f">{t}\n{q}\n")
        print(f"  {t}: query L={len(q)} (from a3m)")
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
echo " Fold sampling : ${N_FOLD_SEEDS} seeds ($FOLD_SEEDS) x ${N_SAMPLE} samples = $((N_FOLD_SEEDS * N_SAMPLE))/target"
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
    --fold_seeds      "$FOLD_SEEDS" \
    --n_sample        "$N_SAMPLE" \
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
        python "$REPO_DIR/scripts/$OST_SCORER" \
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
