#!/bin/bash
# Full ShardStore build from an OpenProteinSet-style a3m tree — native uv,
# no apptainer / no SLURM (same style as run_overfit.sh).
#
# Point it at the a3m root dir. Handles both layouts:
#   {UNIPROT_ID}/a3m/uniclust30.a3m   (OpenProteinSet; key = UNIPROT_ID)
#   {pdb}__{chain}.a3m                (flat; key = file stem)
#
# Usage (a3m dir as the 1st argument OR the A3M_DIR env var):
#   bash scripts/run_preprocess.sh /path/to/openfold/uniclust30
#   A3M_DIR=/path/to/uniclust30 bash scripts/run_preprocess.sh
#
# A full OpenProteinSet run takes days — run it inside tmux, or:
#   nohup bash scripts/run_preprocess.sh /path/to/uniclust30 \
#       > logs/preprocess_native.log 2>&1 &
# RESUME=1 (default) means you can kill / re-run any time and it continues.
#
# Common overrides (env vars):
#   A3M_DIR        (required, or pass as arg 1) a3m root directory
#   OUTPUT_STORE   output store directory      (default: $HOME/data/msaflow_openfold)
#   A3M_NAME       exact a3m filename to use   (default: uniclust30.a3m; ""=every .a3m)
#   A3M_LIST       txt with one a3m path/line  (skips the slow dir scan; the scan
#                  result is auto-cached at <OUTPUT_STORE>.a3m_list.txt anyway)
#   RESUME         1=skip ids already in store (default: 1)
#   PROTEIN_LIST   optional txt, one ID per line, to restrict targets
#   PROTENIX_CKPT  Protenix checkpoint .pt     (default: <repo>/checkpoints/...)
#   CCD_DIR        dir holding components.cif  (default: $HOME/common)
#   N_CYCLE        Protenix recycling cycles   (default: 3)
#   GPU            CUDA device index           (default: 0)
#   MAX_MSA_SEQS   MSA depth cap               (default: 512)
#   MAX_SEQ_LEN    sequence length cap         (default: 1024)
#   SHARD_MAX_GB   shard size cap in GB        (default: 4)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── config ────────────────────────────────────────────────────────────────────
A3M_DIR="${1:-${A3M_DIR:-}}"
if [ -z "$A3M_DIR" ]; then
    echo "ERROR: no a3m directory given."
    echo "  usage: bash scripts/run_preprocess.sh /path/to/openfold/uniclust30"
    echo "     or: A3M_DIR=/path/to/uniclust30 bash scripts/run_preprocess.sh"
    exit 1
fi
OUTPUT_STORE="${OUTPUT_STORE:-$HOME/data/msaflow_openfold}"
A3M_NAME="${A3M_NAME-uniclust30.a3m}"
A3M_LIST="${A3M_LIST:-}"
RESUME="${RESUME:-1}"
PROTEIN_LIST="${PROTEIN_LIST:-}"
PROTENIX_CKPT="${PROTENIX_CKPT:-$REPO_DIR/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}"
CCD_DIR="${CCD_DIR:-$HOME/common}"
N_CYCLE="${N_CYCLE:-3}"
GPU="${GPU:-0}"
MAX_MSA_SEQS="${MAX_MSA_SEQS:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
SHARD_MAX_GB="${SHARD_MAX_GB:-4}"

export PROTENIX_ROOT_DIR="$(dirname "$CCD_DIR")"  # Protenix reads $PROTENIX_ROOT_DIR/common
export CUDA_VISIBLE_DEVICES="$GPU"

# ── checks ────────────────────────────────────────────────────────────────────
A3M_DIR="$(readlink -f "$A3M_DIR")"
[ -d "$A3M_DIR" ] || { echo "ERROR: a3m dir not found: $A3M_DIR"; exit 1; }
if [ -z "$(find -L "$A3M_DIR" -name '*.a3m' -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no .a3m files under $A3M_DIR"; exit 1
fi
[ -f "$PROTENIX_CKPT" ] || { echo "ERROR: checkpoint not found: $PROTENIX_CKPT"; exit 1; }
for f in components.cif components.cif.rdkit_mol.pkl; do
    [ -f "$CCD_DIR/$f" ] || { echo "ERROR: missing CCD file $CCD_DIR/$f"; \
        echo "  download: wget -c https://protenix.tos-cn-beijing.volces.com/common/$f -P $CCD_DIR"; exit 1; }
done
[ -z "$PROTEIN_LIST" ] || [ -f "$PROTEIN_LIST" ] || { echo "ERROR: protein list not found: $PROTEIN_LIST"; exit 1; }
[ -z "$A3M_LIST" ] || [ -f "$A3M_LIST" ] || { echo "ERROR: a3m list not found: $A3M_LIST"; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not found (run scripts/setup_native_uv.sh first)"; exit 1; }

mkdir -p "$(dirname "$OUTPUT_STORE")"

echo "============================================================"
echo " MSAFlow ShardStore build (native uv, no SLURM)"
echo "   a3m_dir : $A3M_DIR"
echo "   a3m_name: ${A3M_NAME:-<all *.a3m>}   resume: $RESUME"
echo "   store   : $OUTPUT_STORE"
echo "   ckpt    : $PROTENIX_CKPT"
echo "   n_cycle : $N_CYCLE   gpu: $GPU   ccd: $CCD_DIR"
echo "   max_msa : $MAX_MSA_SEQS   max_len: $MAX_SEQ_LEN   shard: ${SHARD_MAX_GB}GB"
echo "   list    : ${PROTEIN_LIST:-<none>}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m msaflow.data.preprocessing \
    --a3m_dir "$A3M_DIR" \
    --output "$OUTPUT_STORE" \
    --protenix_checkpoint "$PROTENIX_CKPT" \
    --max_msa_seqs "$MAX_MSA_SEQS" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --device cuda \
    --shard_max_gb "$SHARD_MAX_GB" \
    --n_cycle "$N_CYCLE" \
    ${A3M_NAME:+--a3m_name "$A3M_NAME"} \
    ${A3M_LIST:+--a3m_list "$A3M_LIST"} \
    ${PROTEIN_LIST:+--protein_list "$PROTEIN_LIST"} \
    $([ "$RESUME" = "1" ] && echo --resume)

echo "== done: $OUTPUT_STORE =="
