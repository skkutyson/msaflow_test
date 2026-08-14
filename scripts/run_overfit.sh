#!/bin/bash
# Single-target overfit sanity test — native uv, no apptainer / no SLURM.
#
# Point it at ANY .a3m file (any folder). It builds a 1-target LMDB with the
# faithful Protenix embedding, then overfits the SFM decoder on that one MSA.
# Loss should fall toward ~0 if the pipeline is correct.
#
# Usage (a3m path as the 1st argument OR the A3M_FILE env var):
#   bash scripts/run_overfit.sh /path/to/my_target.a3m
#   A3M_FILE=/path/to/my_target.a3m bash scripts/run_overfit.sh
#
# Common overrides (env vars):
#   A3M_FILE       (required, or pass as arg 1) path to the target .a3m
#   WORK_DIR       output root for LMDB + checkpoints   (default: <repo>/runs/overfit)
#   PROTENIX_CKPT  Protenix checkpoint .pt              (default: <repo>/checkpoints/...)
#   CCD_DIR        dir holding components.cif etc.      (default: $HOME/common)
#   N_CYCLE        Protenix recycling cycles            (default: 1)
#   GPU            CUDA device index                    (default: 0)
#   CONFIG         overfit yaml                         (default: decoder_overfit_single.yaml)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── config ────────────────────────────────────────────────────────────────────
# a3m path: 1st positional arg wins, else the A3M_FILE env var, else error.
A3M_FILE="${1:-${A3M_FILE:-}}"
if [ -z "$A3M_FILE" ]; then
    echo "ERROR: no target a3m given."
    echo "  usage: bash scripts/run_overfit.sh /path/to/target.a3m"
    echo "     or: A3M_FILE=/path/to/target.a3m bash scripts/run_overfit.sh"
    exit 1
fi
WORK_DIR="${WORK_DIR:-$REPO_DIR/runs/overfit}"
PROTENIX_CKPT="${PROTENIX_CKPT:-$REPO_DIR/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}"
CCD_DIR="${CCD_DIR:-$HOME/common}"
N_CYCLE="${N_CYCLE:-1}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-$REPO_DIR/msaflow/configs/decoder_overfit_single.yaml}"
ALPHA="${ALPHA:-}"                # α-geometry: empty=use config (0.0), 0.5=α-Flow (L⁴ sphere)

# ── derive ────────────────────────────────────────────────────────────────────
A3M_FILE="$(readlink -f "$A3M_FILE")"
[ -f "$A3M_FILE" ] || { echo "ERROR: a3m not found: $A3M_FILE"; exit 1; }
TARGET_ID="$(basename "$A3M_FILE" .a3m)"          # a3m stem == store key
STORE="$WORK_DIR/${TARGET_ID}"
# separate output per α so α=0 and α=0.5 runs don't overwrite each other
ALPHA_TAG="${ALPHA:+_a${ALPHA}}"
OUT_DIR="$WORK_DIR/decoder_${TARGET_ID}${ALPHA_TAG}"
# Stage just this one a3m in its own dir so build_lmdb processes ONLY the target
# (no --protein_list / no pre-existing logs dir needed).
STAGE_DIR="$WORK_DIR/a3m_${TARGET_ID}"
export PROTENIX_ROOT_DIR="$(dirname "$CCD_DIR")"  # Protenix reads $PROTENIX_ROOT_DIR/common
export CUDA_VISIBLE_DEVICES="$GPU"

# ── checks ────────────────────────────────────────────────────────────────────
[ -f "$PROTENIX_CKPT" ] || { echo "ERROR: checkpoint not found: $PROTENIX_CKPT"; exit 1; }
for f in components.cif components.cif.rdkit_mol.pkl; do
    [ -f "$CCD_DIR/$f" ] || { echo "ERROR: missing CCD file $CCD_DIR/$f"; \
        echo "  download: wget -c https://protenix.tos-cn-beijing.volces.com/common/$f -P $CCD_DIR"; exit 1; }
done
command -v uv >/dev/null || { echo "ERROR: uv not found (run scripts/setup_native_uv.sh first)"; exit 1; }

mkdir -p "$WORK_DIR" "$OUT_DIR" "$STAGE_DIR"
ln -sf "$A3M_FILE" "$STAGE_DIR/${TARGET_ID}.a3m"

echo "============================================================"
echo " MSAFlow single-target OVERFIT test"
echo "   target : $TARGET_ID"
echo "   a3m    : $A3M_FILE"
echo "   store  : $STORE"
echo "   out    : $OUT_DIR"
echo "   n_cycle: $N_CYCLE   gpu: $GPU   ccd: $CCD_DIR"
echo "   alpha  : ${ALPHA:-config default (0.0)}"
echo "============================================================"

# ── 1) build single-target store (skip if present) ────────────────────────────
if [ -e "$STORE" ]; then
    echo "== [1/2] store exists, skipping preprocess (rm -r '$STORE' to rebuild) =="
else
    echo "== [1/2] build single-target store (faithful Protenix, n_cycle=$N_CYCLE) =="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m msaflow.data.preprocessing \
        --a3m_dir "$STAGE_DIR" \
        --output "$STORE" \
        --protenix_checkpoint "$PROTENIX_CKPT" \
        --n_cycle "$N_CYCLE" \
        --device cuda
fi

# ── 2) overfit the decoder on that one MSA ────────────────────────────────────
echo "== [2/2] overfit decoder (alpha=${ALPHA:-config default}) =="
uv run accelerate launch \
    --config_file "$REPO_DIR/msaflow/configs/accelerate_1gpu.yaml" \
    msaflow/training/train_decoder.py \
    --config "$CONFIG" \
    --store_path "$STORE" \
    --output_dir "$OUT_DIR" \
    ${ALPHA:+--alpha "$ALPHA"}

# ── 3) plot diagnostics (local PNGs, no wandb needed) ─────────────────────────
echo "== [3/3] plotting → $OUT_DIR/*.png =="
uv run python scripts/plot_overfit.py "$OUT_DIR" || echo "  (plot skipped)"

echo "== done: $TARGET_ID  (see $OUT_DIR/train_loss.png, val_recovery.png, val_kl.png) =="
