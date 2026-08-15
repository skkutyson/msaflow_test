#!/bin/bash
# Full SFM decoder training on N local GPUs — native uv, no apptainer / no SLURM
# (same style as run_preprocess.sh). For the ada cluster use the SLURM wrapper
# scripts/train_decoder.sh instead.
#
# Usage (store dir as arg 1 OR the STORE_PATH env var):
#   bash scripts/run_train_decoder.sh ~/data/msaflow_openfold
#   STORE_PATH=~/data/msaflow_openfold bash scripts/run_train_decoder.sh
#
# Long run — launch under tmux, or:
#   nohup bash scripts/run_train_decoder.sh ~/data/msaflow_openfold \
#       > logs/train_decoder.log 2>&1 &
#
# Env overrides:
#   STORE_PATH      (required, or arg 1) train ShardStore directory
#   VAL_STORE_PATH  optional val store; empty = validate on held-in train proteins
#   OUTPUT_DIR      checkpoints/metrics dir     (default: <repo>/runs/decoder)
#   CONFIG          decoder yaml                (default: msaflow/configs/decoder.yaml)
#   NUM_GPUS        processes / GPUs            (default: 4)
#   ALPHA           α-geometry override         (empty=config; 0.0/0.5/-0.5)
#   ACCEL_CONFIG    accelerate yaml             (default: accelerate_4gpu.yaml)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

STORE_PATH="${1:-${STORE_PATH:-}}"
if [ -z "$STORE_PATH" ]; then
    echo "ERROR: no store dir given."
    echo "  usage: bash scripts/run_train_decoder.sh /path/to/store"
    exit 1
fi
STORE_PATH="$(readlink -f "$STORE_PATH")"
VAL_STORE_PATH="${VAL_STORE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/decoder}"
CONFIG="${CONFIG:-$REPO_DIR/msaflow/configs/decoder.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
ALPHA="${ALPHA:-}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$REPO_DIR/msaflow/configs/accelerate_4gpu.yaml}"

# ── checks ────────────────────────────────────────────────────────────────────
[ -f "$STORE_PATH/meta.json" ] || { echo "ERROR: not a store dir (no meta.json): $STORE_PATH"; exit 1; }
[ -f "$STORE_PATH/index.parquet" ] || { echo "ERROR: store has no index.parquet: $STORE_PATH"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -f "$ACCEL_CONFIG" ] || { echo "ERROR: accelerate config not found: $ACCEL_CONFIG"; exit 1; }
[ -z "$VAL_STORE_PATH" ] || [ -f "$VAL_STORE_PATH/meta.json" ] || { echo "ERROR: val store not found: $VAL_STORE_PATH"; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not found (run scripts/setup_native_uv.sh first)"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs"

echo "============================================================"
echo " MSAFlow Decoder Training (native uv, ${NUM_GPUS} GPU)"
echo "   store    : $STORE_PATH"
echo "   val      : ${VAL_STORE_PATH:-<held-in train proteins>}"
echo "   output   : $OUTPUT_DIR"
echo "   config   : $CONFIG"
echo "   alpha    : ${ALPHA:-<config default>}"
echo "   Started  : $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    msaflow/training/train_decoder.py \
    --config      "$CONFIG" \
    --store_path  "$STORE_PATH" \
    --output_dir  "$OUTPUT_DIR" \
    ${VAL_STORE_PATH:+--val_store_path "$VAL_STORE_PATH"} \
    ${ALPHA:+--alpha "$ALPHA"}

echo "== done: $OUTPUT_DIR =="
