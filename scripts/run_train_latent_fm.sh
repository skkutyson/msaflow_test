#!/bin/bash
# Latent FM Encoder training on N local GPUs — native uv, no apptainer / no SLURM.
# Companion to scripts/run_train_decoder.sh (same store, same style). The Latent
# FM learns the ESM2 -> compressed-MSA-embedding flow (esm_emb (L,1280) -> msa_emb
# (L,128)), so it reads BOTH arrays from the ShardStore. For the ada SLURM cluster
# use scripts/train_latent_fm_foldbench.sh instead.
#
# Usage (store dir as arg 1 OR the STORE_PATH env var):
#   bash scripts/run_train_latent_fm.sh ~/data/msaflow_newshard
#   STORE_PATH=~/data/msaflow_newshard bash scripts/run_train_latent_fm.sh
#
# Long run — launch under tmux, or:
#   nohup bash scripts/run_train_latent_fm.sh ~/data/msaflow_newshard \
#       > logs/train_latent_fm.log 2>&1 &
#
# Env overrides:
#   STORE_PATH      (required, or arg 1) train ShardStore directory (needs esm_emb)
#   OUTPUT_DIR      checkpoints/metrics dir     (default: <repo>/runs/latent_fm)
#   CONFIG          latent FM yaml              (default: msaflow/configs/latent_fm.yaml)
#   NUM_GPUS        processes / GPUs            (default: 4)
#   ACCEL_CONFIG    accelerate yaml            (default: accelerate_4gpu.yaml)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

STORE_PATH="${1:-${STORE_PATH:-}}"
if [ -z "$STORE_PATH" ]; then
    echo "ERROR: no store dir given."
    echo "  usage: bash scripts/run_train_latent_fm.sh /path/to/store"
    exit 1
fi
STORE_PATH="$(readlink -f "$STORE_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/latent_fm}"
CONFIG="${CONFIG:-$REPO_DIR/msaflow/configs/latent_fm.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$REPO_DIR/msaflow/configs/accelerate_4gpu.yaml}"

# ── checks ────────────────────────────────────────────────────────────────────
[ -f "$STORE_PATH/meta.json" ]      || { echo "ERROR: not a store dir (no meta.json): $STORE_PATH"; exit 1; }
[ -f "$STORE_PATH/index.parquet" ]  || { echo "ERROR: store has no index.parquet: $STORE_PATH"; exit 1; }
# Latent FM (unlike the decoder) needs esm_emb — verify the store actually has it.
compgen -G "$STORE_PATH/esm_emb/shard_*.bin" >/dev/null \
    || { echo "ERROR: store has no esm_emb shards: $STORE_PATH/esm_emb/ (Latent FM needs esm_emb)"; exit 1; }
[ -f "$CONFIG" ]                    || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -f "$ACCEL_CONFIG" ]             || { echo "ERROR: accelerate config not found: $ACCEL_CONFIG"; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not found (run scripts/setup_native_uv.sh first)"; exit 1; }

# ── GPU sanity (일반 GPU 머신에서 바로 실행 — SLURM/apptainer 없음) ───────────
command -v nvidia-smi >/dev/null 2>&1 \
    || { echo "ERROR: nvidia-smi 없음 — GPU 있는 머신에서 실행하세요 (로그인 노드 X)"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs"

# ── W&B (optional) ────────────────────────────────────────────────────────────
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-$HOME/.wandb_api_key}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_API_KEY_FILE" ]; then
    export WANDB_API_KEY="$(cat "$WANDB_API_KEY_FILE")"
fi
echo "   wandb    : $([ -n "${WANDB_API_KEY:-}" ] && echo enabled || echo 'disabled (no WANDB_API_KEY)')"

echo "============================================================"
echo " MSAFlow Latent FM Training (native uv, ${NUM_GPUS} GPU)"
echo "   store    : $STORE_PATH"
echo "   output   : $OUTPUT_DIR"
echo "   config   : $CONFIG"
echo "   Started  : $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    msaflow/training/train_latent_fm.py \
    --config      "$CONFIG" \
    --store_path  "$STORE_PATH" \
    --output_dir  "$OUTPUT_DIR"

echo "== done: $OUTPUT_DIR/latent_fm_ema_final.pt =="
