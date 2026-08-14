#!/bin/bash
# Multi-GPU ShardStore build: split the a3m list into NUM_GPUS disjoint shards
# and launch one run_preprocess.sh instance per GPU, each writing its OWN store
# part directory (<base>.part<g>). Writers never share files, so there is no
# NFS/GPFS lock problem. Merge the parts afterwards with:
#   uv run python scripts/finalize_store.py --parts <base>.part* --output <base>
#
# Usage:
#   A3M_LIST=/path/to/list.txt bash scripts/run_preprocess_multi.sh /gpfs/database/openfold/uniclust30
#
# Env overrides:
#   NUM_GPUS      number of instances / GPUs        (default: 4)
#   N_CYCLE       Protenix recycling cycles         (default: 1 — full-store speed)
#   OUTPUT_STORE  merged-output store dir           (default: $HOME/data/msaflow_openfold)
#                 parts become <base>.part<g>
#   A3M_LIST      a3m path list, one per line       (default: <OUTPUT_STORE>.a3m_list.txt cache)
# Everything else (MAX_MSA_SEQS, MAX_SEQ_LEN, RESUME, SHARD_MAX_GB, ...) is read
# by run_preprocess.sh from the environment as usual — export to override.
#
# Re-running is safe: the split is deterministic and RESUME=1 (default) makes
# each instance skip ids already in its part store.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

A3M_DIR="${1:-${A3M_DIR:-}}"
if [ -z "$A3M_DIR" ]; then
    echo "ERROR: no a3m directory given."
    echo "  usage: A3M_LIST=list.txt bash scripts/run_preprocess_multi.sh /path/to/uniclust30"
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-4}"
N_CYCLE="${N_CYCLE:-1}"
OUTPUT_STORE="${OUTPUT_STORE:-$HOME/data/msaflow_openfold}"
A3M_LIST="${A3M_LIST:-${OUTPUT_STORE}.a3m_list.txt}"
[ -f "$A3M_LIST" ] || { echo "ERROR: a3m list not found: $A3M_LIST (set A3M_LIST)"; exit 1; }

mkdir -p "$(dirname "$OUTPUT_STORE")" logs

split -n l/"$NUM_GPUS" -d "$A3M_LIST" "${OUTPUT_STORE}.a3mshard_"
echo "Split $(wc -l < "$A3M_LIST") a3m paths into $NUM_GPUS shards"

for ((g = 0; g < NUM_GPUS; g++)); do
    shard="$(printf '%s.a3mshard_%02d' "$OUTPUT_STORE" "$g")"
    out="${OUTPUT_STORE}.part${g}"
    log="logs/preprocess_gpu${g}.log"
    GPU="$g" A3M_LIST="$shard" OUTPUT_STORE="$out" N_CYCLE="$N_CYCLE" \
        nohup bash scripts/run_preprocess.sh "$A3M_DIR" > "$log" 2>&1 &
    echo "  gpu$g  pid=$!  $(wc -l < "$shard") entries  -> $out  (log: $log)"
done

echo
echo "monitor : tail -f logs/preprocess_gpu*.log"
echo "stop all: pkill -f msaflow.data.preprocessing"
echo "merge   : uv run python scripts/finalize_store.py --parts ${OUTPUT_STORE}.part* --output $OUTPUT_STORE"
