#!/bin/bash
#SBATCH --job-name=msaflow-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --nice=0
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/probe_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/probe_%j.err

# ── Probe: faithful Protenix MSA-embedding extraction on ONE example ──────────
# Runs scripts/probe_protenix_msa_encoder.py inside the Apptainer sandbox (which
# has the Protenix env) to verify the featurizer + full model + MSAModule-hook
# path yields a sane (L, 128) embedding before wiring it into build_lmdb.
#
# Submit (fill INPUT_A3M and PROTENIX_CKPT with host paths):
#   INPUT_A3M=/gpfs/.../one.a3m \
#   PROTENIX_CKPT=/gpfs/.../protenix_base_default_v1.0.0.pt \
#   sbatch scripts/probe_protenix.sh

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

# Pick the a3m automatically: pass INPUT_A3M directly, OR pass A3M_DIR and the
# script grabs the first *.a3m it finds under it (recursively).
A3M_DIR=${A3M_DIR:-/gpfs/database/openfold/uniclust30}
if [ -z "${INPUT_A3M:-}" ]; then
    echo "No INPUT_A3M given — picking the first *.a3m under $A3M_DIR ..."
    INPUT_A3M=$(find -L "$A3M_DIR" -name "*.a3m" -print -quit 2>/dev/null)
    [ -n "$INPUT_A3M" ] || { echo "ERROR: no .a3m found under $A3M_DIR (set A3M_DIR or INPUT_A3M)"; exit 1; }
fi
# Protenix checkpoint ships in the repo (1.4GB) — override only if you keep it elsewhere.
PROTENIX_CKPT=${PROTENIX_CKPT:-/home/yjlee4/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt}
N_CYCLE=${N_CYCLE:-5}    # 5 for the per-cycle embedding comparison (model default: 10)

mkdir -p "$REPO_DIR/logs"

# Split into dir + basename so we can bind the parent dirs into the container.
A3M_DIR="$(dirname "$INPUT_A3M")";   A3M_FILE="$(basename "$INPUT_A3M")"
CKPT_DIR="$(dirname "$PROTENIX_CKPT")"; CKPT_FILE="$(basename "$PROTENIX_CKPT")"

echo "============================================================"
echo " MSAFlow Protenix-encoder probe"
echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " GPU         : $CUDA_VISIBLE_DEVICES"
echo " INPUT_A3M   : $INPUT_A3M"
echo " PROTENIX    : $PROTENIX_CKPT"
echo " N_CYCLE     : $N_CYCLE"
echo " Started     : $(date)"
echo "============================================================"

# If a3m and ckpt live in the same dir, bind it once; otherwise bind both.
EXTRA_BIND="--bind ${A3M_DIR}:/mnt/a3m"
CKPT_MOUNT="/mnt/ckpt/${CKPT_FILE}"
if [ "$CKPT_DIR" = "$A3M_DIR" ]; then
    CKPT_MOUNT="/mnt/a3m/${CKPT_FILE}"
else
    EXTRA_BIND="${EXTRA_BIND} --bind ${CKPT_DIR}:/mnt/ckpt"
fi

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    ${EXTRA_BIND} \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW
        uv run python scripts/probe_protenix_msa_encoder.py \
            --input_a3m    /mnt/a3m/${A3M_FILE} \
            --protenix_ckpt ${CKPT_MOUNT} \
            --n_cycle      ${N_CYCLE} \
            --device       cuda
    "

echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
