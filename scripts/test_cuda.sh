#!/bin/bash
#SBATCH --job-name=msaflow-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/preprocess_%j.err
REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox

mkdir -p "$REPO_DIR/logs"

apptainer exec --nv \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    "$SANDBOX" \
    bash -c '
        cd /mnt/MSA_FLOW
        uv run python -c "
import torch
print(\"torch     :\", torch.__version__)
print(\"cuda avail:\", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    sm = cap[0]*10 + cap[1]
    print(\"device    :\", torch.cuda.get_device_name(0))
    print(\"sm        :\", sm)
    print(\"driver    :\", torch.version.cuda)
"
    '
