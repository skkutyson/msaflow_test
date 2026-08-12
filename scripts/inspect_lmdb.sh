#!/bin/bash
#SBATCH --job-name=msaflow-lmdb-inspect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/inspect_lmdb_%j.out
#SBATCH --error=/home/yjlee4/MSA_FLOW/logs/inspect_lmdb_%j.err

REPO_DIR=/home/yjlee4/MSA_FLOW
SANDBOX=/home/yjlee4/msaflow_sandbox
LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_fold.lmdb}
N=${N:-10}

mkdir -p "$REPO_DIR/logs"

apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    --bind "$(dirname "$LMDB_PATH")":/mnt/lmdb \
    "$SANDBOX" \
    bash -c "
        set -euo pipefail
        cd /mnt/MSA_FLOW

        uv run python3 -c \"
import lmdb, pickle, os
path  = '/mnt/lmdb/$(basename "$LMDB_PATH")'
n     = int(os.environ.get('N', '$N'))
env   = lmdb.open(path, readonly=True, lock=False, subdir=False, map_size=500*1024**3)
with env.begin() as txn:
    cur = txn.cursor()
    for i, (k, v) in enumerate(cur):
        if i >= n:
            break
        key  = k.decode()
        data = pickle.loads(v)
        shapes = {field: (val.shape if hasattr(val, 'shape') else val)
                  for field, val in data.items()}
        print(f'[{i:3d}] key={key!r:30s}  {shapes}')
env.close()
\"
    "
