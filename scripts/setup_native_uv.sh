#!/bin/bash
# One-time native uv environment setup (no apptainer / no conda).
#
# The repo was developed inside an Apptainer sandbox where the code lives at
# /mnt/MSA_FLOW. Several files hard-code that path for the vendored Protenix/esm
# path-dependencies (pyproject.toml, uv.lock, Protenix/requirements.txt). On a
# bare node the clone lives elsewhere (e.g. $HOME/MSA_FLOW), so we rewrite those
# absolute paths to *this* clone's location, then build the uv venv.
#
# Idempotent + safe in the container too: there $REPO_DIR == /mnt/MSA_FLOW, so
# the rewrite is a no-op.
#
#   bash scripts/setup_native_uv.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
OLD="/mnt/MSA_FLOW"

echo "== native uv setup =="
echo "  clone dir : $REPO_DIR"

# 0) uv present?
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it first, e.g.:"
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh   (then re-open shell)"
    exit 1
fi

# 1) Point the container path at this clone (no-op if already correct).
#    Mark rewritten tracked files skip-worktree so these node-local edits don't
#    show up as changes and don't block/conflict on future `git pull`.
if [ "$REPO_DIR" != "$OLD" ]; then
    for f in pyproject.toml uv.lock Protenix/requirements.txt; do
        if grep -q "$OLD" "$f" 2>/dev/null; then
            sed -i "s#${OLD}#${REPO_DIR}#g" "$f"
            git update-index --skip-worktree "$f" 2>/dev/null || true
            echo "  patched + skip-worktree : $f"
        fi
    done
    echo "  (to let these files update from git again: git update-index --no-skip-worktree <file>)"
else
    echo "  (running at $OLD — no path rewrite needed)"
fi

# 2) Build the venv from the lock (falls back to re-resolve if the lock drifted)
echo "== uv sync =="
uv sync --frozen 2>/dev/null || uv sync

echo
echo "Done. venv -> $REPO_DIR/.venv"
echo "Run things with 'uv run ...', e.g.:"
echo "  uv run python -c 'import torch; print(torch.cuda.is_available())'"
echo
echo "NOTE: pyproject.toml / uv.lock / Protenix/requirements.txt now differ from"
echo "git (container paths rewritten). That is expected on a bare node — do NOT"
echo "commit these path rewrites back to the shared branch."
