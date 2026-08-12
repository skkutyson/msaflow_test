#!/bin/bash
# Re-apply local submodule modifications after a fresh clone + submodule init.
#
# The superproject only records each submodule's commit SHA, not local edits.
# These patches carry the GPU-compat changes (Protenix Blackwell/sm_120 layer
# norm + fused ops; esm version) that are NOT captured by the superproject.
#
# Usage (from repo root, after cloning):
#   git submodule update --init --recursive
#   bash patches/apply_submodule_patches.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "== expected submodule base commits =="
cat patches/submodule-commits.txt
echo

apply_one() {
    local sub="$1" patch="$2"
    if [ ! -d "$sub/.git" ] && [ ! -f "$sub/.git" ]; then
        echo "SKIP $sub — submodule not initialised (run: git submodule update --init --recursive)"
        return 1
    fi
    if git -C "$sub" apply --check "$REPO_DIR/$patch" 2>/dev/null; then
        git -C "$sub" apply "$REPO_DIR/$patch"
        echo "OK   applied $patch -> $sub"
    elif git -C "$sub" apply --reverse --check "$REPO_DIR/$patch" 2>/dev/null; then
        echo "SKIP $patch already applied to $sub"
    else
        echo "WARN $patch does not apply cleanly to $sub — check submodule base commit / conflicts"
    fi
}

apply_one Protenix patches/protenix-gpu-compat.patch
apply_one esm      patches/esm-version.patch

echo
echo "Done. Verify with: git -C Protenix status --short ; git -C esm status --short"
