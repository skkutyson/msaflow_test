#!/bin/bash
# MSAFlow Sandbox 초기 설정 스크립트
# 새 sandbox 빌드 후 최초 1회 실행.
#
#   bash scripts/setup_sandbox.sh

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
SANDBOX=${SANDBOX:-/home/yjlee4/msaflow_sandbox}

echo "============================================================"
echo " MSAFlow Sandbox Setup"
echo " SANDBOX : $SANDBOX"
echo " REPO    : $REPO_DIR"
echo "============================================================"

# ── [1/4] uv 설치 ─────────────────────────────────────────────────────────────
echo ""
echo "=== [1/4] uv 설치 ==="
apptainer exec --writable --fakeroot \
    "$SANDBOX" \
    bash -c "pip install uv -q && echo \"uv \$(uv --version) installed\""

# ── [2/4] 마운트 포인트 생성 ───────────────────────────────────────────────────
echo ""
echo "=== [2/4] 마운트 포인트 생성 ==="
apptainer exec --writable --fakeroot \
    "$SANDBOX" \
    mkdir -p /mnt/MSA_FLOW /mnt/data /mnt/output

# ── [3/4] Claude Code 설치 ────────────────────────────────────────────────────
echo ""
echo "=== [3/4] Claude Code 설치 ==="
apptainer exec --writable --fakeroot \
    "$SANDBOX" \
    bash -c '
        set -euo pipefail
        # npm 확인
        if ! command -v npm &>/dev/null; then
            echo "npm 없음 — Node.js 설치 중..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
            apt-get install -y nodejs -q
        fi
        echo "Node $(node --version) / npm $(npm --version)"
        npm install -g @anthropic-ai/claude-code -q
        echo "Claude Code $(claude --version) installed"
    '

# ── [4/4] uv sync ─────────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] uv sync ==="
apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    "$SANDBOX" \
    bash -c "cd /mnt/MSA_FLOW && uv sync"

echo ""
echo "============================================================"
echo " Done! 다음 명령어로 검증:"
echo "   bash scripts/verify_env.sh"
echo "============================================================"
