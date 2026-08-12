#!/bin/bash
# MSAFlow Sandbox 이사 스크립트
# nvcr.io/nvidia/pytorch:25.01-py3 기반으로 새 sandbox를 빌드하고
# 기존 sandbox를 교체합니다.
#
#   bash scripts/migrate_sandbox.sh

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/yjlee4/MSA_FLOW}
SANDBOX=${SANDBOX:-/home/yjlee4/msaflow_sandbox}
NEW_IMAGE=${NEW_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}

SANDBOX_NEW="${SANDBOX}_new"
SANDBOX_OLD="${SANDBOX}_old"

echo "============================================================"
echo " MSAFlow Sandbox 이사"
echo " 현재 sandbox : $SANDBOX"
echo " 새 이미지    : $NEW_IMAGE"
echo " 임시 경로    : $SANDBOX_NEW"
echo "============================================================"
echo ""

# ── [1/4] 새 sandbox 빌드 ─────────────────────────────────────────────────────
echo "=== [1/4] 새 sandbox 빌드 (시간 소요) ==="
echo " → $NEW_IMAGE"
apptainer build --sandbox "$SANDBOX_NEW" "docker://$NEW_IMAGE"

# ── [2/4] 기존 sandbox 백업 ───────────────────────────────────────────────────
echo ""
echo "=== [2/4] 기존 sandbox 백업 ==="
if [[ -d "$SANDBOX" ]]; then
    mv "$SANDBOX" "$SANDBOX_OLD"
    echo " → $SANDBOX_OLD 로 이동"
fi
mv "$SANDBOX_NEW" "$SANDBOX"
echo " → 새 sandbox 활성화 완료"

# ── [3/4] uv + Claude 설치 ────────────────────────────────────────────────────
echo ""
echo "=== [3/4] 환경 설치 (setup_sandbox.sh) ==="
REPO_DIR="$REPO_DIR" SANDBOX="$SANDBOX" bash "$REPO_DIR/scripts/setup_sandbox.sh"

# ── [4/4] 검증 ────────────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] 환경 검증 ==="
apptainer exec \
    --bind "$REPO_DIR":/mnt/MSA_FLOW \
    "$SANDBOX" \
    bash -c "cd /mnt/MSA_FLOW && uv run python3 scripts/verify_env.py"

echo ""
echo "============================================================"
echo " 이사 완료!"
echo ""
echo " 이전 sandbox 백업: $SANDBOX_OLD"
echo " 문제 없으면 삭제: rm -rf $SANDBOX_OLD"
echo "============================================================"
