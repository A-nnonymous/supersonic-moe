#!/usr/bin/env bash
# Setup script for dev_b environment to run SonicMoE with ERNIE integration.
#
# Prerequisites:
#   - dev_b venv must already exist with paddle 3.3.0.dev + torch
#   - eb_venv must be accessible (for cutlass-dsl wheel source)
#
# Usage:
#   source docs/setup_dev_b.sh
#   # or:
#   bash docs/setup_dev_b.sh   # prints env vars to eval

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUACK_ROOT="/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
EB_VENV="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/eb_venv"
DEV_B="${DEV_B_VENV:-/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/dev_b}"

# ── Install cutlass-dsl (required by quack GEMM kernels) ────────────────────
echo "[setup_dev_b] Installing nvidia-cutlass-dsl into dev_b..."
if python -c "import cutlass.dsl" 2>/dev/null; then
    echo "  cutlass-dsl already installed, skipping."
else
    pip install nvidia-cutlass-dsl==4.4.2 \
        --find-links "${EB_VENV}/lib/python3.10/site-packages/" \
        --no-deps \
        2>/dev/null \
    || pip install nvidia-cutlass-dsl==4.4.2 --no-deps
    echo "  cutlass-dsl installed."
fi

# ── Environment variables ────────────────────────────────────────────────────
# Add quack and repo to PYTHONPATH
export PYTHONPATH="${QUACK_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# Enable QuACK GEMM backend
export USE_QUACK_GEMM=1

# Assume 128-aligned inputs (DeepEP path guarantees this)
export SONIC_MOE_FP8_ASSUME_ALIGNED=1

echo "[setup_dev_b] Environment configured:"
echo "  PYTHONPATH includes: ${QUACK_ROOT}"
echo "  PYTHONPATH includes: ${REPO_ROOT}"
echo "  USE_QUACK_GEMM=1"
echo "  SONIC_MOE_FP8_ASSUME_ALIGNED=1"
echo ""
echo "To run tests:"
echo "  python tests/ops/test_sonic_moe_func.py"
echo "  python -m pytest tests/ops/test_sonic_moe_func.py -q"
