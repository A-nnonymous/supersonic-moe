#!/bin/bash
# Three-way NSYS timeline collection: Official BF16 vs Fork FP8 vs Ernie MoE
#
# Usage:
#   bash tools/collect_nsys_timelines.sh <GPU_ID>
#   e.g.: bash tools/collect_nsys_timelines.sh 0
#
# Output: timelines saved to /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/
set -euo pipefail

GPU_ID="${1:-0}"
OUTPUT_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output"
SONIC_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
OFFICIAL_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
XFER_ENV="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate"

# Clean distributed env
unset PADDLE_ELASTIC_JOB_ID 2>/dev/null || true
unset PADDLE_TRAINER_ENDPOINTS 2>/dev/null || true
unset DISTRIBUTED_TRAINER_ENDPOINTS 2>/dev/null || true
unset FLAGS_START_PORT 2>/dev/null || true
unset PADDLE_ELASTIC_TIMEOUT 2>/dev/null || true
export NNODES=1
export PADDLE_TRAINERS_NUM=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Three-way NSYS Timeline Collection"
echo "  GPU: $GPU_ID"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

# ---- 1. Official BF16 ----
echo ""
echo "[1/3] Profiling Official SonicMoE BF16..."
source "$XFER_ENV"
cd "$OFFICIAL_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID \
  nsys profile -t cuda,nvtx --gpu-metrics-device=0 \
  --cuda-memory-usage=false -f true \
  -o "$OUTPUT_DIR/official_bf16_timeline" --export=sqlite \
  python "$SONIC_DIR/tools/nsys_profile_official_bf16.py" \
  2>&1 | tail -5

echo "  → Saved: $OUTPUT_DIR/official_bf16_timeline.nsys-rep"

# ---- 2. Fork FP8 ----
echo ""
echo "[2/3] Profiling Fork SonicMoE FP8..."
cd "$SONIC_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID SONIC_MOE_FP8_FUSED_GATED=1 \
  nsys profile -t cuda,nvtx --gpu-metrics-device=0 \
  --cuda-memory-usage=false -f true \
  -o "$OUTPUT_DIR/fork_fp8_timeline" --export=sqlite \
  python tools/nsys_profile_comprehensive.py --mode fp8 \
  2>&1 | tail -5

echo "  → Saved: $OUTPUT_DIR/fork_fp8_timeline.nsys-rep"

# ---- 3. Ernie MoE ----
echo ""
echo "[3/3] Profiling Ernie DeepEPMOELayer..."

CUDA_VISIBLE_DEVICES=$GPU_ID \
  nsys profile -t cuda,nvtx --gpu-metrics-device=0 \
  --cuda-memory-usage=false -f true \
  -o "$OUTPUT_DIR/ernie_moe_timeline" --export=sqlite \
  python "$SONIC_DIR/tools/nsys_profile_ernie_moe.py" \
  2>&1 | tail -5

echo "  → Saved: $OUTPUT_DIR/ernie_moe_timeline.nsys-rep"

# ---- Analysis ----
echo ""
echo "========================================"
echo "  GPU Projection Analysis"
echo "========================================"
source "$XFER_ENV"
cd "$SONIC_DIR"

python tools/analyze_nsys_three_way.py \
  --official "$OUTPUT_DIR/official_bf16_timeline.sqlite" \
  --fork "$OUTPUT_DIR/fork_fp8_timeline.sqlite" \
  --ernie "$OUTPUT_DIR/ernie_moe_timeline.sqlite"

echo ""
echo "Done! Timelines saved to $OUTPUT_DIR/"
echo "  - official_bf16_timeline.nsys-rep"
echo "  - fork_fp8_timeline.nsys-rep"
echo "  - ernie_moe_timeline.nsys-rep"
