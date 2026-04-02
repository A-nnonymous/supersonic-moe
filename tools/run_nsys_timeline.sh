#!/bin/bash
set -e
export VSCODE_SHELL_INTEGRATION=0

# ============================================================================
# SonicMoE nsys timeline collection — BF16 baseline + FP8 frontier
#
# Usage:
#   bash tools/run_nsys_timeline.sh [GPU_ID]       # default GPU 0
#   bash tools/run_nsys_timeline.sh 7               # use GPU 7
#
# Produces:
#   $OUT/timeline_bf16.nsys-rep   — BF16 baseline timeline
#   $OUT/timeline_fp8.nsys-rep    — FP8 frontier timeline
#   $OUT/timeline_bf16_*.csv      — GPU trace + kernel summary
#   $OUT/timeline_fp8_*.csv       — GPU trace + kernel summary
#
# NOTE: Official BF16 env (quack 0.2.7 / SM90-only CUTLASS) cannot run on
#       Blackwell SM100a GPUs. We use the fork codebase with QuACK 0.3.7
#       for BOTH bf16 and fp8, so the comparison isolates FP8 optimizations.
# ============================================================================

source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS \
      FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
export NNODES=1 PADDLE_TRAINERS_NUM=1 USE_QUACK_GEMM=1 PYTHONUNBUFFERED=1

GPU=${1:-0}
OUT=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output
NSYS=/opt/nvidia/nsight-systems-cli/2025.1.1/bin/nsys

if [ ! -f "$NSYS" ]; then
    echo "Installing nsys..."
    dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb 2>/dev/null || true
fi

echo "=============================================="
echo "  SonicMoE nsys Timeline Collection"
echo "  GPU: $GPU"
echo "  env: xfer (quack $(python -c 'import quack;print(quack.__version__)'))"
echo "=============================================="

# ---------- BF16 Baseline ----------
echo ""
echo ">>> [1/2] BF16 Baseline"
CUDA_VISIBLE_DEVICES=$GPU $NSYS profile \
  --gpu-metrics-device=all \
  -t cuda,cudnn,cublas,nvtx \
  --capture-range=cudaProfilerApi \
  --nvtx-capture=profile:BF16-baseline \
  -o $OUT/timeline_bf16 \
  --force-overwrite true \
  python tools/profile_nvtx_timeline.py --mode bf16 \
    --warmup 8 --profile-iters 10 --timing-iters 10

echo ">>> Generating BF16 stats..."
$NSYS stats $OUT/timeline_bf16.nsys-rep --report cuda_gpu_trace  --format csv \
  -o $OUT/timeline_bf16_gputrace   2>&1 | tail -3
$NSYS stats $OUT/timeline_bf16.nsys-rep --report cuda_gpu_kern_sum --format csv \
  -o $OUT/timeline_bf16_gpukernsum 2>&1 | tail -3
$NSYS stats $OUT/timeline_bf16.nsys-rep --report nvtx_sum --format csv \
  -o $OUT/timeline_bf16_nvtx       2>&1 | tail -3

# ---------- FP8 Frontier ----------
echo ""
echo ">>> [2/2] FP8 Frontier"
CUDA_VISIBLE_DEVICES=$GPU $NSYS profile \
  --gpu-metrics-device=all \
  -t cuda,cudnn,cublas,nvtx \
  --capture-range=cudaProfilerApi \
  --nvtx-capture="profile:FP8-frontier" \
  -o $OUT/timeline_fp8 \
  --force-overwrite true \
  python tools/profile_nvtx_timeline.py --mode fp8 \
    --warmup 8 --profile-iters 10 --timing-iters 10

echo ">>> Generating FP8 stats..."
$NSYS stats $OUT/timeline_fp8.nsys-rep --report cuda_gpu_trace  --format csv \
  -o $OUT/timeline_fp8_gputrace   2>&1 | tail -3
$NSYS stats $OUT/timeline_fp8.nsys-rep --report cuda_gpu_kern_sum --format csv \
  -o $OUT/timeline_fp8_gpukernsum 2>&1 | tail -3
$NSYS stats $OUT/timeline_fp8.nsys-rep --report nvtx_sum --format csv \
  -o $OUT/timeline_fp8_nvtx       2>&1 | tail -3

# ---------- Summary ----------
echo ""
echo "=============================================="
echo "  Timeline collection complete!"
echo "  BF16: $OUT/timeline_bf16.nsys-rep"
echo "  FP8:  $OUT/timeline_fp8.nsys-rep"
echo "=============================================="
echo ""

# Quick GPU projection comparison
python3 -c "
import csv, os
OUT = '$OUT'
for tag in ['bf16', 'fp8']:
    fpath = os.path.join(OUT, f'timeline_{tag}_gputrace.csv')
    if not os.path.exists(fpath):
        print(f'{tag}: CSV not found')
        continue
    total_ns = 0
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = int(row.get('Duration (ns)', row.get('Duration', 0)))
            total_ns += dur
    total_us = total_ns / 1000
    per_iter = total_us / 10
    print(f'{tag.upper()}: {total_us:.0f}us total, {per_iter:.0f}us/iter')
" 2>&1 || echo "(CSV comparison skipped)"

echo "=== DONE ==="
