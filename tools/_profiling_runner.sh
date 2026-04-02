#!/bin/bash
# Unified profiling runner for remote SSH execution
# Usage: bash tools/_profiling_runner.sh <mode> [gpu_id]
# Modes: nsys_fp8, nsys_bf16, ncu_fp8, memory, test

set -euo pipefail

MODE="${1:-nsys_fp8}"
GPU="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export VSCODE_SHELL_INTEGRATION=0
export USE_QUACK_GEMM=1

# Unset distributed env vars
unset PADDLE_ELASTIC_JOB_ID 2>/dev/null || true
unset PADDLE_TRAINER_ENDPOINTS 2>/dev/null || true
unset DISTRIBUTED_TRAINER_ENDPOINTS 2>/dev/null || true
unset FLAGS_START_PORT 2>/dev/null || true
unset PADDLE_ELASTIC_TIMEOUT 2>/dev/null || true
export NNODES=1
export PADDLE_TRAINERS_NUM=1

FORK_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
OFFICIAL_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
XFER_ENV="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate"
BF16_ENV="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/activate"

OUT_DIR="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/sonic-moe-profiling"
mkdir -p "$OUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "$MODE" in
    nsys_fp8)
        echo "[nsys_fp8] Starting FP8 nsys profile..."
        source "$XFER_ENV"
        cd "$FORK_DIR"
        export SONIC_MOE_FP8_MODE=perf
        nsys profile --capture-range=cudaProfilerApi --cudabacktrace=none \
            --python-backtrace=none --python-sampling=false -f true \
            -o "$OUT_DIR/nsys_fp8_ernie_${TIMESTAMP}" \
            python tools/gpu_projection_benchmark.py --mode fp8
        echo "[nsys_fp8] Exporting to SQLite..."
        nsys export --type=sqlite --output="$OUT_DIR/nsys_fp8_ernie_${TIMESTAMP}.sqlite" \
            "$OUT_DIR/nsys_fp8_ernie_${TIMESTAMP}.nsys-rep"
        echo "[nsys_fp8] Analyzing..."
        python tools/nsys_full_breakdown.py "$OUT_DIR/nsys_fp8_ernie_${TIMESTAMP}.sqlite" --labels fp8
        echo "[nsys_fp8] Done."
        ;;
    nsys_bf16)
        echo "[nsys_bf16] Starting BF16 (fork code) nsys profile..."
        source "$XFER_ENV"
        cd "$FORK_DIR"
        unset SONIC_MOE_FP8_MODE 2>/dev/null || true
        nsys profile --capture-range=cudaProfilerApi --cudabacktrace=none \
            --python-backtrace=none --python-sampling=false -f true \
            -o "$OUT_DIR/nsys_bf16_ernie_${TIMESTAMP}" \
            python tools/gpu_projection_benchmark.py --mode bf16
        echo "[nsys_bf16] Exporting to SQLite..."
        nsys export --type=sqlite --output="$OUT_DIR/nsys_bf16_ernie_${TIMESTAMP}.sqlite" \
            "$OUT_DIR/nsys_bf16_ernie_${TIMESTAMP}.nsys-rep"
        echo "[nsys_bf16] Analyzing..."
        python tools/nsys_full_breakdown.py "$OUT_DIR/nsys_bf16_ernie_${TIMESTAMP}.sqlite" --labels bf16
        echo "[nsys_bf16] Done."
        ;;
    nsys_official_bf16)
        echo "[nsys_official_bf16] Starting official BF16 nsys profile..."
        source "$BF16_ENV"
        cd "$OFFICIAL_DIR"
        unset SONIC_MOE_FP8_MODE 2>/dev/null || true
        nsys profile --capture-range=cudaProfilerApi --cudabacktrace=none \
            --python-backtrace=none --python-sampling=false -f true \
            -o "$OUT_DIR/nsys_official_bf16_${TIMESTAMP}" \
            python "$FORK_DIR/tools/profile_both.py" --mode bf16
        echo "[nsys_official_bf16] Exporting to SQLite..."
        nsys export --type=sqlite --output="$OUT_DIR/nsys_official_bf16_${TIMESTAMP}.sqlite" \
            "$OUT_DIR/nsys_official_bf16_${TIMESTAMP}.nsys-rep"
        echo "[nsys_official_bf16] Analyzing..."
        source "$XFER_ENV"
        cd "$FORK_DIR"
        python tools/nsys_full_breakdown.py "$OUT_DIR/nsys_official_bf16_${TIMESTAMP}.sqlite" --labels official_bf16
        echo "[nsys_official_bf16] Done."
        ;;
    nsys_fp8_frontier)
        echo "[nsys_fp8_frontier] Starting FP8 frontier nsys profile (profile_both.py)..."
        source "$XFER_ENV"
        cd "$FORK_DIR"
        export SONIC_MOE_FP8_MODE=perf
        nsys profile --capture-range=cudaProfilerApi --cudabacktrace=none \
            --python-backtrace=none --python-sampling=false -f true \
            -o "$OUT_DIR/nsys_fp8_frontier_${TIMESTAMP}" \
            python tools/profile_both.py --mode fp8
        echo "[nsys_fp8_frontier] Exporting to SQLite..."
        nsys export --type=sqlite --output="$OUT_DIR/nsys_fp8_frontier_${TIMESTAMP}.sqlite" \
            "$OUT_DIR/nsys_fp8_frontier_${TIMESTAMP}.nsys-rep"
        echo "[nsys_fp8_frontier] Analyzing..."
        python tools/nsys_full_breakdown.py "$OUT_DIR/nsys_fp8_frontier_${TIMESTAMP}.sqlite" --labels fp8_frontier
        echo "[nsys_fp8_frontier] Done."
        ;;
    mem_fp8)
        echo "[mem_fp8] Measuring FP8 peak memory..."
        source "$XFER_ENV"
        cd "$FORK_DIR"
        export SONIC_MOE_FP8_MODE=perf
        python -c "
import torch, gc
T, H, I, E, K = 8192, 3072, 1536, 8, 8
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = torch.randn(T, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device='cuda', dtype=torch.bfloat16)
for _ in range(3):
    with enable_quack_gemm(True), enable_fp8():
        z, _ = moe(x, use_fp8=True)
    z.backward(dout); x.grad = None; moe.zero_grad(set_to_none=True)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True), enable_fp8():
    z, _ = moe(x, use_fp8=True)
fwd_peak = torch.cuda.max_memory_allocated()
z.backward(dout)
total_peak = torch.cuda.max_memory_allocated()
current = torch.cuda.memory_allocated()
print(f'FP8_FWD_PEAK_MIB={fwd_peak / 1024**2:.1f}')
print(f'FP8_TOTAL_PEAK_MIB={total_peak / 1024**2:.1f}')
print(f'FP8_AFTER_BWD_MIB={current / 1024**2:.1f}')
"
        echo "[mem_fp8] Done."
        ;;
    mem_bf16)
        echo "[mem_bf16] Measuring official BF16 peak memory..."
        source "$BF16_ENV"
        cd "$OFFICIAL_DIR"
        python -c "
import torch, gc
T, H, I, E, K = 8192, 3072, 1536, 8, 8
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = torch.randn(T, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device='cuda', dtype=torch.bfloat16)
for _ in range(3):
    with enable_quack_gemm(True):
        z, _ = moe(x)
    z.backward(dout); x.grad = None; moe.zero_grad(set_to_none=True)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True):
    z, _ = moe(x)
fwd_peak = torch.cuda.max_memory_allocated()
z.backward(dout)
total_peak = torch.cuda.max_memory_allocated()
current = torch.cuda.memory_allocated()
print(f'BF16_FWD_PEAK_MIB={fwd_peak / 1024**2:.1f}')
print(f'BF16_TOTAL_PEAK_MIB={total_peak / 1024**2:.1f}')
print(f'BF16_AFTER_BWD_MIB={current / 1024**2:.1f}')
"
        echo "[mem_bf16] Done."
        ;;
    ncu_fp8)
        echo "[ncu_fp8] Starting FP8 ncu profile (key Triton kernels)..."
        export SONIC_MOE_FP8_MODE=perf
        # Profile Triton quant/dequant/swiglu/gather kernels + CUTLASS GEMMs
        ncu --set full \
            --kernel-name regex:'quantize|dequant|swiglu|gather|pack|token_gather' \
            --launch-skip 50 --launch-count 30 \
            --csv \
            -o "$OUT_DIR/ncu_fp8_triton_${TIMESTAMP}" \
            python tools/gpu_projection_benchmark.py --mode fp8 \
            2>&1 | tee "$OUT_DIR/ncu_fp8_triton_${TIMESTAMP}.log"
        echo "[ncu_fp8] Done."
        ;;
    ncu_gemm)
        echo "[ncu_gemm] Starting GEMM-focused ncu profile..."
        export SONIC_MOE_FP8_MODE=perf
        # Profile CUTLASS GEMMs
        ncu --set full \
            --kernel-name regex:'Gemm|gemm' \
            --launch-skip 50 --launch-count 15 \
            --csv \
            -o "$OUT_DIR/ncu_fp8_gemm_${TIMESTAMP}" \
            python tools/gpu_projection_benchmark.py --mode fp8 \
            2>&1 | tee "$OUT_DIR/ncu_fp8_gemm_${TIMESTAMP}.log"
        echo "[ncu_gemm] Done."
        ;;
    memory)
        echo "[memory] Running memory profiling..."
        export SONIC_MOE_FP8_MODE=perf
        python tools/measure_memory.py 2>&1 | tee "$OUT_DIR/memory_${TIMESTAMP}.log"
        echo "[memory] Running BF16 memory profiling..."
        unset SONIC_MOE_FP8_MODE 2>/dev/null || true
        python tools/measure_memory.py 2>&1 | tee "$OUT_DIR/memory_bf16_${TIMESTAMP}.log"
        echo "[memory] Done."
        ;;
    test)
        echo "[test] Running contract tests..."
        export SONIC_MOE_FP8_MODE=perf
        python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short 2>&1 | \
            tee "$OUT_DIR/test_results_${TIMESTAMP}.log"
        echo "[test] Done."
        ;;
    e2e_bench)
        echo "[e2e_bench] Running end-to-end benchmark..."
        export SONIC_MOE_FP8_MODE=perf
        python tools/bench_aligned_e2e.py 2>&1 | tee "$OUT_DIR/e2e_bench_${TIMESTAMP}.log"
        echo "[e2e_bench] Done."
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
