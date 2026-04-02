"""Benchmark: Native FP8 params vs FP8 frontier vs Official BF16.

Simulates a "native FP8 params" scenario where:
- Input x arrives as FP8 (pre-quantized before MoE)
- Weights are stored as FP8 + ISA-packed scales (no quant/cache needed)
- GemmGated outputs y1 as FP8 (PostAct FP8, scales computed separately)

Usage:
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
        python tools/profile_native_fp8.py [--warmup 10] [--iters 5] [--nsys]
"""
import argparse
import os
import sys
import time

import torch
import torch.cuda

# Ensure fork codebase is on path
FORK_DIR = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
if FORK_DIR not in sys.path:
    sys.path.insert(0, FORK_DIR)

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")


def setup_model_and_data(T=8192, H=3072, I=1536, E=8, K=8):
    """Create MoE model and simulate native FP8 inputs."""
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType

    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x_bf16 = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    return moe, x_bf16, dout


def bench_frontier_fp8(moe, x_bf16, dout, warmup=10, iters=5, use_nsys=False):
    """Benchmark the current FP8 frontier path (with quant overhead)."""
    from sonicmoe import enable_fp8, enable_quack_gemm

    # Warmup
    for _ in range(warmup):
        x_bf16.grad = None
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True), enable_fp8():
            z, _ = moe(x_bf16, use_fp8=True)
        z.backward(dout)

    torch.cuda.synchronize()
    if use_nsys:
        torch.cuda.cudart().cudaProfilerStart()

    for _ in range(iters):
        x_bf16.grad = None
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True), enable_fp8():
            z, _ = moe(x_bf16, use_fp8=True)
        z.backward(dout)

    torch.cuda.synchronize()
    if use_nsys:
        torch.cuda.cudart().cudaProfilerStop()


def bench_native_fp8(moe, x_bf16, dout, warmup=10, iters=5, use_nsys=False):
    """Benchmark the native FP8 params path (no quant overhead for x/weights)."""
    from sonicmoe import enable_native_fp8

    # Warmup
    for _ in range(warmup):
        x_bf16.grad = None
        moe.zero_grad(set_to_none=True)
        with enable_native_fp8():
            z, _ = moe(x_bf16, use_fp8=True)
        z.backward(dout)

    torch.cuda.synchronize()
    if use_nsys:
        torch.cuda.cudart().cudaProfilerStart()

    for _ in range(iters):
        x_bf16.grad = None
        moe.zero_grad(set_to_none=True)
        with enable_native_fp8():
            z, _ = moe(x_bf16, use_fp8=True)
        z.backward(dout)

    torch.cuda.synchronize()
    if use_nsys:
        torch.cuda.cudart().cudaProfilerStop()


def bench_timed(fn, label, warmup=10, iters=5):
    """Run fn for warmup+iters, return per-iter GPU time in µs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    per_iter_us = total_ms * 1000 / iters
    print(f"  {label}: {per_iter_us:.0f} µs/iter ({iters} iters)")
    return per_iter_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--mode", default="both", choices=["frontier", "native", "both", "timed"])
    args = parser.parse_args()

    moe, x_bf16, dout = setup_model_and_data()

    if args.mode == "timed":
        from sonicmoe import enable_fp8, enable_quack_gemm, enable_native_fp8

        print("=== Timed comparison ===")

        def run_frontier():
            x_bf16.grad = None
            moe.zero_grad(set_to_none=True)
            with enable_quack_gemm(True), enable_fp8():
                z, _ = moe(x_bf16, use_fp8=True)
            z.backward(dout)

        def run_native():
            x_bf16.grad = None
            moe.zero_grad(set_to_none=True)
            with enable_native_fp8():
                z, _ = moe(x_bf16, use_fp8=True)
            z.backward(dout)

        t_frontier = bench_timed(run_frontier, "FP8 Frontier", args.warmup, args.iters)
        t_native = bench_timed(run_native, "Native FP8", args.warmup, args.iters)
        if t_frontier > 0:
            print(f"  Native vs Frontier: {t_frontier/t_native:.3f}x")
        return

    if args.mode in ("frontier", "both"):
        print(f"=== FP8 Frontier (warmup={args.warmup}, iters={args.iters}) ===")
        bench_frontier_fp8(moe, x_bf16, dout, args.warmup, args.iters, args.nsys)
        print("Frontier done.")

    if args.mode in ("native", "both"):
        print(f"=== Native FP8 (warmup={args.warmup}, iters={args.iters}) ===")
        bench_native_fp8(moe, x_bf16, dout, args.warmup, args.iters, args.nsys)
        print("Native FP8 done.")


if __name__ == "__main__":
    main()
