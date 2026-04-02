#!/usr/bin/env python3
"""GPU-projection benchmark: generate nsys timelines with proper NVTX + sync.

Generates .nsys-rep files with torch.cuda.synchronize() at NVTX boundaries
so that GPU-projection (sum of CUDA kernel times within an NVTX range)
accurately reflects the GPU work for each iteration.

Usage (run via nsys on remote GPU):
  nsys profile --capture-range=cudaProfilerApi --cudabacktrace=none \
    --python-backtrace=none --python-sampling=false -f true \
    -o <output> python tools/gpu_projection_benchmark.py --mode <bf16|fp8>

Modes:
  bf16: BF16 fork baseline (SM100 CUTLASS path)
  fp8:  FP8 frontier (fused blockscaled gated + zero-mat)
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.pop("SONIC_MOE_FP8_MODE", None)
os.environ.pop("SONIC_MOE_FP8_LEAN", None)
os.environ["USE_QUACK_GEMM"] = "1"

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe import functional

# Ernie production shape
T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP = 10
PROFILE_ITERS = 5


def run_bf16_iter(moe, x):
    """Single BF16 forward+backward iteration."""
    with enable_quack_gemm(True):
        z, _ = moe(x, use_fp8=False)
    z.sum().backward()
    x.grad = None
    moe.zero_grad(set_to_none=True)


def run_fp8_iter(moe, x):
    """Single FP8 forward+backward iteration."""
    with enable_quack_gemm(True), enable_fp8():
        z, _ = moe(x, use_fp8=True)
    z.sum().backward()
    x.grad = None
    moe.zero_grad(set_to_none=True)


def run_profile(mode: str):
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    use_fp8 = (mode == "fp8")
    label = "FP8" if use_fp8 else "BF16"
    run_iter = run_fp8_iter if use_fp8 else run_bf16_iter

    # ── Warmup: no reset, let JIT + caches fully warm ──
    for i in range(WARMUP):
        run_iter(moe, x)
        if i == 0:
            torch.cuda.synchronize()
            print(f"[{label}] warmup iter 0 done (includes JIT)")

    torch.cuda.synchronize()
    print(f"[{label}] {WARMUP} warmup iters done")

    # ── Profiled iterations with NVTX + sync at every boundary ──
    torch.cuda.cudart().cudaProfilerStart()

    for i in range(PROFILE_ITERS):
        torch.cuda.synchronize()  # ensure clean start
        torch.cuda.nvtx.range_push(f"{label}_iter{i}")

        torch.cuda.nvtx.range_push(f"{label}_forward")
        if use_fp8:
            with enable_quack_gemm(True), enable_fp8():
                z, _ = moe(x, use_fp8=True)
        else:
            with enable_quack_gemm(True):
                z, _ = moe(x, use_fp8=False)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # forward

        torch.cuda.nvtx.range_push(f"{label}_backward")
        z.sum().backward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # backward

        x.grad = None
        moe.zero_grad(set_to_none=True)

        torch.cuda.nvtx.range_pop()  # iter

    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    print(f"[{label}] {PROFILE_ITERS} profiled iterations complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "fp8"], required=True)
    args = parser.parse_args()
    run_profile(args.mode)


if __name__ == "__main__":
    main()
