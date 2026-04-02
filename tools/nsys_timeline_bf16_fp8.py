#!/usr/bin/env python3
"""Generate NVTX-annotated nsys timelines for BF16 vs FP8 frontier.

Produces two .nsys-rep files:
  - timeline_bf16.nsys-rep
  - timeline_fp8.nsys-rep

Usage:
  nsys profile -o timeline_bf16 python tools/nsys_timeline_bf16_fp8.py --mode bf16
  nsys profile -o timeline_fp8  python tools/nsys_timeline_bf16_fp8.py --mode fp8
Or run directly for NVTX-marked CUDA profiler traces:
  python tools/nsys_timeline_bf16_fp8.py --mode both
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.pop("SONIC_MOE_FP8_MODE", None)
os.environ["USE_QUACK_GEMM"] = "1"

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe import functional

T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP = 3
PROFILE_ITERS = 3


def reset():
    functional.clear_all_fp8_weight_caches()
    functional._ALIGNMENT_ASSUMED = False
    functional._ALIGNMENT_STREAK = 0
    gc.collect()
    torch.cuda.empty_cache()


def run_profile(mode: str):
    reset()
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    use_fp8 = mode == "fp8"
    label = "FP8-frontier" if use_fp8 else "BF16-fork"

    # Warmup (outside profiler)
    for _ in range(WARMUP):
        if use_fp8:
            with enable_quack_gemm(True), enable_fp8():
                z, _ = moe(x, use_fp8=True)
        else:
            with enable_quack_gemm(True):
                z, _ = moe(x)
        z.sum().backward()
        x.grad = None
        moe.zero_grad(set_to_none=True)
        reset()

    torch.cuda.synchronize()

    # Profiled iterations with NVTX markers
    for i in range(PROFILE_ITERS):
        torch.cuda.nvtx.range_push(f"{label}_iter{i}")

        torch.cuda.nvtx.range_push(f"{label}_forward")
        if use_fp8:
            with enable_quack_gemm(True), enable_fp8():
                z, _ = moe(x, use_fp8=True)
        else:
            with enable_quack_gemm(True):
                z, _ = moe(x)
        torch.cuda.nvtx.range_pop()  # forward

        torch.cuda.nvtx.range_push(f"{label}_backward")
        z.sum().backward()
        torch.cuda.nvtx.range_pop()  # backward

        x.grad = None
        moe.zero_grad(set_to_none=True)

        torch.cuda.nvtx.range_pop()  # iter

    torch.cuda.synchronize()
    print(f"[{label}] {PROFILE_ITERS} profiled iterations complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "fp8", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("bf16", "both"):
        run_profile("bf16")
    if args.mode in ("fp8", "both"):
        run_profile("fp8")


if __name__ == "__main__":
    main()
