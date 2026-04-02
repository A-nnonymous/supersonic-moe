#!/usr/bin/env python3
"""
Detailed nsys timeline profiler for SonicMoE with per-phase NVTX markers.

Usage:
  # BF16 baseline (fork codebase, QuACK kernels)
  python tools/profile_nvtx_timeline.py --mode bf16

  # FP8 frontier
  python tools/profile_nvtx_timeline.py --mode fp8

  # Both (sequential)
  python tools/profile_nvtx_timeline.py --mode both

This script adds NVTX markers around every major phase:
  iter:N / forward:router / forward:up-proj / forward:fp8-boundary /
  forward:down-proj / backward:down-proj / backward:up-proj
"""

import argparse
import os
import sys
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.cuda.nvtx as nvtx

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
import sonicmoe.functional as F
from sonicmoe.functional import clear_all_fp8_weight_caches


# ---------------------------------------------------------------------------
# NVTX instrumentation via monkey-patching
# ---------------------------------------------------------------------------

_orig_up_fwd = F._UpProjection.forward
_orig_up_bwd = F._UpProjection.backward
_orig_down_fwd = F._DownProjection.forward
_orig_down_bwd = F._DownProjection.backward
_orig_router_fwd = F.TC_Softmax_Topk_Router_Function.forward
_orig_router_bwd = F.TC_Softmax_Topk_Router_Function.backward


def _nvtx_up_fwd(ctx, *args, **kwargs):
    nvtx.range_push("forward:up-proj")
    result = _orig_up_fwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def _nvtx_up_bwd(ctx, *args, **kwargs):
    nvtx.range_push("backward:up-proj")
    result = _orig_up_bwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def _nvtx_down_fwd(ctx, *args, **kwargs):
    nvtx.range_push("forward:down-proj")
    result = _orig_down_fwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def _nvtx_down_bwd(ctx, *args, **kwargs):
    nvtx.range_push("backward:down-proj")
    result = _orig_down_bwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def _nvtx_router_fwd(ctx, *args, **kwargs):
    nvtx.range_push("forward:router")
    result = _orig_router_fwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def _nvtx_router_bwd(ctx, *args, **kwargs):
    nvtx.range_push("backward:router")
    result = _orig_router_bwd(ctx, *args, **kwargs)
    nvtx.range_pop()
    return result


def install_nvtx_hooks():
    """Monkey-patch autograd functions with NVTX markers."""
    F._UpProjection.forward = staticmethod(_nvtx_up_fwd)
    F._UpProjection.backward = staticmethod(_nvtx_up_bwd)
    F._DownProjection.forward = staticmethod(_nvtx_down_fwd)
    F._DownProjection.backward = staticmethod(_nvtx_down_bwd)
    F.TC_Softmax_Topk_Router_Function.forward = staticmethod(_nvtx_router_fwd)
    F.TC_Softmax_Topk_Router_Function.backward = staticmethod(_nvtx_router_bwd)


def restore_hooks():
    """Restore original functions."""
    F._UpProjection.forward = staticmethod(_orig_up_fwd)
    F._UpProjection.backward = staticmethod(_orig_up_bwd)
    F._DownProjection.forward = staticmethod(_orig_down_fwd)
    F._DownProjection.backward = staticmethod(_orig_down_bwd)
    F.TC_Softmax_Topk_Router_Function.forward = staticmethod(_orig_router_fwd)
    F.TC_Softmax_Topk_Router_Function.backward = staticmethod(_orig_router_bwd)


# ---------------------------------------------------------------------------
# Uniform routing stub (deterministic, bypasses actual router for profiling)
# ---------------------------------------------------------------------------

def make_uniform_router(T, K, E, device):
    sc = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    idx = ((tok * K + off) % E).to(torch.int32)

    class UniformRouter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, rl, Ea, Ka):
            ctx.save_for_backward(sc, idx)
            ctx.E = Ea
            ctx.d = rl.dtype
            return sc.clone(), idx.clone()

        @staticmethod
        def backward(ctx, gs, _):
            s, _ = ctx.saved_tensors
            return torch.zeros(s.size(0), ctx.E, dtype=ctx.d, device=s.device), None, None

    return UniformRouter


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(mode: str, T=8192, H=3072, I=1536, E=8, K=8,
                  warmup_iters=8, profile_iters=10, timing_iters=10):
    """
    Run a single benchmark configuration.
    mode: "bf16" or "fp8"
    """
    use_fp8 = (mode == "fp8")
    label = f"FP8-frontier" if use_fp8 else "BF16-baseline"
    print(f"\n{'='*60}")
    print(f"  {label}  (T={T}, H={H}, I={I}, E={E}, K={K})")
    print(f"{'='*60}")

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K,
        hidden_size=H, intermediate_size=I,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    moe.train()

    # Uniform routing
    UR = make_uniform_router(T, K, E, "cuda")
    orig_router = F.TC_Softmax_Topk_Router_Function
    F.TC_Softmax_Topk_Router_Function = UR

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)

    # --- Warmup (compile kernels, populate caches) ---
    print(f"Warming up ({warmup_iters} iters)...")
    for i in range(warmup_iters):
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        o, _ = moe(xi, use_fp8=use_fp8)
        o.backward(grad)
    torch.cuda.synchronize()
    print("Warmup done.")

    # --- Timing (cuda events, no nsys overhead) ---
    torch.cuda.reset_peak_memory_stats()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    se.record()
    for _ in range(timing_iters):
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        o, _ = moe(xi, use_fp8=use_fp8)
        o.backward(grad)
    ee.record()
    torch.cuda.synchronize()
    ms = se.elapsed_time(ee) / timing_iters
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"TIMING: {ms:.2f} ms/iter   peak_mem: {peak_gib:.2f} GiB")

    # --- Profiled region with NVTX markers ---
    install_nvtx_hooks()
    torch.cuda.profiler.start()
    nvtx.range_push(f"profile:{label}")
    for i in range(profile_iters):
        nvtx.range_push(f"iter:{i}")

        nvtx.range_push("forward")
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        o, _ = moe(xi, use_fp8=use_fp8)
        nvtx.range_pop()  # forward

        nvtx.range_push("backward")
        o.backward(grad)
        nvtx.range_pop()  # backward

        nvtx.range_pop()  # iter:N
    torch.cuda.synchronize()
    nvtx.range_pop()  # profile:label
    torch.cuda.profiler.stop()
    restore_hooks()
    print(f"NSYS profile done ({profile_iters} iters captured).")

    # Cleanup (precision validated by tests/fp8_large_project_contract_test.py)
    F.TC_Softmax_Topk_Router_Function = orig_router
    del moe, x, grad
    if use_fp8:
        clear_all_fp8_weight_caches()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser(description="SonicMoE NVTX Timeline Profiler")
    parser.add_argument("--mode", choices=["bf16", "fp8", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--timing-iters", type=int, default=10)
    args = parser.parse_args()

    if args.mode in ("bf16", "both"):
        run_benchmark("bf16", warmup_iters=args.warmup,
                      profile_iters=args.profile_iters,
                      timing_iters=args.timing_iters)
    if args.mode in ("fp8", "both"):
        run_benchmark("fp8", warmup_iters=args.warmup,
                      profile_iters=args.profile_iters,
                      timing_iters=args.timing_iters)


if __name__ == "__main__":
    main()
