#!/usr/bin/env python3
"""
Rigorous ground-truth measurement: BF16 vs FP8 at Ernie production shape.

Measures in a SINGLE clean process:
  1. Precision: RRMSE + correlation (FP8 vs BF16, same weights, 3 seeds)
  2. Performance: CUDA event timing (warmup + 20 measured iters)
  3. Memory: peak memory per mode (separate measurement phases with full cleanup)

Usage:
  CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/ground_truth_measure.py
"""

import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.cuda.nvtx as nvtx

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm
import sonicmoe.functional as F
from sonicmoe.functional import clear_all_fp8_weight_caches


# ── Ernie production shape ──
T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP = 10
MEASURE = 20
SEEDS = [42, 123, 777]


def reset_all():
    clear_all_fp8_weight_caches()
    F._ALIGNMENT_ASSUMED = False
    F._ALIGNMENT_STREAK = 0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_model():
    torch.manual_seed(42)
    return MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
        activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)


def run_iter(moe, x, grad, use_fp8):
    moe.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    with enable_quack_gemm(True):
        o, _ = moe(xi, use_fp8=use_fp8)
    o.backward(grad)
    return o, xi


def measure_perf(moe, x, grad, use_fp8, warmup=WARMUP, measure=MEASURE):
    """Returns (avg_ms, peak_gib) with separate warmup."""
    # Warmup
    for _ in range(warmup):
        run_iter(moe, x, grad, use_fp8)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measure):
        run_iter(moe, x, grad, use_fp8)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / measure
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return avg_ms, peak_mib


def measure_precision(moe, x, grad, use_fp8):
    """Returns (fwd_rrmse, fwd_corr, bwd_rrmse, bwd_corr)."""
    reset_all()
    moe.zero_grad(set_to_none=True)
    xb = x.clone().requires_grad_(True)
    with enable_quack_gemm(True):
        ob, _ = moe(xb)
    ob.backward(grad)
    dx_bf16 = xb.grad.float()
    out_bf16 = ob.float()

    reset_all()
    # Need alignment warmup (3+ aligned iters for streak)
    for _ in range(4):
        run_iter(moe, x, grad, True)

    moe.zero_grad(set_to_none=True)
    xf = x.clone().requires_grad_(True)
    with enable_quack_gemm(True):
        of, _ = moe(xf, use_fp8=True)
    of.backward(grad)
    dx_fp8 = xf.grad.float()
    out_fp8 = of.float()

    fwd_rrmse = ((out_fp8 - out_bf16).norm() / out_bf16.norm()).item()
    bwd_rrmse = ((dx_fp8 - dx_bf16).norm() / dx_bf16.norm()).item()
    fwd_corr = torch.corrcoef(torch.stack([out_fp8.flatten(), out_bf16.flatten()]))[0, 1].item()
    bwd_corr = torch.corrcoef(torch.stack([dx_fp8.flatten(), dx_bf16.flatten()]))[0, 1].item()
    return fwd_rrmse, fwd_corr, bwd_rrmse, bwd_corr


def nsys_profiled_run(moe, x, grad, use_fp8, label, iters=10):
    """Run iters under NVTX markers + cudaProfiler for nsys capture."""
    # Install NVTX hooks on autograd functions
    orig_up_fwd = F._UpProjection.forward
    orig_up_bwd = F._UpProjection.backward
    orig_down_fwd = F._DownProjection.forward
    orig_down_bwd = F._DownProjection.backward

    def nvtx_up_fwd(ctx, *a, **kw):
        nvtx.range_push("forward:up-proj")
        r = orig_up_fwd(ctx, *a, **kw)
        nvtx.range_pop()
        return r

    def nvtx_up_bwd(ctx, *a, **kw):
        nvtx.range_push("backward:up-proj")
        r = orig_up_bwd(ctx, *a, **kw)
        nvtx.range_pop()
        return r

    def nvtx_down_fwd(ctx, *a, **kw):
        nvtx.range_push("forward:down-proj")
        r = orig_down_fwd(ctx, *a, **kw)
        nvtx.range_pop()
        return r

    def nvtx_down_bwd(ctx, *a, **kw):
        nvtx.range_push("backward:down-proj")
        r = orig_down_bwd(ctx, *a, **kw)
        nvtx.range_pop()
        return r

    F._UpProjection.forward = staticmethod(nvtx_up_fwd)
    F._UpProjection.backward = staticmethod(nvtx_up_bwd)
    F._DownProjection.forward = staticmethod(nvtx_down_fwd)
    F._DownProjection.backward = staticmethod(nvtx_down_bwd)

    torch.cuda.profiler.start()
    nvtx.range_push(f"profile:{label}")
    for i in range(iters):
        nvtx.range_push(f"iter:{i}")
        nvtx.range_push("forward")
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        with enable_quack_gemm(True):
            o, _ = moe(xi, use_fp8=use_fp8)
        nvtx.range_pop()  # forward
        nvtx.range_push("backward")
        o.backward(grad)
        nvtx.range_pop()  # backward
        nvtx.range_pop()  # iter:N
    torch.cuda.synchronize()
    nvtx.range_pop()  # profile:label
    torch.cuda.profiler.stop()

    # Restore
    F._UpProjection.forward = staticmethod(orig_up_fwd)
    F._UpProjection.backward = staticmethod(orig_up_bwd)
    F._DownProjection.forward = staticmethod(orig_down_fwd)
    F._DownProjection.backward = staticmethod(orig_down_bwd)


def main():
    print("=" * 70)
    print(f"  SonicMoE Ground Truth Measurement")
    print(f"  Shape: T={T} H={H} I={I} E={E} K={K}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Warmup={WARMUP}, Measure={MEASURE}")
    print("=" * 70)

    moe = make_model()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)

    # ── 1. PRECISION (multi-seed) ──
    print("\n── PRECISION ──")
    for seed in SEEDS:
        torch.manual_seed(seed)
        xs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        gs = torch.randn_like(xs)
        fr, fc, br, bc = measure_precision(moe, xs, gs, True)
        status_f = "✓" if fr < 0.10 and fc > 0.99 else "✗"
        status_b = "✓" if br < 0.10 and bc > 0.99 else "✗"
        print(f"  seed={seed}: fwd RRMSE={fr*100:.2f}% corr={fc:.6f} [{status_f}]  "
              f"bwd RRMSE={br*100:.2f}% corr={bc:.6f} [{status_b}]")

    # ── 2. PERFORMANCE ──
    print("\n── PERFORMANCE ──")
    reset_all()
    bf16_ms, bf16_mib = measure_perf(moe, x, grad, use_fp8=False)
    print(f"  BF16: {bf16_ms:.2f} ms/iter   peak={bf16_mib:.0f} MiB")

    reset_all()
    fp8_ms, fp8_mib = measure_perf(moe, x, grad, use_fp8=True)
    print(f"  FP8:  {fp8_ms:.2f} ms/iter   peak={fp8_mib:.0f} MiB")

    speedup = bf16_ms / fp8_ms
    mem_ratio = fp8_mib / bf16_mib
    print(f"\n  Speedup:  {speedup:.3f}x  ({'faster' if speedup > 1 else 'SLOWER'})")
    print(f"  Memory:   {mem_ratio:.3f}x  ({'less' if mem_ratio < 1 else 'MORE'} than BF16)")
    print(f"  Δ memory: {(1-mem_ratio)*100:+.1f}%")

    # ── 3. NSYS PROFILED RUN (if under nsys) ──
    if os.environ.get("_NSYS_CAPTURE", ""):
        print("\n── NSYS CAPTURE ──")
        reset_all()
        for _ in range(WARMUP):
            run_iter(moe, x, grad, False)
        torch.cuda.synchronize()
        nsys_profiled_run(moe, x, grad, False, "BF16", iters=10)
        print("  BF16 capture done")

        reset_all()
        for _ in range(WARMUP):
            run_iter(moe, x, grad, True)
        torch.cuda.synchronize()
        nsys_profiled_run(moe, x, grad, True, "FP8", iters=10)
        print("  FP8 capture done")

    # ── Summary line (machine-parseable) ──
    print(f"\nSUMMARY: bf16={bf16_ms:.2f}ms fp8={fp8_ms:.2f}ms speedup={speedup:.3f}x "
          f"bf16_mem={bf16_mib:.0f}MiB fp8_mem={fp8_mib:.0f}MiB mem_ratio={mem_ratio:.3f}")


if __name__ == "__main__":
    main()
