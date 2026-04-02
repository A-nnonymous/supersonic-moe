#!/usr/bin/env python3
"""Final benchmark: precision, performance, and memory for FP8 vs BF16.

Measures at both contract shape (T=1024) and production shape (T=8192).
Outputs machine-readable results to stdout.
"""
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.pop("SONIC_MOE_FP8_MODE", None)
os.environ["USE_QUACK_GEMM"] = "1"

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe import functional

WARMUP = 5
ITERS = 20
SHAPES = [
    {"name": "contract", "T": 1024, "H": 3072, "I": 1536, "E": 8, "K": 8},
    {"name": "ernie_prod", "T": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8},
]


def reset():
    functional.clear_all_fp8_weight_caches()
    functional._ALIGNMENT_ASSUMED = False
    functional._ALIGNMENT_STREAK = 0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_precision(shape):
    """Measure forward + backward precision of FP8 vs BF16."""
    T, H, I, E, K = shape["T"], shape["H"], shape["I"], shape["E"], shape["K"]
    reset()
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()

    x1 = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = 0.02 * torch.randn_like(x1)

    # BF16 reference
    reset()
    with enable_quack_gemm(True):
        z_bf16, _ = moe(x1)
    z_bf16.backward(dout)
    dx_bf16 = x1.grad.clone()
    x1.grad = None
    moe.zero_grad(set_to_none=True)

    # FP8
    reset()
    x2 = x1.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True), enable_fp8():
        z_fp8, _ = moe(x2, use_fp8=True)
    z_fp8.backward(dout)
    dx_fp8 = x2.grad.clone()

    def rrmse(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def corr(a, b):
        return torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()

    return {
        "fwd_rrmse": rrmse(z_fp8, z_bf16),
        "fwd_corr": corr(z_fp8, z_bf16),
        "bwd_rrmse": rrmse(dx_fp8, dx_bf16),
        "bwd_corr": corr(dx_fp8, dx_bf16),
    }


def measure_perf_memory(shape, mode):
    """Measure wall-clock time and peak memory for a given mode (bf16/fp8)."""
    T, H, I, E, K = shape["T"], shape["H"], shape["I"], shape["E"], shape["K"]
    reset()
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    use_fp8 = mode == "fp8"

    # Warmup (let JIT compile caches warm up, don't clear them)
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

    # Timed runs — do NOT clear compile caches
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(ITERS):
        if use_fp8:
            with enable_quack_gemm(True), enable_fp8():
                z, _ = moe(x, use_fp8=True)
        else:
            with enable_quack_gemm(True):
                z, _ = moe(x)
        z.sum().backward()
        x.grad = None
        moe.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return {
        "wall_ms": (elapsed / ITERS) * 1000,
        "peak_mem_mb": peak_mem,
    }


def main():
    print("=" * 72)
    print("FINAL BENCHMARK: FP8 vs BF16")
    print("=" * 72)

    for shape in SHAPES:
        name = shape["name"]
        print(f"\n{'─' * 72}")
        print(f"Shape: {name} — T={shape['T']}, H={shape['H']}, I={shape['I']}, E={shape['E']}, K={shape['K']}")
        print(f"{'─' * 72}")

        # Precision
        prec = measure_precision(shape)
        print(f"  Precision:")
        print(f"    FWD: RRMSE={prec['fwd_rrmse']:.4f} corr={prec['fwd_corr']:.6f}  {'✓' if prec['fwd_rrmse'] < 0.10 and prec['fwd_corr'] > 0.99 else '✗'}")
        print(f"    BWD: RRMSE={prec['bwd_rrmse']:.4f} corr={prec['bwd_corr']:.6f}  {'✓' if prec['bwd_rrmse'] < 0.10 and prec['bwd_corr'] > 0.99 else '✗'}")

        # BF16 perf
        bf16 = measure_perf_memory(shape, "bf16")
        print(f"  BF16: {bf16['wall_ms']:.2f} ms/iter, peak {bf16['peak_mem_mb']:.0f} MB")

        # FP8 perf
        fp8 = measure_perf_memory(shape, "fp8")
        print(f"  FP8:  {fp8['wall_ms']:.2f} ms/iter, peak {fp8['peak_mem_mb']:.0f} MB")

        speedup = bf16["wall_ms"] / fp8["wall_ms"]
        mem_ratio = fp8["peak_mem_mb"] / bf16["peak_mem_mb"]
        print(f"  Speedup: {speedup:.2f}×")
        print(f"  Memory:  {mem_ratio:.3f}× (FP8/BF16)  {'✓ savings' if mem_ratio < 1.0 else '✗ regression'}")

    print(f"\n{'=' * 72}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
