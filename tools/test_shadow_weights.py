#!/usr/bin/env python3
"""Verify FP8 shadow weights: correctness + performance + memory."""
import gc, os, sys, time, torch

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

torch.manual_seed(42)
device = "cuda"
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02
).to(device=device, dtype=torch.bfloat16)

x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)

# Warmup (JIT compile)
print("Warming up...")
for _ in range(2):
    with enable_quack_gemm(True):
        o = moe(x, use_fp8=True)[0]
    o.backward(dout)
    x.grad = None
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None

gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print("Warmup done.\n")

# === Test 1: Correctness — compare output with and without shadow weights ===
print("=== Test 1: Correctness ===")
# Run without shadow (normal cache path)
moe.clear_fp8_weight_cache()
x1 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o1 = moe(x1, use_fp8=True)[0]
o1.backward(dout)
out_no_shadow = o1.detach().float().cpu()
dx_no_shadow = x1.grad.detach().float().cpu()
x1.grad = None
for p in moe.parameters():
    if p.grad is not None: p.grad = None

# Run with shadow weights
moe.refresh_fp8_shadow_weights()
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
o2.backward(dout)
out_shadow = o2.detach().float().cpu()
dx_shadow = x2.grad.detach().float().cpu()
x2.grad = None
for p in moe.parameters():
    if p.grad is not None: p.grad = None

# Compare
import numpy as np
for name, a, b in [("output", out_no_shadow, out_shadow), ("dx", dx_no_shadow, dx_shadow)]:
    diff = (a - b).abs().max().item()
    rrmse = torch.sqrt(torch.mean((a - b) ** 2)).item() / torch.sqrt(torch.mean(a ** 2)).item() * 100
    print(f"  {name}: max_diff={diff:.2e}, RRMSE={rrmse:.4f}%  {'✓ BIT-IDENTICAL' if diff == 0 else '✓ MATCH' if rrmse < 0.01 else '✗ MISMATCH'}")

# === Test 2: Performance — timing with vs without shadow ===
print("\n=== Test 2: Performance (20 iters, CUDA events) ===")
for mode_name, use_shadow in [("no_shadow", False), ("with_shadow", True)]:
    moe.clear_fp8_weight_cache()
    if use_shadow:
        moe.refresh_fp8_shadow_weights()

    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        xi = x.detach().clone().requires_grad_()
        for p in moe.parameters():
            if p.grad is not None: p.grad = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with enable_quack_gemm(True):
            oi = moe(xi, use_fp8=True)[0]
        oi.backward(dout)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del oi, xi

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {mode_name:15s}: {avg:.2f} ms ± {std:.2f}")

# === Test 3: Memory — peak with vs without shadow ===
print("\n=== Test 3: Memory ===")
for mode_name, use_shadow in [("no_shadow", False), ("with_shadow", True)]:
    moe.clear_fp8_weight_cache()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    if use_shadow:
        moe.refresh_fp8_shadow_weights()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / MiB

    xi = x.detach().clone().requires_grad_()
    for p in moe.parameters():
        if p.grad is not None: p.grad = None

    with enable_quack_gemm(True):
        oi = moe(xi, use_fp8=True)[0]
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() / MiB

    torch.cuda.reset_peak_memory_stats()
    oi.backward(dout)
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated() / MiB

    xi.grad = None
    for p in moe.parameters():
        if p.grad is not None: p.grad = None
    del oi, xi
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    residual = torch.cuda.memory_allocated() / MiB

    print(f"  {mode_name:15s}: base={base:.1f}M  fwd_peak={fwd_peak:.1f}M  bwd_peak={bwd_peak:.1f}M  residual={residual:.1f}M")

print("\nDone.")
