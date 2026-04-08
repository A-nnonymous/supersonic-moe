#!/usr/bin/env python3
"""Verify memory optimization: compare FP8 vs BF16 peak memory.

Usage:
    CUDA_VISIBLE_DEVICES=X USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python tools/verify_memory_optimization.py

Runs in subprocess isolation to avoid FP8 mode contamination.
Reports forward peak, backward peak, and post-backward residual for both paths.
"""

import gc
import os
import subprocess
import sys


def run_single_case(mode: str, device_id: int = 0) -> dict:
    """Run a single memory measurement case in a subprocess."""
    script = f'''
import gc, os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{device_id}"
os.environ["USE_QUACK_GEMM"] = "1"
if "{mode}" == "fp8":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
else:
    os.environ["SONIC_MOE_FP8_MODE"] = "off"

torch.manual_seed(42)
device = "cuda"
MiB = 1024**2

# Ernie shape
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(
    num_experts=E, num_experts_per_tok=K,
    hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU,
    add_bias=False, std=0.02,
).to(device=device, dtype=torch.bfloat16)

x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)

# 2 warmup iterations
for _ in range(2):
    with enable_quack_gemm(True):
        o, _, _ = moe(x)
    o.backward(dout)
    x.grad = None
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

base_alloc = torch.cuda.memory_allocated() / MiB

# Measured iteration
with enable_quack_gemm(True):
    o, _, _ = moe(x)

torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / MiB
fwd_alloc = torch.cuda.memory_allocated() / MiB

torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / MiB
bwd_alloc = torch.cuda.memory_allocated() / MiB

# Clean up and check residual
x.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None
del o
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
residual = torch.cuda.memory_allocated() / MiB

print(f"MODE={mode}")
print(f"base_alloc={base_alloc:.2f}")
print(f"fwd_peak={fwd_peak:.2f}")
print(f"fwd_alloc={fwd_alloc:.2f}")
print(f"bwd_peak={bwd_peak:.2f}")
print(f"bwd_alloc={bwd_alloc:.2f}")
print(f"residual={residual:.2f}")
'''
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"[{mode}] FAILED:\n{result.stderr[-2000:]}")
        return {}

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print(f"=== Memory Verification (GPU {args.gpu}) ===\n")

    bf16 = run_single_case("bf16", args.gpu)
    fp8 = run_single_case("fp8", args.gpu)

    if not bf16 or not fp8:
        print("ERROR: One or both cases failed. Check output above.")
        return

    print(f"\n{'Metric':<25} {'BF16 (MiB)':>12} {'FP8 (MiB)':>12} {'Delta':>12} {'Delta%':>8}")
    print("-" * 70)
    for key in ["base_alloc", "fwd_peak", "fwd_alloc", "bwd_peak", "bwd_alloc", "residual"]:
        b = bf16.get(key, 0)
        f = fp8.get(key, 0)
        delta = f - b
        pct = (delta / b * 100) if b > 0 else 0
        marker = "***" if abs(delta) > 10 else ""
        print(f"{key:<25} {b:>12.2f} {f:>12.2f} {delta:>+12.2f} {pct:>+7.1f}% {marker}")

    fwd_net_bf16 = bf16.get("fwd_peak", 0) - bf16.get("base_alloc", 0)
    fwd_net_fp8 = fp8.get("fwd_peak", 0) - fp8.get("base_alloc", 0)
    bwd_net_bf16 = bf16.get("bwd_peak", 0) - bf16.get("fwd_alloc", 0)
    bwd_net_fp8 = fp8.get("bwd_peak", 0) - fp8.get("fwd_alloc", 0)

    print(f"\n{'Net forward alloc':<25} {fwd_net_bf16:>12.2f} {fwd_net_fp8:>12.2f} {fwd_net_fp8-fwd_net_bf16:>+12.2f}")
    print(f"{'Net backward alloc':<25} {bwd_net_bf16:>12.2f} {bwd_net_fp8:>12.2f} {bwd_net_fp8-bwd_net_bf16:>+12.2f}")

    res_delta = fp8.get("residual", 0) - bf16.get("residual", 0)
    print(f"\n{'Post-bwd residual delta':<25} {res_delta:>+12.2f} MiB")
    if abs(res_delta) < 5:
        print("  ✓ FP8 weight caches properly cleaned (no persistent overhead)")
    elif res_delta > 50:
        print(f"  ✗ FP8 has +{res_delta:.0f} MiB persistent overhead (weight caches not evicted?)")

    # Overall verdict
    bwd_peak_delta = fp8.get("bwd_peak", 0) - bf16.get("bwd_peak", 0)
    print(f"\n=== VERDICT ===")
    print(f"BF16 backward peak: {bf16.get('bwd_peak', 0):.1f} MiB")
    print(f"FP8  backward peak: {fp8.get('bwd_peak', 0):.1f} MiB")
    if bwd_peak_delta < -20:
        print(f"  ✓ FP8 saves {-bwd_peak_delta:.1f} MiB ({-bwd_peak_delta/bf16.get('bwd_peak',1)*100:.1f}%) — GOAL ACHIEVED")
    elif bwd_peak_delta < 20:
        print(f"  ≈ FP8 roughly at parity ({bwd_peak_delta:+.1f} MiB)")
    else:
        print(f"  ✗ FP8 uses {bwd_peak_delta:.1f} MiB MORE than BF16")


if __name__ == "__main__":
    main()
