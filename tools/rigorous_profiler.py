#!/usr/bin/env python3
"""Rigorous GPU-projection kernel profiling + fine-grained memory lifecycle.

Subprocess-isolated per mode (BF16 / FP8). Designed to run on a truly idle GPU.
Outputs JSON with:
  1. Per-kernel CUDA time (torch.profiler, GPU projection only)
  2. Memory lifecycle at sub-stage granularity (torch.cuda.memory_stats)
  3. torch.cuda.memory_summary() snapshots at key points

Usage (local):
  CUDA_VISIBLE_DEVICES=0 python tools/rigorous_profiler.py

Usage (remote idle node via SSH):
  ssh <idle_node_ip> "cd /path/to/sonic-moe && \
    source /path/to/envs/xfer/bin/activate && \
    CUDA_VISIBLE_DEVICES=0 python tools/rigorous_profiler.py"
"""

import json
import os
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# Inner script: runs inside a clean subprocess per mode
# ═══════════════════════════════════════════════════════════════════════════════

_INNER_SCRIPT = r'''
import gc, json, os, sys, time, torch
from collections import defaultdict

MODE = os.environ["_PROFILER_MODE"]
T, H, I, E, K = 8192, 3072, 1536, 8, 8
TK = T * K
device = torch.device("cuda:0")
torch.cuda.set_device(device)

WARMUP_ITERS = 5
PROFILE_ITERS = 10
TIMING_ITERS = 30

# ───────── Memory helpers ─────────

def _mem_mib():
    """Current allocated memory in MiB."""
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated() / (1024**2), 4)

def _peak_mib():
    """Peak allocated memory since last reset, in MiB."""
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / (1024**2), 4)

def _reserved_mib():
    """Current reserved (cached) memory in MiB."""
    torch.cuda.synchronize()
    return round(torch.cuda.memory_reserved() / (1024**2), 4)

def _mem_snapshot():
    """Return a dict of key memory stats at this instant."""
    torch.cuda.synchronize()
    s = torch.cuda.memory_stats()
    return {
        "allocated_MiB": round(s["allocated_bytes.all.current"] / (1024**2), 4),
        "peak_allocated_MiB": round(s["allocated_bytes.all.peak"] / (1024**2), 4),
        "reserved_MiB": round(s["reserved_bytes.all.current"] / (1024**2), 4),
        "peak_reserved_MiB": round(s["reserved_bytes.all.peak"] / (1024**2), 4),
        "num_allocs": s["allocation.all.current"],
        "num_alloc_total": s["allocation.all.allocated"],
    }

def _reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

def _clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def tensor_mib(t):
    if t is None:
        return 0.0
    return round(t.storage().nbytes() / (1024**2), 4)

# ───────── Model setup ─────────

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

_clean()
base_snap = _mem_snapshot()

torch.manual_seed(42)
model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)
model_snap = _mem_snapshot()

param_sizes = {}
for name, p in model.named_parameters():
    param_sizes[name] = tensor_mib(p.data)

x = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
dout = 0.02 * torch.randn_like(x)
input_snap = _mem_snapshot()

use_fp8 = (MODE == "fp8")

# ───────── Warmup ─────────
for _wi in range(WARMUP_ITERS):
    xw = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ow, lw = model(xw, use_fp8=True)
        else:
            ow, lw = model(xw)
    (ow.sum() + lw).backward()
    model.zero_grad(set_to_none=True)
    del ow, lw, xw
_clean()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Fine-grained memory lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

_reset_peak()
pre_fwd_snap = _mem_snapshot()

# Forward pass
with enable_quack_gemm(True):
    if use_fp8:
        with enable_fp8(True):
            out, loss_val = model(x, use_fp8=True)
    else:
        out, loss_val = model(x)
torch.cuda.synchronize()
post_fwd_snap = _mem_snapshot()

_reset_peak()
pre_bwd_snap = _mem_snapshot()

# Backward pass
loss = out.sum() + loss_val
loss.backward()
torch.cuda.synchronize()
post_bwd_snap = _mem_snapshot()

# Collect grad sizes
grad_sizes = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_sizes[name] = tensor_mib(p.grad)

# Collect FP8 weight cache sizes
fp8_cache_after_bwd = {}
if use_fp8:
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _VARLEN_WEIGHT_CACHE, _FUSED_WEIGHT_CACHE,
    )
    for cname, cache in [("VARLEN", _VARLEN_WEIGHT_CACHE), ("FUSED", _FUSED_WEIGHT_CACHE)]:
        for k, v in cache.items():
            entry_total = sum(tensor_mib(t) for t in v if isinstance(t, torch.Tensor))
            shape_key = str(k[2])
            fp8_cache_after_bwd[f"{cname}_{shape_key}"] = entry_total

# Cleanup
del out, loss, loss_val
x.grad = None
model.zero_grad(set_to_none=True)
_clean()
cleanup_snap = _mem_snapshot()

# torch.cuda.memory_summary for the final report
_reset_peak()
x2 = x.detach().clone().requires_grad_(True)
with enable_quack_gemm(True):
    if use_fp8:
        with enable_fp8(True):
            out2, lv2 = model(x2, use_fp8=True)
    else:
        out2, lv2 = model(x2)
(out2.sum() + lv2).backward()
torch.cuda.synchronize()
mem_summary_text = torch.cuda.memory_summary(abbreviated=False)
del out2, lv2, x2
model.zero_grad(set_to_none=True)
_clean()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Kernel profiling (GPU-projection, torch.profiler CUDA activity)
# ═══════════════════════════════════════════════════════════════════════════════

from torch.profiler import profile, ProfilerActivity

kernel_samples = defaultdict(lambda: {"total_us": 0.0, "count": 0, "samples": []})

for trial in range(PROFILE_ITERS):
    xp = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    op, lp = model(xp, use_fp8=True)
            else:
                op, lp = model(xp)
        (op.sum() + lp).backward()
        torch.cuda.synchronize()

    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            name = evt.key
            cuda_us = evt.self_device_time_total  # GPU-projection µs
            cnt = evt.count
            kernel_samples[name]["total_us"] += cuda_us
            kernel_samples[name]["count"] += cnt
            kernel_samples[name]["samples"].append(cuda_us)
    del xp, op, lp

# Compute per-kernel median and mean across trials
kernel_list = []
for kname, kdata in sorted(kernel_samples.items(), key=lambda x: -x[1]["total_us"]):
    avg_us = kdata["total_us"] / PROFILE_ITERS
    avg_count = kdata["count"] / PROFILE_ITERS
    samples = kdata["samples"]
    samples.sort()
    n = len(samples)
    median_us = samples[n // 2] if n % 2 == 1 else (samples[n // 2 - 1] + samples[n // 2]) / 2
    kernel_list.append({
        "name": kname,
        "avg_cuda_us": round(avg_us, 2),
        "median_cuda_us": round(median_us, 2),
        "avg_count": round(avg_count, 1),
        "std_us": round(
            (sum((s - avg_us) ** 2 for s in samples) / max(len(samples) - 1, 1)) ** 0.5, 2
        ),
    })

total_cuda_us = sum(k["avg_cuda_us"] for k in kernel_list)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Wall-clock timing (CUDA Events, for reference only)
# ═══════════════════════════════════════════════════════════════════════════════

fwd_times, bwd_times, total_times = [], [], []
for _ in range(TIMING_ITERS):
    xt = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    s2 = torch.cuda.Event(enable_timing=True)
    s0.record()
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ot, lt = model(xt, use_fp8=True)
        else:
            ot, lt = model(xt)
    s1.record()
    (ot.sum() + lt).backward()
    s2.record()
    torch.cuda.synchronize()
    fwd_times.append(s0.elapsed_time(s1))
    bwd_times.append(s1.elapsed_time(s2))
    total_times.append(s0.elapsed_time(s2))
    del xt, ot, lt

fwd_times.sort()
bwd_times.sort()
total_times.sort()
n = len(fwd_times)
med = lambda lst: lst[n // 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Theoretical tensor sizes (for cross-reference)
# ═══════════════════════════════════════════════════════════════════════════════

def isa_scale_mib(rows, cols):
    row_t = (rows + 127) // 128
    col_t = (cols + 127) // 128
    return round(row_t * col_t * 512 / (1024**2), 4)

theory = {
    "x_input_bf16": round(T * H * 2 / (1024**2), 2),
    "z_bf16_TK_2I": round(TK * 2 * I * 2 / (1024**2), 2),
    "z_fp8_TK_2I": round(TK * 2 * I * 1 / (1024**2), 2),
    "z_scales": isa_scale_mib(TK, 2 * I),
    "y1_bf16_TK_I": round(TK * I * 2 / (1024**2), 2),
    "y1_fp8_TK_I": round(TK * I * 1 / (1024**2), 2),
    "y1_scales": isa_scale_mib(TK, I),
    "y2_bf16_TK_H": round(TK * H * 2 / (1024**2), 2),
    "o_bf16_T_H": round(T * H * 2 / (1024**2), 2),
    "dz_bf16_TK_2I": round(TK * 2 * I * 2 / (1024**2), 2),
    "dz_fp8_TK_2I": round(TK * 2 * I * 1 / (1024**2), 2),
    "dz_scales": isa_scale_mib(TK, 2 * I),
    "dx_expanded_TK_H": round(TK * H * 2 / (1024**2), 2),
    "dx_reduced_T_H": round(T * H * 2 / (1024**2), 2),
    "dw1_E_2I_H": round(E * 2 * I * H * 2 / (1024**2), 2),
    "dw2_E_I_H": round(E * I * H * 2 / (1024**2), 2),
    "x_fp8_T_H": round(T * H * 1 / (1024**2), 2),
    "x_scales_T": isa_scale_mib(T, H),
    "x_scales_TK": isa_scale_mib(TK, H),
    "dout_fp8_T_H": round(T * H * 1 / (1024**2), 2),
    "dout_scales_TK": isa_scale_mib(TK, H),
    "w1_fp8_E_2I_H": round(E * 2 * I * H * 1 / (1024**2), 2),
    "w1_fp8_scales": round(isa_scale_mib(2 * I, H) * E, 4),
    "w2_fp8_E_I_H": round(E * I * H * 1 / (1024**2), 2),
    "w2_fp8_scales": round(isa_scale_mib(I, H) * E, 4),
    "w1T_fp8_E_H_2I": round(E * H * 2 * I * 1 / (1024**2), 2),
    "routing_metadata_total": round((TK * 4 * 5 + T * K * 8 + (E + 1) * 4) / (1024**2), 2),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Assemble output
# ═══════════════════════════════════════════════════════════════════════════════

result = {
    "mode": MODE,
    "shape": {"T": T, "H": H, "I": I, "E": E, "K": K, "TK": TK},
    "memory_lifecycle": {
        "snapshots": {
            "base": base_snap,
            "after_model": model_snap,
            "after_input": input_snap,
            "pre_fwd": pre_fwd_snap,
            "post_fwd": post_fwd_snap,
            "pre_bwd": pre_bwd_snap,
            "post_bwd": post_bwd_snap,
            "cleanup": cleanup_snap,
        },
        "deltas_MiB": {
            "model_params": round(
                model_snap["allocated_MiB"] - base_snap["allocated_MiB"], 4),
            "input_tensors": round(
                input_snap["allocated_MiB"] - model_snap["allocated_MiB"], 4),
            "fwd_activation": round(
                post_fwd_snap["allocated_MiB"] - pre_fwd_snap["allocated_MiB"], 4),
            "fwd_peak_above_pre": round(
                post_fwd_snap["peak_allocated_MiB"] - pre_fwd_snap["allocated_MiB"], 4),
            "bwd_peak_above_pre": round(
                post_bwd_snap["peak_allocated_MiB"] - pre_bwd_snap["allocated_MiB"], 4),
            "bwd_residual": round(
                post_bwd_snap["allocated_MiB"] - pre_bwd_snap["allocated_MiB"], 4),
        },
        "param_sizes_MiB": param_sizes,
        "grad_sizes_MiB": grad_sizes,
        "fp8_cache_after_bwd": fp8_cache_after_bwd,
        "theoretical_sizes_MiB": theory,
    },
    "kernel_profiling": {
        "total_cuda_us": round(total_cuda_us, 2),
        "profile_iters": PROFILE_ITERS,
        "kernels": kernel_list,
    },
    "wall_clock_ms": {
        "median_fwd_ms": round(med(fwd_times), 4),
        "median_bwd_ms": round(med(bwd_times), 4),
        "median_total_ms": round(med(total_times), 4),
        "timing_iters": TIMING_ITERS,
    },
    "memory_summary_text": mem_summary_text,
}

print(json.dumps(result, indent=2))
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Outer driver: subprocess isolation per mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_mode(mode: str, gpu_id: str = "0") -> dict | None:
    env = os.environ.copy()
    env["_PROFILER_MODE"] = mode
    env["USE_QUACK_GEMM"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Clean FP8 env vars
    if mode == "fp8":
        env["SONIC_MOE_FP8_MODE"] = "perf"
        env["SONIC_MOE_FP8_DOUBLE_QUANT"] = "1"
    else:
        env.pop("SONIC_MOE_FP8_MODE", None)
        env.pop("SONIC_MOE_FP8_DOUBLE_QUANT", None)

    print(f"  [{mode.upper()}] Starting subprocess (GPU {gpu_id})...", file=sys.stderr)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-c", _INNER_SCRIPT],
        env=env, capture_output=True, text=True, timeout=900,
    )
    elapsed = time.time() - t0
    print(f"  [{mode.upper()}] Completed in {elapsed:.1f}s (exit={r.returncode})",
          file=sys.stderr)

    if r.returncode != 0:
        print(f"  [{mode.upper()}] FAILED:\n{r.stderr[-4000:]}", file=sys.stderr)
        return None
    if r.stderr:
        # Print first few lines of stderr (CUTLASS JIT, etc.)
        lines = r.stderr.strip().split("\n")
        for line in lines[:5]:
            print(f"  [{mode.upper()}] {line}", file=sys.stderr)
        if len(lines) > 5:
            print(f"  [{mode.upper()}] ... ({len(lines) - 5} more lines)", file=sys.stderr)
    try:
        return json.loads(r.stdout.strip())
    except Exception as e:
        print(f"  [{mode.upper()}] JSON parse error: {e}", file=sys.stderr)
        print(f"  stdout tail: {r.stdout[-2000:]}", file=sys.stderr)
        return None


import time


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    print("=" * 70, file=sys.stderr)
    print("SonicMoE Rigorous Profiler (subprocess-isolated)", file=sys.stderr)
    print(f"  GPU: {gpu}  |  Shape: T=8192 H=3072 I=1536 E=8 K=8", file=sys.stderr)
    print(f"  Warmup: 5  |  Profile: 10  |  Timing: 30", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    bf16 = run_mode("bf16", gpu)
    fp8 = run_mode("fp8", gpu)

    # Assemble combined output
    combined = {"bf16": bf16, "fp8": fp8}

    # Print a summary to stderr
    if bf16 and fp8:
        print("\n" + "=" * 70, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for label, data in [("BF16", bf16), ("FP8", fp8)]:
            kp = data["kernel_profiling"]
            wc = data["wall_clock_ms"]
            ml = data["memory_lifecycle"]["deltas_MiB"]
            print(f"  {label}:", file=sys.stderr)
            print(f"    CUDA kernel total:  {kp['total_cuda_us']:.1f} µs", file=sys.stderr)
            print(f"    Wall-clock (median): fwd={wc['median_fwd_ms']:.3f} "
                  f"bwd={wc['median_bwd_ms']:.3f} total={wc['median_total_ms']:.3f} ms",
                  file=sys.stderr)
            print(f"    Memory: fwd_peak={ml['fwd_peak_above_pre']:.1f} "
                  f"bwd_peak={ml['bwd_peak_above_pre']:.1f} MiB", file=sys.stderr)
        sp = bf16["kernel_profiling"]["total_cuda_us"] / fp8["kernel_profiling"]["total_cuda_us"]
        print(f"\n  GPU-projection speedup: {sp:.3f}×", file=sys.stderr)
        sp_wc = bf16["wall_clock_ms"]["median_total_ms"] / fp8["wall_clock_ms"]["median_total_ms"]
        print(f"  Wall-clock speedup:    {sp_wc:.3f}×", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    # JSON to stdout
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
