#!/usr/bin/env python3
"""Rigorous memory breakdown for FP8 vs BF16 SonicMoE.

Runs BF16 and FP8 in separate subprocesses to avoid process contamination.
Measures at key lifecycle checkpoints via torch.cuda.memory_allocated().

Usage:
  CUDA_VISIBLE_DEVICES=4 python tools/_memory_breakdown.py
"""
import json
import os
import subprocess
import sys

# The inner script runs inside a clean subprocess per mode.
_INNER_SCRIPT = r'''
import gc, json, os, sys, torch

MODE = os.environ["_MEM_MODE"]
T, H, I, E, K = 8192, 3072, 1536, 8, 8
TK = T * K
device = torch.device("cuda:0")
torch.cuda.set_device(device)

def m():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated() / (1024**2), 2)

def pk():
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)

def tensor_mib(t):
    if t is None: return 0.0
    return round(t.storage().nbytes() / (1024**2), 4)

def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

clean()
base = m()

torch.manual_seed(42)
model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)
after_model = m()

param_sizes = {}
for name, p in model.named_parameters():
    param_sizes[name] = tensor_mib(p.data)

x = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
dout = 0.02 * torch.randn_like(x)
after_input = m()

use_fp8 = (MODE == "fp8")

# Warmup (compile kernels, build caches)
for _wi in range(2):
    x_w = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                o_w, l_w = model(x_w, use_fp8=True)
        else:
            o_w, l_w = model(x_w)
    (o_w.sum() + l_w).backward()
    model.zero_grad(set_to_none=True)
    del o_w, l_w, x_w

gc.collect()
torch.cuda.empty_cache()

# ---------- CLEAN MEASUREMENT: FORWARD ----------
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
pre_fwd = m()

with enable_quack_gemm(True):
    if use_fp8:
        with enable_fp8(True):
            out, loss_val = model(x, use_fp8=True)
    else:
        out, loss_val = model(x)

torch.cuda.synchronize()
post_fwd = m()
peak_fwd = pk()

# Check FP8 weight caches
cache_report = {}
if use_fp8:
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _VARLEN_WEIGHT_CACHE, _FUSED_WEIGHT_CACHE,
    )
    for cname, cache in [("VARLEN", _VARLEN_WEIGHT_CACHE), ("FUSED", _FUSED_WEIGHT_CACHE)]:
        for k, v in cache.items():
            entry_total = sum(tensor_mib(t) for t in v if isinstance(t, torch.Tensor))
            shape_key = str(k[2])  # shape tuple
            cache_report[f"{cname}_{shape_key}"] = entry_total

# ---------- CLEAN MEASUREMENT: BACKWARD ----------
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
pre_bwd = m()

loss = out.sum() + loss_val
loss.backward()

torch.cuda.synchronize()
post_bwd = m()
peak_bwd = pk()

grad_sizes = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_sizes[name] = tensor_mib(p.grad)

# Check FP8 caches after backward
cache_after_bwd = {}
if use_fp8:
    for cname, cache in [("VARLEN", _VARLEN_WEIGHT_CACHE), ("FUSED", _FUSED_WEIGHT_CACHE)]:
        for k, v in cache.items():
            entry_total = sum(tensor_mib(t) for t in v if isinstance(t, torch.Tensor))
            shape_key = str(k[2])
            cache_after_bwd[f"{cname}_{shape_key}"] = entry_total

# ---------- CLEANUP ----------
del out, loss, loss_val
x.grad = None
model.zero_grad(set_to_none=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
post_cleanup = m()

# Theoretical sizes
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
    "colvec_reduce_partial_fp32": round(TK * 48 * 4 / (1024**2), 2),
}

result = {
    "mode": MODE,
    "checkpoints": {
        "base": base,
        "after_model": after_model,
        "after_input": after_input,
        "pre_fwd": pre_fwd,
        "post_fwd": post_fwd,
        "peak_fwd": peak_fwd,
        "pre_bwd": pre_bwd,
        "post_bwd": post_bwd,
        "peak_bwd": peak_bwd,
        "post_cleanup": post_cleanup,
    },
    "deltas": {
        "model_params": round(after_model - base, 2),
        "input_tensors": round(after_input - after_model, 2),
        "fwd_activation_delta": round(post_fwd - pre_fwd, 2),
        "fwd_peak_above_pre": round(peak_fwd - pre_fwd, 2),
        "bwd_peak_above_pre": round(peak_bwd - pre_bwd, 2),
        "bwd_residual_delta": round(post_bwd - pre_bwd, 2),
    },
    "param_sizes_mib": param_sizes,
    "grad_sizes_mib": grad_sizes,
    "fp8_caches_after_fwd": cache_report,
    "fp8_caches_after_bwd": cache_after_bwd,
    "theoretical_sizes_mib": theory,
}

print(json.dumps(result, indent=2))
'''


def run_mode(mode, gpu_id="0"):
    env = os.environ.copy()
    env["_MEM_MODE"] = mode
    env["USE_QUACK_GEMM"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    if mode == "fp8":
        env["SONIC_MOE_FP8_MODE"] = "perf"
        env["SONIC_MOE_FP8_DOUBLE_QUANT"] = "1"
    else:
        env.pop("SONIC_MOE_FP8_MODE", None)
        env.pop("SONIC_MOE_FP8_DOUBLE_QUANT", None)

    r = subprocess.run(
        [sys.executable, "-c", _INNER_SCRIPT],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        print(f"[{mode}] FAILED:\n{r.stderr[-3000:]}", file=sys.stderr)
        return None
    # stderr has debug prints
    if r.stderr:
        print(f"[{mode}] stderr:\n{r.stderr[:2000]}", file=sys.stderr)
    try:
        return json.loads(r.stdout.strip())
    except Exception as e:
        print(f"[{mode}] JSON parse error: {e}\nstdout: {r.stdout[:2000]}", file=sys.stderr)
        return None


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    print("=" * 70, file=sys.stderr)
    print("SonicMoE Memory Breakdown (subprocess-isolated)", file=sys.stderr)
    print(f"GPU: {gpu}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    bf16 = run_mode("bf16", gpu)
    fp8 = run_mode("fp8", gpu)

    combined = {"bf16": bf16, "fp8": fp8}
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
