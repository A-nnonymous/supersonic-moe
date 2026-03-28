#!/usr/bin/env python3
"""Correctness test for the custom Triton blockscaled FP8 varlen GEMM kernel.

Compares against BF16 reference GEMM per expert using cu_seqlens boundaries.
Run:  python tools/test_triton_blockscaled_gemm.py
"""

import os
import sys
import torch

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sonicmoe.quack_utils.triton_blockscaled_gemm import (
    blockscaled_fp8_gemm_varlen_triton,
    precompute_weight_fp8_raw_scales,
    quantize_activation_raw,
)


def reference_gemm_varlen(x_bf16, w_bf16, cu_seqlens):
    """Reference BF16 per-expert GEMM: y[start:end] = x[start:end] @ w[:,:,e].T"""
    H, I, E = w_bf16.shape
    total_M = x_bf16.shape[0]
    y = torch.zeros(total_M, H, dtype=torch.bfloat16, device=x_bf16.device)
    for e in range(E):
        s = cu_seqlens[e].item()
        end = cu_seqlens[e + 1].item()
        if end > s:
            # w2[:,:,e] is (H, I), matmul x @ w.T = (n, I) @ (I, H) = (n, H)
            y[s:end] = x_bf16[s:end].float() @ w_bf16[:, :, e].float().T
    return y.to(torch.bfloat16)


def test_small():
    """Small shape test for correctness validation."""
    torch.manual_seed(42)
    device = "cuda"

    E, H, I = 4, 128, 64
    tokens_per_expert = [32, 64, 16, 48]
    total_M = sum(tokens_per_expert)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(tokens_per_expert), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    # Random data
    x = torch.randn(total_M, I, dtype=torch.bfloat16, device=device)
    w = torch.randn(H, I, E, dtype=torch.bfloat16, device=device)

    # Reference
    y_ref = reference_gemm_varlen(x, w, cu_seqlens)

    # Blockscaled FP8
    x_fp8, x_scales = quantize_activation_raw(x)
    w_fp8, w_scales = precompute_weight_fp8_raw_scales(w)
    y_fp8 = blockscaled_fp8_gemm_varlen_triton(
        x_fp8, x_scales, w_fp8, w_scales, cu_seqlens
    )

    # Compare
    err = (y_fp8.float() - y_ref.float()).abs()
    rel_err = err / (y_ref.float().abs().mean() + 1e-6)
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()

    # RelRMSE
    diff = (y_fp8.float() - y_ref.float())
    rmse = (diff ** 2).mean().sqrt()
    ref_rmse = (y_ref.float() ** 2).mean().sqrt()
    relrmse = (rmse / (ref_rmse + 1e-6)).item()

    print(f"[small] E={E}, H={H}, I={I}, total_M={total_M}")
    print(f"  max_rel_err={max_rel:.6f}, mean_rel_err={mean_rel:.6f}, relrmse={relrmse:.6f}")
    assert relrmse < 0.05, f"RelRMSE too high: {relrmse}"
    print("  PASS ✓")


def test_production_shape():
    """Production shape: E=128, H=4096, I=1024."""
    torch.manual_seed(123)
    device = "cuda"

    E, H, I = 128, 4096, 1024
    # Simulated balanced routing: ~515 tokens per expert
    avg_tokens = 515
    tokens_per_expert = [avg_tokens + (i % 7 - 3) for i in range(E)]
    total_M = sum(tokens_per_expert)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(tokens_per_expert), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    x = torch.randn(total_M, I, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(H, I, E, dtype=torch.bfloat16, device=device) * 0.02

    # Reference (sample a few experts for speed)
    print(f"[prod] E={E}, H={H}, I={I}, total_M={total_M}")
    print("  Computing reference (sampling 4 experts)...")
    y_ref_partial = torch.zeros(total_M, H, dtype=torch.bfloat16, device=device)
    sample_experts = [0, 42, 100, 127]
    for e in sample_experts:
        s = cu_seqlens[e].item()
        end = cu_seqlens[e + 1].item()
        if end > s:
            y_ref_partial[s:end] = (
                x[s:end].float() @ w[:, :, e].float().T
            ).to(torch.bfloat16)

    # FP8
    x_fp8, x_scales = quantize_activation_raw(x)
    w_fp8, w_scales = precompute_weight_fp8_raw_scales(w)
    y_fp8 = blockscaled_fp8_gemm_varlen_triton(
        x_fp8, x_scales, w_fp8, w_scales, cu_seqlens
    )

    # Compare only sampled experts
    for e in sample_experts:
        s = cu_seqlens[e].item()
        end = cu_seqlens[e + 1].item()
        if end <= s:
            continue
        diff = (y_fp8[s:end].float() - y_ref_partial[s:end].float())
        ref_norm = (y_ref_partial[s:end].float() ** 2).mean().sqrt()
        rmse = (diff ** 2).mean().sqrt()
        relrmse = (rmse / (ref_norm + 1e-6)).item()
        status = "PASS ✓" if relrmse < 0.05 else f"FAIL ✗ (relrmse={relrmse:.6f})"
        print(f"  expert {e}: relrmse={relrmse:.6f} {status}")
        assert relrmse < 0.05, f"Expert {e} RelRMSE too high: {relrmse}"

    print("  ALL PASS ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Triton Blockscaled FP8 Varlen GEMM — Correctness Test")
    print("=" * 60)

    test_small()
    print()
    test_production_shape()

    print("\n✅ All tests passed!")
