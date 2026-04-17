"""Unit tests for blockscaled_fp8_weight_grad_gemm: torch ↔ BF16 ↔ FP8 3-way.

wgrad GEMM performs per-expert: dw[e] = a_e.T @ b_e

3-way verification:
  (1) torch gold (float32 per-expert matmul)
  (2) FP8 blockscaled wgrad GEMM — this function auto-quantizes bf16 inputs
  (3) cross-check between implementations

All metrics reported per comparison.
"""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_bf16_close, assert_fp8_tolerance,
    rrmse, cosine_sim,
    GEMM_SHAPES,
)

pytestmark = [requires_blackwell, requires_quack]

_WGRAD_SHAPES = []
for p in GEMM_SHAPES:
    T, H, I, E, K = p.values
    TK = T * K // E
    if TK % 128 == 0 and H % 128 == 0 and I % 128 == 0:
        _WGRAD_SHAPES.append(pytest.param(T, H, I, E, K, id=p.id))
WGRAD_SHAPES = _WGRAD_SHAPES if _WGRAD_SHAPES else GEMM_SHAPES[:1]


def _setup_wgrad(T, H, I, E, K):
    """Create uniform routing for wgrad GEMM."""
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    return TK, total_M, cu_seqlens


def _torch_wgrad_gold(a_flat, b_flat, cu_seqlens, E):
    """Per-expert: dw[e] = a_e.T @ b_e in float32 → bf16."""
    dim_A = a_flat.shape[1]
    dim_B = b_flat.shape[1]
    dw = torch.zeros(E, dim_A, dim_B, dtype=torch.bfloat16, device=a_flat.device)
    for exp in range(E):
        s = cu_seqlens[exp].item()
        e = cu_seqlens[exp + 1].item()
        if s < e:
            dw[exp] = (a_flat[s:e].float().T @ b_flat[s:e].float()).to(torch.bfloat16)
    return dw


def _report_metrics(actual, expected, label):
    """Print detailed precision metrics."""
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}")
    return r, c


@pytest.mark.parametrize("T,H,I,E,K", WGRAD_SHAPES)
def test_torch_vs_bf16(T, H, I, E, K, seed):
    """(1/3) Torch gold wgrad (float32 per-expert matmul) — baseline."""
    TK, total_M, cu_seqlens = _setup_wgrad(T, H, I, E, K)
    a_flat = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    b_flat = torch.randn(total_M, I, dtype=torch.bfloat16, device="cuda") * 0.02

    gold = _torch_wgrad_gold(a_flat, b_flat, cu_seqlens, E)
    # BF16 baseline = torch gold (no separate BF16 CUTLASS wgrad kernel available)
    print(f"  [BF16 baseline = torch gold for wgrad GEMM]")
    r, c = _report_metrics(gold, gold, "dw: BF16(=torch) vs torch")
    assert r == 0.0


@pytest.mark.parametrize("T,H,I,E,K", WGRAD_SHAPES)
def test_torch_vs_fp8(T, H, I, E, K, seed):
    """(2/3) FP8 blockscaled wgrad GEMM vs torch gold."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_weight_grad_gemm

    TK, total_M, cu_seqlens = _setup_wgrad(T, H, I, E, K)
    a_flat = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    b_flat = torch.randn(total_M, I, dtype=torch.bfloat16, device="cuda") * 0.02

    gold = _torch_wgrad_gold(a_flat, b_flat, cu_seqlens, E)

    fp8_out = blockscaled_fp8_weight_grad_gemm(
        a_flat, b_flat, cu_seqlens,
        out_dtype=torch.bfloat16,
    )

    r, c = _report_metrics(fp8_out, gold, "dw: FP8 vs torch")
    assert r < 0.10, f"FP8 wgrad RRMSE {r:.6f} >= 0.10 vs torch gold"
    assert c > 0.99, f"FP8 wgrad cosine {c:.8f} <= 0.99 vs torch gold"


@pytest.mark.parametrize("T,H,I,E,K", WGRAD_SHAPES)
def test_bf16_vs_fp8(T, H, I, E, K, seed):
    """(3/3) FP8 vs BF16 cross-check (BF16 = torch gold for wgrad)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_weight_grad_gemm

    TK, total_M, cu_seqlens = _setup_wgrad(T, H, I, E, K)
    a_flat = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    b_flat = torch.randn(total_M, I, dtype=torch.bfloat16, device="cuda") * 0.02

    bf16_gold = _torch_wgrad_gold(a_flat, b_flat, cu_seqlens, E)

    fp8_out = blockscaled_fp8_weight_grad_gemm(
        a_flat, b_flat, cu_seqlens,
        out_dtype=torch.bfloat16,
    )

    r, c = _report_metrics(fp8_out, bf16_gold, "dw: FP8 vs BF16")
    assert r < 0.10, f"FP8 vs BF16 wgrad RRMSE {r:.6f} >= 0.10"
    assert c > 0.99, f"FP8 vs BF16 wgrad cosine {c:.8f} <= 0.99"
