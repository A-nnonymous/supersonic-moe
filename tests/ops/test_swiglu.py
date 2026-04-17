"""Unit tests for SwiGLU forward/backward: torch ↔ BF16 ↔ FP8 3-way cross-validation.

SwiGLU uses interleaved layout: gate = z[:, 0::2], up = z[:, 1::2]
  y1[:, j] = silu(gate[:, j]) * up[:, j]

Forward kernels:
  BF16: swiglu_forward_triton(z) → y1 (bf16)
  FP8:  swiglu_forward_quant_pack_triton(z) → y1_fp8 + ISA-packed scales

Backward kernels:
  BF16: swiglu_backward_triton(dy1, z, s) → dz, y1s, ds
  FP8:  swiglu_backward_from_fp8_quant_pack_triton(dy1, z_fp8, z_scales, s) → dz, dz_fp8, dz_scales, y1s, ds

3-way: torch gold ↔ BF16 ↔ FP8 with detailed metrics on every comparison.
"""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_bf16_close, assert_fp8_tolerance,
    dequant_isa_and_compare,
    unpack_isa_scales, gold_dequant,
    rrmse, cosine_sim,
    SWIGLU_SHAPES,
)

pytestmark = [requires_blackwell, requires_quack]


def _torch_swiglu_fwd(z):
    """Pure torch SwiGLU: interleaved layout.
    gate = z[:, 0::2], up = z[:, 1::2]
    y1[:, j] = gate*sigmoid(gate)*up
    """
    z_f32 = z.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    return (gate * torch.sigmoid(gate) * up).to(z.dtype)


def _torch_swiglu_bwd(dy1, z):
    """Pure torch SwiGLU backward (interleaved). Returns dz (TK, 2I)."""
    z_f32 = z.float()
    dy1_f32 = dy1.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    d_gate = dy1_f32 * up * sig * (1.0 + gate * (1.0 - sig))
    d_up = dy1_f32 * silu_gate
    dz = torch.empty_like(z_f32)
    dz[:, 0::2] = d_gate
    dz[:, 1::2] = d_up
    return dz.to(z.dtype)


def _report(actual, expected, label):
    """Print detailed precision metrics and return (rrmse, cosine)."""
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}")
    return r, c


# =========================================================================
# Forward: 3-way
# =========================================================================

@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_fwd_torch_vs_bf16(TK, I, seed):
    """(1/3) BF16 Triton SwiGLU fwd vs torch gold."""
    from sonicmoe.quack_utils.swiglu_triton import swiglu_forward_triton

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    gold = _torch_swiglu_fwd(z)
    bf16_out = swiglu_forward_triton(z)

    r, c = _report(bf16_out, gold, "y1: BF16 vs torch")
    assert_bf16_close(bf16_out, gold, atol=1e-2)
    if r > 0.005:
        import warnings
        warnings.warn(f"BF16 SwiGLU fwd RRMSE={r:.6f} (>0.5% — investigate)")


@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_fwd_torch_vs_fp8(TK, I, seed):
    """(2/3) FP8 fused SwiGLU fwd vs torch gold: dequant → compare."""
    from sonicmoe.quack_utils.swiglu_triton import swiglu_forward_quant_pack_triton

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    gold = _torch_swiglu_fwd(z)
    y1_fp8, y1_packed_scales = swiglu_forward_quant_pack_triton(z)

    # Dequant FP8 using shared template
    raw_scales = unpack_isa_scales(y1_packed_scales, TK, I)
    y1_dequant = gold_dequant(y1_fp8, raw_scales)

    r, c = _report(y1_dequant, gold, "y1: FP8(dequant) vs torch")
    assert r < 0.10, f"FP8 SwiGLU fwd RRMSE {r:.6f} >= 0.10"
    assert c > 0.99, f"FP8 SwiGLU fwd cosine {c:.8f} <= 0.99"


@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_fwd_bf16_vs_fp8(TK, I, seed):
    """(3/3) FP8 fused SwiGLU fwd vs BF16 Triton cross-check."""
    from sonicmoe.quack_utils.swiglu_triton import (
        swiglu_forward_triton,
        swiglu_forward_quant_pack_triton,
    )

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    bf16_y1 = swiglu_forward_triton(z)
    y1_fp8, y1_packed_scales = swiglu_forward_quant_pack_triton(z)

    raw_scales = unpack_isa_scales(y1_packed_scales, TK, I)
    y1_dequant = gold_dequant(y1_fp8, raw_scales)

    r, c = _report(y1_dequant, bf16_y1, "y1: FP8(dequant) vs BF16")
    assert r < 0.10, f"FP8 vs BF16 SwiGLU fwd RRMSE {r:.6f} >= 0.10"
    assert c > 0.99, f"FP8 vs BF16 SwiGLU fwd cosine {c:.8f} <= 0.99"


# =========================================================================
# Backward: 3-way
# =========================================================================

@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_bwd_torch_vs_bf16(TK, I, seed):
    """(1/3) BF16 Triton SwiGLU bwd vs torch gold."""
    from sonicmoe.quack_utils.swiglu_triton import swiglu_backward_triton

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    dy1 = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda")
    s = torch.ones(TK, dtype=torch.float32, device="cuda")

    gold_dz = _torch_swiglu_bwd(dy1, z)
    dz_bf16, y1s_bf16, ds_bf16 = swiglu_backward_triton(dy1, z, s)

    r, c = _report(dz_bf16, gold_dz, "dz: BF16 vs torch")
    assert_bf16_close(dz_bf16, gold_dz, atol=2e-2)
    if r > 0.005:
        import warnings
        warnings.warn(f"BF16 SwiGLU bwd RRMSE={r:.6f} (>0.5% — investigate)")


@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_bwd_torch_vs_fp8(TK, I, seed):
    """(2/3) FP8 SwiGLU bwd vs torch gold: dz and y1s."""
    from sonicmoe.quack_utils.swiglu_triton import swiglu_backward_from_fp8_quant_pack_triton
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    dy1 = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda")
    s = torch.ones(TK, dtype=torch.float32, device="cuda")

    gold_dz = _torch_swiglu_bwd(dy1, z)
    gold_y1 = _torch_swiglu_fwd(z)

    z_fp8, z_scales = quantize_activation_blockscaled_fast(z)
    dz_bf16_out, dz_fp8, dz_packed_scales, y1s_fp8, ds_fp8 = \
        swiglu_backward_from_fp8_quant_pack_triton(dy1, z_fp8, z_scales, s)

    r_dz, c_dz = _report(dz_bf16_out, gold_dz, "dz: FP8 vs torch")
    r_y1, c_y1 = _report(y1s_fp8, gold_y1, "y1s: FP8 vs torch")

    assert r_dz < 0.10, f"FP8 SwiGLU bwd dz RRMSE {r_dz:.6f} >= 0.10"
    assert c_dz > 0.99, f"FP8 SwiGLU bwd dz cosine {c_dz:.8f} <= 0.99"
    assert r_y1 < 0.10, f"FP8 SwiGLU bwd y1s RRMSE {r_y1:.6f} >= 0.10"
    assert c_y1 > 0.99, f"FP8 SwiGLU bwd y1s cosine {c_y1:.8f} <= 0.99"


@pytest.mark.parametrize("TK,I", SWIGLU_SHAPES)
def test_bwd_bf16_vs_fp8(TK, I, seed):
    """(3/3) FP8 SwiGLU bwd vs BF16 cross-check on dz."""
    from sonicmoe.quack_utils.swiglu_triton import (
        swiglu_backward_triton,
        swiglu_backward_from_fp8_quant_pack_triton,
    )
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast

    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    dy1 = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda")
    s = torch.ones(TK, dtype=torch.float32, device="cuda")

    dz_bf16, y1s_bf16, _ = swiglu_backward_triton(dy1, z, s)

    z_fp8, z_scales = quantize_activation_blockscaled_fast(z)
    dz_fp8_bf16, _, _, y1s_fp8, _ = \
        swiglu_backward_from_fp8_quant_pack_triton(dy1, z_fp8, z_scales, s)

    r_dz, c_dz = _report(dz_fp8_bf16, dz_bf16, "dz: FP8 vs BF16")
    r_y1, c_y1 = _report(y1s_fp8, y1s_bf16, "y1s: FP8 vs BF16")

    assert r_dz < 0.10, f"FP8 vs BF16 dz RRMSE {r_dz:.6f} >= 0.10"
    assert c_dz > 0.99, f"FP8 vs BF16 dz cosine {c_dz:.8f} <= 0.99"
    assert r_y1 < 0.10, f"FP8 vs BF16 y1s RRMSE {r_y1:.6f} >= 0.10"
    assert c_y1 > 0.99, f"FP8 vs BF16 y1s cosine {c_y1:.8f} <= 0.99"
