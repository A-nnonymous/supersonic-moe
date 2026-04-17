"""Unit tests for quantize_and_pack_weight_iso32 (32x32 isotropic blockscaled)."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact,
    gold_e8m0_iso32_quant,
    unpack_isa_scales,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]

# Weight shapes: use QUANT_SHAPES as (M, K) for weights
# ISO32 needs M%32==0 and K%32==0
_WEIGHT_SHAPES = [(M, K) for M, K in
                  [(v.values[0], v.values[1]) if hasattr(v, 'values') else v
                   for v in QUANT_SHAPES]
                  if M % 32 == 0 and K % 32 == 0]
WEIGHT_SHAPES = [pytest.param(M, K, id=f"{M}x{K}") for M, K in _WEIGHT_SHAPES]


@pytest.mark.parametrize("M,K", WEIGHT_SHAPES)
def test_iso32_fp8_vs_gold(M, K, seed):
    """32x32 block amax gold matches kernel FP8 bytes."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_weight_iso32

    w = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    fp8_kernel, _ = quantize_and_pack_weight_iso32(w)
    fp8_gold, _ = gold_e8m0_iso32_quant(w)
    assert_byte_exact(fp8_kernel, fp8_gold)


@pytest.mark.parametrize("M,K", WEIGHT_SHAPES)
def test_iso32_scales_vs_gold(M, K, seed):
    """ISA-packed scales match gold E8M0 iso32 scales."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_weight_iso32

    w = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    _, packed_scales = quantize_and_pack_weight_iso32(w)
    _, gold_scales = gold_e8m0_iso32_quant(w)

    raw_scales = unpack_isa_scales(packed_scales, M, K)
    assert_byte_exact(raw_scales, gold_scales)
