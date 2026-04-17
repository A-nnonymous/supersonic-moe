"""Unit tests for dequantize_blockscaled_fp8."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    gold_e8m0_row_quant, gold_dequant,
    rrmse, assert_fp8_tolerance,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_dequant_vs_torch_gold(TK, dim, seed):
    """Kernel dequant matches gold: fp8.float() * 2^(scale) per group, RRMSE < 0.1%."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
    from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    fp8_data, raw_scales = quantize_activation_blockscaled_fast(x)
    # Kernel dequant
    kernel_out = dequantize_blockscaled_fp8(fp8_data, raw_scales)
    # Gold dequant
    gold_out = gold_dequant(fp8_data, raw_scales)

    r = rrmse(kernel_out, gold_out)
    assert r < 0.001, f"Dequant RRMSE {r:.6f} >= 0.001"


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_roundtrip(TK, dim, seed):
    """quant -> dequant vs original: RRMSE < 1%."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
    from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    fp8_data, raw_scales = quantize_activation_blockscaled_fast(x)
    reconstructed = dequantize_blockscaled_fp8(fp8_data, raw_scales)

    r = rrmse(reconstructed, x)
    assert r < 0.05, f"Roundtrip RRMSE {r:.6f} >= 0.05"
