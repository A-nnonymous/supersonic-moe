"""Unit tests for quantize_and_pack_activation (row-wise blockscaled FP8 quant)."""
import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact, assert_fp8_tolerance,
    gold_e8m0_row_quant, gold_dequant, rrmse,
    unpack_isa_scales, QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_fp8_data_byte_exact(TK, dim, seed):
    """Gold row-quant FP8 bytes == kernel FP8 bytes."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    fp8_kernel, _ = quantize_and_pack_activation(x)
    fp8_gold, _ = gold_e8m0_row_quant(x)
    assert_byte_exact(fp8_kernel, fp8_gold)


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_scales_byte_exact(TK, dim, seed):
    """Unpack ISA scales back to raw and compare to gold E8M0."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    _, packed_scales = quantize_and_pack_activation(x)
    _, gold_scales = gold_e8m0_row_quant(x)

    raw_scales = unpack_isa_scales(packed_scales, TK, dim)
    assert_byte_exact(raw_scales, gold_scales)


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_roundtrip_rrmse(TK, dim, seed):
    """dequant(quant(x)) vs x: RRMSE < 1%."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    fp8_out, packed_scales = quantize_and_pack_activation(x)

    raw_scales = unpack_isa_scales(packed_scales, TK, dim)
    reconstructed = gold_dequant(fp8_out, raw_scales)

    r = rrmse(reconstructed, x)
    assert r < 0.05, f"Roundtrip RRMSE {r:.6f} >= 0.05"
