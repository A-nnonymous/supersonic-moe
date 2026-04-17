"""Unit tests for fused_z_save_y1_quant."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_z_matches_separate(TK, dim, seed):
    """z_fp8, z_raw_scales == quantize_activation_blockscaled_fast(z)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        fused_z_save_y1_quant,
        quantize_activation_blockscaled_fast,
    )

    # z has shape (TK, 2*dim), y1 has shape (TK, dim)
    I = dim
    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    y1 = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda")

    z_fp8_fused, z_scales_fused, _, _ = fused_z_save_y1_quant(z, y1)
    z_fp8_sep, z_scales_sep = quantize_activation_blockscaled_fast(z)

    assert_byte_exact(z_fp8_fused, z_fp8_sep)
    # z_scales_fused is float8_e8m0fnu view of raw uint8; z_scales_sep is also raw uint8
    assert_byte_exact(z_scales_fused, z_scales_sep)


@pytest.mark.parametrize("TK,dim", QUANT_SHAPES)
def test_y1_matches_separate(TK, dim, seed):
    """y1_fp8, y1_isa_scales == quantize_and_pack_activation(y1)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        fused_z_save_y1_quant,
        quantize_and_pack_activation,
    )

    I = dim
    z = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    y1 = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda")

    _, _, y1_fp8_fused, y1_scales_fused = fused_z_save_y1_quant(z, y1)
    y1_fp8_sep, y1_scales_sep = quantize_and_pack_activation(y1)

    assert_byte_exact(y1_fp8_fused, y1_fp8_sep)
    assert_byte_exact(y1_scales_fused, y1_scales_sep)
