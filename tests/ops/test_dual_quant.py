"""Unit tests for dual_quantize_varlen (fused row+col quant)."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]

# dual_quantize_varlen needs TK%32==0 AND dim%32==0
_DUAL_SHAPES = [(TK, dim) for TK, dim in
                [(v.values[0], v.values[1]) if hasattr(v, 'values') else v
                 for v in QUANT_SHAPES]
                if TK % 32 == 0 and dim % 32 == 0]
DUAL_SHAPES = [pytest.param(TK, dim, id=f"{TK}x{dim}") for TK, dim in _DUAL_SHAPES]


@pytest.mark.parametrize("TK,dim", DUAL_SHAPES)
def test_row_output_matches_separate(TK, dim, seed):
    """Row fp8+scales from dual quant == quantize_and_pack_activation output."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        dual_quantize_varlen,
        quantize_and_pack_activation,
    )

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    row_fp8_d, row_scales_d, _, _ = dual_quantize_varlen(x, TK, dim)
    row_fp8_s, row_scales_s = quantize_and_pack_activation(x)

    assert_byte_exact(row_fp8_d, row_fp8_s)
    assert_byte_exact(row_scales_d, row_scales_s)


@pytest.mark.parametrize("TK,dim", DUAL_SHAPES)
def test_col_output_matches_separate(TK, dim, seed):
    """Col fp8+scales from dual quant == colwise_quantize_and_pack output."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        dual_quantize_varlen,
        colwise_quantize_and_pack,
    )

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    _, _, col_fp8_d, col_scales_d = dual_quantize_varlen(x, TK, dim)
    col_fp8_s, col_scales_s = colwise_quantize_and_pack(x, dim, TK)

    assert_byte_exact(col_fp8_d, col_fp8_s)
    assert_byte_exact(col_scales_d, col_scales_s)
