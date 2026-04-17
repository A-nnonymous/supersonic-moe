"""Unit tests for colwise_quantize_and_pack and colwise_quantize_cute."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact,
    gold_e8m0_col_quant,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]

# Colwise requires TK % 32 == 0
_COL_SHAPES = [(TK, dim) for TK, dim in
               [(v.values[0], v.values[1]) if hasattr(v, 'values') else v
                for v in QUANT_SHAPES]
               if TK % 32 == 0]
COL_SHAPES = [pytest.param(TK, dim, id=f"{TK}x{dim}") for TK, dim in _COL_SHAPES]


@pytest.mark.parametrize("TK,dim", COL_SHAPES)
def test_colwise_fp8_vs_gold(TK, dim, seed):
    """Gold col-quant FP8 bytes match Triton colwise kernel."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    # colwise_quantize_and_pack: logical_rows=dim, logical_cols=TK (for ISA packing)
    fp8_kernel, _ = colwise_quantize_and_pack(x, dim, TK)
    fp8_gold, _ = gold_e8m0_col_quant(x)
    assert_byte_exact(fp8_kernel, fp8_gold)


@pytest.mark.parametrize("TK,dim", COL_SHAPES)
def test_colwise_scales_vs_gold(TK, dim, seed):
    """Raw scales from CuTe colwise match gold E8M0."""
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    # CuTe with isa_pack=False returns raw scales (dim, num_groups)
    _, scale_cute = colwise_quantize_cute(x, dim, TK, isa_pack=False)
    _, gold_scales = gold_e8m0_col_quant(x)

    # gold_scales is (dim, num_groups), scale_cute is (dim, num_groups)
    assert_byte_exact(scale_cute, gold_scales)


@pytest.mark.parametrize("TK,dim", COL_SHAPES)
def test_cute_matches_triton(TK, dim, seed):
    """CuTe colwise variant is byte-identical to Triton variant."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    fp8_triton, scales_triton = colwise_quantize_and_pack(x, dim, TK)
    fp8_cute, scales_cute = colwise_quantize_cute(x, dim, TK, isa_pack=True)

    assert_byte_exact(fp8_cute, fp8_triton)
    assert_byte_exact(scales_cute, scales_triton)
