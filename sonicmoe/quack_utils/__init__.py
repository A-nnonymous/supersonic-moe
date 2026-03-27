# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from .blockscaled_fp8_gemm import (
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    blockscaled_fp8_weight_grad_gemm,
    clear_blockscaled_fp8_weight_cache,
    gather_quantize_and_pack_activation,
    make_blockscaled_grouped_reverse_scatter_idx,
    pack_blockscaled_1x32_scales,
    prefetch_blockscaled_w2_fp8,
    precompute_weight_fp8,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
)
from .gemm_interface import gemm_dgated, gemm_gated
