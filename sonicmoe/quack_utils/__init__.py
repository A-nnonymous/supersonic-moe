# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from .blockscaled_fp8_gemm import (
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    blockscaled_fp8_weight_grad_gemm,
    blockscaled_fp8_weight_grad_gemm_fast,
    blockscaled_fp8_wgrad_varlen_k,
    clear_blockscaled_fp8_weight_cache,
    evict_fp8_weight_cache_entry,
    colwise_quantize_and_pack,
    gather_quantize_and_pack_activation,
    make_blockscaled_grouped_reverse_scatter_idx,
    pack_blockscaled_1x32_scales,
    prefetch_blockscaled_w2_fp8,
    precompute_weight_fp8,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
    quantize_and_pack_activation_varlen,
)
from .gemm_interface import gemm_dgated, gemm_gated, gemm_gated_out
from .swiglu_triton import dequantize_blockscaled_fp8
from .sgl_mxfp8_gemm import (
    clear_sgl_weight_cache,
    has_sgl_kernel,
    precompute_weight_fp8_sgl,
    rowmajor_to_sgl_tiled,
    sgl_mxfp8_gemm_varlen,
)
from .triton_blockscaled_gemm import (
    blockscaled_fp8_gemm_varlen_triton,
    clear_raw_weight_cache,
    precompute_weight_fp8_raw_scales,
    quantize_activation_raw,
)
