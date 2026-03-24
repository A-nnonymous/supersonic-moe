# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.1"

from .count_cumsum import count_cumsum
from .enums import KernelBackendMoE
from .functional import (
    FP8ActivationDType,
    FP8Backend,
    apply_activation_fp8_protocol_cutely_fused,
    apply_preact_activation_fp8_protocol_cutely_fused,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    FP8Tensor,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    enable_quack_gemm,
    get_default_fp8_protocol,
    is_blackwell_device,
    moe_general_routing_inputs,
    moe_TC_softmax_topk_layer,
    quantize_activation_reference,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
from .moe import MoE
from .quack_utils import make_blockscaled_grouped_reverse_scatter_idx, pack_blockscaled_1x32_scales
