# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.1"

from .count_cumsum import count_cumsum
from .enums import KernelBackendMoE
from .functional import (
    FP8ActivationDType,
    FP8Backend,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    FP8Tensor,
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
