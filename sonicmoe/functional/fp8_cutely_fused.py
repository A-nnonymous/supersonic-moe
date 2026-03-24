# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from .fp8_protocol import FP8Protocol, validate_fp8_runtime_support
from .fp8_quant import dequantize_activation_blockwise, quantize_activation_blockwise


def apply_activation_fp8_protocol_cutely_fused(
    x: torch.Tensor,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), x.device, quack_enabled=quack_enabled)
    data, scales = quantize_activation_blockwise(x, protocol)
    dequantized = dequantize_activation_blockwise(data, scales, protocol, output_dtype=x.dtype)

    # This adapter keeps the current post-SwiGLU boundary contract stable while we
    # prepare the real pre-SwiGLU fused epilogue integration.
    restored_with_ste = x + (dequantized - x).detach()
    return restored_with_ste, scales
