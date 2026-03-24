# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from dataclasses import dataclass

import torch

from .fp8_protocol import FP8Protocol, validate_fp8_runtime_support
from .fp8_quant import dequantize_activation_blockwise, quantize_activation_blockwise


@dataclass(frozen=True)
class FP8Tensor:
    data: torch.Tensor
    scales: torch.Tensor
    protocol: FP8Protocol


def quantize_activation_reference(
    x: torch.Tensor,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
) -> FP8Tensor:
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), x.device, quack_enabled=quack_enabled)
    data, scales = quantize_activation_blockwise(x, protocol)
    return FP8Tensor(data=data, scales=scales, protocol=protocol)


def dequantize_activation_reference(fp8_tensor: FP8Tensor) -> torch.Tensor:
    validate_fp8_runtime_support(fp8_tensor.protocol, fp8_tensor.data.device, quack_enabled=False if not fp8_tensor.protocol.requires_quack_gemm else True)
    return dequantize_activation_blockwise(fp8_tensor.data, fp8_tensor.scales, fp8_tensor.protocol)
