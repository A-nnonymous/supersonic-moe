# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from .fp8_protocol import FP8Protocol, validate_fp8_protocol


def _pad_last_dim_to_group_size(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    original_last_dim = x.size(-1)
    remainder = original_last_dim % group_size
    if remainder == 0:
        return x, original_last_dim

    pad = group_size - remainder
    padding = torch.zeros(*x.shape[:-1], pad, device=x.device, dtype=x.dtype)
    return torch.cat([x, padding], dim=-1), original_last_dim


def _reshape_last_dim_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    padded_x, original_last_dim = _pad_last_dim_to_group_size(x, group_size)
    grouped_shape = (*padded_x.shape[:-1], padded_x.size(-1) // group_size, group_size)
    return padded_x.reshape(grouped_shape), grouped_shape, original_last_dim


def _safe_scale_from_amax(amax: torch.Tensor, dtype_max: float) -> torch.Tensor:
    raw_scale = amax / dtype_max
    return torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))


def round_scale_to_e8m0(scales: torch.Tensor, protocol: FP8Protocol | None = None) -> torch.Tensor:
    protocol = validate_fp8_protocol(protocol or FP8Protocol())
    if scales.device.type != "cuda":
        raise ValueError("FP8 scale encoding is only supported on CUDA tensors")

    positive_scales = torch.where(
        scales.float() > 0,
        scales.float(),
        torch.full_like(scales.float(), torch.finfo(protocol.scale_torch_dtype).tiny),
    )
    rounded_up_pow2 = torch.pow(2.0, torch.ceil(torch.log2(positive_scales)))
    return rounded_up_pow2.to(protocol.scale_torch_dtype)


def quantize_activation_blockwise(
    x: torch.Tensor,
    protocol: FP8Protocol | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    protocol = validate_fp8_protocol(protocol or FP8Protocol())

    if x.device.type != "cuda":
        raise ValueError("FP8 activation quantization is only supported on CUDA tensors")

    grouped_x, grouped_shape, original_last_dim = _reshape_last_dim_groups(x, protocol.group_size)
    amax = grouped_x.abs().amax(dim=-1).float()
    raw_scale = _safe_scale_from_amax(amax, torch.finfo(protocol.activation_torch_dtype).max)
    encoded_scale = round_scale_to_e8m0(raw_scale, protocol)

    scale_fp32 = encoded_scale.float()
    scale_fp32 = torch.where(scale_fp32 > 0, scale_fp32, torch.ones_like(scale_fp32))
    scale_for_div = scale_fp32.to(dtype=grouped_x.dtype)
    scaled = grouped_x / scale_for_div.unsqueeze(-1)
    quantized = scaled.to(protocol.activation_torch_dtype).reshape(*grouped_shape[:-2], grouped_shape[-2] * grouped_shape[-1])
    quantized = quantized[..., :original_last_dim]

    return quantized, encoded_scale.reshape(grouped_shape[:-1])


def dequantize_activation_blockwise(
    x_fp8: torch.Tensor,
    scales_e8m0: torch.Tensor,
    protocol: FP8Protocol | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    protocol = validate_fp8_protocol(protocol or FP8Protocol())

    grouped_x, grouped_shape, original_last_dim = _reshape_last_dim_groups(x_fp8, protocol.group_size)
    expected_scale_shape = grouped_shape[:-1]

    if tuple(scales_e8m0.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scales shape {expected_scale_shape} for tensor shape {tuple(x_fp8.shape)}, got {tuple(scales_e8m0.shape)}"
        )

    dequantized = (grouped_x.to(dtype=output_dtype) * scales_e8m0.to(dtype=output_dtype).unsqueeze(-1)).reshape(
        *grouped_shape[:-2], grouped_shape[-2] * grouped_shape[-1]
    )
    return dequantized[..., :original_last_dim]
