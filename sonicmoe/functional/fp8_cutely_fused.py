# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from cutify import (
    fused_act_dequant_best,
    fused_weighted_swiglu_act_quant_best,
)

from .fp8_protocol import FP8Protocol, validate_fp8_runtime_support
from .fp8_quant import dequantize_activation_blockwise, quantize_activation_blockwise, round_scale_to_e8m0


def _pad_postact_tensor(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    original_last_dim = x.size(-1)
    remainder = original_last_dim % group_size
    if remainder == 0:
        return x, original_last_dim

    padded_last_dim = original_last_dim + (group_size - remainder)
    padding = torch.zeros(*x.shape[:-1], padded_last_dim - original_last_dim, device=x.device, dtype=x.dtype)
    return torch.cat([x, padding], dim=-1), original_last_dim


def _pad_preact_for_group_size(preact: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    if preact.ndim != 2:
        raise ValueError(f"expected preact to be 2D, got shape {tuple(preact.shape)}")
    if preact.size(-1) % 2 != 0:
        raise ValueError(f"expected even preact width for SwiGLU, got {preact.size(-1)}")

    postact_width = preact.size(-1) // 2
    padded_postact_width = ((postact_width + group_size - 1) // group_size) * group_size
    if padded_postact_width == postact_width:
        return preact.contiguous(), postact_width

    padded_preact_width = padded_postact_width * 2
    padding = torch.zeros(
        preact.size(0),
        padded_preact_width - preact.size(-1),
        device=preact.device,
        dtype=preact.dtype,
    )
    return torch.cat([preact, padding], dim=-1).contiguous(), postact_width


def apply_activation_fp8_protocol_cutely_fused(
    x: torch.Tensor,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
    return_scales: bool = True,
    use_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), x.device, quack_enabled=quack_enabled)
    padded_x, original_last_dim = _pad_postact_tensor(x, protocol.group_size)
    data, scales = quantize_activation_blockwise(padded_x, protocol)
    dequantized = dequantize_activation_blockwise(data, scales, protocol, output_dtype=x.dtype)[..., :original_last_dim]

    restored_with_ste = x + (dequantized - x).detach() if use_ste else dequantized
    return (
        restored_with_ste,
        scales[..., : ((original_last_dim + protocol.group_size - 1) // protocol.group_size)] if return_scales else None,
    )


def apply_preact_activation_fp8_protocol_cutely_fused(
    preact: torch.Tensor,
    postact: torch.Tensor,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
    return_scales: bool = True,
    use_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), preact.device, quack_enabled=quack_enabled)
    if protocol.group_size != 128:
        return apply_activation_fp8_protocol_cutely_fused(
            postact,
            protocol,
            quack_enabled=quack_enabled,
            return_scales=return_scales,
            use_ste=use_ste,
        )
    padded_preact, original_postact_width = _pad_preact_for_group_size(preact, protocol.group_size)
    quantized, packed_scales = fused_weighted_swiglu_act_quant_best(
        padded_preact,
        prob=None,
        block_size=protocol.group_size,
        using_pow2_scaling=True,
        using_ue8m0_scale=False,
    )
    restored = fused_act_dequant_best(quantized, packed_scales, block_size=protocol.group_size)[..., :original_postact_width]
    restored = restored.to(dtype=postact.dtype)

    restored_with_ste = postact + (restored - postact).detach() if use_ste else restored
    if not return_scales:
        return restored_with_ste, None

    scale_cols = (original_postact_width + protocol.group_size - 1) // protocol.group_size
    decoded_scales = round_scale_to_e8m0(packed_scales[:, :scale_cols], protocol)
    return restored_with_ste, decoded_scales
