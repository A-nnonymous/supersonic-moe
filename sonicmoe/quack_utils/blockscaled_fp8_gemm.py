# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.blockscaled_layout as _upstream_blockscaled_utils
import torch
import triton
import triton.language as tl
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_interface import default_config
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase

from ..functional.fp8_protocol import FP8Protocol, FP8ScaleGranularity, validate_fp8_runtime_support
from ..functional.fp8_quant import quantize_activation_blockwise, round_scale_to_e8m0


# ---------------------------------------------------------------------------
# Rank-aware tile_atom_to_shape_SF
# ---------------------------------------------------------------------------
# The upstream CUTLASS DSL function ``tile_atom_to_shape_SF`` hardcodes a
# 3-element order ``(2, 1, 3)`` which assumes rank-3 tensor shapes.  The
# varlen_m scheduler produces rank-2 activation tensors ``(total_M, K)``,
# causing a compile-time rank mismatch.
#
# We provide a rank-aware replacement that uses ``(2, 1)`` for rank-2 and
# ``(2, 1, 3)`` for rank-3 tensors.  This replacement is installed into the
# upstream module namespace so that ``GemmDefaultSm100.__call__`` (which
# references ``blockscaled_utils.tile_atom_to_shape_SF``) picks it up at
# kernel-tracing time.
# ---------------------------------------------------------------------------

@dsl_user_op
def _tile_atom_to_shape_SF_rank_aware(
    Shape: cute.Shape,
    sf_vec_size: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Rank-aware version of ``tile_atom_to_shape_SF``.

    Handles both rank-2 (varlen flat) and rank-3 (grouped/batched) shapes.
    """
    rank = cute.rank(Shape)
    if const_expr(rank == 2):
        # varlen: (total_M, K) → ((Atom_MN, Rest_MN), (Atom_K, Rest_K))
        sf_layout = cute.tile_to_shape(
            BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1)
        )
    else:
        # grouped/batched: (M, K, L) → ((Atom_MN, Rest_MN), (Atom_K, Rest_K), RestL)
        sf_layout = cute.tile_to_shape(
            BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1, 3)
        )
    return sf_layout


# Install into upstream module so GemmDefaultSm100 uses it at trace time.
_upstream_blockscaled_utils.tile_atom_to_shape_SF = _tile_atom_to_shape_SF_rank_aware


_SF_VEC_SIZE = 32
_SF_TILE_M = 128
_SF_TILE_K = 128
_SF_TILE_STORAGE = _SF_TILE_M * (_SF_TILE_K // _SF_VEC_SIZE)
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

_INDEX_CACHE: dict[tuple[int, int, int | None], torch.Tensor] = {}
_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[int, ...], int | None, int, str, str, str, str],
    tuple[torch.Tensor, torch.Tensor],
] = {}
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_TORCH_TO_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    torch.uint8: cutlass.Uint8,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def _div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def _get_blockscaled_expert_capacity() -> int:
    value = os.getenv("SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY")
    if value is None:
        raise RuntimeError(
            "blockscaled_fp8_gemm requires SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY; "
            "the blockscaled path only accepts externally padded/aligned expert groups"
        )
    try:
        capacity = int(value)
    except ValueError as exc:
        raise RuntimeError("SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY must be an integer") from exc
    if capacity <= 0 or capacity % _SF_TILE_M != 0:
        raise RuntimeError(
            f"SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY must be a positive multiple of {_SF_TILE_M}, got {capacity}"
        )
    return capacity


def _validate_blockscaled_capacity(cu_seqlens_m: torch.Tensor, capacity: int) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    max_expert_tokens = int((cu_seqlens_m[1:] - cu_seqlens_m[:-1]).max().item())
    if max_expert_tokens > capacity:
        raise RuntimeError(
            "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY is smaller than the routed expert load: "
            f"capacity={capacity}, max_expert_tokens={max_expert_tokens}"
        )


def make_blockscaled_grouped_reverse_scatter_idx(
    flat_sorted_positions: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    capacity: Optional[int] = None,
    expert_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if flat_sorted_positions.ndim != 1:
        raise ValueError(f"expected 1D flat_sorted_positions, got shape {tuple(flat_sorted_positions.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be a 1D int32 tensor")
    if flat_sorted_positions.device != cu_seqlens_m.device:
        raise ValueError("flat_sorted_positions and cu_seqlens_m must be on the same device")

    if capacity is None:
        capacity = _get_blockscaled_expert_capacity()
    flat_sorted_positions_i64 = flat_sorted_positions.to(torch.int64)
    if expert_ids is None:
        expert_ids_i64 = torch.searchsorted(cu_seqlens_m[1:], flat_sorted_positions_i64, right=True)
    else:
        if expert_ids.ndim != 1:
            raise ValueError(f"expected 1D expert_ids, got shape {tuple(expert_ids.shape)}")
        if expert_ids.device != flat_sorted_positions.device:
            raise ValueError("expert_ids and flat_sorted_positions must be on the same device")
        if expert_ids.numel() != flat_sorted_positions.numel():
            raise ValueError("expert_ids must have the same number of elements as flat_sorted_positions")
        expert_ids_i64 = expert_ids.to(torch.int64)
    expert_starts = cu_seqlens_m.index_select(0, expert_ids_i64).to(torch.int64)
    grouped_positions = expert_ids_i64 * capacity + (flat_sorted_positions_i64 - expert_starts)
    return grouped_positions.to(torch.int32)


def _storage_per_batch(rows: int, cols: int) -> int:
    return _div_up(rows, _SF_TILE_M) * _div_up(cols, _SF_TILE_K) * _SF_TILE_STORAGE


def _scale_pack_index(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    key = (rows, cols, device.index)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    scale_cols = _div_up(cols, _SF_VEC_SIZE)
    k_tiles = _div_up(cols, _SF_TILE_K)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    scale_block_ids = torch.arange(scale_cols, device=device, dtype=torch.int64).unsqueeze(0)

    row_tiles = row_ids // _SF_TILE_M
    row_in_tile = row_ids % _SF_TILE_M
    k_tiles_idx = scale_block_ids // (_SF_TILE_K // _SF_VEC_SIZE)
    k_in_tile = scale_block_ids % (_SF_TILE_K // _SF_VEC_SIZE)

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * _SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    index = (tile_base + row_base + k_in_tile).reshape(-1).contiguous()
    _INDEX_CACHE[key] = index
    return index


def pack_blockscaled_1x32_scales(scales: torch.Tensor, cols: int) -> torch.Tensor:
    if scales.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D scales, got shape {tuple(scales.shape)}")
    if scales.device.type != "cuda":
        raise ValueError("blockscaled scale packing requires CUDA tensors")

    if scales.ndim == 2:
        scales = scales.unsqueeze(0)

    batches, rows, _ = scales.shape
    per_batch_storage = _storage_per_batch(rows, cols)
    packed = torch.ones((batches, per_batch_storage), device=scales.device, dtype=scales.dtype)
    flat_index = _scale_pack_index(rows, cols, scales.device)
    batch_offsets = torch.arange(batches, device=scales.device, dtype=torch.int64).unsqueeze(1) * per_batch_storage
    packed.view(-1)[(batch_offsets + flat_index.unsqueeze(0)).reshape(-1)] = scales.reshape(-1)
    return packed


def _blockscaled_protocol(protocol: FP8Protocol) -> FP8Protocol:
    return replace(protocol, scale_granularity=FP8ScaleGranularity.BLOCK_1X32)


def _is_runtime_fp8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {torch.float8_e4m3fn, torch.float8_e8m0fnu}


def _make_cute_tensor_dynamic(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    if _is_runtime_fp8_tensor(tensor):
        storage = tensor.detach().view(torch.uint8)
        cute_tensor = from_dlpack(storage, assumed_align=16)
        cute_tensor.element_type = _TORCH_TO_CUTLASS_DTYPE[tensor.dtype]
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _weight_cache_key(
    weight: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int | None, int, str, str, str, str]:
    return (
        weight.untyped_storage().data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.device.index,
        weight._version,
        protocol.activation_dtype.value,
        protocol.scale_encoding.value,
        protocol.scale_granularity.value,
        protocol.backend.value,
    )


def _quantize_w2_cached(
    w2: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _weight_cache_key(w2, protocol)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    weight_ehi = w2.permute(2, 0, 1).contiguous()
    weight_fp8_ehi, weight_scales = quantize_activation_blockwise(weight_ehi, protocol)
    packed_scales = pack_blockscaled_1x32_scales(weight_scales, weight_ehi.size(-1))
    _WEIGHT_CACHE[key] = (weight_fp8_ehi, packed_scales)
    return weight_fp8_ehi, packed_scales


def clear_blockscaled_fp8_weight_cache() -> None:
    _WEIGHT_CACHE.clear()


def prefetch_blockscaled_w2_fp8(
    w2: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w2.ndim != 3:
        raise ValueError(f"expected w2 with shape (H, I, E), got {tuple(w2.shape)}")
    if w2.device.type != "cuda":
        raise ValueError("blockscaled FP8 weight prefetch requires CUDA weights")

    blockscaled_protocol = validate_fp8_runtime_support(_blockscaled_protocol(protocol), device=w2.device)
    return _quantize_w2_cached(w2, blockscaled_protocol)


# ---------------------------------------------------------------------------
# Fast fused blockscaled quantization kernel for flat 2D activations
# ---------------------------------------------------------------------------
# Single-pass: read bf16 → compute per-32 amax → E8M0 scale → quantize → write fp8 + scales
# Replaces the Python quantize_activation_blockwise which does ~8 separate kernel launches.

@triton.jit
def _quantize_flat_blockscaled_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_scale_ptr,
    rows,
    cols,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    scale_stride_row,
    scale_stride_col,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Quantize a flat (M, K) bf16 tensor to blockscaled FP8 with 1×GROUP_SIZE scales."""
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group_id = tl.program_id(1)
    col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    row_mask = row_id[:, None] < rows
    col_mask = col_offsets[None, :] < cols
    mask = row_mask & col_mask

    src_ptrs = src_ptr + row_id[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    block_amax = tl.max(tl.abs(values), axis=1)
    raw_scale = block_amax / fp8_max
    positive = raw_scale > 0
    exponent = tl.where(positive, tl.ceil(tl.log2(tl.where(positive, raw_scale, 1.0))), 0.0)
    dequant_scale = tl.exp2(exponent)
    quant_scale = tl.exp2(-exponent)

    quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
    dst_ptrs = dst_fp8_ptr + row_id[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
    tl.store(dst_ptrs, quantized, mask=mask)

    scale_ptrs = dst_scale_ptr + row_id * scale_stride_row + group_id * scale_stride_col
    # Extract E8M0 exponent: float32 → int32 bitcast → shift right 23 → mask 8 bits
    scale_i32 = dequant_scale.to(tl.float32).to(tl.int32, bitcast=True)
    e8m0_byte = ((scale_i32 >> 23) & 0xFF).to(tl.uint8)
    tl.store(scale_ptrs, e8m0_byte, mask=row_id < rows)


def quantize_activation_blockscaled_fast(
    x: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast fused 1×group_size blockscaled quantization using a single Triton kernel.

    Returns (fp8_data, e8m0_scales).
    """
    assert x.is_contiguous(), "Input must be contiguous"
    M, K = x.shape
    num_groups = _div_up(K, group_size)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    # Scale stored as uint8 (E8M0 exponent). Will be view-cast to float8_e8m0fnu for CUTLASS.
    scale_out = torch.empty(M, num_groups, dtype=torch.uint8, device=x.device)

    BLOCK_ROWS = 4
    grid = (_div_up(M, BLOCK_ROWS), num_groups)
    _quantize_flat_blockscaled_kernel[grid](
        x,
        fp8_out,
        scale_out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        scale_out.stride(0),
        scale_out.stride(1),
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
    )
    return fp8_out, scale_out.view(torch.float8_e8m0fnu)


@triton.jit
def _pack_quantize_expert_segments_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_scale_ptr,
    offsets_ptr,
    src_stride_row,
    src_stride_col,
    dst_stride_expert,
    dst_stride_row,
    dst_stride_col,
    dst_scale_stride_expert,
    dst_scale_stride_row,
    dst_scale_stride_col,
    cols,
    fp8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    scale_col = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = scale_col * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    src_rows = expert_start + row_offsets[:, None]
    src_cols = col_offsets[None, :]
    row_mask = row_offsets[:, None] < expert_len
    col_mask = src_cols < cols
    mask = row_mask & col_mask

    src_ptrs = src_ptr + src_rows * src_stride_row + src_cols * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    block_amax = tl.max(tl.abs(values), axis=1)
    positive_scale = tl.where(block_amax > 0, block_amax / fp8_max, 1.0)
    exponent = tl.ceil(tl.log2(positive_scale))
    dequant_scale = tl.where(block_amax > 0, tl.exp2(exponent), 1.0)
    quant_scale = tl.where(block_amax > 0, tl.exp2(-exponent), 1.0)

    quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
    dst_ptrs = (
        dst_fp8_ptr
        + expert_id * dst_stride_expert
        + row_offsets[:, None] * dst_stride_row
        + src_cols * dst_stride_col
    )
    tl.store(dst_ptrs, quantized, mask=mask)

    scale_ptrs = (
        dst_scale_ptr
        + expert_id * dst_scale_stride_expert
        + row_offsets * dst_scale_stride_row
        + scale_col * dst_scale_stride_col
    )
    tl.store(scale_ptrs, dequant_scale, mask=row_offsets < expert_len)


@triton.jit
def _pack_expert_segments_kernel(
    src_ptr,
    dst_ptr,
    offsets_ptr,
    src_stride_row,
    src_stride_col,
    dst_stride_expert,
    dst_stride_row,
    dst_stride_col,
    cols,
    capacity,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    src_rows = expert_start + row_offsets[:, None]
    src_cols = col_offsets[None, :]
    row_mask = row_offsets[:, None] < expert_len
    col_mask = src_cols < cols
    mask = row_mask & col_mask

    src_ptrs = src_ptr + src_rows * src_stride_row + src_cols * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0)

    dst_rows = row_offsets[:, None]
    dst_ptrs = (
        dst_ptr
        + expert_id * dst_stride_expert
        + dst_rows * dst_stride_row
        + src_cols * dst_stride_col
    )
    store_mask = (dst_rows < capacity) & col_mask
    tl.store(dst_ptrs, values, mask=store_mask)


@triton.jit
def _unpack_expert_segments_kernel(
    src_ptr,
    dst_ptr,
    offsets_ptr,
    src_stride_expert,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    row_mask = row_offsets[:, None] < expert_len
    col_mask = col_offsets[None, :] < cols
    mask = row_mask & col_mask

    src_ptrs = (
        src_ptr
        + expert_id * src_stride_expert
        + row_offsets[:, None] * src_stride_row
        + col_offsets[None, :] * src_stride_col
    )
    values = tl.load(src_ptrs, mask=mask, other=0.0)

    dst_rows = expert_start + row_offsets[:, None]
    dst_ptrs = dst_ptr + dst_rows * dst_stride_row + col_offsets[None, :] * dst_stride_col
    tl.store(dst_ptrs, values, mask=mask)


def _pack_grouped_rows(tensor: torch.Tensor, cu_seqlens_m: torch.Tensor, groups: int, capacity: int) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor to pack, got shape {tuple(tensor.shape)}")
    packed = torch.empty((groups, capacity, tensor.size(1)), dtype=tensor.dtype, device=tensor.device)
    grid = (groups, triton.cdiv(capacity, 16), triton.cdiv(tensor.size(1), 128))
    _pack_expert_segments_kernel[grid](
        tensor,
        packed,
        cu_seqlens_m,
        tensor.stride(0),
        tensor.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        tensor.size(1),
        capacity,
        BLOCK_M=16,
        BLOCK_N=128,
    )
    return packed


def _pack_quantize_grouped_rows(
    tensor: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    groups: int,
    capacity: int,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor to pack, got shape {tuple(tensor.shape)}")
    if tensor.size(1) % protocol.group_size != 0:
        raise ValueError(f"expected width divisible by group size {protocol.group_size}, got {tensor.size(1)}")

    quantized = torch.empty((groups, capacity, tensor.size(1)), dtype=protocol.activation_torch_dtype, device=tensor.device)
    dequant_scale_fp32 = torch.ones(
        (groups, capacity, tensor.size(1) // protocol.group_size),
        dtype=torch.float32,
        device=tensor.device,
    )
    grid = (groups, triton.cdiv(capacity, 8), tensor.size(1) // protocol.group_size)
    _pack_quantize_expert_segments_kernel[grid](
        tensor,
        quantized,
        dequant_scale_fp32,
        cu_seqlens_m,
        tensor.stride(0),
        tensor.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        dequant_scale_fp32.stride(0),
        dequant_scale_fp32.stride(1),
        dequant_scale_fp32.stride(2),
        tensor.size(1),
        _FP8_E4M3_MAX,
        BLOCK_M=8,
        BLOCK_N=_SF_VEC_SIZE,
    )
    return quantized, round_scale_to_e8m0(dequant_scale_fp32, protocol)


def _unpack_grouped_rows(grouped: torch.Tensor, cu_seqlens_m: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if grouped.ndim != 3:
        raise ValueError(f"expected 3D grouped tensor, got shape {tuple(grouped.shape)}")
    total_rows = int(out.size(0) if out is not None else cu_seqlens_m[-1].item())
    if out is None:
        out = torch.empty(total_rows, grouped.size(2), dtype=grouped.dtype, device=grouped.device)
    grid = (grouped.size(0), triton.cdiv(grouped.size(1), 16), triton.cdiv(grouped.size(2), 128))
    _unpack_expert_segments_kernel[grid](
        grouped,
        out,
        cu_seqlens_m,
        grouped.stride(0),
        grouped.stride(1),
        grouped.stride(2),
        out.stride(0),
        out.stride(1),
        grouped.size(2),
        BLOCK_M=16,
        BLOCK_N=128,
    )
    return out


def blockscaled_fp8_gemm_grouped(
    a: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol,
    out_dtype: Optional[torch.dtype] = None,
    capacity: Optional[int] = None,
) -> torch.Tensor:
    protocol = validate_fp8_runtime_support(_blockscaled_protocol(protocol), a.device, quack_enabled=True)
    if get_device_capacity(a.device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_gemm currently supports Blackwell only")

    if a.ndim != 2:
        raise ValueError(f"expected activation to be 2D, got shape {tuple(a.shape)}")
    if w2.ndim != 3:
        raise ValueError(f"expected w2 to be 3D, got shape {tuple(w2.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be a 1D int32 tensor")
    if a.size(1) % _SF_TILE_K != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm requires aligned intermediate width multiple of {_SF_TILE_K}")
    if w2.size(0) % _SF_TILE_M != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm requires aligned hidden width multiple of {_SF_TILE_M}")

    if capacity is not None:
        expert_capacity = capacity
    else:
        expert_capacity = _get_blockscaled_expert_capacity()
    _validate_blockscaled_capacity(cu_seqlens_m, expert_capacity)
    num_experts = w2.size(2)
    weight_fp8, weight_scales = _quantize_w2_cached(w2, protocol)
    a_fp8, a_scales = _pack_quantize_grouped_rows(a, cu_seqlens_m, num_experts, expert_capacity, protocol)
    packed_a_scales = pack_blockscaled_1x32_scales(a_scales, a.size(1))

    grouped_out = torch.empty(
        num_experts,
        expert_capacity,
        w2.size(0),
        dtype=(a.dtype if out_dtype is None else out_dtype),
        device=a.device,
    )

    config = default_config(a.device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm does not support swap_ab configs")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(weight_fp8),
        "D": GemmTensorInfo(grouped_out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported FP8 blockscaled type/major combination")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = None
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cutlass_torch.current_stream()
    a_scale_cute = _make_cute_tensor_dynamic(packed_a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(weight_scales, leading_dim=1)

    compile_key = (
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        tuple(tensor_infos["A"].tensor.shape),
        tuple(tensor_infos["A"].tensor.stride()),
        tuple(tensor_infos["B"].tensor.shape),
        tuple(tensor_infos["B"].tensor.stride()),
        tuple(tensor_infos["D"].tensor.shape),
        tuple(tensor_infos["D"].tensor.stride()),
        tuple(packed_a_scales.shape),
        tuple(weight_scales.shape),
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    return grouped_out


def blockscaled_fp8_gemm_varlen(
    a: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Blockscaled FP8 GEMM using the varlen scheduler — no grouped packing.

    Unlike ``blockscaled_fp8_gemm_grouped`` this keeps the activation flat
    (total_M, K) and relies on ``cu_seqlens_m`` to map token-ranges to experts,
    avoiding the expert-capacity env-var, 3-D padding, and unpack step.

    Parameters
    ----------
    a : Tensor (total_M, K) bf16 — activations, contiguous in expert order.
    w2 : Tensor (H, I, E) bf16 — expert weights.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    protocol : FP8Protocol
    out_dtype : optional output dtype (defaults to ``a.dtype``).

    Returns
    -------
    Tensor (total_M, H) — result of per-expert matmuls.

    Notes
    -----
    Uses the rank-aware ``_tile_atom_to_shape_SF_rank_aware`` installed at
    module load time to handle the 2-D activation tensors produced by the
    varlen_m scheduler.
    """
    protocol = validate_fp8_runtime_support(_blockscaled_protocol(protocol), a.device, quack_enabled=True)
    if get_device_capacity(a.device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_gemm_varlen currently supports Blackwell only")

    if a.ndim != 2:
        raise ValueError(f"expected activation to be 2D, got shape {tuple(a.shape)}")
    if w2.ndim != 3:
        raise ValueError(f"expected w2 to be 3D (H, I, E), got shape {tuple(w2.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be a 1D int32 tensor")
    if a.size(1) % _SF_TILE_K != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm_varlen requires intermediate width multiple of {_SF_TILE_K}")
    if w2.size(0) % _SF_TILE_M != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm_varlen requires hidden width multiple of {_SF_TILE_M}")

    total_M, K = a.shape
    H, I, num_experts = w2.shape
    assert K == I, f"K dimension mismatch: a has {K}, w2 has I={I}"
    assert cu_seqlens_m.shape[0] == num_experts + 1, (
        f"cu_seqlens_m length {cu_seqlens_m.shape[0]} != num_experts+1 ({num_experts + 1})"
    )

    # Blockscaled GEMM requires each expert segment ≥ _SF_VEC_SIZE tokens.
    # Smaller segments corrupt the blockscaled scale-factor tile layout.
    if not torch.cuda.is_current_stream_capturing():
        seg_lens = cu_seqlens_m[1:] - cu_seqlens_m[:-1]
        active = seg_lens > 0
        if active.any():
            min_active = int(seg_lens[active].min().item())
            if min_active < _SF_VEC_SIZE:
                raise ValueError(
                    f"blockscaled_fp8_gemm_varlen requires all active expert segments "
                    f"≥ {_SF_VEC_SIZE} tokens, but min active segment has {min_active}. "
                    f"Use the bf16 fallback for small per-expert token counts."
                )

    # --- weight quantization (cached, 3D: E×H×I) ---
    weight_fp8, weight_scales = _quantize_w2_cached(w2, protocol)

    # --- activation quantization (flat 2D: total_M × K) ---
    a_fp8, a_scales_raw = quantize_activation_blockscaled_fast(a.contiguous(), group_size=_SF_VEC_SIZE)
    packed_a_scales = pack_blockscaled_1x32_scales(a_scales_raw, K)

    # --- output tensor (flat 2D: total_M × H) ---
    d_dtype = a.dtype if out_dtype is None else out_dtype
    out = torch.empty(total_M, H, dtype=d_dtype, device=a.device)

    config = default_config(a.device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm_varlen does not support swap_ab configs")

    # --- tensor infos ---
    # For varlen_m: A is 2D (total_M, K), B is 3D (E, H, I), D is 2D (total_M, H)
    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(weight_fp8),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }

    # Permute B from (E, H, I) → (H, I, E)  [the standard (n, k, l) convention]
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for varlen path")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)

    # --- scheduler (persistent for varlen) ---
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )

    # --- varlen args ---
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        cu_seqlens_k=None,
        A_idx=None,
        max_active_clusters=max_active_clusters,
        cluster_shape_mnk=cluster_shape_mnk,
        tensors=tensor_infos,
        num_epi_tensormaps=GemmDefaultSm100.num_epi_tensormaps,
        pingpong=config.pingpong,
    )

    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cutlass_torch.current_stream()
    a_scale_cute = _make_cute_tensor_dynamic(packed_a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(weight_scales, leading_dim=1)

    compile_key = (
        "varlen",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        tuple(tensor_infos["A"].tensor.shape),
        tuple(tensor_infos["A"].tensor.stride()),
        tuple(tensor_infos["B"].tensor.shape),
        tuple(tensor_infos["B"].tensor.stride()),
        tuple(tensor_infos["D"].tensor.shape),
        tuple(tensor_infos["D"].tensor.stride()),
        tuple(packed_a_scales.shape),
        tuple(weight_scales.shape),
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    return out


def _auto_capacity(cu_seqlens_m: torch.Tensor) -> int:
    """Compute minimum expert capacity aligned to tile boundary from cu_seqlens_m."""
    max_tokens = int((cu_seqlens_m[1:] - cu_seqlens_m[:-1]).max().item())
    return int(math.ceil(max_tokens / _SF_TILE_M) * _SF_TILE_M)


def blockscaled_fp8_gemm(
    a: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    capacity = _auto_capacity(cu_seqlens_m)
    grouped_out = blockscaled_fp8_gemm_grouped(
        a,
        w2,
        cu_seqlens_m,
        protocol=protocol,
        out_dtype=(a.dtype if out is None else out.dtype),
        capacity=capacity,
    )
    return _unpack_grouped_rows(grouped_out, cu_seqlens_m, out=out)
