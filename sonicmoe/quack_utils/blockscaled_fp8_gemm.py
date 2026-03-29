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
_PAD_PLAN_CACHE: dict = {}       # content-key → plan
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
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int | None, int, str, str, str, str]:
    return (
        weight.untyped_storage().data_ptr(),
        id(weight),  # guards against CUDA memory reuse after del
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
    # Key on storage identity + version + view layout so that repeated
    # permute() calls on the same underlying Parameter hit cache.
    key = (
        w2.untyped_storage().data_ptr(),
        w2._version,
        tuple(w2.shape),
        tuple(w2.stride()),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    weight_ehi = w2.permute(2, 0, 1).contiguous()
    weight_fp8_ehi, weight_scales = quantize_activation_blockwise(weight_ehi, protocol)
    packed_scales = pack_blockscaled_1x32_scales(weight_scales, weight_ehi.size(-1))
    result = (weight_fp8_ehi, packed_scales)
    if len(_WEIGHT_CACHE) > 2:
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = result
    return result


def clear_blockscaled_fp8_weight_cache() -> None:
    _WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE.clear()
    _PAD_PLAN_CACHE.clear()


def _get_padding_plan(
    cu_seqlens_m: torch.Tensor,
    total_M: int,
) -> tuple[bool, torch.Tensor | None, int, torch.Tensor | None]:
    """Compute (and cache) CTA-tile padding plan for a cu_seqlens_m tensor.

    Content-based cache keyed on cu_seqlens values (one D2H sync per call).
    """
    content_key = tuple(cu_seqlens_m.tolist())
    cached = _PAD_PLAN_CACHE.get(content_key)
    if cached is not None:
        return cached

    seg_lens = cu_seqlens_m[1:] - cu_seqlens_m[:-1]
    remainders = seg_lens % _SF_TILE_M
    needs_pad = bool(remainders.any().item())

    if needs_pad:
        padded_lens = seg_lens + (_SF_TILE_M - remainders) % _SF_TILE_M
        padded_cu = torch.zeros_like(cu_seqlens_m)
        padded_cu[1:] = torch.cumsum(padded_lens, dim=0)
        padded_total = int(padded_cu[-1].item())

        token_idx = torch.arange(total_M, device=cu_seqlens_m.device, dtype=torch.int64)
        expert_ids = torch.searchsorted(cu_seqlens_m, token_idx, right=True) - 1
        local_off = token_idx - cu_seqlens_m[expert_ids].to(torch.int64)
        dst_idx = padded_cu[expert_ids].to(torch.int64) + local_off
        plan = (True, padded_cu, padded_total, dst_idx)
    else:
        plan = (False, None, 0, None)

    if len(_PAD_PLAN_CACHE) > 16:
        _PAD_PLAN_CACHE.clear()
    _PAD_PLAN_CACHE[content_key] = plan
    return plan


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
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """Quantize a flat (M, K) bf16 tensor to blockscaled FP8 with 1×GROUP_SIZE scales.

    Each program processes BLOCK_ROWS × GROUPS_PER_BLOCK groups.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    group_base = tl.program_id(1) * GROUPS_PER_BLOCK

    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    for g in range(GROUPS_PER_BLOCK):
        group_id = group_base + g
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        src_ptrs = src_ptr + row_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        block_amax = tl.max(tl.abs(values), axis=1)

        # Pure-integer E8M0 computation (matches CUTLASS/sgl convention):
        # E8M0 = biased_exponent(amax) - 8 + carry
        # where carry = 1 iff mantissa(amax) > mantissa(fp8_max=448=1.75*2^8)
        # This avoids all log2/ceil/exp2 float precision issues.
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        # fp8_max = 448 = 1.75 * 2^8; mantissa of 1.75 in IEEE754 = 0x600000
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        # Quant scale = 2^(127 - e8m0) = 1 / dequant_scale
        # Construct exact power-of-2 float via bit manipulation.
        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)
        scale_ptrs = dst_scale_ptr + row_ids * scale_stride_row + group_id * scale_stride_col
        tl.store(scale_ptrs, e8m0_byte, mask=row_mask_1d)


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
    scale_out = torch.empty(M, num_groups, dtype=torch.uint8, device=x.device)

    # Process 4 rows × all-groups-per-block to maximize work per block.
    # For (65536, 4096): 128 groups per row, GROUPS_PER_BLOCK=128 → 1 col-block.
    # Grid: (16384, 1) = 16K blocks, each processing 4×128 groups = 16K elements.
    BLOCK_ROWS = 32
    GROUPS_PER_BLOCK = min(num_groups, 128)
    grid = (_div_up(M, BLOCK_ROWS), _div_up(num_groups, GROUPS_PER_BLOCK))
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
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
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


@triton.jit
def _quantize_and_pack_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_packed_scale_ptr,
    rows,
    cols,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused blockscaled quantize + ISA scale pack in a single kernel.

    Each program handles BLOCK_ROWS rows × ALL scale groups (loop over K).
    1D grid: one block per BLOCK_ROWS row chunk. Dramatically reduces grid
    size from M/4 × K/32 (~1M) to M/32 (~1K), cutting launch overhead.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Pre-compute row-dependent ISA layout (invariant across groups)
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        # --- Load bf16 values ---
        src_ptrs = src_ptr + row_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # --- Compute per-row-group scale (E8M0) ---
        block_amax = tl.max(tl.abs(values), axis=1)
        raw_scale = block_amax / fp8_max
        positive = raw_scale > 0
        exponent = tl.where(positive, tl.ceil(tl.log2(tl.where(positive, raw_scale, 1.0))), 0.0)
        quant_scale = tl.exp2(-exponent)

        # --- Quantize to fp8 ---
        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # --- Write E8M0 scale directly into ISA tile layout ---
        dequant_scale = tl.exp2(exponent)
        scale_i32 = dequant_scale.to(tl.float32).to(tl.int32, bitcast=True)
        e8m0_byte = ((scale_i32 >> 23) & 0xFF).to(tl.uint8)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile

        tl.store(dst_packed_scale_ptr + packed_offset, e8m0_byte, mask=row_mask_1d)


@triton.jit
def _gather_quantize_and_pack_kernel(
    src_ptr,          # original (T, K) tensor
    gather_idx_ptr,   # (TK,) int32/int64 gather indices
    dst_fp8_ptr,      # (TK, K) fp8 output
    dst_packed_scale_ptr,  # ISA-packed scales
    rows,             # TK
    cols,             # K
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused gather + blockscaled quantize + ISA scale pack.

    1D grid: each block handles BLOCK_ROWS rows × ALL scale groups (loop).
    Reads from src[gather_idx[row], col] and writes fp8 + ISA-packed scales
    without materializing a bf16 gathered intermediate tensor.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Load gather indices (invariant across groups)
    gather_ids = tl.load(gather_idx_ptr + row_ids, mask=row_mask_1d, other=0)

    # Pre-compute row-dependent ISA layout
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        # Gather from original tensor using indices
        src_ptrs = src_ptr + gather_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Quantize
        block_amax = tl.max(tl.abs(values), axis=1)
        raw_scale = block_amax / fp8_max
        positive = raw_scale > 0
        exponent = tl.where(positive, tl.ceil(tl.log2(tl.where(positive, raw_scale, 1.0))), 0.0)
        quant_scale = tl.exp2(-exponent)
        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # ISA-packed scale
        dequant_scale = tl.exp2(exponent)
        scale_i32 = dequant_scale.to(tl.float32).to(tl.int32, bitcast=True)
        e8m0_byte = ((scale_i32 >> 23) & 0xFF).to(tl.uint8)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile_val = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile_val

        tl.store(dst_packed_scale_ptr + packed_offset, e8m0_byte, mask=row_mask_1d)


def gather_quantize_and_pack_activation(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather + blockscaled FP8 quantize + ISA scale pack.

    Eliminates the bf16 gathered intermediate tensor.
    Instead of: x_gathered = x[gather_idx]; fp8, scales = quantize_and_pack(x_gathered)
    Does:       fp8, scales = gather_quantize_and_pack(x, gather_idx)

    Parameters
    ----------
    x : Tensor (T, K) bf16 — original activation (NOT gathered).
    gather_idx : Tensor (TK,) int32/int64 — token gather indices.

    Returns
    -------
    fp8_data : Tensor (TK, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    TK = gather_idx.shape[0]
    K = x.shape[1]
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    fp8_out = torch.empty(TK, K, dtype=torch.float8_e4m3fn, device=x.device)
    per_batch_storage = _storage_per_batch(TK, K)
    packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=x.device)

    BLOCK_ROWS = 32
    grid = (_div_up(TK, BLOCK_ROWS),)
    _gather_quantize_and_pack_kernel[grid](
        x,
        gather_idx,
        fp8_out,
        packed_scales,
        TK, K,
        x.stride(0), x.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


def quantize_and_pack_activation(
    x: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 activation to blockscaled FP8 with ISA-packed scales.

    Single fused Triton kernel: bf16 → fp8 + ISA-layout packed E8M0 scales.
    Eliminates the intermediate raw_scales tensor and fancy-indexing scatter.

    Parameters
    ----------
    x : Tensor (M, K) bf16/fp16 — contiguous activation tensor.
    group_size : block size for scale computation (default 32).

    Returns
    -------
    fp8_data : Tensor (M, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    x = x.contiguous()
    M, K = x.shape
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)

    per_batch_storage = _storage_per_batch(M, K)
    # Initialize to 127 (E8M0 encoding of scale=1.0: exponent bias 127)
    packed_scales = torch.full(
        (1, per_batch_storage), 127, dtype=torch.uint8, device=x.device
    )

    BLOCK_ROWS = 32
    grid = (_div_up(M, BLOCK_ROWS),)
    _quantize_and_pack_kernel[grid](
        x,
        fp8_out,
        packed_scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


@triton.jit
def _pad_quantize_and_pack_kernel(
    src_ptr,               # original (total_M, K) bf16
    src_idx_ptr,           # (padded_total,) int64 — padded → original row, -1 for pad
    dst_fp8_ptr,           # (padded_total, K) fp8 output
    dst_packed_scale_ptr,  # ISA-packed scales for padded layout
    rows,                  # padded_total
    cols,                  # K
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused pad + blockscaled quantize + ISA scale pack.

    1D grid: each block handles BLOCK_ROWS rows × ALL scale groups (loop).
    For data rows (src_idx >= 0): reads from src, quantizes, packs scales.
    For padding rows (src_idx == -1): writes fp8 zeros + e8m0 scale 0.
    Avoids materializing a bf16 padded intermediate buffer.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Load source indices (invariant across groups)
    src_rows = tl.load(src_idx_ptr + row_ids, mask=row_mask_1d, other=-1)
    is_data = src_rows >= 0
    safe_src_rows = tl.where(is_data, src_rows, 0)

    # Pre-compute row-dependent ISA layout
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        data_mask = is_data[:, None] & col_mask

        # Read from original data (use row 0 for padding rows to avoid OOB)
        src_ptrs = src_ptr + safe_src_rows[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=data_mask, other=0.0).to(tl.float32)

        # Quantize (zeros for padding rows)
        block_amax = tl.max(tl.abs(values), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        # Write fp8 data
        mask = row_mask_1d[:, None] & col_mask
        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Write ISA-packed scale
        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile

        tl.store(dst_packed_scale_ptr + packed_offset, e8m0_byte, mask=row_mask_1d)


_SRC_IDX_CACHE: dict[tuple, torch.Tensor] = {}


def pad_quantize_and_pack_activation(
    a: torch.Tensor,
    padded_total: int,
    dst_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused pad + quantize + ISA-pack: avoids bf16 padded intermediate.

    Instead of allocating a (padded_total, K) bf16 buffer, scattering, then
    quantizing 25% more rows, this function:
    1. Computes the inverse mapping (padded_row → original_row)
    2. Runs a single kernel that reads from original data and writes padded
       fp8 + ISA-packed scales directly

    Parameters
    ----------
    a : Tensor (total_M, K) bf16 — original (non-padded) activation.
    padded_total : int — total padded rows.
    dst_idx : Tensor (total_M,) int64 — original → padded row mapping.

    Returns
    -------
    fp8_data : Tensor (padded_total, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    total_M, K = a.shape
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    # Compute inverse index: padded_row → original_row (cached)
    cache_key = (total_M, padded_total, id(dst_idx))
    src_idx = _SRC_IDX_CACHE.get(cache_key)
    if src_idx is None:
        src_idx = torch.full((padded_total,), -1, device=a.device, dtype=torch.int64)
        src_idx[dst_idx] = torch.arange(total_M, device=a.device, dtype=torch.int64)
        if len(_SRC_IDX_CACHE) > 16:
            _SRC_IDX_CACHE.clear()
        _SRC_IDX_CACHE[cache_key] = src_idx

    fp8_out = torch.empty(padded_total, K, dtype=torch.float8_e4m3fn, device=a.device)
    per_batch_storage = _storage_per_batch(padded_total, K)
    packed_scales = torch.zeros((1, per_batch_storage), dtype=torch.uint8, device=a.device)

    BLOCK_ROWS = 32
    grid = (_div_up(padded_total, BLOCK_ROWS),)
    _pad_quantize_and_pack_kernel[grid](
        a,
        src_idx,
        fp8_out,
        packed_scales,
        padded_total,
        K,
        a.stride(0),
        a.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


def quantize_and_pack_activation_varlen(
    x: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize varlen activation to blockscaled FP8 with per-expert ISA-packed scales.

    Three-step pipeline:
    1. Quantize entire (TK, K) in one kernel → fp8 + raw E8M0 scales (TK, num_groups)
    2. Reshape raw scales to per-expert (E, max_TPE, num_groups) with padding
    3. ISA-pack → (E, per_expert_storage) packed scales

    For uniform TPE, step 2 is a zero-copy reshape.

    Parameters
    ----------
    x : Tensor (TK, K) bf16 — contiguous, expert-sorted activation.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    group_size : block size for scale computation (default 32).

    Returns
    -------
    fp8_data : Tensor (TK, K) float8_e4m3fn
    packed_scales : Tensor (E, per_expert_storage) float8_e8m0fnu in ISA layout
    """
    x = x.contiguous()
    TK, K = x.shape
    E = cu_seqlens_m.numel() - 1

    # Step 1: Quantize entire tensor in one kernel
    fp8_data, raw_scales = quantize_activation_blockscaled_fast(x, group_size)
    # raw_scales: (TK, num_groups) float8_e8m0fnu
    num_groups = raw_scales.size(1)

    # Step 2: Reshape raw scales to per-expert with padding
    offsets_cpu = cu_seqlens_m.cpu()
    lens = offsets_cpu[1:] - offsets_cpu[:-1]
    max_tpe = int(lens.max().item())

    if max_tpe == 0:
        per_batch_storage = max(_storage_per_batch(0, K), 1)
        empty_scales = torch.full(
            (E, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )
        return fp8_data, empty_scales.view(torch.float8_e8m0fnu)

    raw_uint8 = raw_scales.view(torch.uint8)
    min_tpe = int(lens.min().item())

    if min_tpe == max_tpe and TK == E * max_tpe:
        # Uniform TPE: zero-copy reshape
        padded_scales = raw_uint8.view(E, max_tpe, num_groups)
    else:
        # Non-uniform: vectorized scatter into padded tensor
        padded_scales = torch.full(
            (E, max_tpe, num_groups), 127, dtype=torch.uint8, device=x.device
        )
        row_indices = torch.arange(TK, device=x.device)
        # searchsorted with right=True on cu_seqlens_m[1:] gives expert_id for each row
        expert_ids = torch.searchsorted(cu_seqlens_m[1:].long(), row_indices.long(), right=True)
        local_rows = row_indices - cu_seqlens_m[expert_ids]
        padded_scales[expert_ids, local_rows] = raw_uint8

    # Step 3: ISA-pack the 3D scales
    packed = pack_blockscaled_1x32_scales(
        padded_scales.view(torch.float8_e8m0fnu), K
    )
    return fp8_data, packed


def precompute_weight_fp8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize a 3D expert weight to blockscaled FP8 with ISA-packed scales (cached).

    Converts (dim0, dim1, E) bf16 weights to (E, dim0, dim1) fp8 + packed scales.
    Uses the same storage-identity cache as the fused variants so repeated
    calls with the same weight tensor are free.

    Parameters
    ----------
    w : Tensor (dim0, dim1, E) bf16 — expert weights in any layout.

    Returns
    -------
    w_fp8 : Tensor (E, dim0, dim1) float8_e4m3fn
    w_scales : Tensor packed float8_e8m0fnu in ISA layout
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w_ehi = w.permute(2, 0, 1).contiguous()
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    w_fp8, w_scales_raw = quantize_activation_blockwise(w_ehi, proto)
    w_scales_packed = pack_blockscaled_1x32_scales(w_scales_raw, w_ehi.size(-1))
    result = (w_fp8, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 2:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


_FUSED_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def precompute_weight_fp8_for_fused_gated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for fused gemm_gated blockscaled path (cached).

    gemm_gated_tuned expects B in (L, K, N) format and applies .mT internally
    to get (L, N, K).  Blockscaled 1x32 quantisation is done on the physical
    (contiguous) layout which must match what CUTLASS sees after .mT.

    We quantise in (E, N, K) contiguous layout (scales along K), then return
    the .mT view (E, K, N) so that gemm_gated_tuned's internal .mT recovers
    the original contiguous (E, N, K) layout with correctly-aligned scales.

    Parameters
    ----------
    w : Tensor (2I, H, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, K, N) float8_e4m3fn — .mT view of contiguous (E, N, K).
        gemm_gated_tuned's internal B.mT will recover the contiguous (E, N, K).
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K-axis
        of the contiguous layout)
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # w is (2I, H, E) -> (E, 2I, H) contiguous = (E, N, K) physical layout
    w_enk = w.permute(2, 1, 0).mT.contiguous()  # (E, N=2I, K=H) contiguous
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    w_fp8_enk, w_scales_raw = quantize_activation_blockwise(w_enk, proto)
    w_scales_packed = pack_blockscaled_1x32_scales(w_scales_raw, w_enk.size(-1))
    # Return .mT view (E, K, N) so gemm_gated_tuned's B.mT recovers (E, N, K)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 2:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


def precompute_weight_fp8_for_fused_dgated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for fused gemm_dgated blockscaled path (cached).

    gemm_dgated_tuned expects B in (L, K, N) format and applies .mT internally
    to get (L, N, K).  For w2 (H, I, E), the input to gemm_dgated_tuned is
    w2.permute(2, 0, 1) = (E, H, I) = (E, K, N).  After .mT: (E, I, H) = (E, N, K).

    We quantise in (E, N=I, K=H) contiguous layout (scales along K=H), then
    return the .mT view (E, H, I) so gemm_dgated_tuned's internal .mT recovers
    the original contiguous (E, I, H) layout with correctly-aligned scales.

    Parameters
    ----------
    w : Tensor (H, I, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, H, I) float8_e4m3fn — .mT view of contiguous (E, I, H).
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K=H)
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # w is (H, I, E) → (E, I, H) contiguous = (E, N, K) physical layout
    w_enk = w.permute(2, 1, 0).contiguous()  # (E, N=I, K=H) contiguous
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    w_fp8_enk, w_scales_raw = quantize_activation_blockwise(w_enk, proto)
    w_scales_packed = pack_blockscaled_1x32_scales(w_scales_raw, w_enk.size(-1))
    # Return .mT view (E, K=H, N=I) so gemm_dgated_tuned's B.mT recovers (E, N=I, K=H)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 2:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


def blockscaled_fp8_gemm_varlen(
    a: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol | None = None,
    out_dtype: Optional[torch.dtype] = None,
    a_scales: Optional[torch.Tensor] = None,
    w_fp8: Optional[torch.Tensor] = None,
    w_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blockscaled FP8 GEMM using the varlen scheduler.

    Supports two calling conventions:

    **Pre-quantized (preferred)** — pass fp8 data + packed scales directly::

        a_fp8, a_scales = quantize_and_pack_activation(x_bf16)
        w_fp8, w_scales = precompute_weight_fp8(w_bf16)
        out = blockscaled_fp8_gemm_varlen(
            a_fp8, w_bf16, cu,             # w_bf16 only for shape/device
            a_scales=a_scales, w_fp8=w_fp8, w_scales=w_scales,
            out_dtype=torch.bfloat16,
        )

    **Legacy (backward compat)** — pass bf16 and let the function quantize::

        out = blockscaled_fp8_gemm_varlen(x_bf16, w_bf16, cu, protocol=protocol)

    Parameters
    ----------
    a : Tensor (total_M, K) — fp8 (pre-quantized) or bf16 (legacy).
    w : Tensor (H, I, E) bf16 — expert weights (used for shape; ignored
        if w_fp8/w_scales are provided).
    cu_seqlens_m : Tensor (E+1,) int32 — expert boundaries.
    protocol : FP8Protocol — required for legacy bf16 path.
    out_dtype : output dtype (defaults to bf16).
    a_scales : packed ISA-layout scales for ``a`` (pre-quantized path).
    w_fp8 : pre-quantized weight fp8 (E,H,I).
    w_scales : packed ISA-layout scales for weights.
    """
    device = a.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_gemm_varlen requires Blackwell (sm_100)")

    if a.ndim != 2:
        raise ValueError(f"expected activation 2D, got {tuple(a.shape)}")
    if w.ndim != 3:
        raise ValueError(f"expected w 3D (H, I, E), got {tuple(w.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be 1D int32")

    total_M, K = a.shape
    H, I, num_experts = w.shape
    assert K == I, f"K mismatch: a has {K}, w has I={I}"
    assert cu_seqlens_m.shape[0] == num_experts + 1

    if K % _SF_TILE_K != 0:
        raise RuntimeError(f"K must be multiple of {_SF_TILE_K}, got {K}")
    if H % _SF_TILE_M != 0:
        raise RuntimeError(f"H must be multiple of {_SF_TILE_M}, got {H}")

    # --- Determine if inputs are pre-quantized ---
    pre_quantized_act = (a.dtype == torch.float8_e4m3fn and a_scales is not None)
    pre_quantized_wt = (w_fp8 is not None and w_scales is not None)

    # --- CTA tile alignment padding ---
    if not torch.cuda.is_current_stream_capturing():
        needs_pad, padded_cu, padded_total, dst_idx = _get_padding_plan(cu_seqlens_m, total_M)
    else:
        needs_pad = False

    if needs_pad:
        # --- Weight quantization (shared by both paths) ---
        if pre_quantized_wt:
            weight_fp8, weight_scales = w_fp8, w_scales
        else:
            bs_proto = _blockscaled_protocol(protocol or FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32))
            weight_fp8, weight_scales = _quantize_w2_cached(w, bs_proto)

        if pre_quantized_act:
            # Pre-quantized with non-aligned segments: dequant is lossy.
            # This path should be avoided (see _all_segments_128_aligned guard
            # in functional/__init__.py). Fallback: raw cast + re-quantize.
            a_bf16 = a.to(torch.bfloat16)
        else:
            a_bf16 = a

        # Fused pad + quantize + ISA-pack: avoids the bf16 padded buffer.
        padded_fp8, padded_scales = pad_quantize_and_pack_activation(
            a_bf16, padded_total, dst_idx,
        )

        d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        padded_out = _run_cutlass_blockscaled_gemm(
            padded_fp8, padded_scales,
            weight_fp8, weight_scales,
            padded_cu,
            padded_total, K, H, num_experts,
            d_dtype, device,
        )
        return padded_out[dst_idx]

    # --- Weight quantization ---
    if pre_quantized_wt:
        weight_fp8, weight_scales = w_fp8, w_scales
    else:
        # Legacy path: ensure blockscaled protocol for weight quantization
        bs_proto = _blockscaled_protocol(protocol or FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32))
        weight_fp8, weight_scales = _quantize_w2_cached(w, bs_proto)

    # --- Activation quantization ---
    if pre_quantized_act:
        a_fp8_data, packed_a_scales = a, a_scales
    else:
        a_fp8_data, packed_a_scales = quantize_and_pack_activation(
            a.contiguous(), group_size=_SF_VEC_SIZE
        )

    # --- Run CUTLASS GEMM kernel ---
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    return _run_cutlass_blockscaled_gemm(
        a_fp8_data, packed_a_scales,
        weight_fp8, weight_scales,
        cu_seqlens_m,
        total_M, K, H, num_experts,
        d_dtype, device,
    )


def _run_cutlass_blockscaled_gemm(
    a_fp8: torch.Tensor,
    a_scales_packed: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales_packed: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    total_M: int,
    K: int,
    H: int,
    num_experts: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Pure CUTLASS blockscaled FP8 GEMM — all inputs pre-quantized."""
    out = torch.empty(total_M, H, dtype=out_dtype, device=device)

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm_varlen does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(w_fp8),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }

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
        raise TypeError("Unsupported FP8 blockscaled type/major combination")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )

    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        cu_seqlens_k=None,
        A_idx=None,
    )

    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cutlass_torch.current_stream()
    a_scale_cute = _make_cute_tensor_dynamic(a_scales_packed, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(w_scales_packed, leading_dim=1)

    compile_key = (
        "varlen",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        K,
        tuple(tensor_infos["B"].tensor.shape),
        tuple(tensor_infos["B"].tensor.stride()),
        H,
        a_scales_packed.size(1),
        tuple(w_scales_packed.shape),
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


def blockscaled_fp8_weight_grad_gemm(
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Blockscaled FP8 weight-gradient GEMM: D[e] = A_e^T @ B_e per expert.

    Both *a_flat* and *b_flat* are flat 2-D activations in expert-sorted
    token order (``TK, dim``).  ``cu_seqlens_m`` (int32, ``E+1``) marks
    expert boundaries.

    The function:

    1. Packs both operands into ``(E, capacity, dim)`` groups.
    2. Transposes to ``(E, dim, capacity)`` so that K=capacity is the
       last dimension — required for correct 1x32 blockscaled quantisation.
    3. Quantises to blockscaled FP8 (E8M0 scales).
    4. Runs a batched CUTLASS GEMM producing ``D(E, dim_A, dim_B)``.

    Parameters
    ----------
    a_flat : Tensor (TK, dim_A) bf16 — first operand (will be transposed).
    b_flat : Tensor (TK, dim_B) bf16 — second operand.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    out : optional pre-allocated Tensor (E, dim_A, dim_B).
    out_dtype : output element type (default bf16).

    Returns
    -------
    D : Tensor (E, dim_A, dim_B).
    """
    device = a_flat.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm requires Blackwell (sm_100)")
    if a_flat.ndim != 2 or b_flat.ndim != 2:
        raise ValueError(
            f"expected 2D operands, got a={tuple(a_flat.shape)}, b={tuple(b_flat.shape)}"
        )
    if a_flat.size(0) != b_flat.size(0):
        raise ValueError("a_flat and b_flat must have the same number of rows (TK)")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be 1D int32")

    num_experts = cu_seqlens_m.shape[0] - 1
    dim_A = a_flat.size(1)
    dim_B = b_flat.size(1)
    capacity = _auto_capacity(cu_seqlens_m)

    if dim_A % _SF_TILE_M != 0:
        raise RuntimeError(
            f"dim_A ({dim_A}) must be a multiple of {_SF_TILE_M} for blockscaled tiling"
        )
    if dim_B % _SF_TILE_M != 0:
        raise RuntimeError(
            f"dim_B ({dim_B}) must be a multiple of {_SF_TILE_M} for blockscaled tiling"
        )
    # capacity is already tile-aligned by _auto_capacity

    # 1. Pack flat tokens into per-expert groups: (E, cap, dim)
    a_grouped = _pack_grouped_rows(a_flat, cu_seqlens_m, num_experts, capacity)
    b_grouped = _pack_grouped_rows(b_flat, cu_seqlens_m, num_experts, capacity)

    # 2. Transpose so K=capacity is the last dim for correct blockscaling
    a_t = a_grouped.transpose(1, 2).contiguous()   # (E, dim_A, cap)
    b_t = b_grouped.transpose(1, 2).contiguous()   # (E, dim_B, cap)
    del a_grouped, b_grouped

    # 3. Quantise to blockscaled FP8 — reshape to 2-D, quantise, reshape back
    a_2d = a_t.reshape(-1, capacity)                # (E*dim_A, cap)
    a_fp8_2d, a_scales_2d = quantize_activation_blockscaled_fast(a_2d)
    a_fp8 = a_fp8_2d.reshape(num_experts, dim_A, capacity)
    a_scales_3d = a_scales_2d.reshape(num_experts, dim_A, -1)
    del a_t, a_2d, a_fp8_2d, a_scales_2d

    b_2d = b_t.reshape(-1, capacity)                # (E*dim_B, cap)
    b_fp8_2d, b_scales_2d = quantize_activation_blockscaled_fast(b_2d)
    b_fp8 = b_fp8_2d.reshape(num_experts, dim_B, capacity)
    b_scales_3d = b_scales_2d.reshape(num_experts, dim_B, -1)
    del b_t, b_2d, b_fp8_2d, b_scales_2d

    # 4. Pack scales into ISA tile layout
    packed_a_scales = pack_blockscaled_1x32_scales(a_scales_3d, capacity)
    packed_b_scales = pack_blockscaled_1x32_scales(b_scales_3d, capacity)
    del a_scales_3d, b_scales_3d

    # 5. Allocate output — CUTLASS needs a contiguous buffer
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    out_is_alias = out is not None and not out.is_contiguous()
    if out is not None and out.is_contiguous():
        grouped_out = out
    else:
        grouped_out = torch.empty(num_experts, dim_A, dim_B, dtype=d_dtype, device=device)

    # 6. Run CUTLASS grouped blockscaled GEMM
    #    A_cutlass (E, M=dim_A, K=cap)  @  B_cutlass (E, N=dim_B, K=cap)^T
    #    → D (E, M=dim_A, N=dim_B)
    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(b_fp8),
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
        raise TypeError("Unsupported FP8 blockscaled type/major combination for weight-grad GEMM")

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
    b_scale_cute = _make_cute_tensor_dynamic(packed_b_scales, leading_dim=1)

    compile_key = (
        "weight_grad",
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
        tuple(packed_b_scales.shape),
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
    if out_is_alias:
        out.copy_(grouped_out)
        return out
    return grouped_out
