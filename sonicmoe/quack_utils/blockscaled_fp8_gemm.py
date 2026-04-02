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
# Rank-aware tile_atom_to_shape_SF with gather_A SFA override
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
#
# Additionally, when gather_A + blockscaled is active, the SFA M-dimension
# must be TK (not T). A thread-local override allows callers to temporarily
# replace the M-dimension in the Shape passed to this function.
# ---------------------------------------------------------------------------

import threading as _threading

_SFA_M_OVERRIDE = _threading.local()


def set_sfa_m_override(m_override: int | None) -> None:
    """Set a thread-local M-dimension override for SFA layout derivation.

    When set, the next call to tile_atom_to_shape_SF will use this value
    for the M-dimension instead of the one in Shape. Automatically cleared
    after one use.
    """
    _SFA_M_OVERRIDE.value = m_override
    _SFA_M_OVERRIDE.call_count = 0


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
# Fast-path cache: skip validation/tensor-info/compile-key on steady-state calls.
# Maps (total_M, K, H, E, out_dtype, w_shape, w_stride, a_sc_cols, w_sc_shape, dev)
#   → (compiled_fn, scheduler_args, epi_args)
_GEMM_FAST_PATH: dict[tuple, tuple] = {}
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
    vals = _get_cu_seqlens_cpu(cu_seqlens_m)
    max_expert_tokens = max(vals[i + 1] - vals[i] for i in range(len(vals) - 1)) if len(vals) > 1 else 0
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
    weight_fp8_ehi, packed_scales = _quantize_weight_3d_triton(weight_ehi)
    result = (weight_fp8_ehi, packed_scales)
    if len(_WEIGHT_CACHE) > 2:
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = result
    return result


def clear_blockscaled_fp8_weight_cache() -> None:
    _WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE.clear()
    _PAD_PLAN_CACHE.clear()


def evict_fp8_weight_cache_entry(w: torch.Tensor) -> None:
    """Remove cached FP8 data for *w* from all blockscaled weight caches.

    Call this right after a cached FP8 weight has been consumed to release
    GPU memory eagerly instead of waiting for the global cache clear at the
    next optimizer step.  In training the cache would miss anyway (because
    ``w._version`` increments on each in-place update), so the eviction
    adds zero overhead.
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    _FUSED_WEIGHT_CACHE.pop(key, None)
    _VARLEN_WEIGHT_CACHE.pop(key, None)
    _WEIGHT_CACHE.pop(key, None)


def _get_cu_seqlens_cpu(cu_seqlens: torch.Tensor) -> tuple:
    """Return cu_seqlens values as a Python tuple, cached on the tensor object.

    Exactly ONE D2H sync per tensor object lifetime.  All subsequent calls
    with the same tensor are pure Python attribute lookups — zero GPU sync.
    """
    cached = getattr(cu_seqlens, '_cached_cpu_tuple', None)
    if cached is not None:
        return cached
    cpu_tuple = tuple(cu_seqlens.tolist())
    cu_seqlens._cached_cpu_tuple = cpu_tuple
    return cpu_tuple


def _get_padding_plan(
    cu_seqlens_m: torch.Tensor,
    total_M: int,
) -> tuple[bool, torch.Tensor | None, int, torch.Tensor | None]:
    """Compute (and cache) CTA-tile padding plan for a cu_seqlens_m tensor.

    Uses tensor-attribute caching: zero D2H sync after first call per tensor.
    Even on cache miss, all decisions are made from CPU tuple — no `.item()`.
    """
    cpu_tuple = _get_cu_seqlens_cpu(cu_seqlens_m)
    cached = _PAD_PLAN_CACHE.get(cpu_tuple)
    if cached is not None:
        return cached

    # Pure Python arithmetic — zero GPU operations
    cpu_lens = [cpu_tuple[i + 1] - cpu_tuple[i] for i in range(len(cpu_tuple) - 1)]
    remainders = [s % _SF_TILE_M for s in cpu_lens]
    needs_pad = any(r > 0 for r in remainders)

    if needs_pad:
        padded_lens = [s + (_SF_TILE_M - r) % _SF_TILE_M for s, r in zip(cpu_lens, remainders)]
        padded_cu_list = [0]
        for pl in padded_lens:
            padded_cu_list.append(padded_cu_list[-1] + pl)
        padded_total = padded_cu_list[-1]

        # H2D transfer (async, no sync) + GPU scatter
        padded_cu = torch.tensor(padded_cu_list, dtype=torch.int32, device=cu_seqlens_m.device)
        token_idx = torch.arange(total_M, device=cu_seqlens_m.device, dtype=torch.int64)
        expert_ids = torch.searchsorted(cu_seqlens_m, token_idx, right=True) - 1
        local_off = token_idx - cu_seqlens_m[expert_ids].to(torch.int64)
        dst_idx = padded_cu[expert_ids].to(torch.int64) + local_off
        plan = (True, padded_cu, padded_total, dst_idx)
    else:
        plan = (False, None, 0, None)

    if len(_PAD_PLAN_CACHE) > 16:
        _PAD_PLAN_CACHE.clear()
    _PAD_PLAN_CACHE[cpu_tuple] = plan
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

    # 2D grid: (row_blocks, col_blocks) for better SM utilization.
    # BR=32, GPB=12 → ~16K blocks → 43% peak BW on B200 (vs 22% with BR=16, GPB=all).
    BLOCK_ROWS = 32
    GROUPS_PER_BLOCK = min(num_groups, 12)
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


# ---------------------------------------------------------------------------
# Fused transpose + quantize + ISA-pack kernel for weight-gradient data prep
# ---------------------------------------------------------------------------
# Single pass: read flat expert-sorted (TK, dim) bf16 →
#   transpose+quantize → write (E*dim, capacity) fp8 + ISA-packed scales.
# Optional gather_idx fuses the gather step too.
# Eliminates the separate pack_blockscaled_1x32_scales call entirely.

@triton.jit
def _fused_transpose_quantize_kernel(
    src_ptr,           # (TK, dim) bf16
    gather_idx_ptr,    # (TK,) int32 gather index, or unused if HAS_GATHER=False
    dst_fp8_ptr,       # (E*dim, capacity) fp8
    dst_packed_ptr,    # (E, per_batch_storage) u8 — ISA-packed scales
    dim,
    capacity,
    per_batch_storage,
    src_stride_row,
    src_stride_col,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,      # 32
    BLOCK_DIM: tl.constexpr,       # 128
    # ISA layout constants
    SF_TILE_M: tl.constexpr,       # 128
    SF_TILE_K: tl.constexpr,       # 128
    SF_TILE_STORAGE: tl.constexpr, # 512
):
    """Each block processes one (expert, dim_block, quant_group) tile.

    Reads a (GROUP_SIZE, BLOCK_DIM) tile from the input, transposes to
    (BLOCK_DIM, GROUP_SIZE), quantizes with E8M0 blockscaling, and writes
    FP8 data + ISA-packed scales in a single fused pass.
    """
    pid_row = tl.program_id(0)       # expert_id * num_dim_blocks + dim_block
    pid_group = tl.program_id(1)     # quant group along capacity

    num_dim_blocks = tl.cdiv(dim, BLOCK_DIM)
    expert_id = pid_row // num_dim_blocks
    dim_block = pid_row % num_dim_blocks

    cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    # Compute source rows — optionally indirect through gather_idx
    flat_token_ids = expert_id * capacity + cap_offs
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + flat_token_ids).to(tl.int64)
    else:
        src_rows = flat_token_ids.to(tl.int64)

    # Load (GROUP_SIZE, BLOCK_DIM) tile
    dim_mask = dim_offs[None, :] < dim
    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

    # Transpose in registers: (GROUP_SIZE, BLOCK_DIM) → (BLOCK_DIM, GROUP_SIZE)
    values_t = tl.trans(values)

    # Blockscaled E8M0 quantization (per row of BLOCK_DIM, over GROUP_SIZE elements)
    block_amax = tl.max(tl.abs(values_t), axis=1)

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

    quantized = (values_t * quant_scale[:, None]).to(tl.float8e4nv)

    # Write FP8 data: dst_fp8[expert_id*dim + dim_offs, group*GROUP_SIZE + j]
    out_rows = expert_id * dim + dim_offs
    out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    fp8_ptrs = dst_fp8_ptr + out_rows[:, None] * capacity + out_cols[None, :]
    out_mask = dim_offs[:, None] < dim
    tl.store(fp8_ptrs, quantized, mask=out_mask)

    # Write ISA-packed scales directly (inline index computation).
    # ISA layout: tile_base + row_base + k_in_tile
    # where:
    #   row = dim_offs (each element in BLOCK_DIM)
    #   scale_block = pid_group (which capacity group)
    #   row_tiles = row // SF_TILE_M
    #   row_in_tile = row % SF_TILE_M
    #   k_tiles_idx = scale_block // (SF_TILE_K // GROUP_SIZE)
    #   k_in_tile = scale_block % (SF_TILE_K // GROUP_SIZE)
    #   tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    #   row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    #   index = tile_base + row_base + k_in_tile
    k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE  # 4

    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    k_tiles_idx = pid_group // groups_per_k_tile
    k_in_tile = pid_group % groups_per_k_tile

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    isa_index = tile_base + row_base + k_in_tile

    scale_ptrs = dst_packed_ptr + expert_id.to(tl.int64) * per_batch_storage + isa_index.to(tl.int64)
    tl.store(scale_ptrs, e8m0_byte, mask=dim_offs < dim)


def fused_transpose_quantize_for_wgrad(
    flat_sorted: torch.Tensor,
    num_experts: int,
    capacity: int,
    dim: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused transpose+quantize+ISA-pack for wgrad operands.

    Reads flat expert-sorted (TK, dim) bf16, produces:
      - fp8_out: (E, dim, capacity) fp8e4m3
      - packed_scales: ISA-packed scales for CUTLASS (directly, no separate pack step)

    Optionally fuses a gather step via *gather_idx*.
    """
    device = flat_sorted.device
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    BLOCK_DIM = 128

    fp8_flat = torch.empty(num_experts * dim, capacity, dtype=torch.float8_e4m3fn, device=device)

    # Pre-allocate ISA-packed scales directly (fill with 1s = E8M0 exponent 127 = scale 1.0)
    per_batch_storage = _storage_per_batch(dim, capacity)
    packed_scales = torch.ones(num_experts, per_batch_storage, dtype=torch.float8_e8m0fnu, device=device)

    grid = (num_experts * _div_up(dim, BLOCK_DIM), capacity // GROUP_SIZE)

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else flat_sorted  # dummy, unused

    _fused_transpose_quantize_kernel[grid](
        flat_sorted,
        gather_ptr,
        fp8_flat,
        packed_scales.view(torch.uint8),
        dim,
        capacity,
        per_batch_storage,
        flat_sorted.stride(0),
        flat_sorted.stride(1),
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )

    fp8_3d = fp8_flat.reshape(num_experts, dim, capacity)
    return fp8_3d, packed_scales


# ---------------------------------------------------------------------------
# Column-wise blockscaled quantize + ISA-pack for wgrad (varlen_k approach)
# ---------------------------------------------------------------------------
# Groups of 32 along dim 0 of (TK, dim) tensor. ISA-packed scales for
# the logical (dim, TK) layout that CUTLASS sees after .T view.
# Optional gather_idx fuses scatter-read into the quantize pass.

@triton.jit
def _colwise_quantize_and_pack_kernel(
    src_ptr,
    gather_idx_ptr,
    dst_fp8_ptr,
    dst_packed_ptr,
    total_K,
    dim,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_K: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Column-wise blockscaled quantize + ISA-pack.

    2D grid: (num_groups_along_TK, num_blocks_along_dim).
    Quantization groups are along TK (axis 0).
    ISA-packed scales target the LOGICAL (dim, TK) layout.
    """
    pid_group = tl.program_id(0)
    pid_dim = tl.program_id(1)

    k_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_offs = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    k_mask = k_offs < total_K
    dim_mask = dim_offs < dim
    mask = k_mask[:, None] & dim_mask[None, :]

    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + k_offs, mask=k_mask, other=0).to(tl.int64)
    else:
        src_rows = k_offs.to(tl.int64)

    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    block_amax = tl.max(tl.abs(values), axis=0)

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

    quantized = (values * quant_scale[None, :]).to(tl.float8e4nv)

    dst_ptrs = dst_fp8_ptr + k_offs[:, None] * dst_stride_row + dim_offs[None, :] * dst_stride_col
    tl.store(dst_ptrs, quantized, mask=mask)

    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE
    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    k_tiles_idx = pid_group // groups_per_k_tile
    k_in_tile = pid_group % groups_per_k_tile

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    isa_index = tile_base + row_base + k_in_tile

    scale_ptrs = dst_packed_ptr + isa_index.to(tl.int64)
    tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


# ---------------------------------------------------------------------------
# Fused transpose + rowwise quantize for wgrad operands
# ---------------------------------------------------------------------------
# Instead of colwise quant on (TK, dim) → scattered access,
# we transpose + rowwise quant in a single kernel:
#   Read (TK, dim) bf16 → SMEM transpose → write (dim, TK) FP8 + ISA scales
# This converts the cache-hostile colwise pattern into cache-friendly rowwise.

@triton.jit
def _fused_transpose_rowquant_kernel(
    src_ptr,              # (TK, dim) bf16 source, row-major
    gather_idx_ptr,       # (TK,) int32 gather indices (or dummy)
    dst_fp8_ptr,          # (dim, TK) fp8 output, row-major
    dst_packed_scale_ptr, # ISA-packed scales for (dim, TK) layout
    TK,                   # total K dimension
    dim,                  # H or I dimension
    src_stride_row,       # stride of source along TK
    src_stride_col,       # stride of source along dim
    dst_stride_row,       # stride of output along dim (= TK)
    dst_stride_col,       # stride of output along TK (= 1)
    k_tiles,              # ceil(TK / SF_TILE_K) for ISA layout
    HAS_GATHER: tl.constexpr,
    TILE_TK: tl.constexpr,   # tile size along TK (e.g. 128)
    TILE_DIM: tl.constexpr,  # tile size along dim (e.g. 32)
    GROUP_SIZE: tl.constexpr, # 32 for blockscaled
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,  # TILE_TK // GROUP_SIZE (e.g. 4)
):
    """Fused transpose + rowwise blockscaled quantize.

    Grid: (dim_tiles, tk_tiles)
    Each block reads a (TILE_TK, TILE_DIM) tile from src,
    transposes to (TILE_DIM, TILE_TK), then rowwise-quantizes
    with groups of GROUP_SIZE along TK, writing FP8 + ISA-packed scales.

    The output is (dim, TK) row-major FP8 with ISA-packed scales
    for the (dim, TK) logical layout — exactly what wgrad CUTLASS needs.
    """
    pid_dim = tl.program_id(0)   # which dim tile
    pid_tk = tl.program_id(1)    # which TK tile

    dim_offs = pid_dim * TILE_DIM + tl.arange(0, TILE_DIM)
    tk_offs = pid_tk * TILE_TK + tl.arange(0, TILE_TK)
    dim_mask = dim_offs < dim
    tk_mask = tk_offs < TK

    # Load (TILE_TK, TILE_DIM) tile from src with optional gather
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + tk_offs, mask=tk_mask, other=0).to(tl.int64)
    else:
        src_rows = tk_offs.to(tl.int64)

    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    mask_2d = tk_mask[:, None] & dim_mask[None, :]
    values = tl.load(src_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    # values shape: (TILE_TK, TILE_DIM) in registers

    # Transpose: now process as (TILE_DIM, TILE_TK) — each "row" is one dim element
    # For rowwise quant, we need amax over groups of GROUP_SIZE along TK axis
    # Reshape values to (TILE_DIM, TILE_TK) by transposing
    # Triton: values[tk, d] → transposed[d, tk]
    # We process group-by-group along TK within this tile

    # ISA layout for output (dim, TK): dim is M-axis, TK is K-axis
    dst_row_tiles = dim_offs // SF_TILE_M
    dst_row_in_tile = dim_offs % SF_TILE_M
    dst_row_base_offset = (dst_row_in_tile % 32) * 16 + (dst_row_in_tile // 32) * 4

    packed_scale_i32 = tl.zeros([TILE_DIM], dtype=tl.int32)

    for g in tl.range(0, GROUPS_PER_TILE):
        group_tk_start = g * GROUP_SIZE
        group_tk_offs = group_tk_start + tl.arange(0, GROUP_SIZE)
        group_mask = (pid_tk * TILE_TK + group_tk_offs) < TK

        # Extract the (GROUP_SIZE, TILE_DIM) sub-tile from values
        # values[group_tk_offs, :] → (GROUP_SIZE, TILE_DIM)
        subtile = tl.load(
            src_ptr + (src_rows[group_tk_start:group_tk_start + GROUP_SIZE] if HAS_GATHER
                       else (pid_tk * TILE_TK + group_tk_offs).to(tl.int64))[:, None] * src_stride_row
            + dim_offs[None, :] * src_stride_col,
            mask=group_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Amax along GROUP_SIZE (axis=0) → per-dim-element scale
        block_amax = tl.max(tl.abs(subtile), axis=0)  # (TILE_DIM,)

        # Pure-integer E8M0 computation
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_clamped = tl.maximum(e8m0_i32, 0)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        # Quantize: scale then cast
        quantized = (subtile * quant_scale[None, :]).to(tl.float8e4nv)
        # quantized shape: (GROUP_SIZE, TILE_DIM)

        # Write transposed: dst[dim, tk] = quantized^T
        # Output row = dim_offs, output col = actual_tk_offs
        actual_tk_offs = pid_tk * TILE_TK + group_tk_offs
        out_ptrs = dst_fp8_ptr + dim_offs[None, :] * dst_stride_row + actual_tk_offs[:, None] * dst_stride_col
        # But we need to write (GROUP_SIZE, TILE_DIM) transposed as (TILE_DIM, GROUP_SIZE)
        # Use swapped indexing:
        out_ptrs_t = dst_fp8_ptr + dim_offs[:, None] * dst_stride_row + actual_tk_offs[None, :] * dst_stride_col
        tl.store(out_ptrs_t, tl.trans(quantized), mask=dim_mask[:, None] & group_mask[None, :])

        # Pack scale bytes into uint32 (4 groups per TILE_TK=128)
        packed_scale_i32 = packed_scale_i32 | ((e8m0_clamped & 0xFF) << (g * 8))

    # Write packed uint32 ISA scales
    global_group_idx = pid_tk  # which k-tile (each TILE_TK = SF_TILE_K = 128)
    tile_base = (dst_row_tiles * k_tiles + global_group_idx) * SF_TILE_STORAGE
    packed_offset_i32 = (tile_base + dst_row_base_offset) // 4
    scale_ptr_i32 = dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=dim_mask)


def fused_transpose_quantize_and_pack(
    src: torch.Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused transpose + rowwise blockscaled quantize for wgrad operands.

    Replaces colwise_quantize_and_pack with ~3-5x better performance by
    converting the cache-hostile column-wise access pattern into row-wise.

    Input: (TK, dim) bf16 row-major
    Output: (dim, TK) fp8 row-major + ISA-packed scales for (dim, TK) layout

    Parameters are identical to colwise_quantize_and_pack.
    """
    H = logical_rows   # dim
    TK = logical_cols
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    TILE_TK = _SF_TILE_K       # 128 (= SF_TILE_K, matches ISA tile)
    TILE_DIM = 32              # process 32 dim rows per block

    # Output: (dim, TK) row-major FP8
    fp8_out = torch.empty(H, TK, dtype=torch.float8_e4m3fn, device=src.device)

    # ISA-packed scales for (dim, TK) layout
    per_batch_storage = _storage_per_batch(H, TK)
    if H % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0:
        packed_scales = torch.empty((1, per_batch_storage), dtype=torch.uint8, device=src.device)
    else:
        packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)

    k_tiles = _div_up(TK, _SF_TILE_K)
    grid = (_div_up(H, TILE_DIM), _div_up(TK, TILE_TK))
    groups_per_tile = TILE_TK // GROUP_SIZE  # 4

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src

    _fused_transpose_rowquant_kernel[grid](
        src, gather_ptr, fp8_out, packed_scales,
        TK, H,
        src.stride(0), src.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        HAS_GATHER=has_gather,
        TILE_TK=TILE_TK,
        TILE_DIM=TILE_DIM,
        GROUP_SIZE=GROUP_SIZE,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_TILE=groups_per_tile,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


def colwise_quantize_and_pack(
    src: torch.Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-wise blockscaled FP8 quantize + ISA-pack for wgrad operands.

    Quantizes (TK, dim) with groups of 32 along TK (dim 0).
    ISA-packed scales target the logical (dim, TK) layout.

    Parameters
    ----------
    src : (T_or_TK, dim) bf16 source.
    logical_rows : dim — rows of the logical (dim, TK) matrix.
    logical_cols : TK — cols (= total K for varlen_k).
    gather_idx : optional (TK,) int32 gather indices.

    Returns
    -------
    fp8_data : (TK, dim) float8_e4m3fn
    packed_scales : (1, packed_size) float8_e8m0fnu in ISA layout for (dim, TK).
    """
    H = logical_rows
    TK = logical_cols
    GROUP_SIZE = _SF_VEC_SIZE
    BLOCK_DIM = 128

    fp8_out = torch.empty(TK, H, dtype=torch.float8_e4m3fn, device=src.device)
    per_batch_storage = _storage_per_batch(H, TK)
    packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)

    num_groups = _div_up(TK, GROUP_SIZE)
    k_tiles = _div_up(TK, _SF_TILE_K)
    grid = (num_groups, _div_up(H, BLOCK_DIM))

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src

    _colwise_quantize_and_pack_kernel[grid](
        src, gather_ptr, fp8_out, packed_scales,
        TK, H,
        src.stride(0), src.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# varlen_k blockscaled FP8 GEMM for weight gradients
# ---------------------------------------------------------------------------

_COMPILE_CACHE_VK: dict = {}
_GEMM_FAST_PATH_VK: dict = {}


def _run_cutlass_blockscaled_gemm_varlen_k(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    out_dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUTLASS blockscaled FP8 GEMM with varlen_k scheduling.

    D[e] = A[:, K_e] @ B[K_e, :] for each expert e.

    A is logical (M, total_K) M-major — physical a_fp8 is (total_K, M).
    B is logical (N, total_K) N-major — physical b_fp8 is (total_K, N).
    D is (num_experts, M, N).

    If *out* is provided it must be a contiguous (num_experts, M, N) tensor
    of the correct dtype.
    """
    a_logical = a_fp8.T
    b_logical = b_fp8.T
    if out is None:
        out = torch.empty(num_experts, M, N, dtype=out_dtype, device=device)

    fast_key = (
        "vk", M, N, total_K, num_experts, out_dtype,
        a_fp8.shape[0], a_fp8.shape[1],
        b_fp8.shape[0], b_fp8.shape[1],
        a_scales.size(1), b_scales.size(1),
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH_VK.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        d_permuted = out.permute(1, 2, 0)
        a_cute = _make_cute_tensor_dynamic(a_logical, leading_dim=0)
        b_cute = _make_cute_tensor_dynamic(b_logical, leading_dim=0)
        d_cute = _make_cute_tensor_dynamic(d_permuted, leading_dim=1)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
        )
        stream = cutlass_torch.current_stream()
        compiled(
            a_cute, b_cute, d_cute, None,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return out

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled varlen_k does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_logical),
        "B": GemmTensorInfo(b_logical),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)

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
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        Float32, tensor_infos["D"].dtype,
        tensor_infos["A"].major, tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for varlen_k")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None, batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
    )
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cutlass_torch.current_stream()
    a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)

    compile_key = (
        "vk",
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn, cluster_shape_mnk,
        M, N, total_K,
        a_scales.size(1), b_scales.size(1),
        tensor_infos["A"].major, tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong, _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE_VK.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32, tensor_infos["A"].dtype,
            tile_shape_mn, cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE, gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
            epi_args, scheduler_args, varlen_args, current_stream,
            a_scale_cute, b_scale_cute,
        )
        _COMPILE_CACHE_VK[compile_key] = compiled

    _GEMM_FAST_PATH_VK[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
        epi_args, scheduler_args, varlen_args, current_stream,
        a_scale_cute, b_scale_cute,
    )
    return out


def blockscaled_fp8_wgrad_varlen_k(
    a_src: torch.Tensor,
    b_src: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    *,
    a_gather_idx: Optional[torch.Tensor] = None,
    b_gather_idx: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full FP8 weight-gradient: column-wise quant + varlen_k GEMM.

    Computes dW[e] = A_e^T @ B_e for each expert via:
    1. Column-wise quantize A and B (groups of 32 along TK).
    2. varlen_k CUTLASS GEMM with non-contiguous FP8 via TMA.

    Parameters
    ----------
    a_src : (T_or_TK, M) bf16 — e.g., dout, dz, or x.
    b_src : (T_or_TK, N) bf16 — e.g., y1s, dz, or x.
    cu_seqlens_k : (E+1,) int32 expert boundaries.
    M : output rows (e.g., H).
    N : output cols (e.g., I or 2*I).
    a_gather_idx : optional (TK,) int32 gather for A.
    b_gather_idx : optional (TK,) int32 gather for B.
    out_dtype : output dtype.
    out : optional pre-allocated (E, M, N) contiguous output tensor.

    Returns
    -------
    dW : (E, M, N) weight gradient.
    """
    a_rows = a_gather_idx.size(0) if a_gather_idx is not None else a_src.shape[0]
    b_rows = b_gather_idx.size(0) if b_gather_idx is not None else b_src.shape[0]
    if a_rows != b_rows:
        raise ValueError(
            "effective row count must match between operands, "
            f"got a={a_rows}, b={b_rows}"
        )
    TK = a_rows
    num_experts = cu_seqlens_k.shape[0] - 1
    device = b_src.device

    a_fp8, a_scales = colwise_quantize_and_pack(
        a_src, logical_rows=M, logical_cols=TK, gather_idx=a_gather_idx,
    )
    b_fp8, b_scales = colwise_quantize_and_pack(
        b_src, logical_rows=N, logical_cols=TK, gather_idx=b_gather_idx,
    )

    return _run_cutlass_blockscaled_gemm_varlen_k(
        a_fp8, a_scales, b_fp8, b_scales,
        cu_seqlens_k, M, N, TK, num_experts,
        out_dtype, device, out=out,
    )


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
    GROUPS_PER_K_TILE: tl.constexpr = 4,
):
    """Fused blockscaled quantize + ISA scale pack — 2D grid version.

    Each program handles BLOCK_ROWS rows × 1 k-tile (GROUPS_PER_K_TILE groups).
    2D grid: (row_blocks, k_tiles) for maximum SM occupancy.
    Packs GROUPS_PER_K_TILE (4) scale bytes into uint32 per k-tile.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    k_tile_idx = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Pre-compute row-dependent ISA layout (invariant across groups)
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    packed_scale_i32 = tl.zeros([BLOCK_ROWS], dtype=tl.int32)

    for g in tl.range(0, GROUPS_PER_K_TILE):
        group_id = k_tile_idx * GROUPS_PER_K_TILE + g
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        src_ptrs = src_ptr + row_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Pure-integer E8M0 computation
        block_amax = tl.max(tl.abs(values), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_clamped = tl.maximum(e8m0_i32, 0)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Pack scale byte into uint32 (little-endian: byte 0 at bits[0:8])
        packed_scale_i32 = packed_scale_i32 | ((e8m0_clamped & 0xFF) << (g * 8))

    # Write 4 scale bytes as single uint32 (row_base_offset is always 4-byte aligned)
    tile_base = (row_tiles * k_tiles + k_tile_idx) * SF_TILE_STORAGE
    packed_offset_i32 = (tile_base + row_base_offset) // 4
    scale_ptr_i32 = dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=row_mask_1d)


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

        # Pure-integer E8M0 computation (no transcendentals)
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

        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

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
    if TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales = torch.empty(
            (1, per_batch_storage), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales = torch.full(
            (1, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )

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
    # When M and K are tile-aligned, every scale byte is written by the kernel.
    # Skip expensive torch.full fill kernel — use torch.empty instead.
    if M % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales = torch.empty(
            (1, per_batch_storage), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales = torch.full(
            (1, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )

    BLOCK_ROWS = 32
    groups_per_k_tile = _SF_TILE_K // group_size  # 4 for default SF_TILE_K=128, GROUP_SIZE=32
    grid = (_div_up(M, BLOCK_ROWS), k_tiles)
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
        GROUPS_PER_K_TILE=groups_per_k_tile,
    )
    return fp8_out, packed_scales.view(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# Three-step optimized gather: T-quant → fp8_gather → scale_gather
# ---------------------------------------------------------------------------
# Replaces the monolithic gather_quantize_and_pack_activation when the source
# tensor T is small enough to fit in L2 cache (e.g., T=4096..8192, K≤4096).
# 1. quantize_and_pack_activation(x) on (T, K) — ~2-8µs for small T
# 2. fp8 data gather via index_select — ~15-25µs (L2-resident reads)
# 3. ISA-packed scale gather — ~3-8µs (very small I/O)
# Total ~20-40µs vs ~96-99µs for gather_quantize_and_pack_activation.
# Numerically identical: scale for row r is computed from x[r, :] regardless.

@triton.jit
def _gather_isa_packed_scales_kernel(
    src_scale_ptr,     # ISA-packed scales for T rows (uint8)
    gather_idx_ptr,    # (TK,) int32/int64 gather indices
    dst_scale_ptr,     # ISA-packed scales for TK rows (uint8)
    TK,                # number of destination rows
    src_k_tiles: tl.constexpr,   # ceil(K / SF_TILE_K) for T layout
    dst_k_tiles: tl.constexpr,   # ceil(K / SF_TILE_K) for TK layout (same value)
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS_PER_K_TILE: tl.constexpr,
):
    """Gather ISA-packed scales from T-sized layout to TK-sized layout.

    2D grid: (row_blocks, k_tiles).
    Each block processes BLOCK_ROWS destination rows × 1 k-tile (4 scale groups).
    Reads 4 source scale bytes as uint32, writes 4 dest scale bytes as uint32.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    k_tile_idx = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < TK

    # Load gather indices
    gather_ids = tl.load(gather_idx_ptr + row_ids, mask=row_mask, other=0)

    # Source row ISA offsets (T-layout)
    src_row_tiles = gather_ids // SF_TILE_M
    src_row_in_tile = gather_ids % SF_TILE_M
    src_row_base_offset = (src_row_in_tile % 32) * 16 + (src_row_in_tile // 32) * 4
    src_tile_base = (src_row_tiles * src_k_tiles + k_tile_idx) * SF_TILE_STORAGE

    # Dest row ISA offsets (TK-layout)
    dst_row_tiles = row_ids // SF_TILE_M
    dst_row_in_tile = row_ids % SF_TILE_M
    dst_row_base_offset = (dst_row_in_tile % 32) * 16 + (dst_row_in_tile // 32) * 4
    dst_tile_base = (dst_row_tiles * dst_k_tiles + k_tile_idx) * SF_TILE_STORAGE

    # Read 4 scale bytes as uint32 from source, write to dest
    src_offset_i32 = (src_tile_base + src_row_base_offset) // 4
    dst_offset_i32 = (dst_tile_base + dst_row_base_offset) // 4

    src_ptr_i32 = src_scale_ptr.to(tl.pointer_type(tl.int32))
    dst_ptr_i32 = dst_scale_ptr.to(tl.pointer_type(tl.int32))

    packed_val = tl.load(src_ptr_i32 + src_offset_i32, mask=row_mask, other=0)
    tl.store(dst_ptr_i32 + dst_offset_i32, packed_val, mask=row_mask)


def fast_gather_quantize_and_pack_activation(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three-step optimized gather+quantize: T-quant → fp8_gather → scale_gather.

    Numerically identical to gather_quantize_and_pack_activation, but 3-5x
    faster when T << TK (common in MoE with high top-K).

    Step 1: quantize_and_pack_activation on small T-sized tensor (~2-8µs)
    Step 2: gather fp8 data via index_select (~15-25µs, L2-resident source)
    Step 3: gather ISA-packed scales via Triton kernel (~3-8µs)

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
    T, K = x.shape
    k_tiles = _div_up(K, _SF_TILE_K)

    # Step 1: quantize small T-sized tensor
    x_fp8_t, scales_t = quantize_and_pack_activation(x, group_size)

    # Step 2: gather fp8 data (source is T-sized, fits in L2 for typical T)
    gather_idx_i64 = gather_idx.long()
    fp8_out = x_fp8_t.index_select(0, gather_idx_i64)

    # Step 3: gather ISA-packed scales
    per_batch_storage_tk = _storage_per_batch(TK, K)
    if TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales_tk = torch.empty(
            (1, per_batch_storage_tk), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales_tk = torch.full(
            (1, per_batch_storage_tk), 127, dtype=torch.uint8, device=x.device
        )

    BLOCK_ROWS = 32
    grid = (_div_up(TK, BLOCK_ROWS), k_tiles)
    groups_per_k_tile = _SF_TILE_K // group_size  # typically 4
    _gather_isa_packed_scales_kernel[grid](
        scales_t.view(torch.uint8),
        gather_idx,
        packed_scales_tk,
        TK,
        src_k_tiles=k_tiles,
        dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS,
        GROUPS_PER_K_TILE=groups_per_k_tile,
    )

    del x_fp8_t, scales_t
    return fp8_out, packed_scales_tk.view(torch.float8_e8m0fnu)


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

    BLOCK_ROWS = 16  # Production-shape: consistent with gather_quantize optimizations
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
    cpu_tuple = _get_cu_seqlens_cpu(cu_seqlens_m)
    cpu_lens = [cpu_tuple[i + 1] - cpu_tuple[i] for i in range(len(cpu_tuple) - 1)]
    max_tpe = max(cpu_lens) if cpu_lens else 0

    if max_tpe == 0:
        per_batch_storage = max(_storage_per_batch(0, K), 1)
        empty_scales = torch.full(
            (E, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )
        return fp8_data, empty_scales.view(torch.float8_e8m0fnu)

    raw_uint8 = raw_scales.view(torch.uint8)
    min_tpe = min(cpu_lens)

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


def _quantize_weight_3d_triton(
    w_enk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous 3D (E, N, K) weight tensor using the fast Triton kernel.

    Exploits the fact that when N % SF_TILE_M == 0, the ISA scale tile boundaries
    align perfectly at expert boundaries. So (E, N, K) can be reshaped to (E*N, K),
    quantized as a single 2D tensor, and reshaped back — producing identical results
    to per-expert quantization but in a single kernel launch.

    Falls back to per-expert loop when N is not tile-aligned.

    Returns (w_fp8 (E, N, K), packed_scales) with ISA-packed E8M0 scales.
    """
    E, N, K = w_enk.shape
    assert w_enk.is_contiguous(), "Weight must be contiguous (E, N, K)"

    if N % _SF_TILE_M == 0:
        # Fast path: single kernel launch for all experts
        w_2d = w_enk.reshape(E * N, K)
        fp8_2d, packed_scales = quantize_and_pack_activation(w_2d)
        return fp8_2d.reshape(E, N, K), packed_scales
    else:
        # Fallback: per-expert loop (still much faster than PyTorch eager)
        fp8_slices = []
        scale_slices = []
        for e in range(E):
            fp8_e, scales_e = quantize_and_pack_activation(w_enk[e])
            fp8_slices.append(fp8_e)
            scale_slices.append(scales_e)
        return torch.stack(fp8_slices), torch.cat(scale_slices, dim=-1)


def precompute_weight_fp8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize a 3D expert weight to blockscaled FP8 with ISA-packed scales (cached).

    Converts (dim0, dim1, E) bf16 weights to (E, dim0, dim1) fp8 + packed scales.
    Uses a dedicated cache (NOT shared with fused_dgated variants which quantize
    in transposed layout with scales along a different dimension).

    Parameters
    ----------
    w : Tensor (dim0, dim1, E) bf16 — expert weights in any layout.

    Returns
    -------
    w_fp8 : Tensor (E, dim0, dim1) float8_e4m3fn — contiguous row-major.
    w_scales : Tensor packed float8_e8m0fnu in ISA layout
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _VARLEN_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w_ehi = w.permute(2, 0, 1).contiguous()
    w_fp8, w_scales_packed = _quantize_weight_3d_triton(w_ehi)
    result = (w_fp8, w_scales_packed)
    if len(_VARLEN_WEIGHT_CACHE) > 8:
        _VARLEN_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE[key] = result
    return result


_VARLEN_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor],
] = {}


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
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Return .mT view (E, K, N) so gemm_gated_tuned's B.mT recovers (E, N, K)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
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
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Return .mT view (E, K=H, N=I) so gemm_dgated_tuned's B.mT recovers (E, N=I, K=H)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


def precompute_weight_fp8_for_direct_fused_dgated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for direct low-level gemm_dgated blockscaled path.

    Reuses the fused_dgated cache — identical physical (E, N, K) contiguous layout.
    Returns the contiguous tensor (not .mT view) that the low-level kernel consumes
    directly, bypassing gemm_interface's B.mT.contiguous() copy.

    Parameters
    ----------
    w : Tensor (H, I, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, N=I, K=H) float8_e4m3fn — contiguous physical layout.
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K=H).
    """
    key = (
        w.untyped_storage().data_ptr(),
        w._version,
        tuple(w.shape),
        tuple(w.stride()),
    )
    # Check unified fused cache first (shared with precompute_weight_fp8_for_fused_dgated)
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        w_fp8_view, w_scales = cached
        # fused_dgated stores .mT view (E, H, I); we need contiguous (E, I, H)
        if not w_fp8_view.is_contiguous():
            return w_fp8_view.mT, w_scales  # .mT of .mT = contiguous original
        return w_fp8_view, w_scales

    w_enk = w.permute(2, 1, 0).contiguous()  # (E, N=I, K=H) contiguous
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Store contiguous in fused cache; fused_dgated will create .mT view from it
    result_contiguous = (w_fp8_enk, w_scales_packed)
    result_view = (w_fp8_enk.mT, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result_view  # Store .mT view (fused_dgated convention)
    return result_contiguous  # Return contiguous (direct convention)


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
    assume_aligned: bool = False,
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
    assume_aligned : if True, skip the CTA-tile alignment check (avoids
        D2H sync).  Caller must guarantee all expert segments in
        cu_seqlens_m are multiples of the CTA tile size (128).
    """

    # --- Hot path: pre-quantized + assume_aligned (production steady-state) ---
    if (
        assume_aligned
        and a.dtype == torch.float8_e4m3fn
        and a_scales is not None
        and w_fp8 is not None
        and w_scales is not None
    ):
        total_M, K = a.shape
        H = w.shape[0]
        num_experts = w.shape[2]
        d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        return _run_cutlass_blockscaled_gemm(
            a, a_scales, w_fp8, w_scales, cu_seqlens_m,
            total_M, K, H, num_experts, d_dtype, a.device,
        )

    # --- Full validation path (legacy / first few iterations) ---
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
    if assume_aligned or torch.cuda.is_current_stream_capturing():
        needs_pad = False
    else:
        needs_pad, padded_cu, padded_total, dst_idx = _get_padding_plan(cu_seqlens_m, total_M)

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
    """Pure CUTLASS blockscaled FP8 GEMM — all inputs pre-quantized.

    Uses a shape-pinned fast path to skip validation, GemmTensorInfo
    construction, major-order determination, compile-key hashing, and
    _COMPILE_CACHE lookup on steady-state calls.  Only cute-tensor
    creation (unavoidable — different data pointers) and the kernel
    launch remain on the hot path.
    """

    # --- Fast path: steady-state calls with identical problem shapes ---
    fast_key = (
        total_M, K, H, num_experts, out_dtype,
        w_fp8.shape[0], w_fp8.shape[1], w_fp8.shape[2],
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        a_scales_packed.size(1),
        w_scales_packed.shape[0], w_scales_packed.shape[1],
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        out = torch.empty(total_M, H, dtype=out_dtype, device=device)
        # B permute for varlen_m: (H,K,E) → (K,E,H)
        w_permuted = w_fp8.permute(1, 2, 0)
        # Cute tensors — all row-major (leading_dim=1) in production
        a_cute = _make_cute_tensor_dynamic(a_fp8, leading_dim=1)
        b_cute = _make_cute_tensor_dynamic(w_permuted, leading_dim=1)
        d_cute = _make_cute_tensor_dynamic(out, leading_dim=1)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales_packed, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(w_scales_packed, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m, cu_seqlens_k=None, A_idx=None,
        )
        stream = cutlass_torch.current_stream()
        compiled(
            a_cute, b_cute, d_cute, None,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return out

    # --- Slow path (first call for this shape config) ---
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

    # Populate fast-path cache for subsequent calls
    _GEMM_FAST_PATH[fast_key] = (compiled, scheduler_args, epi_args)

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


def blockscaled_fp8_weight_grad_gemm_fast(
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    a_gather_idx: Optional[torch.Tensor] = None,
    b_gather_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Optimized blockscaled FP8 weight-gradient GEMM with fused data prep.

    Same semantics as ``blockscaled_fp8_weight_grad_gemm`` but:
      - For aligned routing, skips separate pack (zero-copy reshape)
      - Fuses transpose + quantize + optional gather into one Triton kernel
      - ~2-3x faster data preparation than the decomposed pipeline

    Parameters
    ----------
    a_flat : Tensor (TK, dim_A) bf16 — first operand (will be transposed).
    b_flat : Tensor (TK, dim_B) bf16 — second operand.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    a_gather_idx : optional (TK,) int32 — gather index for a_flat.
    b_gather_idx : optional (TK,) int32 — gather index for b_flat.
    out : optional pre-allocated Tensor (E, dim_A, dim_B).
    out_dtype : output element type (default bf16).
    """
    device = a_flat.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm_fast requires Blackwell (sm_100)")
    if a_flat.ndim != 2 or b_flat.ndim != 2:
        raise ValueError(f"expected 2D operands, got a={tuple(a_flat.shape)}, b={tuple(b_flat.shape)}")

    # When gather indices are provided, effective row count comes from the
    # gather index length (TK), not from a_flat/b_flat which may be shorter (T).
    TK = int(cu_seqlens_m[-1].item())
    a_rows = a_gather_idx.size(0) if a_gather_idx is not None else a_flat.size(0)
    b_rows = b_gather_idx.size(0) if b_gather_idx is not None else b_flat.size(0)
    if a_rows != TK or b_rows != TK:
        raise ValueError(
            f"effective row count must match cu_seqlens_m[-1]={TK}, "
            f"got a={a_rows}, b={b_rows}"
        )

    num_experts = cu_seqlens_m.shape[0] - 1
    dim_A = a_flat.size(1)
    dim_B = b_flat.size(1)
    capacity = _auto_capacity(cu_seqlens_m)

    if dim_A % _SF_TILE_M != 0:
        raise RuntimeError(f"dim_A ({dim_A}) must be a multiple of {_SF_TILE_M}")
    if dim_B % _SF_TILE_M != 0:
        raise RuntimeError(f"dim_B ({dim_B}) must be a multiple of {_SF_TILE_M}")

    # Fused transpose + quantize (+ optional gather) in a single kernel pass
    a_fp8, packed_a_scales = fused_transpose_quantize_for_wgrad(
        a_flat, num_experts, capacity, dim_A, gather_idx=a_gather_idx,
    )
    b_fp8, packed_b_scales = fused_transpose_quantize_for_wgrad(
        b_flat, num_experts, capacity, dim_B, gather_idx=b_gather_idx,
    )

    # Allocate output
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    out_is_alias = out is not None and not out.is_contiguous()
    if out is not None and out.is_contiguous():
        grouped_out = out
    else:
        grouped_out = torch.empty(num_experts, dim_A, dim_B, dtype=d_dtype, device=device)

    # CUTLASS grouped blockscaled GEMM
    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm_fast does not support swap_ab")

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
        "weight_grad_fast",
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
