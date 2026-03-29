"""Fused Triton kernels for interleaved SwiGLU forward and backward.

The interleaved layout stores gate/up pairs as z(TK, 2I) where:
  z[:, 0::2] = gate,  z[:, 1::2] = up

Provides five kernel variants:
  1. SwiGLU → bf16 (unfused, for non-fp8 paths)
  2. SwiGLU → blockscaled fp8 + raw e8m0 scales (fused forward quant)
  3. dSwiGLU → blockscaled fp8 + raw e8m0 scales (fused backward quant)
  4. SwiGLU → blockscaled fp8 + ISA-packed e8m0 scales (fused forward quant+pack)
  5. dSwiGLU → blockscaled fp8 + ISA-packed e8m0 scales (fused backward quant+pack)

Variants 4 & 5 eliminate the intermediate raw-scale tensor and
produce scales directly in the ISA tile layout consumed by CUTLASS
blockscaled GEMMs, removing the separate pack_blockscaled_1x32_scales
step entirely.
"""

import torch
import triton
import triton.language as tl

_GROUP_SIZE: tl.constexpr = 32  # 1×32 blockscaled granularity

# ISA tile layout constants (must match blockscaled_fp8_gemm.py)
_SF_VEC_SIZE = 32
_SF_TILE_M = 128
_SF_TILE_K = 128
_SF_TILE_STORAGE = _SF_TILE_M * (_SF_TILE_K // _SF_VEC_SIZE)  # 512


# ===================================================================
# Forward: z(TK, 2I) → y1(TK, I) bf16  (unfused, legacy)
# ===================================================================

@triton.jit
def _swiglu_fwd_kernel(
    Z_ptr, Y1_ptr,
    TK: tl.constexpr, I: tl.constexpr,
    stride_z_row: tl.constexpr, stride_y_row: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Each program handles one row, iterating over I in BLOCK_I chunks."""
    row = tl.program_id(0)
    z_row_base = Z_ptr + row * stride_z_row
    y_row_base = Y1_ptr + row * stride_y_row

    for j_start in range(0, I, BLOCK_I):
        j_offs = j_start + tl.arange(0, BLOCK_I)
        mask = j_offs < I

        gate = tl.load(z_row_base + j_offs * 2, mask=mask).to(tl.float32)
        up = tl.load(z_row_base + j_offs * 2 + 1, mask=mask).to(tl.float32)

        sig = tl.sigmoid(gate)
        y1 = gate * sig * up

        tl.store(y_row_base + j_offs, y1.to(tl.bfloat16), mask=mask)


def swiglu_forward_triton(z: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward: z(TK, 2I) → y1(TK, I)."""
    TK, two_I = z.shape
    assert two_I % 2 == 0
    I = two_I // 2

    y1 = torch.empty(TK, I, dtype=torch.bfloat16, device=z.device)

    BLOCK_I = min(1024, triton.next_power_of_2(I))

    _swiglu_fwd_kernel[(TK,)](
        z, y1,
        TK, I,
        z.stride(0), y1.stride(0),
        BLOCK_I=BLOCK_I,
    )
    return y1


# ===================================================================
# Forward fused: z(TK, 2I) → y1_fp8(TK, I) + e8m0_scales(TK, I//32)
# ===================================================================

@triton.jit
def _swiglu_fwd_quant_kernel(
    Z_ptr, Y1_FP8_ptr, SCALE_ptr,
    TK: tl.constexpr, I: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    stride_scale_row: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Fused SwiGLU + blockscaled fp8 quantize.

    Each program processes one row. Iterates over I in GROUP_SIZE (32)
    chunks, computing SwiGLU then immediately quantizing to fp8 with
    per-group e8m0 scale factor. Zero intermediate bf16 materialisation.
    """
    row = tl.program_id(0)
    z_base = Z_ptr + row * stride_z_row
    y_base = Y1_FP8_ptr + row * stride_y_row
    sc_base = SCALE_ptr + row * stride_scale_row

    num_groups: tl.constexpr = I // GROUP_SIZE

    for g in range(num_groups):
        j_offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

        gate = tl.load(z_base + j_offs * 2).to(tl.float32)
        up = tl.load(z_base + j_offs * 2 + 1).to(tl.float32)

        sig = tl.sigmoid(gate)
        y1 = gate * sig * up

        # Blockscaled quantize: e8m0 pow2 scale
        amax = tl.max(tl.abs(y1))
        positive = amax > 0
        exponent = tl.where(positive,
                            tl.ceil(tl.log2(tl.where(positive, amax / fp8_max, 1.0))),
                            0.0)
        quant_scale = tl.exp2(-exponent)
        y1_fp8 = (y1 * quant_scale).to(tl.float8e4nv)
        tl.store(y_base + j_offs, y1_fp8)

        # E8M0 exponent byte
        dequant = tl.exp2(exponent).to(tl.float32)
        e8m0 = ((dequant.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
        tl.store(sc_base + g, e8m0)


def swiglu_forward_quant_triton(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU forward + blockscaled fp8 quantize.

    Args:
        z: (TK, 2I) bf16 pre-activation (interleaved gate/up).

    Returns:
        y1_fp8:  (TK, I) float8_e4m3fn — quantized SwiGLU output.
        scales:  (TK, I//32) uint8 — e8m0 scale per group of 32.
    """
    TK, two_I = z.shape
    assert two_I % 2 == 0
    I = two_I // 2
    assert I % 32 == 0, f"I={I} must be multiple of 32 for blockscaled"

    num_groups = I // 32
    y1_fp8 = torch.empty(TK, I, dtype=torch.float8_e4m3fn, device=z.device)
    scales = torch.empty(TK, num_groups, dtype=torch.uint8, device=z.device)

    _swiglu_fwd_quant_kernel[(TK,)](
        z, y1_fp8, scales,
        TK, I,
        z.stride(0), y1_fp8.stride(0), scales.stride(0),
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=32,
    )
    return y1_fp8, scales


# ===================================================================
# Forward fused+packed: z(TK, 2I) → y1_fp8(TK, I) + ISA-packed scales
# ===================================================================

def _div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def _storage_per_batch(rows: int, cols: int) -> int:
    return _div_up(rows, _SF_TILE_M) * _div_up(cols, _SF_TILE_K) * _SF_TILE_STORAGE


@triton.jit
def _swiglu_fwd_quant_pack_kernel(
    Z_ptr, Y1_FP8_ptr, PACKED_SCALE_ptr,
    rows, I_dim: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused SwiGLU + blockscaled fp8 quantize + ISA scale pack.

    1D grid: each program processes BLOCK_ROWS rows × ALL scale groups,
    looping over groups internally. Hoists row-dependent ISA layout
    computations outside the group loop.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < rows

    # Hoist row-dependent ISA layout computations outside loop
    z_row_ptrs = Z_ptr + row_ids[:, None] * stride_z_row
    y_row_ptrs = Y1_FP8_ptr + row_ids[:, None] * stride_y_row
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < I_dim
        mask = row_mask[:, None] & col_mask

        # Load interleaved gate/up from z
        gate = tl.load(z_row_ptrs + col_offsets[None, :] * 2, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(z_row_ptrs + col_offsets[None, :] * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        # SwiGLU: y1 = silu(gate) * up = gate * sigmoid(gate) * up
        sig = tl.sigmoid(gate)
        y1 = gate * sig * up

        # Pure-integer E8M0 blockscaled quantize (no transcendentals)
        block_amax = tl.max(tl.abs(y1), axis=1)
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
        y1_fp8 = (y1 * quant_scale[:, None]).to(tl.float8e4nv)

        # Store fp8 data
        tl.store(y_row_ptrs + col_offsets[None, :], y1_fp8, mask=mask)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile_val = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile_val

        tl.store(PACKED_SCALE_ptr + packed_offset, e8m0_byte, mask=row_mask)


@triton.jit
def _swiglu_fwd_quant_pack_zsave_kernel(
    Z_ptr, Y1_FP8_ptr, PACKED_SCALE_ptr, Z_FP8_ptr, Z_SCALE_ptr,
    rows, I_dim: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    stride_zfp8_row: tl.constexpr,
    stride_zscale_row: tl.constexpr,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused SwiGLU + y1 quantize/ISA-pack + z-fp8 save.

    Reads z ONCE, writes both y1_fp8+ISA scales and z_fp8+flat scales.
    Eliminates the separate quantize_activation_blockscaled_fast(z) call.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < rows

    z_row_ptrs = Z_ptr + row_ids[:, None] * stride_z_row
    y_row_ptrs = Y1_FP8_ptr + row_ids[:, None] * stride_y_row
    zfp8_row_ptrs = Z_FP8_ptr + row_ids[:, None] * stride_zfp8_row
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < I_dim
        mask = row_mask[:, None] & col_mask

        # Load interleaved gate/up from z
        gate = tl.load(z_row_ptrs + col_offsets[None, :] * 2, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(z_row_ptrs + col_offsets[None, :] * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        # SwiGLU: y1 = silu(gate) * up
        sig = tl.sigmoid(gate)
        y1 = gate * sig * up

        # --- y1 quantize: ISA-packed scales for down-proj GEMM ---
        y1_amax = tl.max(tl.abs(y1), axis=1)
        y1_bits = y1_amax.to(tl.int32, bitcast=True)
        y1_bexp = (y1_bits >> 23) & 0xFF
        y1_mant = y1_bits & 0x7FFFFF
        y1_e8m0 = y1_bexp - 8 + tl.where(y1_mant > 0x600000, 1, 0)
        y1_e8m0 = tl.where(y1_bexp > 0, y1_e8m0, 0)
        y1_e8m0 = tl.maximum(y1_e8m0, 0)

        y1_qbexp = tl.maximum(tl.minimum(254 - y1_e8m0, 254), 1)
        y1_qscale = (y1_qbexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        y1_fp8 = (y1 * y1_qscale[:, None]).to(tl.float8e4nv)
        tl.store(y_row_ptrs + col_offsets[None, :], y1_fp8, mask=mask)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile_val = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile_val
        tl.store(PACKED_SCALE_ptr + packed_offset, y1_e8m0.to(tl.uint8), mask=row_mask)

        # --- z quantize: flat scales for backward dequantize ---
        # Each y1 group (32 cols) spans 64 z cols = 2 flat z groups.
        # z_group_lo: gate[0:16]+up[0:16], z_group_hi: gate[16:32]+up[16:32]
        half: tl.constexpr = GROUP_SIZE // 2  # 16
        lane = tl.arange(0, GROUP_SIZE)
        z_abs = tl.maximum(tl.abs(gate), tl.abs(up))  # (BLOCK_ROWS, 32)

        z_abs_lo = tl.where(lane[None, :] < half, z_abs, 0.0)
        z_abs_hi = tl.where(lane[None, :] >= half, z_abs, 0.0)
        z_amax_lo = tl.max(z_abs_lo, axis=1)
        z_amax_hi = tl.max(z_abs_hi, axis=1)

        # Integer E8M0 for lo group
        z_lo_bits = z_amax_lo.to(tl.int32, bitcast=True)
        z_lo_bexp = (z_lo_bits >> 23) & 0xFF
        z_lo_mant = z_lo_bits & 0x7FFFFF
        z_lo_e8m0 = z_lo_bexp - 8 + tl.where(z_lo_mant > 0x600000, 1, 0)
        z_lo_e8m0 = tl.where(z_lo_bexp > 0, z_lo_e8m0, 0)
        z_lo_e8m0 = tl.maximum(z_lo_e8m0, 0)

        # Integer E8M0 for hi group
        z_hi_bits = z_amax_hi.to(tl.int32, bitcast=True)
        z_hi_bexp = (z_hi_bits >> 23) & 0xFF
        z_hi_mant = z_hi_bits & 0x7FFFFF
        z_hi_e8m0 = z_hi_bexp - 8 + tl.where(z_hi_mant > 0x600000, 1, 0)
        z_hi_e8m0 = tl.where(z_hi_bexp > 0, z_hi_e8m0, 0)
        z_hi_e8m0 = tl.maximum(z_hi_e8m0, 0)

        # Quant scales: select lo for lanes [0..15], hi for [16..31]
        z_lo_qbexp = tl.maximum(tl.minimum(254 - z_lo_e8m0, 254), 1)
        z_lo_qscale = (z_lo_qbexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        z_hi_qbexp = tl.maximum(tl.minimum(254 - z_hi_e8m0, 254), 1)
        z_hi_qscale = (z_hi_qbexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        z_qscale = tl.where(lane[None, :] < half, z_lo_qscale[:, None], z_hi_qscale[:, None])

        z_gate_fp8 = (gate * z_qscale).to(tl.float8e4nv)
        z_up_fp8 = (up * z_qscale).to(tl.float8e4nv)

        tl.store(zfp8_row_ptrs + col_offsets[None, :] * 2, z_gate_fp8, mask=mask)
        tl.store(zfp8_row_ptrs + col_offsets[None, :] * 2 + 1, z_up_fp8, mask=mask)
        tl.store(Z_SCALE_ptr + row_ids * stride_zscale_row + 2 * group_id, z_lo_e8m0.to(tl.uint8), mask=row_mask)
        tl.store(Z_SCALE_ptr + row_ids * stride_zscale_row + 2 * group_id + 1, z_hi_e8m0.to(tl.uint8), mask=row_mask)


def swiglu_forward_quant_pack_triton(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU forward + blockscaled fp8 quantize + ISA scale pack.

    Single Triton kernel: z(TK, 2I) → y1_fp8(TK, I) + ISA-packed scales.
    No intermediate bf16 y1 tensor. Scales are in ISA tile layout ready
    for direct consumption by CUTLASS blockscaled GEMMs.

    Args:
        z: (TK, 2I) bf16 pre-activation (interleaved gate/up).

    Returns:
        y1_fp8:       (TK, I) float8_e4m3fn — quantized SwiGLU output.
        packed_scales: (1, packed_size) float8_e8m0fnu — ISA-packed scales.
    """
    TK, two_I = z.shape
    assert two_I % 2 == 0
    I = two_I // 2
    assert I % 32 == 0, f"I={I} must be multiple of 32 for blockscaled"

    num_groups = I // 32
    k_tiles = _div_up(I, _SF_TILE_K)

    y1_fp8 = torch.empty(TK, I, dtype=torch.float8_e4m3fn, device=z.device)

    per_batch_storage = _storage_per_batch(TK, I)
    packed_scales = torch.full(
        (1, per_batch_storage), 127, dtype=torch.uint8, device=z.device
    )

    BLOCK_ROWS = 2  # Optimal at production shape: 22 regs, 256 blocks for 160 SMs
    grid = (_div_up(TK, BLOCK_ROWS),)
    _swiglu_fwd_quant_pack_kernel[grid](
        z, y1_fp8, packed_scales,
        TK, I,
        z.stride(0), y1_fp8.stride(0),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=32,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return y1_fp8, packed_scales.view(torch.float8_e8m0fnu)


def swiglu_forward_quant_pack_zsave_triton(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SwiGLU forward + y1 quantize + z-fp8 save in one kernel.

    Reads z(TK, 2I) only ONCE, producing:
      - y1_fp8 + ISA-packed scales (for down-proj GEMM)
      - z_fp8 + flat scales (for backward z dequantize)

    Saves ~128MB DRAM reads vs separate SwiGLU+quant + z-fp8-save.

    Args:
        z: (TK, 2I) bf16 pre-activation (interleaved gate/up).

    Returns:
        y1_fp8:       (TK, I) float8_e4m3fn
        packed_scales: (1, packed_size) float8_e8m0fnu — ISA-packed y1 scales.
        z_fp8:        (TK, 2I) float8_e4m3fn — z quantized for backward.
        z_scales:     (TK, num_z_groups) float8_e8m0fnu — flat z scales.
    """
    TK, two_I = z.shape
    assert two_I % 2 == 0
    I = two_I // 2
    assert I % 32 == 0, f"I={I} must be multiple of 32 for blockscaled"

    num_groups = I // 32  # groups for y1 (I columns, 32 each)
    num_z_groups = two_I // 32  # groups for z (2I columns, 32 each)
    k_tiles = _div_up(I, _SF_TILE_K)

    y1_fp8 = torch.empty(TK, I, dtype=torch.float8_e4m3fn, device=z.device)
    per_batch_storage = _storage_per_batch(TK, I)
    packed_scales = torch.full(
        (1, per_batch_storage), 127, dtype=torch.uint8, device=z.device
    )
    z_fp8 = torch.empty(TK, two_I, dtype=torch.float8_e4m3fn, device=z.device)
    z_scales = torch.empty(TK, num_z_groups, dtype=torch.uint8, device=z.device)

    BLOCK_ROWS = 2  # 40 regs/thread → small BR for better wave distribution
    grid = (_div_up(TK, BLOCK_ROWS),)
    _swiglu_fwd_quant_pack_zsave_kernel[grid](
        z, y1_fp8, packed_scales, z_fp8, z_scales,
        TK, I,
        z.stride(0), y1_fp8.stride(0),
        z_fp8.stride(0), z_scales.stride(0),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=32,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return (
        y1_fp8,
        packed_scales.view(torch.float8_e8m0fnu),
        z_fp8,
        z_scales.view(torch.float8_e8m0fnu),
    )


# ===================================================================
# Backward: (dy1, z, s) → (dz, y1s, ds) bf16  (unfused, legacy)
# ===================================================================

@triton.jit
def _swiglu_bwd_kernel(
    DY1_ptr, Z_ptr, S_ptr,
    DZ_ptr, Y1S_ptr, DS_ptr,
    TK: tl.constexpr, I: tl.constexpr,
    stride_dy_row: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_dz_row: tl.constexpr,
    stride_y1s_row: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Fused SwiGLU backward + router score weighting.

    Each program processes one row across all I columns.
    """
    row = tl.program_id(0)
    dy_base = DY1_ptr + row * stride_dy_row
    z_base = Z_ptr + row * stride_z_row
    dz_base = DZ_ptr + row * stride_dz_row
    y1s_base = Y1S_ptr + row * stride_y1s_row

    s_val = tl.load(S_ptr + row).to(tl.float32)

    # Accumulate ds = sum_j(dy1[j] * y1[j]) across columns
    ds_acc = 0.0

    for j_start in range(0, I, BLOCK_I):
        j_offs = j_start + tl.arange(0, BLOCK_I)
        mask = j_offs < I

        # Load inputs
        gate = tl.load(z_base + j_offs * 2, mask=mask).to(tl.float32)
        up = tl.load(z_base + j_offs * 2 + 1, mask=mask).to(tl.float32)
        dy1_val = tl.load(dy_base + j_offs, mask=mask).to(tl.float32)

        # Forward recomputation
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        y1 = silu_gate * up

        # ds accumulation (before s weighting)
        ds_acc = ds_acc + tl.sum(dy1_val * y1, axis=0)

        # Apply router score to gradient
        dy1_s = dy1_val * s_val

        # Backward SwiGLU
        d_up = dy1_s * silu_gate
        d_gate = dy1_s * up * sig * (1.0 + gate * (1.0 - sig))

        # Store dz interleaved
        tl.store(dz_base + j_offs * 2, d_gate.to(tl.bfloat16), mask=mask)
        tl.store(dz_base + j_offs * 2 + 1, d_up.to(tl.bfloat16), mask=mask)

        # y1s = y1 * s (for weight grad)
        y1s_out = y1 * s_val
        tl.store(y1s_base + j_offs, y1s_out.to(tl.bfloat16), mask=mask)

    # Store ds
    tl.store(DS_ptr + row, ds_acc)


def swiglu_backward_triton(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SwiGLU backward: (dy1, z, s) → (dz, y1s, ds).

    Args:
        dy1: (TK, I) gradient w.r.t. post-activation y1
        z:   (TK, 2I) pre-activation (interleaved gate/up)
        s:   (TK,) router scores

    Returns:
        dz:  (TK, 2I) gradient w.r.t. pre-activation
        y1s: (TK, I) forward SwiGLU output weighted by s (for weight grad)
        ds:  (TK,) gradient w.r.t. router scores
    """
    TK, I = dy1.shape

    dz = torch.empty_like(z)
    y1s = torch.empty(TK, I, dtype=torch.bfloat16, device=z.device)
    ds = torch.empty(TK, dtype=torch.float32, device=z.device)

    BLOCK_I = min(1024, triton.next_power_of_2(I))

    _swiglu_bwd_kernel[(TK,)](
        dy1, z, s,
        dz, y1s, ds,
        TK, I,
        dy1.stride(0), z.stride(0), dz.stride(0), y1s.stride(0),
        BLOCK_I=BLOCK_I,
    )
    return dz, y1s, ds


# ===================================================================
# Backward fused: (dy1, z, s) → dz_fp8 + scales, y1s(bf16), ds
# ===================================================================

@triton.jit
def _swiglu_bwd_quant_kernel(
    DY1_ptr, Z_ptr, S_ptr,
    DZ_FP8_ptr, DZ_SCALE_ptr, Y1S_ptr, DS_ptr,
    TK: tl.constexpr, I: tl.constexpr,
    stride_dy_row: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_dz_row: tl.constexpr,
    stride_dz_scale_row: tl.constexpr,
    stride_y1s_row: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Fused dSwiGLU + router score + blockscaled fp8 quantize.

    dz is interleaved [d_gate0, d_up0, d_gate1, ...] with 2I columns.
    Each blockscaled group of 32 covers 16 interleaved (d_gate, d_up) pairs.
    y1s is output in bf16 (needed for weight grads via quack.gemm).
    """
    row = tl.program_id(0)
    dy_base = DY1_ptr + row * stride_dy_row
    z_base = Z_ptr + row * stride_z_row
    dz_base = DZ_FP8_ptr + row * stride_dz_row
    sc_base = DZ_SCALE_ptr + row * stride_dz_scale_row
    y1s_base = Y1S_ptr + row * stride_y1s_row

    s_val = tl.load(S_ptr + row).to(tl.float32)
    ds_acc = 0.0

    # 2I columns → 2I/32 groups for dz quantization
    # Each group of 32 dz elements = 16 (d_gate, d_up) pairs
    PAIRS_PER_GROUP: tl.constexpr = GROUP_SIZE // 2  # 16

    num_groups: tl.constexpr = (2 * I) // GROUP_SIZE

    for g in range(num_groups):
        pair_start = g * PAIRS_PER_GROUP
        pair_offs = pair_start + tl.arange(0, PAIRS_PER_GROUP)
        mask = pair_offs < I

        gate = tl.load(z_base + pair_offs * 2, mask=mask).to(tl.float32)
        up = tl.load(z_base + pair_offs * 2 + 1, mask=mask).to(tl.float32)
        dy1_val = tl.load(dy_base + pair_offs, mask=mask).to(tl.float32)

        # Forward recomputation
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        y1 = silu_gate * up

        ds_acc = ds_acc + tl.sum(dy1_val * y1, axis=0)

        dy1_s = dy1_val * s_val

        d_up = dy1_s * silu_gate
        d_gate = dy1_s * up * sig * (1.0 + gate * (1.0 - sig))

        # y1s in bf16
        tl.store(y1s_base + pair_offs, (y1 * s_val).to(tl.bfloat16), mask=mask)

        # Blockscaled quantize over 32 interleaved dz elements
        amax_gate = tl.max(tl.where(mask, tl.abs(d_gate), 0.0))
        amax_up = tl.max(tl.where(mask, tl.abs(d_up), 0.0))
        amax = tl.maximum(amax_gate, amax_up)

        positive = amax > 0
        exponent = tl.where(positive,
                            tl.ceil(tl.log2(tl.where(positive, amax / fp8_max, 1.0))),
                            0.0)
        quant_scale = tl.exp2(-exponent)

        dg_fp8 = (d_gate * quant_scale).to(tl.float8e4nv)
        du_fp8 = (d_up * quant_scale).to(tl.float8e4nv)
        tl.store(dz_base + pair_offs * 2, dg_fp8, mask=mask)
        tl.store(dz_base + pair_offs * 2 + 1, du_fp8, mask=mask)

        # E8M0 scale
        dequant = tl.exp2(exponent).to(tl.float32)
        e8m0 = ((dequant.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
        tl.store(sc_base + g, e8m0)

    tl.store(DS_ptr + row, ds_acc)


def swiglu_backward_quant_triton(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SwiGLU backward + blockscaled fp8 quantize for dz.

    Args:
        dy1: (TK, I) gradient w.r.t. post-activation y1
        z:   (TK, 2I) pre-activation (interleaved gate/up)
        s:   (TK,) router scores

    Returns:
        dz_fp8:    (TK, 2I) float8_e4m3fn — quantized gradient
        dz_scales: (TK, 2I//32) uint8 — e8m0 scales for dz
        y1s:       (TK, I) bf16 — forward output weighted by s
        ds:        (TK,) float32 — gradient w.r.t. router scores
    """
    TK, I = dy1.shape
    two_I = z.shape[1]
    assert two_I == 2 * I
    assert two_I % 32 == 0, f"2I={two_I} must be multiple of 32"

    num_groups = two_I // 32
    dz_fp8 = torch.empty(TK, two_I, dtype=torch.float8_e4m3fn, device=z.device)
    dz_scales = torch.empty(TK, num_groups, dtype=torch.uint8, device=z.device)
    y1s = torch.empty(TK, I, dtype=torch.bfloat16, device=z.device)
    ds = torch.empty(TK, dtype=torch.float32, device=z.device)

    _swiglu_bwd_quant_kernel[(TK,)](
        dy1, z, s,
        dz_fp8, dz_scales, y1s, ds,
        TK, I,
        dy1.stride(0), z.stride(0), dz_fp8.stride(0),
        dz_scales.stride(0), y1s.stride(0),
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=32,
    )
    return dz_fp8, dz_scales, y1s, ds


# ===================================================================
# Dequantize: blockscaled fp8 (TK, D) + e8m0 scales → bf16 (TK, D)
# ===================================================================

@triton.jit
def _dequant_blockscaled_fp8_kernel(
    FP8_ptr, SCALES_ptr, OUT_ptr,
    stride_fp8_row, stride_scale_row, stride_out_row,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Dequantize one row of blockscaled FP8 data to bfloat16.

    Each group of GROUP_SIZE fp8 elements shares one e8m0 scale.
    Actual value = fp8_raw * 2^(e8m0 - 127).
    """
    row = tl.program_id(0)
    num_groups: tl.constexpr = D // GROUP_SIZE

    for g in range(num_groups):
        e8m0_val = tl.load(SCALES_ptr + row * stride_scale_row + g)
        # Reconstruct float32 scale from e8m0: place exponent bits in IEEE754
        scale_f32 = (e8m0_val.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        col_offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = col_offs < D
        fp8_vals = tl.load(FP8_ptr + row * stride_fp8_row + col_offs, mask=mask)
        bf16_vals = (fp8_vals.to(tl.float32) * scale_f32).to(tl.bfloat16)
        tl.store(OUT_ptr + row * stride_out_row + col_offs, bf16_vals, mask=mask)


def dequantize_blockscaled_fp8(
    fp8_data: torch.Tensor,
    scales_uint8: torch.Tensor,
) -> torch.Tensor:
    """Dequantize blockscaled FP8 tensor to bfloat16.

    Args:
        fp8_data:     (TK, D) float8_e4m3fn — raw FP8 values.
        scales_uint8: (TK, D//32) uint8 — e8m0 scale per group of 32.

    Returns:
        (TK, D) bfloat16 — properly dequantized values.
    """
    TK, D = fp8_data.shape
    assert D % _GROUP_SIZE == 0, f"D={D} must be multiple of {_GROUP_SIZE}"
    out = torch.empty(TK, D, dtype=torch.bfloat16, device=fp8_data.device)
    _dequant_blockscaled_fp8_kernel[(TK,)](
        fp8_data, scales_uint8, out,
        fp8_data.stride(0), scales_uint8.stride(0), out.stride(0),
        D=D, GROUP_SIZE=_GROUP_SIZE,
    )
    return out


# ===================================================================
# Backward fused+packed: (dy1, z, s) → dz_fp8 + ISA-packed scales,
#                         y1s(bf16), ds
# ===================================================================

@triton.jit
def _swiglu_bwd_quant_pack_kernel(
    DY1_ptr, Z_ptr, S_ptr,
    DZ_FP8_ptr, DZ_PACKED_SCALE_ptr, Y1S_ptr, DS_ptr,
    rows, I_dim: tl.constexpr,
    stride_dy_row: tl.constexpr,
    stride_z_row: tl.constexpr,
    stride_dz_row: tl.constexpr,
    stride_y1s_row: tl.constexpr,
    k_tiles_dz,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PAIRS_PER_GROUP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused dSwiGLU + router score + blockscaled fp8 quantize + ISA pack.

    1D grid: each program processes BLOCK_ROWS rows × ALL dz scale groups,
    looping over groups internally. ds is accumulated in registers across
    all groups and written once — no atomic operations needed.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < rows

    # Load router scores once (used by all groups)
    s_vals = tl.load(S_ptr + row_ids, mask=row_mask, other=0.0).to(tl.float32)

    # Hoist row-dependent ISA layout computations outside loop
    z_base = Z_ptr + row_ids[:, None] * stride_z_row
    dy_base = DY1_ptr + row_ids[:, None] * stride_dy_row
    dz_base = DZ_FP8_ptr + row_ids[:, None] * stride_dz_row
    y1s_base = Y1S_ptr + row_ids[:, None] * stride_y1s_row
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    # Accumulate ds across ALL groups in registers (no atomics!)
    ds_acc = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

    for group_id in tl.range(0, NUM_GROUPS):
        # This group covers pairs [pair_start, pair_start + PAIRS_PER_GROUP)
        pair_start = group_id * PAIRS_PER_GROUP
        pair_offs = pair_start + tl.arange(0, PAIRS_PER_GROUP)
        pair_mask = pair_offs[None, :] < I_dim
        mask = row_mask[:, None] & pair_mask

        # Load inputs
        gate = tl.load(z_base + pair_offs[None, :] * 2, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(z_base + pair_offs[None, :] * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        dy1_val = tl.load(dy_base + pair_offs[None, :], mask=mask, other=0.0).to(tl.float32)

        # Forward recomputation
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        y1 = silu_gate * up

        # ds accumulation in registers
        ds_acc += tl.sum(dy1_val * y1, axis=1)

        # Apply router score to gradient
        dy1_s = dy1_val * s_vals[:, None]

        # Backward SwiGLU
        d_up = dy1_s * silu_gate
        d_gate = dy1_s * up * sig * (1.0 + gate * (1.0 - sig))

        # y1s = y1 * s (for weight grad)
        y1s_out = y1 * s_vals[:, None]
        tl.store(y1s_base + pair_offs[None, :], y1s_out.to(tl.bfloat16), mask=mask)

        # Pure-integer E8M0 blockscaled quantize for dz (no transcendentals)
        amax_gate = tl.max(tl.where(pair_mask, tl.abs(d_gate), 0.0), axis=1)
        amax_up = tl.max(tl.where(pair_mask, tl.abs(d_up), 0.0), axis=1)
        amax = tl.maximum(amax_gate, amax_up)

        amax_bits = amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        dg_fp8 = (d_gate * quant_scale[:, None]).to(tl.float8e4nv)
        du_fp8 = (d_up * quant_scale[:, None]).to(tl.float8e4nv)

        tl.store(dz_base + pair_offs[None, :] * 2, dg_fp8, mask=mask)
        tl.store(dz_base + pair_offs[None, :] * 2 + 1, du_fp8, mask=mask)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile_val = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles_dz + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile_val

        tl.store(DZ_PACKED_SCALE_ptr + packed_offset, e8m0_byte, mask=row_mask)

    # Write ds once — no atomics needed
    tl.store(DS_ptr + row_ids, ds_acc, mask=row_mask)


def swiglu_backward_quant_pack_triton(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SwiGLU backward + blockscaled fp8 quantize + ISA pack for dz.

    Single kernel produces dz_fp8 with ISA-packed scales, y1s(bf16), ds.

    Args:
        dy1: (TK, I) gradient w.r.t. post-activation y1
        z:   (TK, 2I) pre-activation (interleaved gate/up)
        s:   (TK,) router scores

    Returns:
        dz_fp8:          (TK, 2I) float8_e4m3fn — quantized gradient
        dz_packed_scales: (1, packed_size) float8_e8m0fnu — ISA-packed scales
        y1s:             (TK, I) bf16 — forward output weighted by s
        ds:              (TK,) float32 — gradient w.r.t. router scores
    """
    TK, I = dy1.shape
    two_I = z.shape[1]
    assert two_I == 2 * I
    assert two_I % 32 == 0, f"2I={two_I} must be multiple of 32"

    GROUP_SIZE = 32
    PAIRS_PER_GROUP = GROUP_SIZE // 2  # 16
    num_groups = two_I // GROUP_SIZE
    k_tiles_dz = _div_up(two_I, _SF_TILE_K)

    dz_fp8 = torch.empty(TK, two_I, dtype=torch.float8_e4m3fn, device=z.device)
    per_batch_storage = _storage_per_batch(TK, two_I)
    dz_packed_scales = torch.full(
        (1, per_batch_storage), 127, dtype=torch.uint8, device=z.device
    )
    y1s = torch.empty(TK, I, dtype=torch.bfloat16, device=z.device)
    ds = torch.empty(TK, dtype=torch.float32, device=z.device)

    BLOCK_ROWS = 4
    grid = (_div_up(TK, BLOCK_ROWS),)
    _swiglu_bwd_quant_pack_kernel[grid](
        dy1, z, s,
        dz_fp8, dz_packed_scales, y1s, ds,
        TK, I,
        dy1.stride(0), z.stride(0), dz_fp8.stride(0), y1s.stride(0),
        k_tiles_dz,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=GROUP_SIZE,
        PAIRS_PER_GROUP=PAIRS_PER_GROUP,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return dz_fp8, dz_packed_scales.view(torch.float8_e8m0fnu), y1s, ds
