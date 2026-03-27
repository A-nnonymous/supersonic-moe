"""Fused Triton kernels for interleaved SwiGLU forward and backward.

The interleaved layout stores gate/up pairs as z(TK, 2I) where:
  z[:, 0::2] = gate,  z[:, 1::2] = up

Provides three kernel variants:
  1. SwiGLU → bf16 (unfused, for non-fp8 paths)
  2. SwiGLU → blockscaled fp8 + e8m0 scales (fused forward quant)
  3. dSwiGLU → blockscaled fp8 + e8m0 scales (fused backward quant)

The fused variants eliminate a full read-write pass between SwiGLU and
quantize_and_pack, matching the reference fused_weighted_swiglu_act_quant
data flow.
"""

import torch
import triton
import triton.language as tl

_GROUP_SIZE: tl.constexpr = 32  # 1×32 blockscaled granularity


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
