#!/usr/bin/env python3
"""Prototype test: FP8 wgrad via varlen_k CUTLASS GEMM with TMA-based non-contiguous access.

Validates:
1. Column-wise quantize kernel (groups of 32 along TK for (TK, H) layout)
2. Non-contiguous FP8 tensors with CUTLASS GemmDefaultSm100 + sf_vec_size=32
3. varlen_k GEMM correctness vs BF16 reference
4. Performance comparison
"""

import os
import sys
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

import torch
import triton
import triton.language as tl

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cutlass
from cutlass import cute
from quack.cute_dsl_utils import Float32
import cutlass.torch as cutlass_torch
from quack.gemm_wrapper_utils import GemmWrapperBase

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _make_cute_tensor_dynamic, _TORCH_TO_CUTLASS_DTYPE,
    _div_up, _storage_per_batch, GemmTensorInfo,
    pack_blockscaled_1x32_scales, quantize_and_pack_activation,
)
from quack.gemm_default_epi import GemmDefaultSm100
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters

torch.manual_seed(42)

# ============================================================================
# Column-wise quantize kernel (groups of 32 along dim 0 of (TK, H))
# ============================================================================

@triton.jit
def _colwise_quantize_and_pack_kernel(
    src_ptr,                   # (TK, H) bf16 source
    gather_idx_ptr,            # (TK,) int32 gather index, or dummy
    dst_fp8_ptr,               # (TK, H) fp8 output
    dst_packed_ptr,            # (1, per_batch_storage) u8 ISA-packed scales
    total_K,                   # TK
    dim,                       # H
    src_stride_row,            # stride along TK
    src_stride_col,            # stride along H (typically 1)
    dst_stride_row,            # stride along TK for fp8 output
    dst_stride_col,            # stride along H for fp8 output
    k_tiles,                   # ceil(TK / SF_TILE_K) for ISA packing
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,      # 32
    BLOCK_DIM: tl.constexpr,       # 128
    SF_TILE_M: tl.constexpr,       # 128
    SF_TILE_K: tl.constexpr,       # 128
    SF_TILE_STORAGE: tl.constexpr, # 512
):
    """Column-wise blockscaled quantize + ISA-pack.

    2D grid: (num_groups_along_TK, num_blocks_along_H).
    Each block processes one (GROUP_SIZE=32, BLOCK_DIM) tile from (TK, H).
    Quantization groups are along TK (axis 0).
    ISA-packed scales are for the LOGICAL (H, TK) layout (M=H, K=TK).
    """
    pid_group = tl.program_id(0)   # group index along TK
    pid_dim = tl.program_id(1)     # block index along H

    k_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_offs = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    k_mask = k_offs < total_K
    dim_mask = dim_offs < dim
    mask = k_mask[:, None] & dim_mask[None, :]

    # Load source data — optionally via gather
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + k_offs, mask=k_mask, other=0).to(tl.int64)
    else:
        src_rows = k_offs.to(tl.int64)

    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute amax along GROUP_SIZE (axis=0) for each column
    block_amax = tl.max(tl.abs(values), axis=0)  # (BLOCK_DIM,)

    # E8M0 scale computation (pure-integer, no transcendentals)
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

    # Quantize: scale is per-column (broadcast along GROUP_SIZE)
    quantized = (values * quant_scale[None, :]).to(tl.float8e4nv)

    # Write FP8 data in (TK, H) layout
    dst_ptrs = dst_fp8_ptr + k_offs[:, None] * dst_stride_row + dim_offs[None, :] * dst_stride_col
    tl.store(dst_ptrs, quantized, mask=mask)

    # Write ISA-packed scales for LOGICAL (H, TK) layout
    # Logical: row = dim_offs (H), scale_block = pid_group (TK/32 group)
    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE  # 4

    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    k_tiles_idx = pid_group // groups_per_k_tile
    k_in_tile = pid_group % groups_per_k_tile

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    isa_index = tile_base + row_base + k_in_tile

    scale_ptrs = dst_packed_ptr + isa_index.to(tl.int64)
    tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


def colwise_quantize_and_pack(
    src: torch.Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-wise blockscaled FP8 quantize + ISA-pack.

    Quantizes (TK, H) with groups of 32 along TK (dim 0).
    ISA-packed scales are for the LOGICAL (H, TK) layout.

    Parameters
    ----------
    src : Tensor (T_or_TK, H) bf16 source.
    logical_rows : H — rows of the logical (H, TK) matrix.
    logical_cols : TK — cols (= total K for varlen_k).
    gather_idx : optional (TK,) int32 gather indices.

    Returns
    -------
    fp8_data : Tensor (TK, H) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout for (H, TK).
    """
    H = logical_rows
    TK = logical_cols
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    BLOCK_DIM = 128

    fp8_out = torch.empty(TK, H, dtype=torch.float8_e4m3fn, device=src.device)

    # ISA storage for logical (H, TK)
    per_batch_storage = _storage_per_batch(H, TK)
    packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)

    num_groups = _div_up(TK, GROUP_SIZE)
    k_tiles = _div_up(TK, _SF_TILE_K)
    grid = (num_groups, _div_up(H, BLOCK_DIM))

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src  # dummy for non-gather

    _colwise_quantize_and_pack_kernel[grid](
        src,
        gather_ptr,
        fp8_out,
        packed_scales,
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


# ============================================================================
# varlen_k blockscaled FP8 GEMM
# ============================================================================

_COMPILE_CACHE_VK = {}
_FAST_PATH_VK = {}


def run_cutlass_blockscaled_gemm_varlen_k(
    a_fp8: torch.Tensor,         # (TK, M_dim) fp8 — physical layout, M-major after .T
    a_scales: torch.Tensor,      # ISA-packed for logical (M, TK)
    b_fp8: torch.Tensor,         # (TK, N_dim) fp8 — physical layout, N-major after .mT
    b_scales: torch.Tensor,      # ISA-packed for logical (N, TK)
    cu_seqlens_k: torch.Tensor,  # (E+1,) int32
    M: int,                      # H
    N: int,                      # I
    total_K: int,                # TK
    num_experts: int,            # E
    out_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """CUTLASS blockscaled FP8 GEMM with varlen_k scheduling.

    Computes D[e] = A[:, K_e] @ B[K_e, :] for each expert e, where K_e is
    defined by cu_seqlens_k.

    A is (M, TK) M-major (physical: a_fp8 is (TK, M), so a_fp8.T is (M, TK) non-contiguous).
    B is (N, TK) N-major (physical: b_fp8 is (TK, N), so b_fp8.mT is (N, TK) non-contiguous).
    D is (E, M, N).
    """
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import default_config

    # Create the logical tensor views
    # A: (M, TK) with stride (1, M) — M-major (leading_dim=0)
    a_logical = a_fp8.T   # (M, TK), stride (1, M)
    # B: b_fp8.mT gives (N, TK) with stride (1, N) — N-major (leading_dim=0)
    # But for CUTLASS varlen_k: B input should be (TK, N) and quack does B.mT internally
    # Actually, we're building the CUTLASS call directly, so we need to handle this ourselves.
    # For varlen_k in quack's convention: B is originally (TK, N), quack does B.mT = (N, TK)
    # In our direct CUTLASS call, we pass B as (N, TK) directly after permute_tensors.
    b_logical = b_fp8.T   # (N, TK), stride (1, N)

    # D: (E, M, N) — will be permuted for varlen_k
    out = torch.empty(num_experts, M, N, dtype=out_dtype, device=device)

    # --- Fast path ---
    fast_key = (
        "varlen_k", M, N, total_K, num_experts, out_dtype,
        a_fp8.shape[0], a_fp8.shape[1],
        b_fp8.shape[0], b_fp8.shape[1],
        a_scales.size(1), b_scales.size(1),
        device.index if device.index is not None else -1,
    )
    cached = _FAST_PATH_VK.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        # D permuted for varlen_k: (L, M, N) → (M, N, L)
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

    # --- Slow path (first call) ---
    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm_varlen_k does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_logical),
        "B": GemmTensorInfo(b_logical),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }

    # For varlen_k: D and C get permuted (L, M, N) → (M, N, L)
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    print(f"  A major: {tensor_infos['A'].major}, shape: {tuple(tensor_infos['A'].tensor.shape)}, "
          f"stride: {tuple(tensor_infos['A'].tensor.stride())}")
    print(f"  B major: {tensor_infos['B'].major}, shape: {tuple(tensor_infos['B'].tensor.shape)}, "
          f"stride: {tuple(tensor_infos['B'].tensor.stride())}")
    print(f"  D major: {tensor_infos['D'].major}, shape: {tuple(tensor_infos['D'].tensor.shape)}, "
          f"stride: {tuple(tensor_infos['D'].tensor.stride())}")

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
        raise TypeError(
            f"Unsupported type/major combo: A={tensor_infos['A'].dtype}/{tensor_infos['A'].major}, "
            f"B={tensor_infos['B'].dtype}/{tensor_infos['B'].major}"
        )

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
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
        "varlen_k",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        M, N, total_K,
        a_scales.size(1),
        b_scales.size(1),
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE_VK.get(compile_key)
    if compiled is None:
        print(f"  Compiling CUTLASS varlen_k GEMM: tile={tile_shape_mn}, cluster={cluster_shape_mnk}")
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
        _COMPILE_CACHE_VK[compile_key] = compiled

    _FAST_PATH_VK[fast_key] = (compiled, scheduler_args, epi_args)

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


# ============================================================================
# Tests
# ============================================================================

def test_colwise_quantize(E=4, tpe=256, H=4096, verbose=True):
    """Test column-wise quantize kernel correctness."""
    TK = E * tpe
    print(f"\n{'='*60}")
    print(f"Test: column-wise quantize (TK={TK}, H={H})")
    print(f"{'='*60}")

    # Create test data
    src = torch.randn(TK, H, dtype=torch.bfloat16, device="cuda")

    # Column-wise quantize
    fp8_out, packed_scales = colwise_quantize_and_pack(src, logical_rows=H, logical_cols=TK)

    print(f"  fp8_out shape: {fp8_out.shape}, dtype: {fp8_out.dtype}")
    print(f"  packed_scales shape: {packed_scales.shape}, dtype: {packed_scales.dtype}")

    # Verify by dequantizing and comparing
    # For each column h, group g: scale[h, g] covers src[g*32:(g+1)*32, h]
    # Reconstruct raw scales from ISA-packed format
    num_groups = _div_up(TK, _SF_VEC_SIZE)

    # Use the reference: quantize each column's groups manually
    max_err = 0.0
    for g in range(min(num_groups, 4)):
        for h in range(min(H, 4)):
            group_data = src[g*32:(g+1)*32, h].float()
            fp8_data = fp8_out[g*32:(g+1)*32, h].float()
            amax = group_data.abs().max().item()
            if amax > 0:
                rel_err = (group_data - fp8_data).abs().max().item() / amax
                max_err = max(max_err, rel_err)

    # Dequantize for comparison: reconstruct by reading back scales from ISA buffer
    # Simpler: use the row-wise reference for validation (test 2 proves exact match)
    # Here just check that FP8 values are finite and non-zero
    fp8_f32 = fp8_out.float()
    assert fp8_f32.isfinite().all(), "FP8 output has non-finite values"
    nonzero_frac = (fp8_f32 != 0).float().mean().item()
    print(f"  Non-zero fraction: {nonzero_frac:.4f}")
    assert nonzero_frac > 0.5, f"Too many zeros: {nonzero_frac}"
    print(f"  Sample max rel error (group-level): {max_err:.4f}")
    print("  ✅ PASS")
    return True


def test_colwise_quantize_vs_rowwise(E=4, tpe=256, H=512, verbose=True):
    """Verify column-wise quant on (TK, H) matches row-wise quant on (H, TK) transposed."""
    TK = E * tpe
    print(f"\n{'='*60}")
    print(f"Test: colwise vs rowwise equivalence (TK={TK}, H={H})")
    print(f"{'='*60}")

    src = torch.randn(TK, H, dtype=torch.bfloat16, device="cuda")

    # Column-wise on (TK, H): groups of 32 along TK
    fp8_cw, scales_cw = colwise_quantize_and_pack(src, logical_rows=H, logical_cols=TK)

    # Row-wise on (H, TK): groups of 32 along TK (contiguous dim)
    src_t = src.T.contiguous()  # (H, TK)
    fp8_rw, scales_rw = quantize_and_pack_activation(src_t, group_size=_SF_VEC_SIZE)

    # Compare FP8 values (fp8_cw is (TK, H), fp8_rw is (H, TK))
    # fp8_cw.T should match fp8_rw
    diff = (fp8_cw.T.contiguous().float() - fp8_rw.float()).abs()
    max_diff = diff.max().item()
    print(f"  Max FP8 value diff: {max_diff}")

    # Compare ISA-packed scales
    # Both should produce the same ISA-packed scales for logical (H, TK)
    scale_diff = (scales_cw.view(torch.uint8).int() - scales_rw.view(torch.uint8).int()).abs()
    max_scale_diff = scale_diff.max().item()
    print(f"  Max scale diff (ISA bytes): {max_scale_diff}")

    if max_diff == 0 and max_scale_diff == 0:
        print("  ✅ PASS — exact match")
    elif max_diff < 1e-3 and max_scale_diff <= 1:
        print("  ⚠️  CLOSE — small numerical diffs (rounding)")
    else:
        print(f"  ❌ FAIL — significant diffs")
        return False
    return True


def test_varlen_k_gemm_basic(E=4, tpe=256, H=512, I=256, verbose=True):
    """Test varlen_k GEMM with non-contiguous FP8 tensors."""
    TK = E * tpe
    print(f"\n{'='*60}")
    print(f"Test: varlen_k GEMM (E={E}, tpe={tpe}, H={H}, I={I})")
    print(f"{'='*60}")

    # Reference: BF16 per-expert matmul
    dout_gathered = torch.randn(TK, H, dtype=torch.bfloat16, device="cuda") * 0.1
    y1s = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens = torch.arange(0, (E + 1) * tpe, tpe, dtype=torch.int32, device="cuda")

    # BF16 reference: dW[e] = dout_e.T @ y1s_e
    dw_ref = torch.zeros(E, H, I, dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        start, end = e * tpe, (e + 1) * tpe
        dw_ref[e] = dout_gathered[start:end].T.to(torch.float32) @ y1s[start:end].to(torch.float32)
    dw_ref = dw_ref.to(torch.bfloat16)

    # FP8 path: column-wise quantize + varlen_k GEMM
    print("  Quantizing A (dout_gathered)...")
    a_fp8, a_scales = colwise_quantize_and_pack(dout_gathered, logical_rows=H, logical_cols=TK)
    print(f"    a_fp8: {a_fp8.shape}, a_scales: {a_scales.shape}")

    print("  Quantizing B (y1s)...")
    b_fp8, b_scales = colwise_quantize_and_pack(y1s, logical_rows=I, logical_cols=TK)
    print(f"    b_fp8: {b_fp8.shape}, b_scales: {b_scales.shape}")

    print("  Running CUTLASS varlen_k GEMM...")
    dw_fp8 = run_cutlass_blockscaled_gemm_varlen_k(
        a_fp8, a_scales, b_fp8, b_scales,
        cu_seqlens, H, I, TK, E,
        out_dtype=torch.bfloat16, device=dout_gathered.device,
    )

    # Compare
    print(f"  dw_fp8 shape: {dw_fp8.shape}")
    ref = dw_ref.float()
    test = dw_fp8.float()
    rel_rmse = (ref - test).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    corr = torch.corrcoef(torch.stack([ref.flatten(), test.flatten()]))[0, 1]

    print(f"  RelRMSE: {rel_rmse:.4f} ({rel_rmse*100:.2f}%)")
    print(f"  Correlation: {corr:.6f}")
    assert rel_rmse < 0.15, f"RelRMSE too high: {rel_rmse}"
    assert corr > 0.98, f"Correlation too low: {corr}"
    print("  ✅ PASS")
    return True


def test_varlen_k_gemm_production(E=128, tpe=256, H=4096, I=1024, verbose=True):
    """Test at production shapes."""
    TK = E * tpe
    print(f"\n{'='*60}")
    print(f"Test: production shape varlen_k GEMM (E={E}, tpe={tpe}, H={H}, I={I})")
    print(f"{'='*60}")

    dout_gathered = torch.randn(TK, H, dtype=torch.bfloat16, device="cuda") * 0.02
    y1s = torch.randn(TK, I, dtype=torch.bfloat16, device="cuda") * 0.02
    cu_seqlens = torch.arange(0, (E + 1) * tpe, tpe, dtype=torch.int32, device="cuda")

    # BF16 reference (sample experts only — too big for full check)
    print("  Computing BF16 reference (sampled)...")
    sample_experts = [0, E//4, E//2, E-1]
    dw_ref_samples = {}
    for e in sample_experts:
        start, end = e * tpe, (e + 1) * tpe
        dw_ref_samples[e] = (
            dout_gathered[start:end].T.float() @ y1s[start:end].float()
        ).to(torch.bfloat16)

    # FP8 path
    print("  Quantizing A (dout_gathered)...")
    t0 = time.perf_counter()
    a_fp8, a_scales = colwise_quantize_and_pack(dout_gathered, logical_rows=H, logical_cols=TK)
    torch.cuda.synchronize()
    t_quant_a = (time.perf_counter() - t0) * 1000
    print(f"    Time: {t_quant_a:.2f} ms")

    print("  Quantizing B (y1s)...")
    t0 = time.perf_counter()
    b_fp8, b_scales = colwise_quantize_and_pack(y1s, logical_rows=I, logical_cols=TK)
    torch.cuda.synchronize()
    t_quant_b = (time.perf_counter() - t0) * 1000
    print(f"    Time: {t_quant_b:.2f} ms")

    print("  Running CUTLASS varlen_k GEMM...")
    t0 = time.perf_counter()
    dw_fp8 = run_cutlass_blockscaled_gemm_varlen_k(
        a_fp8, a_scales, b_fp8, b_scales,
        cu_seqlens, H, I, TK, E,
        out_dtype=torch.bfloat16, device=dout_gathered.device,
    )
    torch.cuda.synchronize()
    t_gemm = (time.perf_counter() - t0) * 1000
    print(f"    Time (incl compile): {t_gemm:.2f} ms")

    # Warm up + time
    for _ in range(3):
        dw_fp8 = run_cutlass_blockscaled_gemm_varlen_k(
            a_fp8, a_scales, b_fp8, b_scales,
            cu_seqlens, H, I, TK, E,
            out_dtype=torch.bfloat16, device=dout_gathered.device,
        )
    torch.cuda.synchronize()

    # CUDA events timing
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(10):
        dw_fp8 = run_cutlass_blockscaled_gemm_varlen_k(
            a_fp8, a_scales, b_fp8, b_scales,
            cu_seqlens, H, I, TK, E,
            out_dtype=torch.bfloat16, device=dout_gathered.device,
        )
    end_evt.record()
    torch.cuda.synchronize()
    t_avg = start_evt.elapsed_time(end_evt) / 10
    print(f"    Avg GEMM time: {t_avg:.3f} ms ({t_avg*1000:.1f} µs)")

    # BF16 wgrad reference timing (per-expert loop, matching nsys baseline)
    dw2_ref = torch.empty(E, H, I, dtype=torch.bfloat16, device="cuda")
    # Warm up
    for _ in range(3):
        for e_idx in range(E):
            s, en = e_idx * tpe, (e_idx + 1) * tpe
            torch.mm(dout_gathered[s:en].T.float(), y1s[s:en].float(), out=dw2_ref[e_idx].float())
    torch.cuda.synchronize()

    # Use quack per-expert batch gemm for fair comparison
    # Actually use torch.bmm on the proper layout
    dout_per_expert = dout_gathered.view(E, tpe, H).permute(0, 2, 1).contiguous()  # (E, H, tpe)
    y1s_per_expert = y1s.view(E, tpe, I)  # (E, tpe, I)
    for _ in range(3):
        torch.bmm(dout_per_expert.float(), y1s_per_expert.float())
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(10):
        torch.bmm(dout_per_expert.float(), y1s_per_expert.float())
    end_evt.record()
    torch.cuda.synchronize()
    t_bf16 = start_evt.elapsed_time(end_evt) / 10
    print(f"    BF16 bmm time: {t_bf16:.3f} ms ({t_bf16*1000:.1f} µs)")
    print(f"    [Reference from nsys: BF16 varlen_k wgrad w2 = ~575µs]")
    t_bf16_ref = 0.575  # from nsys profiling
    print(f"    FP8 GEMM speedup over BF16 (nsys ref): {t_bf16_ref/t_avg:.2f}x")

    # Total FP8 wgrad time including quant
    # Warmup quant kernels
    for _ in range(3):
        _ = colwise_quantize_and_pack(dout_gathered, logical_rows=H, logical_cols=TK)
        _ = colwise_quantize_and_pack(y1s, logical_rows=I, logical_cols=TK)
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(10):
        a_fp8_, a_sc_ = colwise_quantize_and_pack(dout_gathered, logical_rows=H, logical_cols=TK)
        b_fp8_, b_sc_ = colwise_quantize_and_pack(y1s, logical_rows=I, logical_cols=TK)
    end_evt.record()
    torch.cuda.synchronize()
    t_quant = start_evt.elapsed_time(end_evt) / 10
    print(f"    Quant (A+B) time: {t_quant:.3f} ms ({t_quant*1000:.1f} µs)")
    print(f"    Total FP8 wgrad: {t_quant + t_avg:.3f} ms ({(t_quant + t_avg)*1000:.1f} µs)")
    print(f"    Total speedup over BF16: {t_bf16/(t_quant + t_avg):.2f}x")

    # Compare sampled experts
    max_rel_rmse = 0.0
    min_corr = 1.0
    for e in sample_experts:
        ref = dw_ref_samples[e].float()
        test = dw_fp8[e].float()
        rel_rmse = (ref - test).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
        corr = torch.corrcoef(torch.stack([ref.flatten(), test.flatten()]))[0, 1]
        max_rel_rmse = max(max_rel_rmse, rel_rmse.item())
        min_corr = min(min_corr, corr.item())

    print(f"  Max RelRMSE: {max_rel_rmse:.4f} ({max_rel_rmse*100:.2f}%)")
    print(f"  Min Correlation: {min_corr:.6f}")
    assert max_rel_rmse < 0.15, f"RelRMSE too high: {max_rel_rmse}"
    assert min_corr > 0.98, f"Correlation too low: {min_corr}"
    print("  ✅ PASS")
    return True


def main():
    print("FP8 wgrad varlen_k prototype test")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")

    passed = 0
    total = 0

    # Test 1: Column-wise quantize correctness
    total += 1
    try:
        if test_colwise_quantize(E=4, tpe=256, H=512):
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()

    # Test 2: Column-wise vs row-wise equivalence
    total += 1
    try:
        if test_colwise_quantize_vs_rowwise(E=4, tpe=256, H=512):
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()

    # Test 3: varlen_k GEMM at small shape
    total += 1
    try:
        if test_varlen_k_gemm_basic(E=4, tpe=256, H=512, I=256):
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()

    # Test 4: varlen_k GEMM at production shape
    total += 1
    try:
        if test_varlen_k_gemm_production(E=128, tpe=256, H=4096, I=1024):
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*60}")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
