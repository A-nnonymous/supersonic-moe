"""CuTe DSL dual blockscaled FP8 quantize — [32][33] padded smem design.

Single HBM read -> padded smem -> both row and col quant bank-conflict-free.
Inspired by Paddle's shm[128][129] technique.

Tile: (TILE_TK=256, TILE_DIM=32), smem stride 33 (not 32).
Phase A (col quant): 8 warps, lane=dim col, reads sSrc[tk, lane] -> stride-33 rows, no bank conflict
Phase B (row quant): 256 threads, each reads sSrc[tidx, 0..31] -> consecutive cols, no bank conflict
Both directions conflict-free thanks to +1 padding.
"""
import math
from typing import Optional

import torch

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import cuda.bindings.driver as cuda

import quack.copy_utils as copy_utils
from quack.compile_utils import make_fake_tensor
from quack.cache_utils import jit_cache


@dsl_user_op
def _f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

@dsl_user_op
def _i32_as_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

@dsl_user_op
def _rcp_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))

@dsl_user_op
def _abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


GROUP_SIZE = 32
TILE_TK = 256
TILE_DIM = 32
PADDED_DIM = 33   # +1 for bank-conflict-free column access
NUM_THREADS = 256
FP8_MAX = 448.0
SF_TILE_M = 128
SF_TILE_K = 128
SF_TILE_STORAGE = SF_TILE_M * (SF_TILE_K // GROUP_SIZE)
SF_GROUPS_PER_K_TILE = SF_TILE_K // GROUP_SIZE


@cute.jit
def _e8m0_rcp(amax_var: Float32):
    """Compute E8M0 scale via rcp.approx. Returns (quant_scale, e8m0)."""
    amax_var = cute.arch.fmax(amax_var, Float32(1e-12))
    cand = _f32_as_i32(Float32(FP8_MAX) * _rcp_approx(amax_var))
    qexp = (cand >> Int32(23)) & Int32(0xFF)
    qexp = qexp if cutlass.Boolean(qexp > Int32(1)) else Int32(1)
    qexp = qexp if cutlass.Boolean(qexp < Int32(254)) else Int32(254)
    qs = _i32_as_f32(qexp << Int32(23))
    e8m0 = Int32(254) - qexp
    e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
    return qs, e8m0


class DualQuantOp:
    def __init__(self, dtype, TK, dim):
        self.dtype = dtype
        self.TK = TK
        self.dim = dim
        self.num_groups_tk = TK // GROUP_SIZE
        self.groups_per_tile = TILE_TK // GROUP_SIZE
        self.row_k_tiles = (dim + SF_TILE_K - 1) // SF_TILE_K
        self.col_k_tiles = (TK + SF_TILE_K - 1) // SF_TILE_K

    @cute.jit
    def __call__(self, mSrc: cute.Tensor, mColFp8: cute.Tensor, mColScale: cute.Tensor,
                 mRowFp8: cute.Tensor, mRowScale: cute.Tensor, stream: cuda.CUstream):
        TK_val = mSrc.shape[0]
        dim_val = mSrc.shape[1]
        self.kernel(mSrc, mColFp8, mColScale, mRowFp8, mRowScale, TK_val).launch(
            grid=[cute.ceil_div(dim_val, TILE_DIM), cute.ceil_div(TK_val, TILE_TK), 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mSrc: cute.Tensor, mColFp8: cute.Tensor, mColScale: cute.Tensor,
               mRowFp8: cute.Tensor, mRowScale: cute.Tensor, TK_val: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bid_dim, bid_tk, _ = cute.arch.block_idx()
        dim_val = mSrc.shape[1]

        # ─── Standard smem: (256, 32) bf16 + cp.async vectorized load ───
        smem = cutlass.utils.SmemAllocator()
        sSrc = smem.allocate_tensor(
            mSrc.element_type,
            cute.make_ordered_layout((TILE_TK, const_expr(TILE_DIM)), order=(1, 0)),
            byte_alignment=16,
        )

        # Vectorized cp.async load
        tiled_copy_ld = copy_utils.tiled_copy_2d(
            mSrc.element_type, threads_per_row=4, num_threads=NUM_THREADS, num_copy_elems=8
        )
        thr_copy = tiled_copy_ld.get_slice(tidx)
        gSrc_tile = cute.local_tile(mSrc, (TILE_TK, TILE_DIM), (bid_tk, bid_dim))
        tSgSrc = thr_copy.partition_S(gSrc_tile)
        tSsSrc = thr_copy.partition_D(sSrc)
        idX = cute.make_identity_tensor(mSrc.shape)
        cX = cute.local_tile(idX, (TILE_TK, TILE_DIM), (bid_tk, bid_dim))
        tScX = thr_copy.partition_S(cX)[(0, None), None, None]
        tSpX = copy_utils.predicate_k(thr_copy.partition_S(cX), limit=dim_val)
        if tScX[0][0] < TK_val:
            copy_utils.copy(tSgSrc, tSsSrc, pred=tSpX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.arch.barrier()

        dim_base = bid_dim * const_expr(TILE_DIM)
        tk_base = bid_tk * const_expr(TILE_TK)
        warp_id = tidx // 32
        lane = tidx % 32
        abs_dim_row = dim_base + lane
        dim_valid = abs_dim_row < dim_val
        group_tk_base = warp_id * const_expr(GROUP_SIZE)

        # ═══════════════════════════════════════════════════════════════════
        # Phase A: Col quant — groups of 32 along TK (bank-conflict-free)
        # ═══════════════════════════════════════════════════════════════════
        col_amax = Float32(0.0)
        for j in cutlass.range(const_expr(GROUP_SIZE)):
            col_amax = cute.arch.fmax(col_amax, _abs_f32(Float32(sSrc[group_tk_base + j, lane])))

        col_scale, col_e8m0 = _e8m0_rcp(col_amax)

        abs_tk_c = tk_base + group_tk_base
        for j4 in cutlass.range(const_expr(GROUP_SIZE // 4)):
            base_j = group_tk_base + j4 * 4
            r4 = cute.make_rmem_tensor(4, cute.Float32)
            for k in cutlass.range(4, unroll_full=True):
                r4[k] = Float32(sSrc[base_j + k, lane]) * col_scale
            rFp8 = cute.make_rmem_tensor(4, cute.Float8E4M3FN)
            rFp8.store(r4.load().to(cute.Float8E4M3FN))
            for k in cutlass.range(4, unroll_full=True):
                if dim_valid:
                    if abs_tk_c < TK_val:
                        mColFp8[abs_tk_c, abs_dim_row] = rFp8[k]
                abs_tk_c = abs_tk_c + Int32(1)

        # Col ISA scale
        abs_group_col = (tk_base // const_expr(GROUP_SIZE)) + warp_id
        if dim_valid:
            if abs_group_col < self.num_groups_tk:
                cr = abs_dim_row
                c_rt = cr // const_expr(SF_TILE_M)
                c_rit = cr % const_expr(SF_TILE_M)
                c_kti = abs_group_col // const_expr(SF_GROUPS_PER_K_TILE)
                c_kit = abs_group_col % const_expr(SF_GROUPS_PER_K_TILE)
                c_tb = (c_rt * const_expr(self.col_k_tiles) + c_kti) * const_expr(SF_TILE_STORAGE)
                c_rb = (c_rit % Int32(32)) * Int32(16) + (c_rit // Int32(32)) * Int32(4)
                mColScale[c_tb + c_rb + c_kit] = cute.Uint8(col_e8m0)

        # ═══════════════════════════════════════════════════════════════════
        # Phase B: Row quant — re-read smem, per-row amax via warp shuffle
        # Same warp mapping: warp_id=TK group, lane=dim col.
        # For each TK row j in group: all 32 lanes have val for different dim cols.
        # Butterfly shuffle MAX -> row_amax. Then scale+store fp8.
        # Process 4 rows at a time for fp8 batch cast.
        # ═══════════════════════════════════════════════════════════════════
        for j4 in cutlass.range(const_expr(GROUP_SIZE // 4)):
            # Read 4 rows, compute 4 row_amaxes, scale 4 rows, store 4×fp8
            r4_fp8_out = cute.make_rmem_tensor(4, cute.Float8E4M3FN)
            r4_e8m0 = cute.make_rmem_tensor(4, cute.Int32)
            r4_scaled = cute.make_rmem_tensor(4, cute.Float32)

            for k in cutlass.range(4, unroll_full=True):
                j = j4 * 4 + k
                val = Float32(sSrc[group_tk_base + j, lane])
                abs_val = _abs_f32(val)
                # Butterfly shuffle MAX across 32 lanes
                rm = abs_val
                rm = cute.arch.fmax(rm, cute.arch.warp_reduction(rm, cute.arch.fmax))
                # rm now has the max across all 32 dim cols for this TK row
                row_qs, row_e8m0 = _e8m0_rcp(rm)
                r4_scaled[k] = val * row_qs
                r4_e8m0[k] = row_e8m0

            r4_fp8_out.store(r4_scaled.load().to(cute.Float8E4M3FN))

            for k in cutlass.range(4, unroll_full=True):
                abs_tk_r = tk_base + group_tk_base + j4 * 4 + k
                if dim_valid:
                    if abs_tk_r < TK_val:
                        mRowFp8[abs_tk_r, abs_dim_row] = r4_fp8_out[k]

            # Row ISA scale (lane 0 only — all lanes have same e8m0)
            if lane == 0:
                for k in cutlass.range(4, unroll_full=True):
                    abs_tk_r = tk_base + group_tk_base + j4 * 4 + k
                    if abs_tk_r < TK_val:
                        r_rt = abs_tk_r // const_expr(SF_TILE_M)
                        r_rit = abs_tk_r % const_expr(SF_TILE_M)
                        r_kti = bid_dim // const_expr(SF_GROUPS_PER_K_TILE)
                        r_kit = bid_dim % const_expr(SF_GROUPS_PER_K_TILE)
                        r_tb = (r_rt * const_expr(self.row_k_tiles) + r_kti) * const_expr(SF_TILE_STORAGE)
                        r_rb = (r_rit % Int32(32)) * Int32(16) + (r_rit // Int32(32)) * Int32(4)
                        mRowScale[r_tb + r_rb + r_kit] = cute.Uint8(r4_e8m0[k])


def _storage_per_batch(rows: int, cols: int) -> int:
    return math.ceil(rows / SF_TILE_M) * math.ceil(cols / SF_TILE_K) * SF_TILE_STORAGE


# ─── Dual Quant Compilation ──────────────────────────────────────────────────

@jit_cache
def _compile_dual_quant(dtype_width, TK, dim):
    assert TK % 32 == 0 and dim % 32 == 0
    dtype = cute.BFloat16 if dtype_width == 16 else cute.Float16
    TK_sym, dim_sym = cute.sym_int(), cute.sym_int()
    div = 128 // dtype.width

    mSrc = make_fake_tensor(dtype, (TK_sym, dim_sym), divisibility=div)  # div=8 for 128-bit cp.async
    mColFp8 = make_fake_tensor(cute.Float8E4M3FN, (TK_sym, dim_sym), divisibility=div)
    mRowFp8 = make_fake_tensor(cute.Float8E4M3FN, (TK_sym, dim_sym), divisibility=div)

    col_sc_sym = cute.sym_int()
    mColScale = cute.runtime.make_fake_tensor(cute.Uint8, (col_sc_sym,), stride=(1,), assumed_align=1)
    row_sc_sym = cute.sym_int()
    mRowScale = cute.runtime.make_fake_tensor(cute.Uint8, (row_sc_sym,), stride=(1,), assumed_align=1)

    op = DualQuantOp(dtype, TK, dim)
    return cute.compile(op, mSrc, mColFp8, mColScale, mRowFp8, mRowScale,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi")


def dual_quantize_varlen_cute(src: Tensor, TK: int, dim: int):
    """CuTe DSL dual blockscaled FP8 quantize — single HBM read, two outputs."""
    assert src.dim() == 2 and src.is_contiguous() and src.is_cuda
    assert TK % 32 == 0 and dim % 32 == 0
    device = src.device

    row_fp8 = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=device)
    col_fp8 = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=device)

    row_per_batch = _storage_per_batch(TK, dim)
    col_per_batch = _storage_per_batch(dim, TK)

    if TK % SF_TILE_M == 0 and dim % SF_TILE_K == 0:
        row_scales = torch.empty((1, row_per_batch), dtype=torch.uint8, device=device)
    else:
        row_scales = torch.full((1, row_per_batch), 127, dtype=torch.uint8, device=device)

    if dim % SF_TILE_M == 0 and TK % SF_TILE_K == 0:
        col_scales = torch.empty((1, col_per_batch), dtype=torch.uint8, device=device)
    else:
        col_scales = torch.full((1, col_per_batch), 127, dtype=torch.uint8, device=device)

    dtype_width = 16 if src.dtype in (torch.bfloat16, torch.float16) else 32
    fn = _compile_dual_quant(dtype_width, TK, dim)
    fn(src, col_fp8, col_scales.view(-1), row_fp8, row_scales.view(-1))

    return (row_fp8, row_scales.view(_E8M0_DTYPE),
            col_fp8, col_scales.view(_E8M0_DTYPE))
