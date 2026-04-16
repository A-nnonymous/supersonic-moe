"""CuTe DSL colwise blockscaled FP8 quantize — v5 with gather + ISA packing.

NCU-guided: 30 regs, 89% occ, bank-conflict-free, rcp.approx E8M0, abs.f32.
Supports: ISA-packed scales, optional gather_idx for indirect addressing.
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
NUM_THREADS = 256
FP8_MAX = 448.0
SF_TILE_M = 128
SF_TILE_K = 128
SF_TILE_STORAGE = SF_TILE_M * (SF_TILE_K // GROUP_SIZE)
SF_GROUPS_PER_K_TILE = SF_TILE_K // GROUP_SIZE


class ColwiseQuantOp:
    def __init__(self, dtype, TK, isa_pack=True, has_gather=False):
        self.dtype = dtype
        self.TK = TK
        self.num_groups_total = TK // GROUP_SIZE
        self.groups_per_tile = TILE_TK // GROUP_SIZE
        self.isa_pack = isa_pack
        self.has_gather = has_gather

    @cute.jit
    def __call__(self, mSrc: cute.Tensor, mFp8: cute.Tensor, mScale: cute.Tensor,
                 mGatherIdx: cute.Tensor, stream: cuda.CUstream):
        TK_val = mSrc.shape[0] if const_expr(not self.has_gather) else mFp8.shape[0]
        dim_val = mSrc.shape[1]
        self.kernel(mSrc, mFp8, mScale, mGatherIdx, TK_val).launch(
            grid=[cute.ceil_div(dim_val, TILE_DIM), cute.ceil_div(TK_val, TILE_TK), 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mSrc: cute.Tensor, mFp8: cute.Tensor, mScale: cute.Tensor,
               mGatherIdx: cute.Tensor, TK_val: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bid_dim, bid_tk, _ = cute.arch.block_idx()

        dim_val = mSrc.shape[1]

        # ─── Smem ───
        smem = cutlass.utils.SmemAllocator()
        sSrc = smem.allocate_tensor(
            mSrc.element_type,
            cute.make_ordered_layout((TILE_TK, TILE_DIM), order=(1, 0)),
            byte_alignment=16,
        )

        # ─── Load: either cp.async (no gather) or element-wise (gather) ───
        if const_expr(not self.has_gather):
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
        else:
            # Cooperative coalesced gather: warp-level row loads.
            # Each warp loads 1 full row per iteration (32 consecutive cols
            # by 32 lanes -> perfectly coalesced along dim axis).
            # 8 warps × 32 iterations = 256 rows = TILE_TK.
            # NCU-guided: fixes 2.1 bytes/sector LD efficiency -> ~32 bytes/sector.
            warp_ld = tidx // 32
            lane_ld = tidx % 32
            dim_col_ld = bid_dim * const_expr(TILE_DIM) + lane_ld
            col_valid = dim_col_ld < dim_val
            for li in cutlass.range(const_expr(TILE_TK // (NUM_THREADS // 32))):
                lr = li * const_expr(NUM_THREADS // 32) + warp_ld
                atk = bid_tk * const_expr(TILE_TK) + lr
                if atk < TK_val:
                    sr = Int32(mGatherIdx[atk])
                    if col_valid:
                        sSrc[lr, lane_ld] = mSrc[sr, dim_col_ld]
                    else:
                        sSrc[lr, lane_ld] = mSrc.element_type(0)
                else:
                    sSrc[lr, lane_ld] = mSrc.element_type(0)

        cute.arch.barrier()

        # ─── Thread mapping ───
        dim_base = bid_dim * const_expr(TILE_DIM)
        warp_id = tidx // 32
        lane = tidx % 32
        abs_dim_row = dim_base + lane
        dim_valid = abs_dim_row < dim_val
        group_tk_base = warp_id * const_expr(GROUP_SIZE)
        tk_base = bid_tk * const_expr(TILE_TK)

        # ─── Phase 1: amax ───
        amax = Float32(0.0)
        for j in cutlass.range(const_expr(GROUP_SIZE)):
            amax = cute.arch.fmax(amax, _abs_f32(Float32(sSrc[group_tk_base + j, lane])))

        # ─── Phase 2: E8M0 via rcp.approx ───
        amax = cute.arch.fmax(amax, Float32(1e-12))
        cand_bits = _f32_as_i32(Float32(FP8_MAX) * _rcp_approx(amax))
        qexp = (cand_bits >> Int32(23)) & Int32(0xFF)
        qexp = qexp if cutlass.Boolean(qexp > Int32(1)) else Int32(1)
        qexp = qexp if cutlass.Boolean(qexp < Int32(254)) else Int32(254)
        quant_scale = _i32_as_f32(qexp << Int32(23))
        e8m0 = Int32(254) - qexp
        e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)

        # ─── Phase 3: scale, cast, store fp8 ───
        abs_tk = tk_base + group_tk_base  # pre-init outside loop
        for j4 in cutlass.range(const_expr(GROUP_SIZE // 4)):
            base_j = group_tk_base + j4 * 4
            r4 = cute.make_rmem_tensor(4, cute.Float32)
            for k in cutlass.range(4, unroll_full=True):
                r4[k] = Float32(sSrc[base_j + k, lane]) * quant_scale
            rFp8 = cute.make_rmem_tensor(4, cute.Float8E4M3FN)
            rFp8.store(r4.load().to(cute.Float8E4M3FN))
            for k in cutlass.range(4, unroll_full=True):
                if dim_valid:
                    if abs_tk < TK_val:
                        mFp8[abs_tk, abs_dim_row] = rFp8[k]
                abs_tk = abs_tk + Int32(1)

        # ─── Scale store ───
        abs_group = (tk_base // const_expr(GROUP_SIZE)) + warp_id
        if dim_valid:
            if abs_group < self.num_groups_total:
                if const_expr(self.isa_pack):
                    dim_row = abs_dim_row
                    group_idx = abs_group
                    row_tiles = dim_row // const_expr(SF_TILE_M)
                    row_in_tile = dim_row % const_expr(SF_TILE_M)
                    k_tiles_idx = group_idx // const_expr(SF_GROUPS_PER_K_TILE)
                    k_in_tile = group_idx % const_expr(SF_GROUPS_PER_K_TILE)
                    k_tiles = const_expr((self.TK + SF_TILE_K - 1) // SF_TILE_K)
                    tile_base = (row_tiles * k_tiles + k_tiles_idx) * const_expr(SF_TILE_STORAGE)
                    row_base = (row_in_tile % Int32(32)) * Int32(16) + (row_in_tile // Int32(32)) * Int32(4)
                    isa_index = tile_base + row_base + k_in_tile
                    mScale[isa_index] = cute.Uint8(e8m0)
                else:
                    mScale[abs_group, abs_dim_row] = cute.Uint8(e8m0)


# ─── Compilation ──────────────────────────────────────────────────────────────

@jit_cache
def _compile_colwise_quant(dtype_width, TK, dim, isa_pack=True, has_gather=False):
    assert TK % 32 == 0
    dtype = cute.BFloat16 if dtype_width == 16 else cute.Float16
    TK_sym, dim_sym = cute.sym_int(), cute.sym_int()
    div = 128 // dtype.width

    if has_gather:
        # Src shape is (T, dim) where T may != TK; gather_idx is (TK,) int32
        T_sym = cute.sym_int()
        mSrc = make_fake_tensor(dtype, (T_sym, dim_sym), divisibility=1)
    else:
        mSrc = make_fake_tensor(dtype, (TK_sym, dim_sym), divisibility=div)

    mFp8 = make_fake_tensor(cute.Float8E4M3FN, (TK_sym, dim_sym), divisibility=div)

    if isa_pack:
        scale_sym = cute.sym_int()
        mScale = cute.runtime.make_fake_tensor(
            cute.Uint8, (scale_sym,), stride=(1,), assumed_align=1)
    else:
        ng_sym = cute.sym_int()
        mScale = make_fake_tensor(cute.Uint8, (ng_sym, dim_sym), divisibility=div)

    if has_gather:
        mGatherIdx = cute.runtime.make_fake_tensor(
            cute.Int32, (TK_sym,), stride=(1,), assumed_align=4)
    else:
        # Dummy: 1-element tensor (not accessed)
        mGatherIdx = cute.runtime.make_fake_tensor(
            cute.Int32, (1,), stride=(1,), assumed_align=4)

    op = ColwiseQuantOp(dtype, TK, isa_pack=isa_pack, has_gather=has_gather)
    return cute.compile(op, mSrc, mFp8, mScale, mGatherIdx,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi")


# ─── PyTorch Entry Point ─────────────────────────────────────────────────────

def _storage_per_batch(rows: int, cols: int) -> int:
    return math.ceil(rows / SF_TILE_M) * math.ceil(cols / SF_TILE_K) * SF_TILE_STORAGE


def colwise_quantize_cute(
    src: Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: Optional[Tensor] = None,
    isa_pack: bool = False,
) -> tuple[Tensor, Tensor]:
    """CuTe DSL colwise blockscaled FP8 quantize.

    Args:
        src: (T_or_TK, dim) bf16 contiguous
        logical_rows: dim (rows of logical (dim, TK) matrix)
        logical_cols: TK
        gather_idx: optional (TK,) int32 — indirect row indices into src
        isa_pack: output ISA-packed scales for CUTLASS GEMM

    Returns:
        fp8_data: (TK, dim) fp8
        scales: ISA-packed (1, storage) e8m0fnu if isa_pack, else (dim, num_groups) uint8
    """
    assert src.dim() == 2 and src.is_contiguous() and src.is_cuda
    dim = src.shape[1]
    TK = logical_cols
    has_gather = gather_idx is not None
    assert TK % 32 == 0
    num_groups = TK // 32

    fp8_out = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=src.device)
    dtype_width = 16 if src.dtype in (torch.bfloat16, torch.float16) else 32

    # Dummy gather_idx if not provided (1-element, never accessed in kernel)
    gather_dummy = gather_idx if has_gather else torch.zeros(1, dtype=torch.int32, device=src.device)

    compiled_fn = _compile_colwise_quant(dtype_width, TK, dim, isa_pack=isa_pack, has_gather=has_gather)

    if isa_pack:
        per_batch_storage = _storage_per_batch(logical_rows, logical_cols)
        if dim % SF_TILE_M == 0 and TK % SF_TILE_K == 0:
            scale_out = torch.empty((1, per_batch_storage), dtype=torch.uint8, device=src.device)
        else:
            scale_out = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)
        compiled_fn(src, fp8_out, scale_out.view(-1), gather_dummy)
        return fp8_out, scale_out.view(_E8M0_DTYPE)
    else:
        scale_out = torch.empty(num_groups, dim, dtype=torch.uint8, device=src.device)
        compiled_fn(src, fp8_out, scale_out, gather_dummy)
        return fp8_out, scale_out.t().contiguous()
