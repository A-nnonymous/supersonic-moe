"""CuTe DSL colwise blockscaled FP8 quantize — optimized version.

NCU-guided: bank conflicts eliminated via row-major smem reads.
Optimizations: rcp.approx E8M0, minimal register usage, coalesced stores.

Each CTA: (TILE_TK, TILE_DIM) tile. Vectorized cp.async load, row-major smem reads,
per-group E8M0 via rcp.approx, fp8 cast, coalesced gmem stores.
"""
import math
from functools import partial

import torch
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
    """rcp.approx.f32 — reciprocal without FTZ. Bit-exact for E8M0 (mantissa masked)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


GROUP_SIZE = 32
TILE_TK = 256
TILE_DIM = 32
NUM_THREADS = 256
# FP8E4M3 max representable = 448 = 1.75 * 2^8
FP8_MAX = 448.0


class ColwiseQuantOp:

    def __init__(self, dtype, TK, use_rcp=True):
        self.dtype = dtype
        self.TK = TK
        self.num_groups_total = TK // GROUP_SIZE
        self.groups_per_tile = TILE_TK // GROUP_SIZE
        self.use_rcp = use_rcp

    @cute.jit
    def __call__(self, mSrc: cute.Tensor, mFp8: cute.Tensor, mScale: cute.Tensor,
                 stream: cuda.CUstream):
        TK_val = mSrc.shape[0]
        dim_val = mSrc.shape[1]
        self.kernel(mSrc, mFp8, mScale).launch(
            grid=[cute.ceil_div(dim_val, TILE_DIM), cute.ceil_div(TK_val, TILE_TK), 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mSrc: cute.Tensor, mFp8: cute.Tensor, mScale: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bid_dim, bid_tk, _ = cute.arch.block_idx()

        TK_val = mSrc.shape[0]
        dim_val = mSrc.shape[1]
        groups_per_tile = const_expr(self.groups_per_tile)

        # ─── Smem ───
        smem = cutlass.utils.SmemAllocator()
        sSrc = smem.allocate_tensor(
            mSrc.element_type,
            cute.make_ordered_layout((TILE_TK, TILE_DIM), order=(1, 0)),
            byte_alignment=16,
        )

        # ─── Vectorized cp.async load ───
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

        row = tScX[0][0]
        if row < TK_val:
            copy_utils.copy(tSgSrc, tSsSrc, pred=tSpX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ─── Compute: each warp handles 1 group ───
        dim_base = bid_dim * TILE_DIM
        warp_id = tidx // 32
        lane = tidx % 32
        abs_dim_row = dim_base + lane
        dim_valid = abs_dim_row < dim_val
        group_tk_start = warp_id * GROUP_SIZE
        tk_base = bid_tk * TILE_TK

        # Pass 1: compute amax (runtime loop, no unroll → minimal regs)
        amax = Float32(0.0)
        for j in cutlass.range(GROUP_SIZE):
            val_f32 = Float32(sSrc[group_tk_start + j, lane])
            abs_val = cute.arch.fmax(val_f32, Float32(0.0) - val_f32)
            amax = cute.arch.fmax(amax, abs_val)

        # ─── E8M0 via rcp.approx (bit-exact: mantissa masked away) ───
        if const_expr(self.use_rcp):
            # quant_scale ≈ fp8_max / amax, rounded down to power of 2
            # rcp.approx(amax) ≈ 1/amax with ≤1 ULP mantissa error
            # After extracting exponent (masking mantissa), error vanishes
            amax = cute.arch.fmax(amax, Float32(1e-12))
            inv_amax = _rcp_approx(amax)
            candidate = Float32(FP8_MAX) * inv_amax
            # Extract exponent, zero mantissa → round down to power of 2
            cand_bits = _f32_as_i32(candidate)
            qexp = (cand_bits >> Int32(23)) & Int32(0xFF)
            # Clamp to valid range
            qexp = qexp if cutlass.Boolean(qexp > Int32(1)) else Int32(1)
            qexp = qexp if cutlass.Boolean(qexp < Int32(254)) else Int32(254)
            quant_scale = _i32_as_f32(qexp << Int32(23))
            e8m0 = Int32(254) - qexp
            e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
        else:
            # Original integer bitops approach
            amax = cute.arch.fmax(amax, Float32(1e-12))
            amax_bits = _f32_as_i32(amax)
            biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
            mantissa = amax_bits & Int32(0x7FFFFF)
            carry = Int32(1) if cutlass.Boolean(mantissa > Int32(0x600000)) else Int32(0)
            e8m0 = biased_exp - Int32(8) + carry
            e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
            qexp = Int32(254) - e8m0
            qexp = qexp if cutlass.Boolean(qexp < Int32(254)) else Int32(254)
            qexp = qexp if cutlass.Boolean(qexp > Int32(1)) else Int32(1)
            quant_scale = _i32_as_f32(qexp << Int32(23))

        # Pass 2: re-read smem, scale, cast fp8, store
        # Runtime loop (not unrolled) to minimize register pressure
        # Process 4 elements at a time for 32-bit fp8 alignment
        for j4 in cutlass.range(const_expr(GROUP_SIZE // 4)):
            r4 = cute.make_rmem_tensor(4, cute.Float32)
            for k in cutlass.range(4, unroll_full=True):
                r4[k] = Float32(sSrc[group_tk_start + j4 * 4 + k, lane]) * quant_scale
            rFp8_4 = cute.make_rmem_tensor(4, cute.Float8E4M3FN)
            rFp8_4.store(r4.load().to(cute.Float8E4M3FN))
            for k in cutlass.range(4, unroll_full=True):
                abs_tk_pos = tk_base + group_tk_start + j4 * 4 + k
                if dim_valid:
                    if abs_tk_pos < TK_val:
                        mFp8[abs_tk_pos, abs_dim_row] = rFp8_4[k]

        # Store scale — (num_groups, dim) layout: 32 lanes write consecutive dim addresses
        # mScale[abs_group, dim_base + lane] → coalesced!
        abs_group = (tk_base // GROUP_SIZE) + warp_id
        if dim_valid:
            if abs_group < self.num_groups_total:
                mScale[abs_group, abs_dim_row] = cute.Uint8(e8m0)


@jit_cache
def _compile_colwise_quant(dtype_width, TK, dim, use_rcp=True):
    assert TK % 32 == 0
    dtype = cute.BFloat16 if dtype_width == 16 else cute.Float16

    TK_sym = cute.sym_int()
    dim_sym = cute.sym_int()
    num_groups_sym = cute.sym_int()

    div = 128 // dtype.width
    mSrc = make_fake_tensor(dtype, (TK_sym, dim_sym), divisibility=div)
    mFp8 = make_fake_tensor(cute.Float8E4M3FN, (TK_sym, dim_sym), divisibility=div)
    mScale = make_fake_tensor(cute.Uint8, (num_groups_sym, dim_sym), divisibility=div)

    op = ColwiseQuantOp(dtype, TK, use_rcp=use_rcp)
    return cute.compile(
        op, mSrc, mFp8, mScale,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def colwise_quantize_cute(src: Tensor, logical_rows: int, logical_cols: int,
                           use_rcp: bool = True):
    assert src.dim() == 2 and src.is_contiguous() and src.is_cuda
    TK, dim = src.shape
    assert TK % 32 == 0
    num_groups = TK // 32

    fp8_out = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=src.device)
    # Scale layout: (num_groups, dim) for coalesced stores
    scale_out = torch.empty(num_groups, dim, dtype=torch.uint8, device=src.device)

    dtype_width = 16 if src.dtype in (torch.bfloat16, torch.float16) else 32
    compiled_fn = _compile_colwise_quant(dtype_width, TK, dim, use_rcp=use_rcp)
    compiled_fn(src, fp8_out, scale_out)

    return fp8_out, scale_out.t().contiguous()  # transpose to (dim, num_groups) for API compat
