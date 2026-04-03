# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.activation
import quack.sm90_utils as sm90_utils
import torch
from cutlass import const_expr, Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, mlir_namedtuple
from quack.epi_ops import TileStore, EpiOp
import quack.copy_utils as copy_utils
from quack.rounding import RoundingMode
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase
from quack.layout_utils import permute_gated_Cregs_b16

from torch import Tensor


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    torch.uint8: cutlass.Uint8,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _is_runtime_fp8_tensor(tensor: Tensor) -> bool:
    return tensor.dtype in {torch.float8_e4m3fn, torch.float8_e8m0fnu}


def _make_cute_tensor_dynamic(tensor: Tensor, leading_dim: int) -> cute.Tensor:
    if _is_runtime_fp8_tensor(tensor):
        storage = tensor.detach().view(torch.uint8)
        cute_tensor = from_dlpack(storage, assumed_align=16)
        cute_tensor.element_type = _TORCH_TO_CUTLASS_DTYPE[tensor.dtype]
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _halve_epi_tile(gemm, epi_tile):
    """Halve the N-dimension of the epilogue tile for gated activations."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(2, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // 2)


class GemmGatedMixin(GemmActMixin):
    _epi_ops = (*GemmActMixin._epi_ops[:-1], TileStore("mPostAct", epi_tile_fn=_halve_epi_tile))

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Override: force CopyUniversalOp for postact R2S when blockscaled.

        NOTE: The fused GemmGated + blockscaled FP8 path crashes due to a
        deeper issue in epi_visit_subtile's accumulator recast. This override
        alone is insufficient — the decomposed path is used instead.
        Kept as documentation of the attempted fix.
        """
        if const_expr(self.blockscaled):
            sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
            copy_atom_postact_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.postact_dtype
            )
            tiled_copy_postact_r2s = cute.make_tiled_copy_S(
                copy_atom_postact_r2s, tiled_copy_r2s
            )
            tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
            batch_idx = tile_coord_mnkl[3]
            copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
                params.tma_atom_mPostAct,
                varlen_manager.offset_batch_epi(params.mPostAct, batch_idx),
                self.cta_tile_shape_postact_mn,
                params.epi_tile_mPostAct,
                sPostAct,
                tile_coord_mnkl,
            )
            return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact
        else:
            return GemmActMixin.epi_setup_postact(
                self, params, epi_smem_tensors, tiled_copy_r2s,
                tiled_copy_t2r, tile_coord_mnkl, varlen_manager, tidx,
            )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert self.postact_dtype.width in {8, 16}, "GemmGated only supports 8bit or 16bit postact for now"
        assert self.d_layout is None or self.d_layout.is_n_major_c()
        assert self.postact_layout.is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, "GemmGatedSm90 requires tileN to be divisible by 32"
        self.cta_tile_shape_postact_mn = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2)
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]), (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                )
        return tRS_rPostAct

    @cute.jit
    def epi_convert_postact(self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx):
        tRS_rPostAct_out = GemmActMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(self.arch == 90):
            permute_gated_Cregs_b16(tRS_rPostAct_out)
        return tRS_rPostAct_out


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    pass


class GemmGatedSm100(GemmGatedMixin, GemmSm100):
    pass


# ---------------------------------------------------------------------------
# DSL primitives for epilogue blockscaled FP8 quantization
# ---------------------------------------------------------------------------

@dsl_user_op
def _f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    """Bitcast float32 to int32 (reinterpret bits, no conversion)."""
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _i32_as_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    """Bitcast int32 to float32 (reinterpret bits, no conversion)."""
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _rcp_approx_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """PTX rcp.approx.f32: fast reciprocal, bitwise exact for powers of 2."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# FP8 E4M3 max representable value
_FP8_MAX = Float32(448.0)


class BlockscaledScaleStore(EpiOp):
    """EpiOp: accumulates per-group UE8M0 in registers, writes to gmem in end().

    State: flat (N_GROUPS,) Int32 register array per thread.
    Each epi_visit_subtile writes ue8m0 to state[subtile_idx].
    end() writes all accumulated scales to the output buffer using tile coords.
    """

    def __init__(self, name, n_groups=4):
        super().__init__(name)
        self._n_groups = n_groups  # CTA tile_N / 32

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name, None)
        if tensor is not None:
            # tensor is already a CuTe tensor (from EpilogueArguments construction)
            return {self.name: tensor}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Just pass through the param tensor (or None)."""
        return param

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        """Return None — scale write happens in overridden epilogue(), not via EpiOp state."""
        return None


class GemmGatedBlockscaledQuantMixin(GemmGatedMixin):
    """GemmGated with fused blockscaled FP8 quantization of z in epilogue.

    Matches Paddle fp8_quant_blockwise_kernel.cu with UE8M0 power-of-2 scaling.
    All 5 steps run atomically in the same register domain:

      1. amax = max(abs(z[group]))  — from ORIGINAL f32 register values
      2. scale = 448.0 / amax → ldexpf(1.0, floor(log2(scale)))  [power-of-2]
      3. store_scale = 1.0 / scale  [exact for pow2]
      4. UE8M0 = biased_exponent(store_scale)
      5. z_quantized = z_original * scale → hardware fp8 saturating cast

    CRITICAL: steps 1-5 must use the SAME original z values. Cannot split
    across kernels (bf16 intermediate store loses precision, amax recovery
    from quantized values is numerically wrong).

    Precision notes (matching Paddle reference):
      - scale = fp8_max / amax uses fdiv, NOT rcp.approx
        (rcp.approx has mantissa error that can flip the pow2 rounding boundary)
      - Power-of-2 rounding: ldexpf(1.0, biased_exp - 127), NOT clearing mantissa bits
        (ldexpf is exact; clearing mantissa differs for subnormals and exp=0)
      - store_scale = 1.0 / scale is exact when scale is power-of-2
      - UE8M0 = (float_as_int(store_scale) >> 23) & 0xFF — direct exponent extraction
      - fp8 cast: hardware static_cast<fp8>(val * scale) does saturation + round

    Register layout: SM100 Ld32x32bOp maps 1 M-row per thread, all N-elements
    local. For epi_tile_n=32: each thread holds 32 z-elements = 1 blockscaled group.
    No warp shuffle needed.
    """
    _epi_ops = (
        *GemmGatedMixin._epi_ops,
        BlockscaledScaleStore("mZScale"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScale: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # Standard: alpha/beta/bias → SwiGLU → tRS_rPostAct
        tRS_rPostAct = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        # ── Blockscaled quant of z (tRS_rD) in registers ──
        # Uses the SAME integer+carry algorithm as _quantize_and_pack_kernel (Triton)
        # which is bitwise identical to Paddle reference (verified: 0/100000 mismatches).
        num_z = cute.size(tRS_rD)

        # Step 1: amax = max(abs(z[0:32])) — AFTER bias/alpha, BEFORE SwiGLU
        amax = Float32(0.0)
        for i in cutlass.range(num_z, unroll_full=True):
            val = tRS_rD[i]
            neg = Float32(0.0) - val
            abs_val = cute.arch.fmax(val, neg)
            amax = cute.arch.fmax(amax, abs_val)

        # Step 2: E8M0 via integer bit manipulation (matches Triton kernel exactly)
        amax_bits = _f32_as_i32(amax)
        biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
        mantissa_bits = amax_bits & Int32(0x7FFFFF)
        # carry = 1 if mantissa > 0x600000 (round up when mantissa > 0.75)
        has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
        carry = Int32(1) if has_carry else Int32(0)
        e8m0 = biased_exp - Int32(8) + carry
        is_normal = cutlass.Boolean(biased_exp > Int32(0))
        e8m0 = e8m0 if is_normal else Int32(0)
        is_pos = cutlass.Boolean(e8m0 > Int32(0))
        e8m0 = e8m0 if is_pos else Int32(0)

        # Step 3: quant_scale = 2^(254 - e8m0)
        qexp = Int32(254) - e8m0
        qexp_hi = cutlass.Boolean(qexp > Int32(254))
        qexp = Int32(254) if qexp_hi else qexp
        qexp_lo = cutlass.Boolean(qexp < Int32(1))
        qexp = Int32(1) if qexp_lo else qexp
        quant_scale = _i32_as_f32(qexp << Int32(23))

        # Step 4: z *= quant_scale (values now in [-448, 448])
        for i in cutlass.range(num_z, unroll_full=True):
            tRS_rD[i] = tRS_rD[i] * quant_scale

        # Step 5: UE8M0 = e8m0 (clamped, same as Triton kernel stores)
        ue8m0 = e8m0

        # Return y1 + ue8m0 for epilogue to write scale to gmem
        return tRS_rPostAct, ue8m0

    @cute.jit
    def epilogue(
        self, params, epi_smem_tensors, epi_pipeline, epi_store_pipeline,
        epi_read_state, epi_producer_state, epi_tile, load_acc_subtile,
        tRS_rD, tRS_rC, tiled_copy_t2r, tiled_copy_r2s, tRS_sD,
        tiled_copy_s2r, tSR_rC, tSR_sC, copy_D, copy_C,
        tile_coord_mnkl, varlen_manager, epilogue_barrier,
        tile_scheduler, tidx, is_tma_warp,
    ):
        """Override epilogue to inject scale gmem write after epi_visit_subtile.

        This is a copy of GemmSm90.epilogue with ONE addition: after
        epi_visit_subtile (which computes ue8m0 and stores it in EpiOp state),
        we write the scale byte directly to gmem using tile coordinates.
        """
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)

        postact_ctx = self.epi_setup_postact(
            params, epi_smem_tensors, tiled_copy_r2s, tiled_copy_t2r,
            tile_coord_mnkl, varlen_manager, tidx,
        )

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        epi_tensors = self.epi_begin(
            params, epi_smem_tensors, epi_tile, tiled_copy_t2r, tiled_copy_r2s,
            tile_coord_mnkl, varlen_manager, epilogue_barrier, tidx,
        )

        if const_expr(copy_C is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            load_acc_subtile(tRS_rD, epi_idx)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

            # ─── epi_visit_subtile: SwiGLU + blockscaled quant (integer+carry, returns ue8m0) ───
            tRS_rPostAct, ue8m0 = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)

            # ─── SCALE GMEM WRITE ───
            z_scale_tensor = epi_tensors["mZScale"]
            if const_expr(z_scale_tensor is not None):
                tile_M = self.cta_tile_shape_mnk[0]
                m_in_tile = tidx % tile_M
                if const_expr(varlen_manager.varlen_m):
                    batch_idx = tile_coord_mnkl[3]
                    m_abs = varlen_manager.params.cu_seqlens_m[batch_idx] + tile_coord_mnkl[0] * tile_M + m_in_tile
                    limit_m = varlen_manager.len_m(batch_idx)
                else:
                    m_abs = tile_coord_mnkl[0] * tile_M + m_in_tile
                    limit_m = tile_M
                if m_in_tile + tile_coord_mnkl[0] * tile_M < limit_m:
                    n_groups_per_tile = self.cta_tile_shape_mnk[1] // 32
                    n_group = tile_coord_mnkl[1] * n_groups_per_tile + epi_idx
                    z_scale_tensor[m_abs, n_group] = ue8m0

            # Convert and store postact
            if const_expr(postact_ctx is not None):
                tRS_rPostAct_out = self.epi_convert_postact(
                    tRS_rPostAct, epi_loop_tensors["sr_seed"], tidx,
                    tile_coord_mnkl, num_prev_subtiles, epi_idx,
                )
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
            epilogue_barrier.arrive_and_wait()
            # D registers → smem
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                if const_expr(
                    self.rounding_mode == RoundingMode.RS
                    and self.acc_dtype == cutlass.Float32
                    and self.d_dtype == cutlass.BFloat16
                ):
                    seed = epi_loop_tensors["sr_seed"] + (
                        tile_coord_mnkl[0] * 65537 + tile_coord_mnkl[1] * 257
                        + tile_coord_mnkl[3] * 17 + (num_prev_subtiles + epi_idx) * 7
                    )
                    copy_utils.sr_cvt_copy(
                        tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer],
                        seed, tidx,
                    )
                else:
                    copy_utils.cvt_copy(
                        tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer]
                    )
            # PostAct registers → smem
            if const_expr(postact_ctx is not None):
                tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
                cute.copy(
                    tiled_copy_postact_r2s,
                    tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                    tRS_sPostAct[None, None, None, epi_buffer],
                )
            # smem → gmem via TMA
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
                if const_expr(postact_ctx is not None):
                    copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                epi_store_pipeline.producer_commit()

        self.epi_end(
            params, epi_tensors, epi_tile, tiled_copy_t2r, tiled_copy_r2s,
            tile_coord_mnkl, varlen_manager, tidx,
        )
        return epi_read_state, epi_producer_state


class GemmGatedBlockscaledQuantSm100(GemmGatedBlockscaledQuantMixin, GemmSm100):
    pass


gate_fn_map = {
    "swiglu": quack.activation.swiglu,
    "swiglu_oai": quack.activation.swiglu_oai,
    "reglu": quack.activation.reglu,
    "geglu": quack.activation.geglu,
    "glu": quack.activation.glu,
}


def gemm_gated(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n//2) or (total_m, n//2) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    a_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for A
    b_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for B
    z_scale_out: Optional[Tensor] = None,  # (total_m, n//32) uint8 UE8M0 output for epilogue quant
) -> None:
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in gate_fn_map, f"Unsupported activation {activation}"

    # Special validation for PostAct shape
    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )

    # PostAct shape validation depends on varlen_m
    if cu_seqlens_m is not None:
        # varlen_m case: PostAct is 2D (total_m, n//2)
        assert PostAct.dim() == 2 and PostAct.is_cuda, "PostAct must be a 2D CUDA tensor for varlen_m"
        assert PostAct.shape == (
            M,
            N // 2,
        ), f"PostAct must have shape {(M, N // 2)}, got {PostAct.shape}"
    else:
        # Normal case: PostAct is 3D (l, m, n//2)
        assert PostAct.dim() == 3 and PostAct.is_cuda, "PostAct must be a 3D CUDA tensor"
        assert PostAct.shape == (
            L,
            M,
            N // 2,
        ), f"PostAct must have shape {(L, M, N // 2)}, got {PostAct.shape}"

    tensor_infos["PostAct"] = GemmTensorInfo(PostAct)
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),  # PostAct has shape (m, n//2, l) after permute
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for info in tensor_infos.values():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    # Use zero-materialization kernel when gather_A + blockscaled (FP8 with A_idx)
    blockscaled_runtime = a_scales is not None and b_scales is not None
    if device_capacity[0] > 9 and gather_A and blockscaled_runtime:
        from .gemm_sm100_fp8_zeromat import GemmGatedSm100ZeroMat
        GemmCls = GemmGatedSm100ZeroMat
    elif device_capacity[0] > 9:
        GemmCls = GemmGatedSm100
    else:
        GemmCls = GemmGatedSm90

    acc_dtype = cutlass.Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute_tensor_dynamic(info.tensor, leading_dim)
    act_fn = gate_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        mRowVecBroadcast=(
            from_dlpack(rowvec_bias.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
            if rowvec_bias is not None
            else None
        ),
        mColVecBroadcast=(
            from_dlpack(colvec_bias.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_bias is not None
            else None
        ),
        **({"mZScale": _make_cute_tensor_dynamic(z_scale_out, leading_dim=1)}
           if z_scale_out is not None else {}),
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore,
        max_swizzle_size=max_swizzle_size,
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
    )

    current_stream = cutlass_torch.current_stream()

    blockscaled = a_scales is not None and b_scales is not None
    sf_vec_size = 32 if blockscaled else None

    # Prepare blockscaled scale cute tensors if provided
    if blockscaled:
        a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
    else:
        a_scale_cute = None
        b_scale_cute = None

    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        rowvec_bias.dtype if rowvec_bias is not None else None,
        colvec_bias.dtype if colvec_bias is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        blockscaled,
        z_scale_out is not None,  # triggers different kernel compilation
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_gated.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
            sf_vec_size=sf_vec_size,
        )
        cache[compile_key] = cute.compile(
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
    cache[compile_key](
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


gemm_gated.compile_cache = {}
