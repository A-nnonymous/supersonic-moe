# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
from typing import Callable, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import quack.activation
import quack.layout_utils as layout_utils
import quack.utils as utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_ops import ColVecReduce, TileStore, EpiOp, assume_stride_divisibility
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_wrapper_utils import GemmWrapperBase
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


class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"), ColVecReduce("mColVecReduce"))
    _extra_param_fields = (
        ("act_bwd_fn", cutlass.Constexpr, None),
        ("implicit_dtype", cutlass.Constexpr, None),
    )

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

        Same fix as GemmGatedMixin — avoids StMatrix/smem layout mismatch
        in blockscaled mode.
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

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0  # RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert args.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        assert tRS_rC is not None
        implicit_dtype = params.implicit_dtype
        assert implicit_dtype.width == 16, "GemmDGatedMixin only supports 16bit for now"
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):  # Scale D by colvec
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            # Need to multiply before D is scaled by colvec_scale
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1])
                    )
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):  # Scale Out by colvec
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        # Write dXY (packed f16x2 as f32) back to D
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, implicit_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        # Return PostAct in acc_dtype; conversion happens in epi_convert_postact
        return tRS_rOut


class GemmDGatedSm90(GemmDGatedMixin, GemmSm90):
    pass


class GemmDGatedSm100(GemmDGatedMixin, GemmSm100):
    pass


# ---------------------------------------------------------------------------
# FP8 PreAct: load z_fp8 + scales directly in epilogue, skip C tensor
# ---------------------------------------------------------------------------

@dsl_user_op
def _fp8e4m3_to_f32(x, *, loc=None, ip=None) -> Float32:
    """Convert scalar f8E4M3FN to f32 via PTX: fp8 → f16 → f32."""
    from cutlass._mlir.dialects import arith as _arith
    x_i8 = llvm.bitcast(T.i8(), x.ir_value(loc=loc, ip=ip) if hasattr(x, 'ir_value') else x,
                         loc=loc, ip=ip)
    x_i16 = llvm.zext(T.i16(), x_i8, loc=loc, ip=ip)
    f16_val = llvm.inline_asm(
        T.f16(), [x_i16],
        "{ .reg .b8 s; mov.b16 {s, _}, $1; cvt.rn.f16.e4m3 $0, s; }",
        "=h,h",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    f32_val = _arith.extf(T.f32(), f16_val, loc=loc, ip=ip)
    return Float32(f32_val)


@dsl_user_op
def _f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _i32_as_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


class FP8PreActLoad(EpiOp):
    """EpiOp: loads fp8 z + UE8M0 scales from gmem, dequants in registers.

    Param is a tuple (z_fp8_tensor, z_scales_tensor) passed as a single field.
    begin(): unpacks and captures coordinates.
    begin_loop(): computes subtile coordinates.
    The mixin's epi_visit_subtile loads fp8 bytes + scales and dequants.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        return 0

    def to_params(self, gemm, args):
        fp8 = getattr(args, self.name + "_fp8", None)
        scales = getattr(args, self.name + "_scales", None)
        if fp8 is not None and scales is not None:
            return {self.name: (assume_stride_divisibility(fp8), assume_stride_divisibility(scales))}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            fp8_tensor, scales_tensor = param
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]

            # Compute varlen M offset
            if const_expr(ctx.varlen_manager.varlen_m):
                m_offset = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
            else:
                m_offset = Int32(0)
            m_base = ctx.tile_coord_mnkl[0] * tile_M

            # Identity tensor partitioned for this thread's epilogue elements
            # This gives the exact (row, col) for each register position
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((tile_M, tile_N))
            )

            # N base in fp8 logical coordinates (tile_N f32 = 2*tile_N bf16 = 2*tile_N fp8)
            n_base_logical = ctx.tile_coord_mnkl[1] * tile_N * 2

            return (fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base_logical)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base = state
            # Extract this subtile's identity coordinates
            tDcD_sub = cute.group_modes(tDcD, 3, cute.rank(tDcD))[None, None, None, epi_coord]
            return (fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base)
        return None


class GemmDGatedFP8PreActMixin(GemmDGatedMixin):
    """GemmDGated with fp8 PreAct: loads z_fp8 + scales in epilogue, no C tensor.

    When mFP8PreAct_fp8 is provided, tRS_rC is ignored (can be None).
    The epilogue loads fp8 z bytes + UE8M0 scale bytes via LDG, dequants
    in registers, and constructs tRS_rXY_f32x2 for dSwiGLU computation.

    Memory saving: eliminates 384MB z_bf16 temporary buffer.
    """
    _epi_ops = (
        *GemmDGatedMixin._epi_ops,
        FP8PreActLoad("mFP8PreAct"),
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        fp8_mode = getattr(args, "mFP8PreAct_fp8", None) is not None
        if not fp8_mode:
            assert args.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
            assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]

        fp8_preact_info = epi_loop_tensors["mFP8PreAct"]

        if const_expr(fp8_preact_info is not None):
            # ── FP8 PreAct path: use identity tensor for correct coordinates ──
            fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base = fp8_preact_info

            # tDcD_sub[i] gives (row_in_tile, col_in_tile) for each D register element
            # col is C's physical N (f32 = bf16x2). Each f32 maps to 2 fp8 bytes.
            num_d = cute.size(tDcD_sub)

            # Allocate fp8 register tensor (2x D elements for gate+up pairs)
            tRS_rXY_bf16_layout = cute.recast_tensor(tRS_rD, cutlass.BFloat16).layout
            tRS_rFP8 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, cutlass.Float8E4M3FN)

            # Load fp8 bytes using identity-derived coordinates
            # For each D[i] at (row, col), load fp8[row, col*2] and fp8[row, col*2+1]
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                n0 = n_base + col * 2
                tRS_rFP8[2 * i] = fp8_tensor[m_abs, n0]
                tRS_rFP8[2 * i + 1] = fp8_tensor[m_abs, n0 + Int32(1)]

            # Vectorized fp8→f32 (DSL auto-packs vec4 → nvgpu.cvt_fpext)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rFP8.load().to(Float32))

            # Blockscaled dequant using identity coordinates for correct group index
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                n0 = n_base + col * 2
                group_0 = n0 >> Int32(5)
                group_1 = (n0 + Int32(1)) >> Int32(5)
                scale_0 = _i32_as_f32(Int32(scales_tensor[m_abs, group_0]) << Int32(23))
                scale_1 = _i32_as_f32(Int32(scales_tensor[m_abs, group_1]) << Int32(23))
                tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 PreAct path ──
            assert tRS_rC is not None
            implicit_dtype = params.implicit_dtype
            tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
            tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

        # ── dSwiGLU + colvec scale/reduce (unchanged from parent) ──
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1])
                    )
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )

        # Write dXY (packed f16x2 as f32) back to D
        if const_expr(fp8_preact_info is not None):
            pack_dtype = cutlass.BFloat16
        else:
            pack_dtype = params.implicit_dtype
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut


class GemmDGatedFP8CLoadMixin(GemmDGatedMixin):
    """GemmDGated with TMA-based fp8 C load.

    Loads fp8 z (PreAct) via TMA to smem, then fp8→f32 conversion in registers.
    Eliminates 384MB z_bf16 temporary buffer.

    C tensor: z_fp8 (TK, 2I) Float8E4M3FN
    Scale tensor: z_scales (TK, 2I/32) uint8 — loaded via EpiOp LDG

    Key overrides:
    - epilog_smem_load_and_partition: double the register layout for fp8 (2N fp8 vs N f32)
    - epi_visit_subtile: fp8→f32 vectorized conversion + blockscaled dequant
    - epi_to_underlying_arguments: handle fp8 c_dtype
    """

    _epi_ops = (
        *GemmDGatedMixin._epi_ops,
        FP8PreActLoad("mFP8PreAct"),
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        # fp8 C: c_dtype is Float8E4M3FN, skip bf16 assertions
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        # c_dtype may be fp8 (8-bit) — allow it
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    def epilog_smem_load_and_partition(
        self, tiled_copy_t2r_or_mma, c_layout, dtype, sC, tRS_rD_layout, tidx
    ):
        """Override: when C is fp8, create register tensor with 2x elements.

        Standard: tRS_rC has N f32 elements (same as D).
        FP8: tRS_rC has 2N fp8 elements (each f32 covers 2 bf16 = 2 fp8).
        After fp8→f32 conversion: 2N f32 = correct for dSwiGLU.
        """
        if const_expr(dtype == cutlass.Float8E4M3FN):
            # fp8 C: double the register layout
            # tRS_rD_layout has N elements; we need 2N fp8 elements
            tRS_rC_fp8_layout = cute.recast_tensor(
                cute.make_rmem_tensor(tRS_rD_layout, cutlass.BFloat16),
                cutlass.Float8E4M3FN
            ).layout
            # Use standard copy atom but with fp8 dtype
            from quack import copy_utils
            tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_copy_t2r_or_mma)
            copy_atom_s2r = copy_utils.sm90_get_smem_load_op(c_layout, dtype)
            tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
            thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
            tSR_sC = thr_copy_s2r.partition_S(sC)
            tRS_rC = cute.make_rmem_tensor(tRS_rC_fp8_layout, dtype)
            tSR_rC = thr_copy_s2r.retile(tRS_rC)
            return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC
        else:
            return GemmDGatedMixin.epilog_smem_load_and_partition(
                self, tiled_copy_t2r_or_mma, c_layout, dtype, sC, tRS_rD_layout, tidx
            )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]

        if const_expr(self.c_dtype == cutlass.Float8E4M3FN):
            # ── FP8 C path: tRS_rC has 2N fp8 elements from TMA ──
            # Vectorized fp8→f32 conversion
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rC.layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rC.load().to(Float32))

            # Blockscaled dequant: multiply by 2^(e8m0 << 23) per group
            # Scale info from EpiOp
            fp8_preact_info = epi_loop_tensors["mFP8PreAct"]
            if const_expr(fp8_preact_info is not None):
                # Scales loaded via EpiOp (small data, LDG is fine)
                fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base = fp8_preact_info
                num_d = cute.size(tDcD_sub)
                for i in cutlass.range(num_d, unroll_full=True):
                    coord = tDcD_sub[i]
                    row, col = coord[0], coord[1]
                    m_abs = m_offset + m_base + row
                    n0 = n_base + col * 2
                    group_0 = n0 >> Int32(5)
                    group_1 = (n0 + Int32(1)) >> Int32(5)
                    scale_0 = _i32_as_f32(Int32(scales_tensor[m_abs, group_0]) << Int32(23))
                    scale_1 = _i32_as_f32(Int32(scales_tensor[m_abs, group_1]) << Int32(23))
                    tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                    tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 C path ──
            assert tRS_rC is not None
            implicit_dtype = params.implicit_dtype
            tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
            tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

        # ── dSwiGLU + colvec (shared between both paths) ──
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rD_scaled_mn[m, 2*n], tRS_rD_scaled_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                tRS_rdXY_f32x2[2*i], tRS_rdXY_f32x2[2*i+1], tRS_rOut[i] = params.act_bwd_fn(
                    tRS_rXY_f32x2[2*i], tRS_rXY_f32x2[2*i+1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (tRS_rdXY_f32x2[4*i], tRS_rdXY_f32x2[4*i+2]), \
                (tRS_rdXY_f32x2[4*i+1], tRS_rdXY_f32x2[4*i+3]), \
                (tRS_rOut[2*i], tRS_rOut[2*i+1]) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4*i], tRS_rXY_f32x2[4*i+2]),
                    (tRS_rXY_f32x2[4*i+1], tRS_rXY_f32x2[4*i+3]),
                    (tRS_rD_scaled[2*i], tRS_rD_scaled[2*i+1]),
                )
        if const_expr(tDrColVecReduce is not None):
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1]))
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]), row_sum)
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        # Write dXY back to D
        if const_expr(self.c_dtype == cutlass.Float8E4M3FN):
            pack_dtype = cutlass.BFloat16
        else:
            pack_dtype = params.implicit_dtype
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut


class GemmDGatedFP8CLoadSm100(GemmDGatedFP8CLoadMixin, GemmSm100):
    pass


dgate_fn_map = {
    "swiglu": quack.activation.dswiglu,
    "swiglu_oai": quack.activation.dswiglu_oai,
    "reglu": quack.activation.dreglu,
    "geglu": quack.activation.dgeglu,
    "glu": quack.activation.dglu,
}


def gemm_dgated(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PreAct: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    a_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for A
    b_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for B
    preact_fp8: Optional[Tensor] = None,  # (total_m, 2n) fp8 — replaces PreAct when provided
    preact_scales: Optional[Tensor] = None,  # (total_m, 2n//32) uint8 — blockscaled scales for preact_fp8
) -> None:
    """If tile_count_semaphore is provided, it must already be zero'ed out."""
    fp8_preact_mode = preact_fp8 is not None and preact_scales is not None
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        if not fp8_preact_mode:
            assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in dgate_fn_map, f"Unsupported activation {activation}"

    # Special handling for Out and PreAct
    AB_swapped = not Out.stride(-1) == 1
    if fp8_preact_mode:
        # FP8 PreAct: C = z_fp8 (TK, 2I) fp8 directly — TMA loads fp8 to smem
        implicit_dtype = cutlass.BFloat16  # for D output packing
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        if cu_seqlens_m is not None or not AB_swapped:
            Out = Out.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
        # PreAct stays as fp8 — passed directly to CUTLASS
        # Shape: (TK, 2I) fp8 ≠ (TK, I) f32, but TMA handles the different shape
        PreAct = preact_fp8  # (TK, 2I) Float8E4M3FN
    else:
        assert Out.dtype == PreAct.dtype
        implicit_dtype = torch2cute_dtype_map[Out.dtype]
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
        if cu_seqlens_m is not None or not AB_swapped:
            Out = Out.view(torch.float32)
            PreAct = PreAct.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
            PreAct = PreAct.mT.view(torch.float32).mT

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A,
        B,
        Out,
        None if fp8_preact_mode else PreAct,  # skip C validation for fp8 (different shape)
        additional_tensors={"PostAct": PostAct},
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    if fp8_preact_mode:
        # Manually set C tensor info for fp8 PreAct (TK, 2I) fp8
        from quack.gemm_wrapper_utils import GemmTensorInfo
        tensor_infos["C"] = GemmTensorInfo(PreAct)
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    # Use zero-materialization kernel when gather_A + blockscaled (FP8 with A_idx)
    blockscaled_runtime = a_scales is not None and b_scales is not None
    if fp8_preact_mode:
        assert device_capacity[0] > 9, "FP8 PreAct only supported on SM100+"
        if gather_A and blockscaled_runtime:
            from .gemm_sm100_fp8_zeromat import GemmDGatedSm100ZeroMat
            # TODO: ZeroMat + FP8CLoad variant
            GemmCls = GemmDGatedFP8CLoadSm100
        else:
            GemmCls = GemmDGatedFP8CLoadSm100
    elif device_capacity[0] > 9 and gather_A and blockscaled_runtime:
        from .gemm_sm100_fp8_zeromat import GemmDGatedSm100ZeroMat
        GemmCls = GemmDGatedSm100ZeroMat
    elif device_capacity[0] > 9:
        GemmCls = GemmDGatedSm100
    else:
        GemmCls = GemmDGatedSm90

    acc_dtype = Float32
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
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )
    act_fn = dgate_fn_map[activation]
    epi_kwargs = {}
    if fp8_preact_mode:
        epi_kwargs["mFP8PreAct_fp8"] = _make_cute_tensor_dynamic(preact_fp8, leading_dim=1)
        epi_kwargs["mFP8PreAct_scales"] = _make_cute_tensor_dynamic(preact_scales, leading_dim=1)
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        implicit_dtype=implicit_dtype,
        mColVecBroadcast=(
            from_dlpack(colvec_scale.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_scale is not None
            else None
        ),
        mColVecReduce=(
            from_dlpack(colvec_reduce.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=2 if cu_seqlens_m is None else 1
            )
            if colvec_reduce is not None
            else None
        ),
        **epi_kwargs,
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(max_active_clusters, tile_count_semaphore)

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
    )

    current_stream = cutlass_torch.current_stream()

    blockscaled = a_scales is not None and b_scales is not None
    sf_vec_size = 32 if blockscaled else None
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
        colvec_scale.dtype if colvec_scale is not None else None,
        colvec_reduce.dtype if colvec_reduce is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        blockscaled,
        fp8_preact_mode,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_dgated.compile_cache
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
            tensor_infos["D"].cute_tensor,  # Out
            tensor_infos["C"].cute_tensor,  # PreAct
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
        tensor_infos["D"].cute_tensor,  # Out
        tensor_infos["C"].cute_tensor,  # PreAct
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )


gemm_dgated.compile_cache = {}
