# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os

import torch
import torch.nn.functional as F
from ..count_cumsum import count_cumsum
from ..enums import ActivationType, is_glu
from ..quack_utils import (
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    blockscaled_fp8_gemm_varlen_triton,
    blockscaled_fp8_weight_grad_gemm,
    clear_raw_weight_cache,
    clear_sgl_weight_cache,
    gather_quantize_and_pack_activation,
    gemm_dgated,
    gemm_gated,
    gemm_gated_out,
    has_sgl_kernel,
    make_blockscaled_grouped_reverse_scatter_idx,
    precompute_weight_fp8,
    precompute_weight_fp8_for_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_raw_scales,
    quantize_activation_raw,
    quantize_and_pack_activation,
    quantize_and_pack_activation_varlen,
    sgl_mxfp8_gemm_varlen,
)
from quack.gemm_interface import gemm
from ..quack_utils.fp8_quack_patch import apply_fp8_quack_patch

apply_fp8_quack_patch()

from .backward import (
    _down_projection_backward_act,
    _down_projection_backward_weight,
    _softmax_topk_bwd,
    _token_broadcast_backward,
    _up_projection_backward_act,
    _up_projection_backward_weight,
)
from .fp8_protocol import (
    FP8ActivationDType,
    FP8Backend,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    get_default_fp8_protocol,
    is_blackwell_device,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
from .fp8_cutely_fused import apply_activation_fp8_protocol_cutely_fused
from .fp8_cutely_fused import apply_preact_activation_fp8_protocol_cutely_fused
from .fp8_reference import (
    FP8Tensor,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    quantize_activation_reference,
)
from .forward import _down_projection_forward, _router_forward, _softmax_topk_fwd, _up_projection_forward
from .triton_kernels import TC_topk_router_metadata_triton
from .utils import enable_quack_gemm, is_using_quack_gemm


# ---------------------------------------------------------------------------
# Standalone SwiGLU forward/backward (for blockscaled split path)
# ---------------------------------------------------------------------------
# SonicMoE stores w1 interleaved: [gate_row0, up_row0, gate_row1, ...].
# The GEMM output z thus has interleaved layout: columns 0,2,4,...=gate,
# columns 1,3,5,...=up.

from ..quack_utils.swiglu_triton import (
    swiglu_forward_triton,
    swiglu_backward_triton,
    swiglu_forward_quant_triton,
    swiglu_backward_quant_triton,
    swiglu_forward_quant_pack_triton,
    swiglu_backward_quant_pack_triton,
    swiglu_backward_from_fp8_triton,
    dequantize_blockscaled_fp8,
)
from ..quack_utils.blockscaled_fp8_gemm import (
    pack_blockscaled_1x32_scales,
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)


def _swiglu_forward_interleaved(z: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU on interleaved pre-activation z(TK, 2I) → y1(TK, I)."""
    return swiglu_forward_triton(z)


def _swiglu_backward_interleaved(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward SwiGLU + router score weighting on interleaved layout."""
    return swiglu_backward_triton(dy1, z, s)


def _use_fused_swiglu_quant() -> bool:
    """Check if fused SwiGLU+quantize kernels are enabled (default: enabled)."""
    return os.getenv("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _save_z_fp8() -> bool:
    """Check if z tensor should be stored in FP8 format to save memory (default: enabled).

    When enabled, z(TK, 2I) is quantized to blockscaled FP8 at end of forward
    and dequantized at start of backward, saving ~50% of z's memory footprint.
    """
    return os.getenv("SONIC_MOE_FP8_SAVE_Z_FP8", "1").lower() in {"1", "true", "yes", "on"}


def _use_fused_blockscaled_gated() -> bool:
    """Check if fused gemm_gated + blockscaled FP8 is enabled (default: enabled).

    When enabled, the blockscaled FP8 path uses fused gemm_gated/gemm_dgated
    (single CUTLASS kernel: GEMM + SwiGLU + blockscaled descale) instead of
    separate blockscaled_fp8_gemm_varlen + standalone SwiGLU.
    This provides ~1.8x speedup over BF16 fused at production shapes.
    """
    return os.getenv("SONIC_MOE_FP8_FUSED_GATED", "1").lower() in {"1", "true", "yes", "on"}


# Transfer pre-packed blockscaled scales between autograd Function boundaries.
# Each entry maps a tag to (fp8_tensor, packed_scales) or
# (fp8_tensor, packed_scales, raw_scales_uint8).  The consumer checks
# that its input tensor ``is`` the stored fp8_tensor before using the scales,
# which avoids id-reuse hazards.
# "fwd": _UpProjection.forward -> _DownProjection.forward
# "bwd": _DownProjection.backward -> _UpProjection.backward
_PREQUANTIZED_SCALES: dict[str, tuple] = {}


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


def _min_expert_segment(expert_frequency_offset: torch.Tensor) -> int:
    """Return minimum active expert segment length from cu_seqlens.

    Pure Python arithmetic on cached CPU values — zero GPU sync after first call.
    """
    if torch.cuda.is_current_stream_capturing():
        return 0
    vals = _get_cu_seqlens_cpu(expert_frequency_offset)
    active = [vals[i + 1] - vals[i] for i in range(len(vals) - 1) if vals[i + 1] > vals[i]]
    return min(active) if active else 0


_ALIGNMENT_STREAK: int = 0
_ALIGNMENT_ASSUMED: bool = (
    os.getenv("SONIC_MOE_FP8_ASSUME_ALIGNED", "").lower() in {"1", "true", "yes", "on"}
)
_ALIGNMENT_STREAK_THRESHOLD: int = 3


def _all_segments_128_aligned(cu_seqlens: torch.Tensor) -> bool:
    """Return True if all expert segments are 128-aligned (no GEMM padding needed).

    Pre-quantized activation input to blockscaled_fp8_gemm_varlen is only
    beneficial when no padding is required, because the padding fallback must
    dequantize → pad → re-quantize which is very expensive.

    After ``_ALIGNMENT_STREAK_THRESHOLD`` consecutive aligned iterations, the
    check is skipped entirely (zero D2H sync).  The env var
    ``SONIC_MOE_FP8_ASSUME_ALIGNED=1`` forces immediate zero-sync mode.
    """
    global _ALIGNMENT_STREAK, _ALIGNMENT_ASSUMED
    if _ALIGNMENT_ASSUMED:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    vals = _get_cu_seqlens_cpu(cu_seqlens)
    result = all((vals[i + 1] - vals[i]) % 128 == 0 for i in range(len(vals) - 1))
    if result:
        _ALIGNMENT_STREAK += 1
        if _ALIGNMENT_STREAK >= _ALIGNMENT_STREAK_THRESHOLD:
            _ALIGNMENT_ASSUMED = True
    else:
        _ALIGNMENT_STREAK = 0
    return result


def _use_cutely_fused_fp8_adapter() -> bool:
    return os.getenv("SONIC_MOE_FP8_CUTELY_FUSED", "").lower() in {"1", "true", "yes", "on"}


def _parse_runtime_precision(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise RuntimeError(f"{name} must be one of {{{allowed_list}}}, but got {value!r}")
    return value


def _legacy_blockscaled_fp8_downproj_enabled() -> bool:
    return os.getenv("SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ", "").lower() in {"1", "true", "yes", "on"}


def _upproj_epilogue_precision() -> str:
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION",
        default="fp8",
        allowed={"bf16", "fp8"},
    )


def _downproj_mainloop_precision() -> str:
    legacy_default = "fp8-blockscaled" if _legacy_blockscaled_fp8_downproj_enabled() else "bf16"
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION",
        default=legacy_default,
        allowed={"bf16", "fp8-blockscaled"},
    )


def _downproj_weight_precision() -> str:
    default = "fp8" if _downproj_mainloop_precision() == "fp8-blockscaled" else "bf16"
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION",
        default=default,
        allowed={"bf16", "fp8"},
    )


def _use_blockscaled_fp8_downproj() -> bool:
    return _downproj_mainloop_precision() == "fp8-blockscaled"


def _use_native_fp8_upproj() -> bool:
    return os.getenv("SONIC_MOE_OPT_NATIVE_FP8_UPPROJ", "").lower() in {"1", "true", "yes", "on"}


def _use_dummy_fp8_postact_buffer() -> bool:
    return os.getenv("SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER", "").lower() in {"1", "true", "yes", "on"}


def _use_mixed_dtype_downproj_dw2() -> bool:
    return os.getenv("SONIC_MOE_OPT_MIXED_DTYPE_DOWNPROJ_DW2", "").lower() in {"1", "true", "yes", "on"}


def _fp8_mode() -> str:
    """Return FP8 mode: 'off', 'perf' (cache+speed), or 'mem' (no-cache+savings)."""
    mode = os.getenv("SONIC_MOE_FP8_MODE", "").strip().lower()
    if mode in ("perf", "mem"):
        return mode
    # Backward compat: individual flags → perf mode
    if _use_native_fp8_upproj() or _use_mixed_dtype_downproj_dw2():
        return "perf"
    return "off"


def _fp8_enabled() -> bool:
    return _fp8_mode() != "off"


def _get_blockscaled_protocol() -> FP8Protocol:
    """Return FP8Protocol with 1×32 blockscaling for Blackwell hardware-native descaling."""
    return FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)


# ---------------------------------------------------------------------------
# FP8 weight helpers
# ---------------------------------------------------------------------------
_FP8_WEIGHT_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}

# Permuted + contiguous caches for gemm_gated / gemm_dgated custom kernels
_TAG_PERM = {
    "w1_ekh": (2, 1, 0),  # (2I,H,E) → (E,H,2I) contiguous — gemm_gated
    "w2_ehi": (2, 0, 1),  # (H,I,E)  → (E,H,I)  contiguous — gemm_dgated
}


def _make_fp8_weight(w: torch.Tensor, tag: str) -> torch.Tensor:
    """Create an fp8 copy of *w* with the permutation for *tag*.
    Single allocation: no intermediate bf16 contiguous copy."""
    perm = _TAG_PERM[tag]
    target_shape = tuple(w.shape[p] for p in perm)
    fp8_w = torch.empty(target_shape, dtype=torch.float8_e4m3fn, device=w.device)
    fp8_w.copy_(w.permute(*perm))
    return fp8_w


# Flag for one-shot lazy eviction when switching to blockscaled path.
_PER_TENSOR_EVICTED: bool = False


def _get_cached_fp8_weight(w: torch.Tensor, tag: str) -> torch.Tensor:
    """Return a cached fp8 copy of *w*. Always cached (essential for fused kernels)."""
    global _PER_TENSOR_EVICTED
    key = (w.untyped_storage().data_ptr(), w._version, tag)
    cached = _FP8_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    fp8_w = _make_fp8_weight(w, tag)
    if len(_FP8_WEIGHT_CACHE) >= 4:
        oldest = next(iter(_FP8_WEIGHT_CACHE))
        del _FP8_WEIGHT_CACHE[oldest]
    _FP8_WEIGHT_CACHE[key] = fp8_w
    # Per-tensor cache is being populated again; allow future eviction.
    _PER_TENSOR_EVICTED = False
    return fp8_w


# Original-layout fp8 cache for quack.gemm paths (permute views at call site)
_FP8_ORIG_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _get_fp8_weight_orig(w: torch.Tensor) -> torch.Tensor:
    """Return fp8 copy of *w* in original layout. Cached in perf mode."""
    global _PER_TENSOR_EVICTED
    if _fp8_mode() != "perf":
        return w.to(torch.float8_e4m3fn)
    key = (w.untyped_storage().data_ptr(), w._version)
    cached = _FP8_ORIG_CACHE.get(key)
    if cached is not None:
        return cached
    fp8_w = w.to(torch.float8_e4m3fn)
    if len(_FP8_ORIG_CACHE) >= 4:
        oldest = next(iter(_FP8_ORIG_CACHE))
        del _FP8_ORIG_CACHE[oldest]
    _FP8_ORIG_CACHE[key] = fp8_w
    # Per-tensor cache is being populated again; allow future eviction.
    _PER_TENSOR_EVICTED = False
    return fp8_w


def clear_fp8_native_weight_cache() -> None:
    """Call between steps if weights change (e.g. optimizer step)."""
    global _PER_TENSOR_EVICTED
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = False


def _evict_per_tensor_caches_once() -> None:
    """Clear per-tensor FP8 weight caches when transitioning to blockscaled path.

    Called once when the blockscaled path is first taken; subsequent calls are no-ops
    until the flag is reset (e.g. by clear_all_fp8_weight_caches).
    """
    global _PER_TENSOR_EVICTED
    if _PER_TENSOR_EVICTED:
        return
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = True


def clear_all_fp8_weight_caches() -> None:
    """Clear every FP8 weight cache (per-tensor + blockscaled).

    Intended for MoE.clear_fp8_weight_cache() and optimizer-step boundaries.
    """
    global _PER_TENSOR_EVICTED
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = False
    # Also clear the blockscaled weight cache in blockscaled_fp8_gemm.py
    from ..quack_utils import clear_blockscaled_fp8_weight_cache as _clear_bs
    _clear_bs()
    # Clear the Triton raw-scale weight cache
    clear_raw_weight_cache()
    # Clear the sgl-kernel weight cache
    clear_sgl_weight_cache()


def _validate_runtime_precision_switches(fp8_protocol: FP8Protocol | None) -> None:
    upproj_precision = _upproj_epilogue_precision()
    downproj_mainloop_precision = _downproj_mainloop_precision()
    downproj_weight_precision = _downproj_weight_precision()

    if fp8_protocol is None:
        return

    if downproj_weight_precision == "fp8" and downproj_mainloop_precision != "fp8-blockscaled":
        raise RuntimeError(
            "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION=fp8 currently requires "
            "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION=fp8-blockscaled"
        )


def _stage_memory_debug_enabled() -> bool:
    return os.getenv("SONIC_MOE_STAGEWISE_MEMORY", "").lower() in {"1", "true", "yes", "on"}


def _reset_stage_memory_probe() -> None:
    if not _stage_memory_debug_enabled() or torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _log_stage_memory(stage: str) -> None:
    if not _stage_memory_debug_enabled() or torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    mib = 1024**2
    print(
        f"[stage-memory] {stage}: "
        f"alloc_mib={torch.cuda.memory_allocated() / mib:.2f}, "
        f"reserved_mib={torch.cuda.memory_reserved() / mib:.2f}, "
        f"peak_alloc_mib={torch.cuda.max_memory_allocated() / mib:.2f}, "
        f"peak_reserved_mib={torch.cuda.max_memory_reserved() / mib:.2f}"
    )


def general_routing_router_metadata(
    router_scores_selected: torch.Tensor, sorted_selected_T: torch.Tensor, selected_E: torch.Tensor, T: int, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = router_scores_selected.device

    expert_frequency, expert_frequency_offset = count_cumsum(selected_E, E, do_cumsum=True)
    expert_frequency_offset = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), expert_frequency_offset])

    s_scatter_idx = selected_E.argsort().int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.size(0), device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    x_gather_idx = sorted_selected_T[s_scatter_idx]

    if T % 4 == 0 and T <= 50000:
        _, num_activated_expert_per_token_offset = count_cumsum(sorted_selected_T, T, do_cumsum=True)
    else:
        num_activated_expert_per_token_offset = torch.bincount(sorted_selected_T, minlength=T).cumsum(0).int()

    num_activated_expert_per_token_offset = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), num_activated_expert_per_token_offset]
    )

    return (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits: torch.Tensor, E: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        # change this to router_logits.dtype (bfloat16) increase another 5 tflops at fwd at the cost of numerical accuracy
        topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
        topk_router_indices = torch.empty(T, K, dtype=torch.int32, device=router_logits.device)

        _softmax_topk_fwd(router_logits, topk_router_score, topk_router_indices, E, K)

        ctx.save_for_backward(topk_router_score, topk_router_indices)
        ctx.E = E
        ctx.dtype = router_logits.dtype

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, K = dtopk_score.size()

        topk_router_score, topk_router_indices = ctx.saved_tensors
        dlogits = torch.zeros(T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device)

        _softmax_topk_bwd(dlogits, None, dtopk_score, topk_router_score, topk_router_indices, K)

        return dlogits, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        use_low_precision_postact_buffer: bool,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        use_quack_gemm = is_using_quack_gemm()

        if use_quack_gemm:
            assert not torch.compiler.is_compiling()
            assert is_glu_activation, "QuACK GEMM does not support non GLU activation yet"
            if _fp8_enabled():
                _evict_per_tensor_caches_once()
                aligned = _all_segments_128_aligned(expert_frequency_offset)
                w1_fp8, w1_scales = precompute_weight_fp8(w1)

                if aligned:
                    # All segments 128-aligned: use fused gather+quantize
                    # and pre-quantized GEMM (no padding overhead).
                    x_fp8, x_scales = gather_quantize_and_pack_activation(
                        x, x_gather_idx
                    )
                    z = blockscaled_fp8_gemm_varlen(
                        x_fp8, w1, expert_frequency_offset,
                        a_scales=x_scales,
                        w_fp8=w1_fp8, w_scales=w1_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=aligned,
                    )
                    del x_fp8, x_scales
                else:
                    # Non-aligned segments: bf16 gather, let GEMM
                    # handle padding + quantize internally.
                    x_gathered = x[x_gather_idx]
                    z = blockscaled_fp8_gemm_varlen(
                        x_gathered, w1, expert_frequency_offset,
                        w_fp8=w1_fp8, w_scales=w1_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=False,
                    )
                    del x_gathered

                # Fused SwiGLU+quant only when down-proj won't need padding
                if aligned and _use_fused_swiglu_quant():
                    if _save_z_fp8():
                        # Fused SwiGLU+y1_quant+z_save: read z ONCE
                        from sonicmoe.quack_utils.swiglu_triton import swiglu_forward_quant_pack_zsave_triton
                        y1_fp8, y1_packed_scales, z_fp8, z_raw_scales = (
                            swiglu_forward_quant_pack_zsave_triton(z)
                        )
                        _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                    else:
                        y1_fp8, y1_packed_scales = swiglu_forward_quant_pack_triton(z)
                    _PREQUANTIZED_SCALES["fwd"] = (y1_fp8, y1_packed_scales)
                    y1 = y1_fp8
                else:
                    y1 = _swiglu_forward_interleaved(z)
            else:
                z, y1 = gemm_gated(
                    x,
                    w1.permute(2, 1, 0),
                    activation="swiglu",
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    postact_dtype=(torch.float8_e4m3fn if use_low_precision_postact_buffer else None),
                    dynamic_scheduler=False,
                )
        else:
            z = torch.empty(TK, (2 * I if is_glu_activation else I), dtype=x.dtype, device=x.device)
            y1 = torch.empty(TK, I, dtype=x.dtype, device=x.device)
            _up_projection_forward(
                x=x,
                w1=w1,
                z=z,
                y1=y1,
                b1=b1,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
                activation_type=activation_type.value,
                is_glu_activation=is_glu_activation,
                is_inference_mode_enabled=is_inference_mode_enabled,
            )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_varlen_K = is_varlen_K
        ctx.is_glu_activation = is_glu_activation
        ctx.stream_id = stream_id
        ctx.use_quack_gemm = use_quack_gemm

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            None if use_quack_gemm else s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.mark_non_differentiable(y1)
        ctx.set_materialize_grads(False)

        return y1, z

    @staticmethod
    def backward(ctx, _: None, dz: torch.Tensor):
        is_compiling = torch.compiler.is_compiling()

        if not is_compiling:
            assert _ is None

        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_varlen_K = ctx.is_varlen_K
        stream_id = ctx.stream_id
        use_quack_gemm = ctx.use_quack_gemm

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            assert not is_compiling

            if _fp8_enabled():
                # Blockscaled FP8 act-grad + BF16 weight-grad.
                # Weight-grad stays BF16: bandwidth-bound (small K = TPE).

                # Weight grad: BF16 varlen GEMM (x^T × dz → dw1 per expert)
                dz_bf16 = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)
                gemm(
                    x.T,
                    dz_bf16,
                    out=dw1.permute(2, 1, 0),
                    cu_seqlens_k=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    batch_idx_permute=None,
                    dynamic_scheduler=False,
                )

                # Act grad: native blockscaled FP8 GEMM (dz × w1^T → dx_expanded)
                w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
                if _ALIGNMENT_ASSUMED:
                    # Pre-quantize dz to skip the _get_padding_plan sync
                    # inside blockscaled_fp8_gemm_varlen (safe because we've
                    # verified alignment for N consecutive iterations).
                    dz_fp8, dz_scales = quantize_and_pack_activation(
                        dz_bf16.contiguous(), group_size=32
                    )
                    dx_expanded = blockscaled_fp8_gemm_varlen(
                        dz_fp8, w1.permute(1, 0, 2), expert_frequency_offset,
                        a_scales=dz_scales,
                        w_fp8=w1T_fp8, w_scales=w1T_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=True,
                    )
                    del dz_fp8, dz_scales
                else:
                    dx_expanded = blockscaled_fp8_gemm_varlen(
                        dz_bf16, w1.permute(1, 0, 2), expert_frequency_offset,
                        w_fp8=w1T_fp8, w_scales=w1T_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=False,
                    )
                del dz_bf16
            else:
                gemm(
                    x.T,
                    dz,
                    out=dw1.permute(2, 1, 0),
                    cu_seqlens_k=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    batch_idx_permute=None,
                    dynamic_scheduler=False,
                )
                dx_expanded = gemm(dz, w1.permute(2, 0, 1), cu_seqlens_m=expert_frequency_offset, dynamic_scheduler=False)
        else:
            dx_expanded = torch.empty(TK, H, dtype=dz.dtype, device=dz.device)

            _up_projection_backward_act(
                w1=w1,
                dx_expanded=dx_expanded,
                dz=dz,
                db1=db1,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                is_glu_activation=is_glu_activation,
                stream_id=stream_id,
            )

            _up_projection_backward_weight(
                x=x,
                dw1=dw1,
                dz=dz,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                is_glu_activation=is_glu_activation,
                stream_id=stream_id,
            )

        _log_stage_memory("backward:up-proj-core")
        _reset_stage_memory_probe()
        dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )
        _log_stage_memory("backward:token-reduce")

        return dx_reduced, dw1, db1, *[None] * 12


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y1: torch.Tensor,
        z: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        fp8_protocol: FP8Protocol | None,
    ) -> torch.Tensor:
        TK = y1.size(0)
        H, I, E = w2.shape

        use_quack_gemm = is_using_quack_gemm()

        if use_quack_gemm:
            assert not torch.compiler.is_compiling()

            assert b2 is None
            if _fp8_enabled():
                # Blockscaled FP8 down-proj: use pre-quantized y1 if available
                # from fused SwiGLU+quant in _UpProjection.forward.
                w2_fp8, w2_scales = precompute_weight_fp8(w2)
                prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                if prequant is not None and prequant[0] is y1:
                    y1_fp8, y1_packed_scales = prequant
                    y2 = blockscaled_fp8_gemm_varlen(
                        y1_fp8, w2, expert_frequency_offset,
                        a_scales=y1_packed_scales,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=_ALIGNMENT_ASSUMED,
                    )
                else:
                    y2 = blockscaled_fp8_gemm_varlen(
                        y1, w2, expert_frequency_offset,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=False,
                    )
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            else:
                y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_frequency_offset)
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
        else:
            y2 = torch.empty(TK, H, dtype=y1.dtype, device=y1.device)
            _down_projection_forward(
                w2=w2,
                y1=y1,
                y2=y2,
                b2=b2,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
            )
            router_perm = s_reverse_scatter_idx
            y2_for_router = y2

        o = torch.empty(T, H, device=z.device, dtype=z.dtype)
        topk_scores = topk_scores.flatten()

        _router_forward(
            y2=y2_for_router,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=router_perm,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.stream_id = stream_id
        ctx.use_quack_gemm = use_quack_gemm

        # Memory optimization: store z in FP8 to save ~50% of z's memory.
        z_is_fp8 = (_fp8_enabled() and use_quack_gemm and _save_z_fp8()
                    and z.dtype == torch.bfloat16)
        ctx._z_is_fp8 = z_is_fp8

        if z_is_fp8:
            precomputed_z_fp8 = _PREQUANTIZED_SCALES.pop("z_fp8", None)
            if precomputed_z_fp8 is not None:
                z_fp8, z_raw_scales = precomputed_z_fp8
            else:
                z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
            ctx.save_for_backward(
                z_fp8,
                z_raw_scales,
                w2,
                b2,
                topk_scores,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
            )
        else:
            ctx.save_for_backward(
                z,
                w2,
                b2,
                topk_scores,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
            )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        stream_id = ctx.stream_id
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type
        use_quack_gemm = ctx.use_quack_gemm

        # Ensure dout is contiguous (expanded tensors from e.g. sum().backward()
        # have stride (0,0) which violates GEMM k-major assertions)
        dout = dout.contiguous()

        if ctx._z_is_fp8:
            (
                z_fp8,
                z_raw_scales,
                w2,
                b2,
                topk_scores,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
            ) = ctx.saved_tensors
            z_raw_scales_u8 = z_raw_scales.view(torch.uint8)
            # Defer dequantize: FP8 path uses fused kernel, others lazy-dequant
            z = None
        else:
            (
                z,
                w2,
                b2,
                topk_scores,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
            ) = ctx.saved_tensors
            z_fp8 = z_raw_scales_u8 = None

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            assert not torch.compiler.is_compiling()
            assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

            s = topk_scores[s_scatter_idx]
            if _fp8_enabled():
                aligned = _all_segments_128_aligned(expert_frequency_offset)
                w2_actgrad = w2.permute(1, 0, 2)  # (I, H, E)
                w2_fp8, w2_scales = precompute_weight_fp8(w2_actgrad)

                if aligned:
                    dout_fp8, dout_scales = gather_quantize_and_pack_activation(
                        dout, x_gather_idx
                    )
                    dy1 = blockscaled_fp8_gemm_varlen(
                        dout_fp8, w2_actgrad, expert_frequency_offset,
                        a_scales=dout_scales,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=aligned,
                    )
                    del dout_fp8, dout_scales
                else:
                    dout_gathered = dout[x_gather_idx]
                    dy1 = blockscaled_fp8_gemm_varlen(
                        dout_gathered, w2_actgrad, expert_frequency_offset,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=False,
                    )
                    del dout_gathered

                # Step 3: SwiGLU backward
                if z_fp8 is not None:
                    # Fused: read fp8 z directly, skip bf16 materialization
                    dz, y1s, ds = swiglu_backward_from_fp8_triton(
                        dy1, z_fp8, z_raw_scales_u8, s
                    )
                    del z_fp8, z_raw_scales_u8
                else:
                    dz, y1s, ds = _swiglu_backward_interleaved(dy1, z, s)
                del dy1

                _log_stage_memory("backward:down-proj-dgated")
                _reset_stage_memory_probe()

                # Weight-grad: BF16 varlen GEMM
                y1s_wgrad = y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                gemm(
                    dout.T,
                    y1s_wgrad,
                    out=dw2.permute(2, 0, 1),
                    cu_seqlens_k=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    batch_idx_permute=None,
                    dynamic_scheduler=False,
                )
                del y1s_wgrad
                _log_stage_memory("backward:down-proj-weight")
                ds = ds[s_reverse_scatter_idx]
            else:
                # BF16 path: needs bf16 z for gemm_dgated
                if z is None:
                    z = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                    del z_fp8, z_raw_scales_u8
                # BF16 path: cast colvec_scale to fp32 to avoid QuACK varlen
                # alignment bug (domain_offset on bf16 ptr reduces to 16-bit
                # alignment, but async copy requires 32-bit; fp32 is always
                # 32-bit aligned after any integer offset)
                dz = torch.empty_like(z)
                _, y1s, ds = gemm_dgated(
                    dout,
                    w2.permute(2, 0, 1),
                    PreAct=z,
                    activation="swiglu",
                    dx_out=dz,
                    colvec_scale=s.float(),
                    colvec_reduce=True,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                )
                _log_stage_memory("backward:down-proj-dgated")
                _reset_stage_memory_probe()

                y1s_wgrad = y1s.to(torch.bfloat16) if y1s.dtype == torch.float8_e4m3fn else y1s
                gemm(
                    dout.T,
                    y1s_wgrad,
                    out=dw2.permute(2, 0, 1),
                    cu_seqlens_k=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    batch_idx_permute=None,
                    dynamic_scheduler=False,
                )
                _log_stage_memory("backward:down-proj-weight")
                ds = ds[s_reverse_scatter_idx]
        else:
            # Non-quack path: needs bf16 z
            if z is None:
                z = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                del z_fp8, z_raw_scales_u8
            ds = torch.empty_like(topk_scores)
            dz = torch.empty_like(z)

            I = w2.size(1)
            TK = x_gather_idx.size(0)

            y1s = torch.empty(TK, I, dtype=z.dtype, device=z.device)
            is_glu_activation = is_glu(activation_type)

            _down_projection_backward_act(
                dout=dout,
                z=z,
                w2=w2,
                dz=dz,
                ds=ds,
                b2=b2,
                db2=db2,
                y1s=y1s,
                topk_scores=topk_scores,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                is_glu_activation=is_glu_activation,
                activation_type=activation_type.value,
                stream_id=stream_id,
            )
            _log_stage_memory("backward:down-proj-dgated")
            _reset_stage_memory_probe()

            _down_projection_backward_weight(
                dout=dout,
                y1s=y1s,
                dw2=dw2,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
            )
            _log_stage_memory("backward:down-proj-weight")

        _reset_stage_memory_probe()
        del y1s
        _log_stage_memory("backward:down-proj-postact-release")
        # TC top-K routing
        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dz, dw2, db2, ds, None, *[None] * 11


def _moe_tc_softmax_topk_layer_quack_inference(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType,
    fp8_protocol: FP8Protocol | None,
    use_low_precision_postact_buffer: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E = router_w.size(0)
    T = x.size(0)
    H = w2.size(0)
    TK = T * K
    device = x.device

    with torch.no_grad():
        _reset_stage_memory_probe()
        router_logits = F.linear(x, router_w)
        topk_scores = torch.empty(T, K, dtype=torch.float32, device=device)
        topk_indices = torch.empty(T, K, dtype=torch.int32, device=device)
        _softmax_topk_fwd(router_logits, topk_scores, topk_indices, E, K)

        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        )
        _log_stage_memory("forward:router-metadata")

        needs_preact = fp8_protocol is not None and _upproj_epilogue_precision() == "fp8"
        if _fp8_enabled():
            x_fp8 = x if x.dtype == torch.float8_e4m3fn else x.to(torch.float8_e4m3fn)
            w1_fp8 = _get_cached_fp8_weight(w1, "w1_ekh")
            z, y1 = gemm_gated(
                x_fp8,
                w1_fp8,
                activation="swiglu",
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                out_dtype=torch.bfloat16,
                postact_dtype=torch.float8_e4m3fn,
                store_preact=needs_preact,
                dynamic_scheduler=False,
            )
        else:
            z, y1 = gemm_gated(
                x,
                w1.permute(2, 1, 0),
                activation="swiglu",
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                postact_dtype=(torch.float8_e4m3fn if use_low_precision_postact_buffer else None),
                store_preact=needs_preact,
                dynamic_scheduler=False,
            )
        _log_stage_memory("forward:up-proj")

        # In full-pipeline FP8, y1 stays fp8 for down-proj.
        # In legacy/blockscaled FP8, convert to bf16.
        if _fp8_enabled() and not needs_preact:
            pass  # y1 stays fp8
        elif _fp8_enabled() and needs_preact:
            # Preact path with fp8 enabled: skip dequant round-trip
            if y1.dtype != x.dtype:
                y1 = y1.to(x.dtype)
        elif _use_native_fp8_upproj() and y1.dtype != x.dtype:
            y1 = y1.to(x.dtype)

        if needs_preact:
            _reset_stage_memory_probe()
            if _fp8_enabled():
                # y1 was computed via FP8 tensor cores; convert to bf16 and
                # skip the quant→dequant round-trip.
                if y1.dtype != x.dtype:
                    y1 = y1.to(x.dtype)
            else:
                restored_out = None
                if y1.size(-1) % fp8_protocol.group_size == 0:
                    if use_low_precision_postact_buffer:
                        restored_out = torch.empty(y1.shape, dtype=x.dtype, device=device)
                    else:
                        restored_out = y1
                y1, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                    z,
                    None,
                    fp8_protocol,
                    quack_enabled=True,
                    return_scales=False,
                    use_ste=False,
                    restored_out=restored_out,
                    output_dtype=x.dtype,
                )
            _log_stage_memory("forward:fp8-boundary")

        del z
        _reset_stage_memory_probe()
        if fp8_protocol is not None and _use_blockscaled_fp8_downproj():
            y2 = blockscaled_fp8_gemm_grouped(
                y1,
                w2,
                expert_frequency_offset,
                protocol=fp8_protocol,
            )
            router_perm = make_blockscaled_grouped_reverse_scatter_idx(
                s_reverse_scatter_idx,
                expert_frequency_offset,
                expert_ids=topk_indices.reshape(-1),
            )
            y2_for_router = y2.view(-1, H)
        else:
            if _fp8_enabled():
                y1_fp8 = y1 if y1.dtype == torch.float8_e4m3fn else y1.to(torch.float8_e4m3fn)
                w2_fp8 = _get_fp8_weight_orig(w2)
                y2 = gemm(y1_fp8, w2_fp8.permute(2, 1, 0),
                          cu_seqlens_m=expert_frequency_offset,
                          out_dtype=torch.bfloat16)
            else:
                y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_frequency_offset)
            router_perm = s_reverse_scatter_idx
            y2_for_router = y2

        del y1
        o = torch.empty(T, H, device=device, dtype=y2_for_router.dtype)
        topk_scores = topk_scores.flatten()
        _router_forward(
            y2=y2_for_router,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=router_perm,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=H,
            is_varlen_K=False,
        )
        _log_stage_memory("forward:down-proj-router")

    return o, router_logits, expert_frequency


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    fp8_protocol: FP8Protocol | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    _validate_runtime_precision_switches(fp8_protocol)
    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    use_low_precision_postact_buffer = (
        fp8_protocol is not None
        and _upproj_epilogue_precision() == "fp8"
        and is_using_quack_gemm()
        and _use_dummy_fp8_postact_buffer()
    )
    if is_inference_mode_enabled and is_using_quack_gemm():
        return _moe_tc_softmax_topk_layer_quack_inference(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            K,
            stream_id,
            activation_type,
            fp8_protocol,
            use_low_precision_postact_buffer,
        )

    E = router_w.size(0)
    _reset_stage_memory_probe()
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
    )
    _log_stage_memory("forward:router-metadata")

    T = x.size(0)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        T * K,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
        use_low_precision_postact_buffer,
    )
    _log_stage_memory("forward:up-proj")

    if fp8_protocol is not None and _upproj_epilogue_precision() == "fp8":
        _reset_stage_memory_probe()
        if _use_native_fp8_upproj() and is_using_quack_gemm():
            # y1 was computed via FP8 tensor cores and already converted to
            # bf16 inside _UpProjection.  Skip the quant→dequant round-trip;
            # the GEMM epilogue already applied the activation at fp8 precision.
            pass
        elif is_using_quack_gemm():
            restored_out = None
            if y1.size(-1) % fp8_protocol.group_size == 0:
                if use_low_precision_postact_buffer:
                    restored_out = torch.empty(y1.shape, dtype=z.dtype, device=z.device)
                else:
                    restored_out = y1
            with torch.no_grad():
                y1, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                    z,
                    None,
                    fp8_protocol,
                    quack_enabled=True,
                    return_scales=False,
                    use_ste=False,
                    restored_out=restored_out,
                    output_dtype=z.dtype,
                )
        else:
            fp8_adapter = apply_activation_fp8_protocol_cutely_fused if _use_cutely_fused_fp8_adapter() else apply_activation_fp8_protocol
            y1, _ = fp8_adapter(
                y1,
                fp8_protocol,
                quack_enabled=False,
                return_scales=False,
                use_ste=not is_inference_mode_enabled,
            )
        _log_stage_memory("forward:fp8-boundary")

    _reset_stage_memory_probe()
    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        topk_scores,
        topk_indices,
        expert_frequency_offset,
        T,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
        fp8_protocol,
    )
    _log_stage_memory("forward:down-proj-router")

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Weight format requirements:
# - w1_weight: Shape (2*I, H, E), stride order (2, 0, 1), must be interleaved [gate_row0, up_row0, gate_row1, up_row1, ...]
# - w2_weight: Shape (H, I, E), stride order (2, 0, 1)


# We assume token_indices is already SORTED ascendingly !!!
#   and len(token_indices) = len(expert_indices) = len(router_scores)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"

    T = x.size(0)
    TK = router_scores.size(0)
    E = w2.size(-1)
    (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = general_routing_router_metadata(router_scores, token_indices, expert_indices, T, E)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
        False,  # use_low_precision_postact_buffer
    )

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_scores,
        expert_indices,
        expert_frequency_offset,
        T,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
        None,
    )

    return o, expert_frequency
