# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import collections
import os

import torch
import torch.nn.functional as F
from ..count_cumsum import count_cumsum
from ..enums import ActivationType, is_glu
from ..quack_utils import (
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    clear_raw_weight_cache,
    clear_sgl_weight_cache,
    fast_gather_quantize_and_pack_activation,
    gemm_dgated,
    gemm_gated,
    make_blockscaled_grouped_reverse_scatter_idx,
    precompute_weight_fp8,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
)
from quack.gemm_interface import default_config, gemm
from ..quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
from ..quack_utils.fp8_quack_patch import apply_fp8_quack_patch

apply_fp8_quack_patch()

from .backward import (
    _softmax_topk_bwd,
    _token_broadcast_backward,
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
from .forward import _router_forward, _softmax_topk_fwd
from .triton_kernels import TC_topk_router_metadata_triton
from .utils import enable_fp8, enable_quack_gemm, is_fp8_active, is_using_quack_gemm


# ---------------------------------------------------------------------------
# Standalone SwiGLU forward/backward (for blockscaled split path)
# ---------------------------------------------------------------------------
# SonicMoE stores w1 interleaved: [gate_row0, up_row0, gate_row1, ...].
# The GEMM output z thus has interleaved layout: columns 0,2,4,...=gate,
# columns 1,3,5,...=up.

from ..quack_utils.swiglu_triton import (
    swiglu_forward_triton,
    swiglu_backward_triton,
    swiglu_forward_quant_pack_triton,
    swiglu_backward_quant_pack_triton,
    swiglu_backward_from_fp8_triton,
    dequantize_blockscaled_fp8,
)
from ..quack_utils.blockscaled_fp8_gemm import (
    pack_blockscaled_1x32_scales,
    quantize_activation_blockscaled_fast,
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


def _fused_blockscaled_gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run blockscaled GEMM+SwiGLU with zero-materialization FP8.

    Zero-materialization path (SonicMoE design principle):
    1. quantize_and_pack_activation(x) on T-sized tensor (~2-8µs)
    2. ISA-packed scale gather T→TK (~3-8µs, tiny I/O)
    3. Custom GemmGatedSm100ZeroMat kernel: T-FP8 + A_idx + TK-scales
    No TK-sized FP8 activation is materialized in HBM.

    Falls back to three-step pipeline if custom kernel fails.

    Parameters
    ----------
    w1_fp8_pre : optional pre-computed (w1_fp8, w1_scales) tuple.
        When provided, skips the global cache lookup (used in stash mode
        when the cache key may not match the modified parameter data_ptr).
    """
    from ..quack_utils.blockscaled_fp8_gemm import (
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )

    if w1_fp8_pre is not None:
        w1_fp8, w1_scales = w1_fp8_pre
    elif "w1_fused" in _STASHED_FP8_WEIGHTS:
        w1_fp8, w1_scales = _STASHED_FP8_WEIGHTS["w1_fused"]
    else:
        w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)

    # Step 1: Quantize at T-size (NOT TK)
    x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Step 2: Gather ISA-packed scales T→TK (~3-8µs)
    TK = x_gather_idx.shape[0]
    K = x.shape[1]
    k_tiles = _div_up(K, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(TK, K)
    x_scales_tk = (
        torch.empty((1, per_batch_tk), dtype=torch.uint8, device=x.device)
        if (TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0)
        else torch.full((1, per_batch_tk), 127, dtype=torch.uint8, device=x.device)
    )
    BLOCK_ROWS = 32
    _gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK_ROWS), k_tiles)](
        x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    x_scales_tk_e8m0 = x_scales_tk.view(torch.float8_e8m0fnu)
    del x_scales_t

    # Step 3: Zero-materialization GEMM via standard interface.
    # gemm_gated() with A_idx auto-selects GemmGatedSm100ZeroMat on SM100,
    # which gathers A rows inside the kernel (no TK FP8 materialization).
    # When epilogue quant is enabled, D output is fp8 directly (no bf16 round-trip).
    # The epilogue multiplies z by quant_scale in registers → hardware fp8 saturating
    # cast writes z_fp8 to D. This eliminates the standalone z.to(fp8) cast kernel
    # and halves D bandwidth (192MB fp8 vs 384MB bf16).
    cfg = _get_fp8_config()
    epilogue_quant = cfg.epilogue_quant and cfg.save_z_fp8
    if epilogue_quant:
        N = w1.shape[0]  # (2I, H, E) → w1.shape[0] = 2I
        z_scale_out = torch.empty(TK, N // 32, dtype=torch.uint8, device=x.device)
    else:
        z_scale_out = None

    z, y1 = gemm_gated(
        x_fp8, w1_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        dynamic_scheduler=False,
        tuned=False,
        z_scale_out=z_scale_out,
    )
    del x_fp8, x_scales_tk_e8m0

    if epilogue_quant:
        # Epilogue wrote scales + modified D (scaled z values).
        # Cast bf16 → fp8 for ctx memory saving.
        z_fp8 = z.to(torch.float8_e4m3fn)
        _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_scale_out.view(torch.float8_e8m0fnu))

    return z, y1


def _use_epilogue_quant() -> bool:
    """Check if epilogue blockscaled quant of z is enabled (default: enabled).

    When enabled, the GemmGated epilogue computes blockscaled FP8 quantization
    of z in registers (integer+carry E8M0, matching Triton/Paddle reference).
    This eliminates the standalone quantize_activation_blockscaled_fast kernel
    for z (−122 µs) and allows earlier z_bf16 freeing (−384 MiB transient).
    """
    return os.getenv("SONIC_MOE_FP8_EPILOGUE_QUANT", "0").lower() in {"1", "true", "yes", "on"}


def _use_fused_swiglu_quant() -> bool:
    """Check if fused SwiGLU+quantize kernels are enabled (default: enabled)."""
    return os.getenv("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _use_fused_zy1_quant() -> bool:
    """Check if fused z+y1 quantization is enabled (default: disabled).

    When enabled, z (flat scales) and y1 (ISA-packed scales) are quantized
    in a single fused Triton kernel launch, saving ~3µs launch overhead.
    Cost: +96 MiB forward peak (z_fp8 + y1_fp8 coexist during kernel).
    Enable with SONIC_MOE_FP8_FUSED_ZY1_QUANT=1.
    """
    return os.getenv("SONIC_MOE_FP8_FUSED_ZY1_QUANT", "").lower() in {"1", "true", "yes", "on"}


def _use_fp8_wgrad() -> bool:
    """Check if FP8 weight gradients are enabled.

    When enabled, wgrad GEMMs use blockscaled FP8 via varlen_k CUTLASS scheduling.
    This allows freeing dz_bf16 before wgrad (−384 MiB at Ernie shape).
    Performance is neutral at I=1536, positive at I≥2048.
    Default: disabled at I≤1536 (colwise quant overhead 977µs > bf16 GEMM 1125µs).
    Enable with SONIC_MOE_FP8_WGRAD=1 for I≥2048 where GEMM savings exceed quant cost.
    """
    return os.getenv("SONIC_MOE_FP8_WGRAD", "0").lower() in {"1", "true", "yes", "on"}


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
    separate blockscaled_fp8_gemm_varlen + standalone SwiGLU.  This is the
    best-performing FP8 up-proj path on Blackwell and is enabled by default.
    """
    return os.getenv("SONIC_MOE_FP8_FUSED_GATED", "1").lower() in {"1", "true", "yes", "on"}


# Transfer pre-packed blockscaled scales between autograd Function boundaries.
# Each entry maps a tag to (fp8_tensor, packed_scales) or
# (fp8_tensor, packed_scales, raw_scales_uint8).  The consumer checks
# that its input tensor shares the same storage/view metadata as the stored
# tensor before using the scales. Custom autograd boundaries may wrap the same
# storage in a fresh Tensor object, so object identity alone is too strict.
# "fwd": _UpProjection.forward -> _DownProjection.forward  (3-tuple: ref, fp8, scales)
# "bwd": _DownProjection.backward -> _UpProjection.backward (3-tuple: ref, fp8, scales)
_PREQUANTIZED_SCALES: dict[str, tuple] = {}

# Stashed FP8 weight references — populated by MoE.stash_bf16_to_cpu(),
# consumed by _fused_blockscaled_gated_forward and _DownProjection.forward
# to bypass global cache lookups when bf16 param storage has been freed.
# Keys: "w1_fused", "w2_varlen", "w2_dgated", "w1T_varlen"
_STASHED_FP8_WEIGHTS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

# Counter for pre-quantization hits (testing/diagnostics).
_PREQUANT_HIT_COUNT: dict[str, int] = collections.defaultdict(int)

# Side stream for overlapping wgrad with actgrad in _UpProjection.backward.
_WGRAD_STREAM: torch.cuda.Stream | None = None


def _get_wgrad_stream() -> torch.cuda.Stream:
    global _WGRAD_STREAM
    if _WGRAD_STREAM is None:
        _WGRAD_STREAM = torch.cuda.Stream()
    return _WGRAD_STREAM


# Side stream for overlapping z-dequant with dout-quant in _DownProjection.backward.
_DEQUANT_STREAM: torch.cuda.Stream | None = None


def _get_dequant_stream() -> torch.cuda.Stream:
    global _DEQUANT_STREAM
    if _DEQUANT_STREAM is None:
        _DEQUANT_STREAM = torch.cuda.Stream()
    return _DEQUANT_STREAM


def _matches_prequant_tensor(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> bool:
    if lhs is None or rhs is None:
        return False
    return (
        lhs.device == rhs.device
        and lhs.dtype == rhs.dtype
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and lhs.storage_offset() == rhs.storage_offset()
        and lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()
    )


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
    """Deprecated: always returns False. The cutely-fused adapter is only used
    for non-quack-gemm fallback which is a dead path on the frontier."""
    return False


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
    """Deprecated legacy flag — use SONIC_MOE_FP8_MODE=perf instead."""
    return False


def _use_dummy_fp8_postact_buffer() -> bool:
    """Deprecated test-only flag — always disabled on the frontier path."""
    return False


def _use_mixed_dtype_downproj_dw2() -> bool:
    """Deprecated legacy flag — use SONIC_MOE_FP8_MODE=perf instead."""
    return False


def _fp8_mode() -> str:
    """Return FP8 mode: 'off', 'perf' (cache+speed), or 'mem' (no-cache+savings).

    Checks the programmatic ``enable_fp8()`` flag first, then falls back to
    the ``SONIC_MOE_FP8_MODE`` environment variable.
    """
    if is_fp8_active():
        return "perf"
    mode = os.getenv("SONIC_MOE_FP8_MODE", "").strip().lower()
    if mode in ("perf", "mem"):
        return mode
    return "off"


def _fp8_enabled() -> bool:
    return _fp8_mode() != "off"


# ---------------------------------------------------------------------------
# FP8 runtime config — resolved once per forward, passed via ctx
# ---------------------------------------------------------------------------
class _FP8Config:
    """Snapshot of all FP8 flags, resolved once at forward entry.

    Eliminates repeated os.getenv() calls in the hot path.  Instances are
    cheap (no tensors), picklable, and stored on autograd ctx for backward.
    """
    __slots__ = (
        "enabled", "fused_gated", "save_z_fp8", "fused_swiglu_quant",
        "epilogue_quant", "fp8_wgrad", "alignment_assumed",
    )

    def __init__(self) -> None:
        self.enabled: bool = _fp8_enabled()
        self.fused_gated: bool = _use_fused_blockscaled_gated()
        self.save_z_fp8: bool = _save_z_fp8()
        self.fused_swiglu_quant: bool = _use_fused_swiglu_quant()
        self.epilogue_quant: bool = _use_epilogue_quant()
        self.fp8_wgrad: bool = _use_fp8_wgrad()
        self.alignment_assumed: bool = False  # set after alignment check

    @staticmethod
    def disabled() -> "_FP8Config":
        """Return a config where everything is off (BF16 path)."""
        cfg = _FP8Config.__new__(_FP8Config)
        cfg.enabled = False
        cfg.fused_gated = False
        cfg.save_z_fp8 = False
        cfg.fused_swiglu_quant = False
        cfg.epilogue_quant = False
        cfg.fp8_wgrad = False
        cfg.alignment_assumed = False
        return cfg


# Module-level singleton, refreshed per forward call.
_fp8_cfg: _FP8Config = _FP8Config.disabled()


def _get_fp8_config() -> _FP8Config:
    """Return the current FP8 config (resolved at forward entry)."""
    return _fp8_cfg


def _refresh_fp8_config() -> _FP8Config:
    """Re-read all env vars and return a fresh config. Call at forward entry."""
    global _fp8_cfg
    _fp8_cfg = _FP8Config()
    return _fp8_cfg


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
            cfg = _get_fp8_config()
            if cfg.enabled:
                global _ALIGNMENT_ASSUMED
                _evict_per_tensor_caches_once()
                aligned = _all_segments_128_aligned(expert_frequency_offset)
                _ALIGNMENT_ASSUMED = aligned
                cfg.alignment_assumed = aligned

                if aligned and cfg.fused_gated:
                    z, y1 = _fused_blockscaled_gated_forward(
                        x, w1, expert_frequency_offset, x_gather_idx
                    )
                    if cfg.save_z_fp8 and "z_fp8" not in _PREQUANTIZED_SCALES:
                        if _use_fused_zy1_quant():
                            # Fused z+y1 quantization: single kernel launch, ~3µs
                            # less launch overhead, but +96 MiB peak (z_fp8 + y1_fp8
                            # coexist with z_bf16 + y1_bf16 during the kernel).
                            from ..quack_utils.blockscaled_fp8_gemm import fused_z_save_y1_quant
                            z_fp8, z_raw_scales, y1_fp8, y1_packed_scales = (
                                fused_z_save_y1_quant(z, y1)
                            )
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            z.untyped_storage().resize_(0)
                        else:
                            # Split quantization: z first, free z bf16, then y1.
                            # This avoids z_bf16+y1_bf16+z_fp8+y1_fp8 all coexisting
                            # and reduces forward peak by ~96 MiB at Ernie shape.
                            from ..quack_utils.blockscaled_fp8_gemm import (
                                quantize_activation_blockscaled_fast,
                            )
                            z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            z.untyped_storage().resize_(0)
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                    else:
                        # z_fp8 already populated (epilogue quant wrote it inside
                        # _fused_blockscaled_gated_forward).  Free z_bf16 now (−384 MiB).
                        if cfg.save_z_fp8:
                            z.untyped_storage().resize_(0)
                        y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                    _PREQUANTIZED_SCALES["fwd"] = (y1, y1_fp8, y1_packed_scales)
                    y1.untyped_storage().resize_(0)
                elif aligned:
                    w1_fp8, w1_scales = precompute_weight_fp8(w1)
                    # All segments 128-aligned: use fused gather+quantize
                    # and pre-quantized GEMM (no padding overhead).
                    x_fp8, x_scales = fast_gather_quantize_and_pack_activation(
                        x, x_gather_idx
                    )
                    z = blockscaled_fp8_gemm_varlen(
                        x_fp8, w1, expert_frequency_offset,
                        a_scales=x_scales,
                        w_fp8=w1_fp8, w_scales=w1_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=True,
                    )
                    del x_fp8, x_scales

                    # Fused SwiGLU+quant only when segments are aligned
                    if cfg.fused_swiglu_quant:
                        if cfg.save_z_fp8:
                            # Fused SwiGLU+y1_quant+z_save: read z ONCE
                            from sonicmoe.quack_utils.swiglu_triton import swiglu_forward_quant_pack_zsave_triton
                            y1_fp8, y1_packed_scales, z_fp8, z_raw_scales = (
                                swiglu_forward_quant_pack_zsave_triton(z)
                            )
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                        else:
                            y1_fp8, y1_packed_scales = swiglu_forward_quant_pack_triton(z)
                        _PREQUANTIZED_SCALES["fwd"] = (y1_fp8, y1_fp8, y1_packed_scales)
                        y1 = y1_fp8
                    else:
                        y1 = _swiglu_forward_interleaved(z)
                else:
                    # Non-aligned: fall back to BF16 fused path (gemm_gated).
                    # FP8-with-padding is 2-8x slower than BF16 fused due to
                    # per-expert padding overhead (128 copy+pad+GEMM+unpad ops).
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
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
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
        # Store FP8 config snapshot for backward (avoids os.getenv in backward).
        ctx._fp8_cfg = cfg if (use_quack_gemm and cfg.enabled) else _FP8Config.disabled()
        # Legacy compat: keep individual flags for code that reads them directly.
        ctx._fp8_enabled = ctx._fp8_cfg.enabled
        ctx._alignment_assumed = ctx._fp8_cfg.alignment_assumed

        # Weight decoupling: in FP8+aligned mode, backward doesn't need bf16 w1 data
        # (only uses fp8 cache + metadata). This enables stash_bf16_to_cpu() to
        # resize_(0) the bf16 param storage without breaking backward.
        _fp8_aligned = (use_quack_gemm and cfg.enabled and cfg.alignment_assumed)
        ctx._w1_decoupled = _fp8_aligned
        if _fp8_aligned:
            # Store metadata needed for dw1 allocation
            ctx._w1_shape = w1.shape  # (2I, H, E)
            ctx._w1_dtype = w1.dtype
            ctx._w1_device = w1.device
            # Eagerly lookup w1T fp8 cache — will be used in backward actgrad.
            # This is a cache hit (zero compute) since forward already populated the fused cache.
            _w1T_fp8, _w1T_scales = _STASHED_FP8_WEIGHTS.get("w1T_varlen", None) or precompute_weight_fp8(w1.permute(1, 0, 2))
            ctx._w1T_fp8 = _w1T_fp8
            ctx._w1T_scales = _w1T_scales
            ctx.save_for_backward(
                x,
                # w1 omitted — backward uses ctx._w1T_fp8 + metadata
                b1,
                expert_frequency_offset,
                x_gather_idx,
                None if use_quack_gemm else s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            )
        else:
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

        # Keep w1 FP8 cache — backward hits cache (~112µs savings) at ~74MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

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

        if ctx._w1_decoupled:
            # FP8+aligned: w1 not in saved_tensors; use metadata + fp8 cache.
            (
                x,
                b1,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            ) = ctx.saved_tensors
            w1_shape = ctx._w1_shape   # (2I, H, E)
            w1_dtype = ctx._w1_dtype
            w1_device = ctx._w1_device
        else:
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
            w1_shape = w1.shape
            w1_dtype = w1.dtype
            w1_device = w1.device

        # Defer dw1 allocation for FP8 wgrad path (blockscaled_fp8_wgrad_varlen_k
        # allocates its own output).  BF16 path allocates below.
        dw1_base = dw1 = None
        db1 = None if b1 is None else torch.empty_like(b1)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            assert not is_compiling

            if ctx._fp8_enabled and ctx._alignment_assumed:
                # Blockscaled FP8 act-grad + weight-grad.
                # Memory-optimized: run wgrad first, free dz bf16 (~384 MiB),
                # then run actgrad using FP8 dz from prequant cache.
                # This serializes the two GEMMs but avoids dz_bf16 + dx_expanded
                # coexisting, reducing backward peak by ~384 MiB.
                dz_bf16 = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)

                # Prepare actgrad resources first (cache lookup, no alloc).
                if ctx._w1_decoupled:
                    # w1T fp8 was pre-looked-up in forward; use directly.
                    w1T_fp8 = ctx._w1T_fp8
                    w1T_scales = ctx._w1T_scales
                else:
                    w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
                prequant_dz = _PREQUANTIZED_SCALES.pop("bwd", None)
                if ctx._fp8_cfg.fp8_wgrad:
                    # FP8 wgrad: dz_bf16 was already freed in DownProj via dual-quant.
                    # Skip _matches_prequant_tensor (dz storage is 0).
                    has_prequant = prequant_dz is not None
                else:
                    has_prequant = (
                        prequant_dz is not None
                        and _matches_prequant_tensor(prequant_dz[0], dz)
                    )

                # Phase 1: Wgrad.
                if ctx._fp8_cfg.fp8_wgrad:
                    # FP8 wgrad with early dz_bf16 release.
                    # dz_col_fp8 was pre-computed in DownProj via dual_quantize_varlen
                    # (single HBM read of dz produced both row+col fp8).
                    from ..quack_utils.blockscaled_fp8_gemm import (
                        colwise_quantize_and_pack,
                        _run_cutlass_blockscaled_gemm_varlen_k,
                    )
                    bwd_col = _PREQUANTIZED_SCALES.pop("bwd_col", None)
                    if bwd_col is not None:
                        # Use pre-computed col-fp8 from dual quant (zero extra HBM read)
                        dz_col_fp8, dz_col_scales = bwd_col
                    else:
                        # Fallback: compute col-fp8 now
                        dz_col_fp8, dz_col_scales = colwise_quantize_and_pack(
                            dz_bf16, logical_rows=w1_shape[0], logical_cols=TK,
                        )
                    # FREE dz_bf16 NOW (−384 MiB before wgrad GEMM!)
                    dz.untyped_storage().resize_(0)
                    del dz_bf16
                    # Colwise-quant x (small: T × H = 48 MiB bf16)
                    x_col_fp8, x_col_scales = colwise_quantize_and_pack(
                        x, logical_rows=H, logical_cols=TK,
                        gather_idx=x_gather_idx,
                    )
                    # CUTLASS wgrad GEMM
                    dw1_base = _run_cutlass_blockscaled_gemm_varlen_k(
                        dz_col_fp8, dz_col_scales,
                        x_col_fp8, x_col_scales,
                        expert_frequency_offset,
                        M=w1_shape[0], N=H, total_K=TK,
                        num_experts=E, out_dtype=w1_dtype, device=x.device,
                    )
                    dw1 = dw1_base.permute(1, 2, 0)
                    del dz_col_fp8, dz_col_scales, x_col_fp8, x_col_scales
                else:
                    dw1_base = torch.empty((E, w1_shape[0], w1_shape[1]), dtype=w1_dtype, device=w1_device)
                    dw1 = dw1_base.permute(1, 2, 0)
                    gemm(
                        x.T,
                        dz_bf16,
                        out=dw1_base.permute(0, 2, 1),
                        cu_seqlens_k=expert_frequency_offset,
                        A_idx=x_gather_idx,
                        batch_idx_permute=None,
                        dynamic_scheduler=False,
                    )

                # Phase 2: Free dz bf16 storage (~384 MiB at Ernie shape).
                # FP8 wgrad already freed it in step 2 above; BF16 path frees here.
                if not ctx._fp8_cfg.fp8_wgrad:
                    dz.untyped_storage().resize_(0)
                    del dz_bf16

                # Phase 3: Actgrad using FP8 dz (avoids dz_bf16 + dx_expanded coexistence).
                if has_prequant:
                    _PREQUANT_HIT_COUNT["bwd"] += 1
                    _, dz_fp8, dz_packed_scales = prequant_dz
                    if ctx._w1_decoupled:
                        # w1 not in saved_tensors; call low-level GEMM directly
                        # with shape metadata (avoids needing a weight tensor).
                        from ..quack_utils.blockscaled_fp8_gemm import (
                            _run_cutlass_blockscaled_gemm,
                        )
                        dx_expanded = _run_cutlass_blockscaled_gemm(
                            dz_fp8, dz_packed_scales,
                            w1T_fp8, w1T_scales,
                            expert_frequency_offset,
                            total_M=dz_fp8.shape[0],
                            K=dz_fp8.shape[1],
                            H=w1_shape[1],       # w1 is (2I, H, E), H=shape[1]
                            num_experts=E,
                            out_dtype=torch.bfloat16,
                            device=dz_fp8.device,
                        )
                    else:
                        dx_expanded = blockscaled_fp8_gemm_varlen(
                            dz_fp8, w1.permute(1, 0, 2), expert_frequency_offset,
                            a_scales=dz_packed_scales,
                            w_fp8=w1T_fp8, w_scales=w1T_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                    del dz_fp8, dz_packed_scales
                    # Keep w1T FP8 cache (~74 MiB) — avoids 308µs permute+contiguous
                    # on next iter.  Cache auto-invalidates via w._version at optimizer step.
                    del w1T_fp8, w1T_scales
                else:
                    # No prequant: quantize dz inline (dz storage was freed;
                    # this path should not be reached with fused gated).
                    raise RuntimeError(
                        "dz storage freed but no bwd prequant — cannot quantize. "
                        "Ensure _DownProjection backward creates bwd prequant."
                    )
            else:
                dw1_base = torch.empty((E, w1_shape[0], w1_shape[1]), dtype=w1_dtype, device=w1_device)
                dw1 = dw1_base.permute(1, 2, 0)
                gemm(
                    x.T,
                    dz,
                    out=dw1_base.permute(0, 2, 1),
                    cu_seqlens_k=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    batch_idx_permute=None,
                    dynamic_scheduler=False,
                )
                dx_expanded = gemm(dz, w1.permute(2, 0, 1), cu_seqlens_m=expert_frequency_offset, dynamic_scheduler=False)
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
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
            cfg = _get_fp8_config()
            if cfg.enabled and cfg.alignment_assumed:
                if cfg.fused_gated:
                    # Use pre-quantized y1 from _UpProjection if available
                    # (zero quant overhead — y1 was quantized while hot in L2).
                    # Format: 3-tuple (bf16_ref, fp8_data, packed_scales).
                    prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                    has_prequant = (
                        prequant is not None
                        and len(prequant) == 3
                        and _matches_prequant_tensor(prequant[0], y1)
                    )
                    if has_prequant:
                        _PREQUANT_HIT_COUNT["fwd"] += 1
                        w2_fp8, w2_scales = _STASHED_FP8_WEIGHTS.get("w2_varlen", None) or precompute_weight_fp8(w2)
                        _, y1_fp8, y1_packed_scales = prequant
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_packed_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                        del y1_fp8, y1_packed_scales
                    else:
                        # Fallback: inline FP8 quant (prequant cache miss)
                        w2_fp8, w2_scales = _STASHED_FP8_WEIGHTS.get("w2_varlen", None) or precompute_weight_fp8(w2)
                        y1_fp8, y1_scales = quantize_and_pack_activation(y1)
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                        del y1_fp8, y1_scales
                else:
                    # Blockscaled FP8 down-proj: use pre-quantized y1 if available
                    # from fused SwiGLU+quant in _UpProjection.forward.
                    w2_fp8, w2_scales = precompute_weight_fp8(w2)
                    prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                    has_prequant = (
                        prequant is not None
                        and len(prequant) == 3
                        and _matches_prequant_tensor(prequant[0], y1)
                    )
                    if has_prequant:
                        _PREQUANT_HIT_COUNT["fwd"] += 1
                        _, y1_fp8, y1_packed_scales = prequant
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_packed_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                    else:
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1, w2, expert_frequency_offset,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                # Keep w2 varlen cache — iso32 re-quant is expensive (~87µs/iter).
                # Cache auto-invalidates via w._version at optimizer step.
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            else:
                y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_frequency_offset)
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

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
        # Store FP8 config snapshot for backward.
        ctx._fp8_cfg = cfg if (use_quack_gemm and cfg.enabled) else _FP8Config.disabled()
        # Legacy compat aliases
        ctx._fp8_enabled_flag = ctx._fp8_cfg.enabled
        ctx._alignment_assumed_flag = ctx._fp8_cfg.alignment_assumed
        ctx._use_fused_blockscaled_gated_flag = ctx._fp8_cfg.fused_gated

        # Memory optimization: store z in FP8 to save ~50% of z's memory.
        # At Ernie shape (TK=65536, 2I=3072), z is 384MB BF16 → ~213MB FP8 = ~171MB saved.
        # Accept fp8 z when prequant cache already holds the fp8+scales pair
        # (e.g. epilogue quant produced them), even if z.dtype is no longer bf16.
        z_has_prequant = "z_fp8" in _PREQUANTIZED_SCALES
        z_is_fp8 = (cfg.enabled and use_quack_gemm and cfg.save_z_fp8
                    and cfg.alignment_assumed
                    and (z.dtype == torch.bfloat16 or z_has_prequant))
        ctx._z_is_fp8 = z_is_fp8

        # w2 decoupling: in FP8+aligned+fused_gated mode, backward doesn't
        # read bf16 w2 data (uses fp8 dgated cache + metadata).  This enables
        # stash_bf16_to_cpu() to resize_(0) the bf16 param storage.
        _w2_decouple = z_is_fp8 and cfg.fused_gated
        ctx._w2_decoupled = _w2_decouple

        if z_is_fp8:
            precomputed_z_fp8 = _PREQUANTIZED_SCALES.pop("z_fp8", None)
            if precomputed_z_fp8 is not None:
                z_fp8, z_raw_scales = precomputed_z_fp8
            else:
                assert z.nelement() > 0, (
                    "z storage was freed for memory optimization but prequant "
                    "cache miss — this should not happen"
                )
                assert z.dtype == torch.bfloat16, (
                    f"z_is_fp8=True but no prequant cache and z.dtype={z.dtype} "
                    f"(expected bf16 for inline quantization)"
                )
                z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
            if _w2_decouple:
                # Eagerly look up w2 dgated fp8 cache for backward.
                _w2_dgated_fp8, _w2_dgated_scales = _STASHED_FP8_WEIGHTS.get("w2_dgated", None) or precompute_weight_fp8_for_direct_fused_dgated(w2)
                ctx._w2_dgated_fp8 = _w2_dgated_fp8
                ctx._w2_dgated_scales = _w2_dgated_scales
                ctx._w2_shape = w2.shape  # (H, I, E)
                ctx._w2_dtype = w2.dtype
                ctx._w2_device = w2.device
                ctx.save_for_backward(
                    z_fp8,
                    z_raw_scales,
                    # w2 omitted — backward uses ctx._w2_dgated_fp8 + metadata
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                )
            else:
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

        # Keep w2 FP8 cache — backward hits cache (~38µs savings) at ~37MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

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
            if ctx._w2_decoupled:
                (
                    z_fp8,
                    z_raw_scales,
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                ) = ctx.saved_tensors
                w2_shape = ctx._w2_shape   # (H, I, E)
                w2_dtype = ctx._w2_dtype
                w2_device = ctx._w2_device
            else:
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
                w2_shape = w2.shape
                w2_dtype = w2.dtype
                w2_device = w2.device
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
            w2_shape = w2.shape
            w2_dtype = w2.dtype
            w2_device = w2.device
            z_fp8 = z_raw_scales_u8 = None

        # Defer dw2 allocation: in the fused_gated path, dw2 is not needed
        # until the wgrad GEMM (~384 MiB after dgated outputs dz+y1s).
        # Allocating here adds 72 MiB to the dgated peak unnecessarily.
        dw2_base = dw2 = None  # allocated just before wgrad in each path
        db2 = None if b2 is None else torch.empty_like(b2)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            assert not torch.compiler.is_compiling()
            assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

            s = topk_scores[s_scatter_idx]
            if ctx._fp8_enabled_flag and ctx._alignment_assumed_flag:
                # All segments aligned: use blockscaled FP8 path.
                if ctx._use_fused_blockscaled_gated_flag:
                    # Zero-materialization FP8 dgated: T-quant + scale_gather + A_idx
                    from ..quack_utils.blockscaled_fp8_gemm import (
                        _gather_isa_packed_scales_kernel,
                        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE,
                        _SF_VEC_SIZE, _storage_per_batch,
                    )

                    # --- Phase 3.1: FP8 PreAct eliminates z dequant + 384MB temp ---
                    # When z is fp8 (from ctx), pass directly to GemmDGated.
                    # The kernel loads fp8 z + scales in its epilogue via EpiOp LDG,
                    # avoiding the standalone dequant kernel and z_bf16 allocation.
                    use_fp8_preact = (z is None and z_fp8 is not None)

                    if not use_fp8_preact:
                        # Fallback: standalone dequant (when z is already bf16)
                        s_float = s.float()
                        if z is None:
                            _ds = _get_dequant_stream()
                            _ds.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(_ds):
                                z = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                                del z_fp8, z_raw_scales_u8
                    else:
                        s_float = s.float()

                    # dout-quant + scale_gather on default stream (parallel with dequant if fallback).
                    dout_fp8, dout_scales_t = quantize_and_pack_activation(dout)
                    TK_bwd = x_gather_idx.shape[0]
                    K_bwd = dout.shape[1]
                    k_tiles_bwd = _div_up(K_bwd, _SF_TILE_K)
                    per_batch_bwd = _storage_per_batch(TK_bwd, K_bwd)
                    dout_scales_tk = (
                        torch.empty((1, per_batch_bwd), dtype=torch.uint8, device=dout.device)
                        if (TK_bwd % _SF_TILE_M == 0 and K_bwd % _SF_TILE_K == 0)
                        else torch.full((1, per_batch_bwd), 127, dtype=torch.uint8, device=dout.device)
                    )
                    _gather_isa_packed_scales_kernel[(_div_up(TK_bwd, 32), k_tiles_bwd)](
                        dout_scales_t.view(torch.uint8), x_gather_idx, dout_scales_tk, TK_bwd,
                        src_k_tiles=k_tiles_bwd, dst_k_tiles=k_tiles_bwd,
                        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
                        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
                    )
                    dout_scales = dout_scales_tk.view(torch.float8_e8m0fnu)
                    del dout_scales_t, dout_scales_tk

                    # Synchronize: gemm_dgated_kernel needs dout_fp8/dout_scales
                    # (and z_bf16 if not using fp8 preact).
                    if not use_fp8_preact:
                        torch.cuda.current_stream().wait_stream(_get_dequant_stream())

                    if ctx._w2_decoupled:
                        w2_fp8_enk = ctx._w2_dgated_fp8
                        w2_scales = ctx._w2_dgated_scales
                    else:
                        w2_fp8_enk, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)
                    config = default_config(dout.device)
                    total_m = x_gather_idx.shape[0]  # TK (not T — dout_fp8 is T-sized)
                    n = w2_fp8_enk.shape[-2]
                    dz = torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=dout.device)
                    y1s = torch.empty((total_m, n), dtype=torch.bfloat16, device=dout.device)
                    colvec_reduce_partial = torch.empty(
                        (total_m, (n + config.tile_n - 1) // config.tile_n),
                        dtype=torch.float32,
                        device=dout.device,
                    )
                    gemm_dgated_kernel(
                        dout_fp8,
                        w2_fp8_enk,
                        dz,
                        dz,  # PreAct: ignored when preact_fp8 is set (C=None in wrapper)
                        y1s,
                        None,
                        "swiglu",
                        config.tile_m,
                        config.tile_n,
                        config.cluster_m,
                        config.cluster_n,
                        config.pingpong,
                        persistent=True,
                        max_swizzle_size=config.max_swizzle_size,
                        colvec_scale=s_float,
                        colvec_reduce=colvec_reduce_partial,
                        cu_seqlens_m=expert_frequency_offset,
                        A_idx=x_gather_idx,
                        a_scales=dout_scales,
                        b_scales=w2_scales,
                        preact_fp8=z_fp8 if use_fp8_preact else None,
                        preact_scales=z_raw_scales_u8 if use_fp8_preact else None,
                    )
                    ds = colvec_reduce_partial.sum(dim=-1)
                    del dout_fp8, dout_scales, z, colvec_reduce_partial
                    # Release FP8 preact tensors from ctx (z_fp8 ~192 MiB + scales ~6 MiB).
                    # The dgated GEMM is done; these are no longer needed.
                    if use_fp8_preact:
                        del z_fp8, z_raw_scales_u8
                    # Keep w2 fused cache — avoids ~40µs re-quant on next iter.
                    del w2_fp8_enk, w2_scales

                    # Weight-grad: dw2 = dout.T @ y1s (per expert).
                    _log_stage_memory("backward:down-proj-dgated")
                    _reset_stage_memory_probe()
                    if ctx._fp8_cfg.fp8_wgrad:
                        from ..quack_utils import blockscaled_fp8_wgrad_varlen_k
                        # dW2[e] = dout_e^T @ y1s_e = (H, TK_e) @ (TK_e, I) = (H, I)
                        # blockscaled_fp8_wgrad_varlen_k: a=(TK,M), b=(TK,N) → (E,M,N)
                        # So a=y1s (TK,I=M), b=dout (T,H=N) with b_gather_idx
                        # Result: (E, I, H). But we need dw2 as (H, I, E).
                        # Swap: a=dout, b=y1s → (E, H, I) → dw2 = .permute(1,2,0)
                        dw2_base = blockscaled_fp8_wgrad_varlen_k(
                            dout, y1s, cu_seqlens_k=expert_frequency_offset,
                            M=dout.shape[1], N=y1s.shape[1],
                            a_gather_idx=x_gather_idx,
                        )
                        dw2 = dw2_base.permute(1, 2, 0)
                    else:
                        dw2_base = torch.empty((w2_shape[2], w2_shape[0], w2_shape[1]), dtype=w2_dtype, device=w2_device)
                        dw2 = dw2_base.permute(1, 2, 0)
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
                    del y1s
                    _log_stage_memory("backward:down-proj-weight")

                    # Pre-quantize dz: single HBM read produces BOTH:
                    # - row_fp8 + scales (for actgrad in UpProj backward)
                    # - col_fp8 + scales (for FP8 wgrad in UpProj backward)
                    if ctx._fp8_cfg.fp8_wgrad:
                        from ..quack_utils.blockscaled_fp8_gemm import dual_quantize_varlen
                        dz_fp8, dz_packed_scales, dz_col_fp8, dz_col_scales = (
                            dual_quantize_varlen(dz, TK=dz.shape[0], dim=dz.shape[1])
                        )
                        # Free dz_bf16 NOW — FP8 wgrad uses col_fp8, actgrad uses row_fp8.
                        # Neither needs dz_bf16 after dual-quant.
                        _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                        dz.untyped_storage().resize_(0)  # −384 MiB!
                        _PREQUANTIZED_SCALES["bwd_col"] = (dz_col_fp8, dz_col_scales)
                    else:
                        dz_fp8, dz_packed_scales = quantize_and_pack_activation(dz)
                        _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                    ds = ds[s_reverse_scatter_idx]
                else:
                    w2_actgrad = w2.permute(1, 0, 2)  # (I, H, E)
                    w2_fp8, w2_scales = precompute_weight_fp8(w2_actgrad)

                    dout_fp8, dout_scales = fast_gather_quantize_and_pack_activation(
                        dout, x_gather_idx
                    )
                    dy1 = blockscaled_fp8_gemm_varlen(
                        dout_fp8, w2_actgrad, expert_frequency_offset,
                        a_scales=dout_scales,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=True,
                    )
                    del dout_fp8, dout_scales
                    # Eagerly release w2 FP8 cache (~37 MiB) — actgrad GEMM done.
                    del w2_fp8, w2_scales
                    # Keep w2 varlen cache — avoids re-quant on next iter.

                    # Step 3: SwiGLU backward
                    if z_fp8 is not None:
                        if ctx._fp8_cfg.fused_swiglu_quant:
                            # Decomposed path (faster than fully-fused):
                            # 1. Dequant z_fp8 → z_bf16  (~0.046ms, BLOCK_ROWS=16)
                            # 2. dSwiGLU + quant + ISA-pack + dz_bf16  (~0.36ms, single kernel)
                            # Total ~0.41ms vs fused 0.47ms (12% faster)
                            from sonicmoe.quack_utils.swiglu_triton import (
                                swiglu_backward_quant_pack_triton,
                            )
                            z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                            dz_fp8, dz_packed_scales, y1s, ds, dz = (
                                swiglu_backward_quant_pack_triton(
                                    dy1, z_bf16, s, return_dz_bf16=True
                                )
                            )
                            del z_bf16
                            _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                        else:
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
                    dw2_base = torch.empty((w2_shape[2], w2_shape[0], w2_shape[1]), dtype=w2_dtype, device=w2_device)
                    dw2 = dw2_base.permute(1, 2, 0)
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
                dw2_base = torch.empty((w2.shape[2], w2.shape[0], w2.shape[1]), dtype=w2.dtype, device=w2.device)
                dw2 = dw2_base.permute(1, 2, 0)
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
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        _reset_stage_memory_probe()
        y1s = None  # may already be freed by fused dgated path
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
        # Inference routing is independent of training: always do a real
        # alignment check instead of trusting _ALIGNMENT_ASSUMED.
        if _fp8_enabled() and _use_fused_blockscaled_gated():
            _vals = _get_cu_seqlens_cpu(expert_frequency_offset)
            aligned = all((_vals[i + 1] - _vals[i]) % 128 == 0 for i in range(len(_vals) - 1))
        else:
            aligned = False
        if _fp8_enabled() and _use_fused_blockscaled_gated() and aligned:
            # Blockscaled FP8 path: reuse the same CUTLASS kernel as training.
            w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)
            x_fp8, x_scales = fast_gather_quantize_and_pack_activation(x, x_gather_idx)
            z, y1 = gemm_gated(
                x_fp8,
                w1_fp8,
                activation="swiglu",
                out_dtype=torch.bfloat16,
                postact_dtype=torch.bfloat16,
                cu_seqlens_m=expert_frequency_offset,
                dynamic_scheduler=False,
                a_scales=x_scales,
                b_scales=w1_scales,
                tuned=False,
            )
            del x_fp8, x_scales
        elif _fp8_enabled():
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
        if _fp8_enabled() and not needs_preact:
            pass  # y1 stays fp8
        elif _fp8_enabled() and needs_preact:
            # Preact path with fp8 enabled: skip dequant round-trip
            if y1.dtype != x.dtype:
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
            if _fp8_enabled() and _use_fused_blockscaled_gated() and aligned:
                # Blockscaled FP8 down-proj: same path as training.
                y1_fp8, y1_scales = quantize_and_pack_activation(y1)
                w2_fp8, w2_scales = precompute_weight_fp8(w2)
                y2 = blockscaled_fp8_gemm_varlen(
                    y1_fp8, w2, expert_frequency_offset,
                    a_scales=y1_scales,
                    w_fp8=w2_fp8, w_scales=w2_scales,
                    out_dtype=torch.bfloat16,
                    assume_aligned=True,
                )
            elif _fp8_enabled():
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
    # Resolve all FP8 flags once at entry — eliminates repeated os.getenv in hot path.
    _refresh_fp8_config()
    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    use_low_precision_postact_buffer = False
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
        cfg = _get_fp8_config()
        if cfg.enabled and cfg.fused_gated and cfg.alignment_assumed and is_using_quack_gemm():
            # Blockscaled FP8 path: y1 was already quantized inside _UpProjection
            # (prequant cache holds fp8+scales).  Skip the adapter's quant→dequant
            # round-trip which costs ~250µs and is redundant.
            pass
        elif cfg.alignment_assumed and is_using_quack_gemm():
            # Aligned non-fused-gated path: cutify's fused SwiGLU+quant expects
            # z in stacked [gate|value] layout.  Both blockscaled_fp8_gemm_varlen
            # and fused_gated produce z compatible with this convention.
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
        elif is_using_quack_gemm():
            # Unaligned QuACK path: gemm_gated outputs z in interleaved layout
            # [g0,v0,g1,v1,…] which is incompatible with cutify's stacked
            # SwiGLU+quant.  The down-projection also falls back to BF16 for
            # unaligned segments, so FP8 quantization adds no benefit.  Skip.
            pass
        else:
            y1, _ = apply_activation_fp8_protocol(
                y1,
                fp8_protocol,
                quack_enabled=False,
                return_scales=False,
                use_ste=not is_inference_mode_enabled,
            )
        _log_stage_memory("forward:fp8-boundary")

    # ── Memory optimization: eagerly release forward transients ──────────
    # z bf16 and y1 bf16 storage was already freed inside _UpProjection
    # via untyped_storage().resize_(0).  Clear w1 FUSED cache (~74 MiB)
    # which is forward-only.  Keep VARLEN cache — entries auto-invalidate
    # via w._version at optimizer step, and hit in no-optimizer benchmarks.
    if _get_fp8_config().enabled and _get_fp8_config().alignment_assumed:
        from ..quack_utils.blockscaled_fp8_gemm import clear_fused_weight_cache
        clear_fused_weight_cache()

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
    _refresh_fp8_config()

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

    # ── Eagerly release forward transients (same as moe_TC_softmax_topk_layer) ──
    # z/y1 bf16 storage freed inside _UpProjection; clear w1 FUSED cache only.
    if _fp8_enabled() and _ALIGNMENT_ASSUMED:
        from ..quack_utils.blockscaled_fp8_gemm import clear_fused_weight_cache
        clear_fused_weight_cache()

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
