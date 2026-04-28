"""SonicMoE ↔ ERNIE integration: MlpNode + PyLayer with fully manual fwd/bwd.

Contains:
  - Weight cache & stacking: ``stack_ernie_w1``, ``stack_ernie_w2``,
    ``invalidate_weight_caches``
  - Native-layout grad accumulation: ``_accumulate_w1``, ``_accumulate_w2``,
    ``flush_native_grads``
  - ``SonicMoEFunc``: ERNIE-core style PyLayer (argsort-based, legacy test path)
  - ``SonicMoEMlpNode``: drop-in MlpNode replacement using DeepEP topk
    metadata (the production path)

``SonicMoEMlpNode`` packages unzip + FP8 FFN + zip into a single callable,
using the DeepEP topk metadata path to eliminate argsort overhead.

Drop-in replacement for ERNIE's MlpNode:
    ``forward(dispatched_hidden_states, tokens_per_expert,
              dispatched_indices, dispatched_probs) → expert_output``

Gradient contract (matching ERNIE FusionMoePyLayer):
  - dx flows back to the caller via Paddle autograd (required by
    FusedDispatch.backward for the reverse A2A communication).
  - dw1/dw2 are accumulated into per-expert ``main_grad`` buffers via
    native-layout accumulators; they do NOT flow through autograd.
  - ds (d/d(score)) flows back to the caller via autograd.
"""

from __future__ import annotations

from typing import Any, Sequence

import paddle
import torch
import triton
import triton.language as tl

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat.deepep_metadata import (
    deepep_topk_to_sonic_metadata,
    invalidate_topk_cache,
)
from sonicmoe.functional import (
    _DownProjection,
    _UpProjection,
    _refresh_fp8_config,
    clear_all_fp8_weight_caches,
    general_routing_router_metadata,
)
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.quack_utils import precompute_weight_fp8_warmup


# ── Weight layout cache ──────────────────────────────────────────────────────

_W_CACHE: dict[tuple, torch.Tensor] = {}


def _cache_key(*tensors: torch.Tensor) -> tuple:
    def _ver(t):
        v = getattr(t, '_inplace_version', None)
        if v is not None:
            return v() if callable(v) else v
        return getattr(t, '_version', 0)
    return tuple((t.data_ptr(), _ver(t)) for t in tensors)


def invalidate_weight_caches() -> None:
    """Call after optimizer step."""
    _W_CACHE.clear()
    clear_all_fp8_weight_caches()
    invalidate_topk_cache()


def stack_ernie_w1(experts: Sequence[Any], H: int, I: int) -> torch.Tensor:
    """Per-expert ``[H, 2I]`` split-half → logical ``[2I, H, E]`` interleaved.

    Physical layout ``[E, 2I, H]`` (contiguous), presented as ``[2I, H, E]``
    via ``.permute(1, 2, 0)`` — matching the stride order QuACK GEMM expects.

    Side effect: lazily allocates a single fp32 ``main_grad`` buffer of shape
    ``[E, H, 2I]`` (split-half, matching per-expert ``weight.shape``) and
    aliases each ``expert.up_gate_proj.weight.main_grad`` to a strided slice of
    it. Per-iter ``_accumulate_w1`` then reduces wgrad into this single buffer
    with two fused cast+add kernels (gate+up) instead of an 8-expert Python loop.
    """
    key = ("w1",) + _cache_key(*(e.up_gate_proj.weight for e in experts))
    if key in _W_CACHE:
        return _W_CACHE[key]
    E = len(experts)
    stacked = torch.stack([e.up_gate_proj.weight for e in experts])  # [E, H, 2I]
    gate, up = stacked[:, :, :I], stacked[:, :, I:]
    physical = torch.stack([gate, up], dim=3) \
                    .reshape(E, H, 2 * I) \
                    .permute(0, 2, 1).contiguous()  # [E, 2I, H]
    w1 = physical.permute(1, 2, 0)  # logical [2I, H, E]
    w1.stop_gradient = False
    _W_CACHE[key] = w1

    # Stacked fp32 main_grad: [E, H, 2I] split-half (cols 0..I-1=gate, I..2I-1=up).
    # Each per-expert weight.main_grad is a strided [H, 2I] view into this buffer,
    # so the optimizer sees the same ERNIE-style per-expert main_grad without copy.
    mg = paddle.zeros([E, H, 2 * I], dtype="float32")
    _W_CACHE[("w1_mg",) + key[1:]] = mg
    for e_idx, exp in enumerate(experts):
        exp.up_gate_proj.weight.main_grad = mg[e_idx]
    return w1


def stack_ernie_w2(experts: Sequence[Any]) -> torch.Tensor:
    """Per-expert ``[I, H]`` → logical ``[H, I, E]``.

    Physical ``[E, H, I]`` (contiguous) via ``.permute(1, 2, 0)``.

    Side effect: aliases each ``expert.down_proj.weight.main_grad`` to a strided
    slice of a single shared ``[E, I, H]`` fp32 buffer (see ``stack_ernie_w1``).
    """
    key = ("w2",) + _cache_key(*(e.down_proj.weight for e in experts))
    if key in _W_CACHE:
        return _W_CACHE[key]
    physical = torch.stack([e.down_proj.weight for e in experts]) \
                    .permute(0, 2, 1).contiguous()  # [E, H, I]
    w2 = physical.permute(1, 2, 0)  # logical [H, I, E]
    w2.stop_gradient = False
    _W_CACHE[key] = w2

    E = len(experts)
    I = w2.shape[1]
    H = w2.shape[0]
    mg = paddle.zeros([E, I, H], dtype="float32")
    _W_CACHE[("w2_mg",) + key[1:]] = mg
    for e_idx, exp in enumerate(experts):
        exp.down_proj.weight.main_grad = mg[e_idx]
    return w2


def _w1_mg_stacked(experts: Sequence[Any]) -> torch.Tensor:
    return _W_CACHE[("w1_mg",) + _cache_key(*(e.up_gate_proj.weight for e in experts))]


def _w2_mg_stacked(experts: Sequence[Any]) -> torch.Tensor:
    return _W_CACHE[("w2_mg",) + _cache_key(*(e.down_proj.weight for e in experts))]


# ── Input format conversion ──────────────────────────────────────────────────

def prepare_sonic_inputs(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    x: torch.Tensor,
    n_experts: int,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """ERNIE ``[T, K]`` → SonicMoE flat sorted format.

    Returns ``(x_padded, token_indices, expert_indices, router_scores, T_orig)``.
    """
    T = x.shape[0]
    tok_ids = torch.arange(T, dtype=torch.int32, device=x.device) \
                   .unsqueeze(1).expand_as(dispatched_indices)
    valid = dispatched_indices >= 0
    tok_flat = tok_ids[valid]
    exp_flat = dispatched_indices[valid].int()
    scr_flat = dispatched_probs[valid].float()

    exp_counts = torch.bincount(exp_flat.long(), minlength=n_experts).int()
    pad_counts = (block - exp_counts % block) % block
    max_pad = int(pad_counts.max().item())

    if max_pad == 0:
        return x, tok_flat, exp_flat, scr_flat, T

    row_ids = torch.arange(max_pad, dtype=torch.int32, device=x.device) \
                   .unsqueeze(1).expand(max_pad, n_experts)
    exp_ids = torch.arange(n_experts, dtype=torch.int32, device=x.device) \
                   .unsqueeze(0).expand(max_pad, n_experts)
    active = row_ids < pad_counts.unsqueeze(0)

    token_indices = torch.cat([tok_flat, (T + row_ids[active]).int()])
    expert_indices = torch.cat([exp_flat, exp_ids[active].int()])
    router_scores = torch.cat([scr_flat,
                               torch.zeros(active.sum(), dtype=torch.float32,
                                           device=x.device)])
    x_padded = torch.cat([x, torch.zeros(max_pad, x.shape[1],
                                         dtype=x.dtype, device=x.device)])
    return x_padded, token_indices, expert_indices, router_scores, T


# ── Weight-grad → main_grad ─────────────────────────────────────────────────

# ── Native-layout accumulators (zero-transpose path, per-layer) ──────────────
# We accumulate dw1/dw2 in SonicMoE's native physical layout
# (W1: [E, 2I, H], W2: [E, H, I]) and defer the layout conversion to ERNIE's
# per-expert layout (W1: [E, H, 2I] split-half, W2: [E, I, H]) until
# ``flush_native_grads()`` is called at optimizer-step time. This eliminates
# the per-iter TilingSwapDim/contiguous kernels from the profiled path.
#
# Critically, the buffer is the *same storage* as the per-expert
# ``weight.main_grad`` allocated by ``stack_ernie_w{1,2}`` (a single
# contiguous fp32 tensor sliced into per-expert views). We just re-interpret
# that storage as the native shape during accumulation, then transpose it
# back into the ERNIE shape in-place at flush time. Benefits:
#   1. Per-layer binding "for free" — main_grad is already per-weight, so
#      models with multiple MoE layers each get their own buffer (the old
#      single-global ``_NATIVE_W1_GRAD`` silently lost grads when a second
#      layer's backward overwrote the global).
#   2. Zero extra permanent allocation (saves E*H*2I*4 + E*I*H*4 fp32 bytes
#      per layer — ~288 MB for the production E=8/H=3072/I=1536 shape).
#
# Layer registry: each backward that touched native layout records its
# ``experts`` list here; ``flush_native_grads`` iterates and converts every
# pending layer. List (not set) to preserve order; identity comparison
# because ``experts`` is typically a Paddle LayerList that isn't hashable.
#
# Caveat for callers: between accumulation and ``flush_native_grads()``,
# ``param.main_grad`` contains *native-layout* data (scrambled relative to
# ERNIE's expected [H, 2I] / [I, H] view). Always call ``node.step()`` (or
# ``flush_native_grads()`` directly) before reading or optimizing main_grad.

_PENDING_FLUSH_LAYERS: list = []  # list of `experts` lists awaiting flush

# ── Deprecated compatibility shims ────────────────────────────────────────────
# The native-layout buffer used to be a single global tensor; it's now backed
# by per-expert main_grad storage (see _w1_native_view / _w2_native_view).
# These aliases remain so tests/benchmarks that "reset" the globals to None
# (a no-op cleanup) don't break. Nothing in the live path reads them.
_NATIVE_W1_GRAD: torch.Tensor | None = None
_NATIVE_W2_GRAD: torch.Tensor | None = None
_NATIVE_GRAD_EXPERTS: list | None = None
_NATIVE_GRAD_I: int = 0


def _ensure_native_grads(*_args, **_kwargs) -> None:
    """Deprecated no-op: native buffers now live in per-expert main_grad."""
    return


def _w1_native_view(experts) -> torch.Tensor:
    """View per-expert main_grad storage as native [E, 2I, H] contiguous fp32."""
    mg = _w1_mg_stacked(experts)  # [E, H, 2I] contig fp32 (aliases per-expert main_grad)
    E, H, two_I = mg.shape
    return mg.view(E, two_I, H)


def _w2_native_view(experts) -> torch.Tensor:
    """View per-expert main_grad storage as native [E, H, I] contiguous fp32."""
    mg = _w2_mg_stacked(experts)  # [E, I, H] contig fp32
    E, I, H = mg.shape
    return mg.view(E, H, I)


def _mark_pending_flush(experts) -> None:
    for e in _PENDING_FLUSH_LAYERS:
        if e is experts:
            return
    _PENDING_FLUSH_LAYERS.append(experts)


def _accumulate_w1(dw1: torch.Tensor, experts: list, I: int) -> None:
    """BF16-fallback accumulator: dw1 [2I, H, E] → native [E, 2I, H] storage.

    Only used when the FP8 wgrad path returns a non-None dw1 (i.e., the
    BF16 fallback). Cost: 1 TilingSwapDim + 1 MultiPrecisionAdd.
    """
    native = _w1_native_view(experts)
    native.add_(dw1.permute(2, 0, 1).contiguous())
    _mark_pending_flush(experts)


def _accumulate_w2(dw2: torch.Tensor, experts: list) -> None:
    """BF16-fallback accumulator: dw2 [H, I, E] → native [E, H, I] storage."""
    native = _w2_native_view(experts)
    native.add_(dw2.permute(2, 0, 1).contiguous())
    _mark_pending_flush(experts)


def flush_native_grads() -> None:
    """Convert every pending layer's native-layout main_grad → ERNIE layout in-place.

    Call this at optimizer-step time (typically via ``node.step()``).
    For each pending layer:
      W1: storage [E, 2I, H] interleaved (gate0,up0,gate1,up1,...)
          → ERNIE [E, H, 2I] split-half (gates then ups)
      W2: storage [E, H, I] → ERNIE [E, I, H]
    Conversion is in-place via a single contiguous() scratch (one alloc per
    flush, freed after copy_), then the pending list is cleared.
    """
    global _PENDING_FLUSH_LAYERS
    if not _PENDING_FLUSH_LAYERS:
        return

    pending, _PENDING_FLUSH_LAYERS = _PENDING_FLUSH_LAYERS, []
    for experts in pending:
        # ── W1 ──────────────────────────────────────────────────────────────
        mg1 = _w1_mg_stacked(experts)  # alias view [E, H, 2I]; storage holds [E, 2I, H] data
        E, H, two_I = mg1.shape
        I = two_I // 2
        native = mg1.view(E, two_I, H)  # correct view of the native data
        # native [E, 2I, H] is interleaved: 2I groups as (I, 2) → split-half [E, H, 2, I]
        rhs = native.view(E, I, 2, H).permute(0, 3, 2, 1).contiguous()
        mg1.view(E, H, 2, I).copy_(rhs)

        # ── W2 ──────────────────────────────────────────────────────────────
        mg2 = _w2_mg_stacked(experts)  # alias view [E, I, H]; storage holds [E, H, I] data
        E2, I2, H2 = mg2.shape
        native2 = mg2.view(E2, H2, I2)
        rhs2 = native2.permute(0, 2, 1).contiguous()  # [E, I, H]
        mg2.copy_(rhs2)


# ── The legacy PyLayer (argsort-based) ────────────────────────────────────────

class _FakeCtx:
    """Minimal ctx stub for calling _UpProjection / _DownProjection static methods.

    Supports ``save_for_backward``, ``saved_tensor``, ``mark_non_differentiable``,
    ``set_materialize_grads``, and arbitrary attribute access (for ctx.T, ctx._fp8_cfg, etc.).
    """
    def save_for_backward(self, *args): self._saved = args
    def saved_tensor(self): return self._saved
    def mark_non_differentiable(self, *args): pass
    def set_materialize_grads(self, value): pass


class SonicMoEFunc(paddle.autograd.PyLayer):
    """ERNIE-core style PyLayer with fully manual forward/backward.

    Calls ``_UpProjection.forward`` / ``_DownProjection.forward`` directly
    (as plain functions via FakeCtx), saves all state manually, and calls
    their ``.backward`` counterparts in our own backward — no inner autograd.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,       # [T_padded, H] bf16  (grad)
        router_scores: torch.Tensor,        # [TK] float32        (grad)
        token_indices: torch.Tensor,        # [TK] int32          (no grad)
        expert_indices: torch.Tensor,       # [TK] int32          (no grad)
        w1: torch.Tensor,                   # [2I, H, E] bf16     (no grad — wgrad via main_grad)
        w2: torch.Tensor,                   # [H, I, E] bf16      (no grad — wgrad via main_grad)
        experts: Sequence[Any] = None,
        n_experts: int = 0,
        T_orig: int = 0,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ) -> torch.Tensor:
        _refresh_fp8_config()
        T = hidden_states.shape[0]
        E = n_experts

        # ── Routing metadata ─────────────────────────────────────────────
        (
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = general_routing_router_metadata(
            router_scores, token_indices, expert_indices, T, E,
        )
        TK = s_scatter_idx.shape[0]

        # ── UpProjection forward (via FakeCtx) ───────────────────────────
        up_ctx = _FakeCtx()
        with enable_fp8(True):
            y1, z = _UpProjection.forward(
                up_ctx,
                hidden_states, w1, None,        # x, w1, b1
                expert_frequency_offset, TK, None, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                False,                          # is_inference_mode_enabled
                False,                          # use_low_precision_postact_buffer
            )

        # ── DownProjection forward (via FakeCtx) ─────────────────────────
        down_ctx = _FakeCtx()
        with enable_fp8(True):
            out = _DownProjection.forward(
                down_ctx,
                y1, z, w2, None,                # y1, z, w2, b2
                router_scores, expert_indices, expert_frequency_offset,
                T, None, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                None,                           # fp8_protocol
            )

        # ── Save everything for backward ─────────────────────────────────
        # Outer PyLayer's save_for_backward only takes tensors and detaches
        # them, so we store the ctx objects as Python attributes instead.
        ctx._up_ctx = up_ctx
        ctx._down_ctx = down_ctx
        ctx._experts = experts
        ctx._I = w1.shape[0] // 2
        ctx._T_orig = T_orig
        return out[:T_orig]

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        up_ctx = ctx._up_ctx
        down_ctx = ctx._down_ctx
        T_orig = ctx._T_orig
        T_padded = up_ctx.T if hasattr(up_ctx, 'T') else output_grad.shape[0]

        # Pad output_grad back to T_padded if needed
        if T_padded > T_orig:
            output_grad = torch.cat([
                output_grad,
                torch.zeros(T_padded - T_orig, output_grad.shape[1],
                            dtype=output_grad.dtype, device=output_grad.device),
            ])

        # ── DownProjection backward ──────────────────────────────────────
        # Returns: (dy1=None, dz, dw2, db2?, ds, None_selected_experts,
        #           None_efo, None x N metadata)
        down_grads = _DownProjection.backward(down_ctx, output_grad)
        # dz is at index 1, dw2 at index 2
        dz = down_grads[1]    # [TK, 2I] bf16
        dw2 = down_grads[2]   # [H, I, E] w2_dtype

        # ds: at index 3 (b2=None → no db2) or 4 (b2 present → db2 at 3)
        ds_idx = 4 if getattr(down_ctx, '_has_b2', False) else 3
        ds = down_grads[ds_idx]

        # ── UpProjection backward ────────────────────────────────────────
        # Input to backward: (dy1=None, dz)
        up_grads = _UpProjection.backward(up_ctx, None, dz)
        # Returns: (dx, dw1, db1?, None x N metadata)
        dx_full = up_grads[0]   # [T_padded, H] bf16
        dw1 = up_grads[1]       # [2I, H, E] w1_dtype

        # ── Weight grads → per-expert main_grad ──────────────────────────
        if dw1 is not None:
            _accumulate_w1(dw1, ctx._experts, ctx._I)
        if dw2 is not None:
            _accumulate_w2(dw2, ctx._experts)

        # ── Return grads for tensor inputs ───────────────────────────────
        # Tensor inputs: hidden_states, router_scores, token_indices,
        #                expert_indices, w1, w2
        dx = dx_full[:T_orig] if dx_full is not None else None

        return dx, ds, None, None, None, None


# ── Differentiable router-scores reconstruction ──────────────────────────────

@triton.jit
def _build_score_src_idx_kernel(
    INDICES_ptr,  # [N_recv, topk] int32 — dispatched_indices (read-only)
    NAEPT_ptr,    # [N_recv+1] int32
    OUT_ptr,      # [TK] int64
    WORK_ptr,     # [N_recv * TOPK] int32 — scratch (same size as INDICES)
    TOPK: tl.constexpr,
):
    """Build score_src_idx: per-token ascending sort of expert IDs.

    Pure scalar ops — compatible with Triton 3.5.0 + SM103a (no tl.min).
    """
    t = tl.program_id(0)
    start = tl.load(NAEPT_ptr + t)
    end = tl.load(NAEPT_ptr + t + 1)
    n_valid = end - start
    if n_valid == 0:
        return

    base = t * TOPK
    # Copy to scratch; replace invalid (-1) with 0x7FFF
    for k in tl.static_range(TOPK):
        e = tl.load(INDICES_ptr + base + k)
        tl.store(WORK_ptr + base + k, tl.where(e >= 0, e, 0x7FFF))

    # Selection sort: for each output position, find argmin in scratch
    for j in range(TOPK):
        if j < n_valid:
            best_e: tl.int32 = 0x7FFF + 1
            best_c: tl.int32 = 0
            for k in tl.static_range(TOPK):
                ek = tl.load(WORK_ptr + base + k)
                better = (ek < best_e) | ((ek == best_e) & (k < best_c))
                best_e = tl.where(better, ek, best_e)
                best_c = tl.where(better, k, best_c)
            tl.store(OUT_ptr + start + j, (t * TOPK + best_c).to(tl.int64))
            tl.store(WORK_ptr + base + best_c, 0x7FFF + 1)


def _differentiable_router_scores(
    dispatched_probs: torch.Tensor,    # [N_recv, topk] float32
    dispatched_indices: torch.Tensor,  # [N_recv, topk] int32, -1 = masked
    naept: torch.Tensor,               # [N_recv+1] int32 — prefix-sum of valid counts
    TK: int,                           # number of real (non-padding) entries
    TK_padded: int,                    # TK + padding for 128-alignment
    E: int,                            # number of experts
    score_src_idx: torch.Tensor | None = None,  # [TK] int32 from CUDA kernel
) -> torch.Tensor:
    """Reconstruct router_scores from dispatched_probs with autograd connection.

    The CUDA metadata kernel produces topk_scores as a non-differentiable data
    copy.  This function builds the same [TK_padded] tensor using differentiable
    fancy-indexing into dispatched_probs, so that ds gradients flow back through
    Paddle autograd to the caller's gate / dispatched_probs tensor.

    Within each token, entries are ordered by ascending expert ID, matching the
    CUDA kernel's rank-based ordering in ``scatter_and_fixup_kernel``.

    Zero CPU-GPU synchronization.

    Fast path: when ``score_src_idx`` is provided (CUDA metadata path) it is
    used directly — bit-exact with the standalone Triton _build_score_src_idx
    kernel and saves a separate launch.  Falls back to the Triton kernel only
    when CUDA path is unavailable (Python metadata fallback).
    """
    N_recv, topk = dispatched_indices.shape
    device = dispatched_indices.device

    if dispatched_indices.stride(1) != 1:
        raise ValueError("dispatched_indices must be contiguous in last dim")
    if "int32" not in str(dispatched_indices.dtype):
        raise ValueError(f"dispatched_indices: expected int32, got {dispatched_indices.dtype}")
    if naept.stride(0) != 1:
        raise ValueError("naept must be contiguous 1D")
    if naept.shape[0] != N_recv + 1:
        raise ValueError(f"naept: expected shape ({N_recv+1},), got {naept.shape}")

    if TK == 0:
        return torch.zeros(TK_padded, dtype=dispatched_probs.dtype, device=device)

    if score_src_idx is not None:
        # Provided by scatter_and_fixup_kernel — int32, already in token-major
        # + ascending-expert order matching the metadata path.
        if score_src_idx.shape[0] != TK:
            raise ValueError(
                f"score_src_idx: expected shape ({TK},), got {score_src_idx.shape}"
            )
        gather_idx = score_src_idx.to(torch.int64)
    else:
        # Build score_src_idx: [TK] int64 flat indices into dispatched_probs.reshape(-1)
        # Uses Triton kernel — no boolean indexing, no dynamic shapes, no D2H sync.
        built = torch.empty(TK, dtype=torch.int64, device=device)
        work = torch.empty_like(dispatched_indices)  # scratch buffer
        _build_score_src_idx_kernel[(N_recv,)](
            dispatched_indices, naept, built, work,
            TOPK=triton.next_power_of_2(topk),
        )
        gather_idx = built

    # Differentiable gather from dispatched_probs
    gathered = dispatched_probs.reshape(-1)[gather_idx]  # [TK]

    if TK_padded > TK:
        padding = torch.zeros(
            TK_padded - TK, dtype=gathered.dtype, device=device,
        )
        return torch.cat([gathered, padding])
    return gathered


# ── SonicMoEMlpNode (production path using DeepEP metadata) ──────────────────

class SonicMoEMlpNode:
    """Drop-in replacement for ERNIE's MlpNode (topk DeepEP dispatch path).

    Packages unzip + FP8 FFN + zip into a single callable.
    Uses DeepEP's topk dispatch metadata for zero-sync metadata conversion.

    Gradient contract:
      - dx flows back through Paddle autograd to the caller (no detach).
      - dw1/dw2 accumulate into per-expert main_grad via native-layout buffers.
      - ds (d/d(score)) flows back through Paddle autograd.

    Performance optimizations (zero migration overhead):
      - Route-level padding: x is passed directly (no padding, no cat, no copy).
      - Native-layout grad accumulation (1 add_ kernel per weight, no transpose).

    Parameters
    ----------
    experts : list
        Per-expert modules, each with ``up_gate_proj.weight [H, 2I]``
        and ``down_proj.weight [I, H]``.
    n_experts : int
        Number of local experts (E).
    hidden_size : int
        Model hidden dimension (H).
    intermediate_size : int
        FFN intermediate dimension (I). ``up_gate_proj`` has 2*I columns.
    activation_type : ActivationType
        Activation function (default SWIGLU).
    stream_id : int
        CUDA stream index for FP8 ops (default 0).
    """

    def __init__(
        self,
        experts: Sequence[Any],
        n_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ):
        self._experts = list(experts)
        self._E = n_experts
        self._H = hidden_size
        self._I = intermediate_size
        self._activation_type = activation_type
        self._stream_id = stream_id
        self._warmed_for_step = False

    def prequantize_weights(self) -> None:
        """Fused single-pass FP8 prequantize for w1/w2 (all 4 layouts).

        Reads each BF16 weight ONCE and writes both transposed FP8 layouts +
        ISA-packed scales in a single Triton kernel per weight.  ~3x faster
        than letting the four ``precompute_weight_fp8_*`` helpers fire lazily
        inside the first microbatch's forward.

        Idempotent within a step: the second call is a cheap cache lookup.
        ``step()`` clears the flag so the next step re-quantizes the freshly
        updated weights.
        """
        if self._warmed_for_step:
            return
        w1 = stack_ernie_w1(self._experts, self._H, self._I)  # [2I, H, E]
        w2 = stack_ernie_w2(self._experts)                    # [H, I, E]
        precompute_weight_fp8_warmup(w1, w2)
        self._warmed_for_step = True

    def warmup(self, total_K_list: list[int] | None = None, max_workers: int = 0):
        """Pre-compile all JIT kernels. Call once after model construction.

        After compile_key dynamic-dim fix, a single warmup covers all seqlens.
        """
        from sonicmoe.jit_warmup import warmup_jit
        warmup_jit(
            self._E, self._H, self._I,
            device="cuda",
            fp8=True,
            total_K_list=total_K_list,
            max_workers=max_workers,
        )

    def forward(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
        dispatched_indices: torch.Tensor | None = None,
        dispatched_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FP8 expert FFN on DeepEP-dispatched tokens.

        Parameters
        ----------
        dispatched_hidden_states : Tensor [N_recv, H] bf16
            Received tokens from DeepEP dispatch (topk layout).
        tokens_per_expert : list[int] or Tensor [E]
            Per-expert token counts from DeepEP ``buffer.dispatch()``.
        dispatched_indices : Tensor [N_recv, topk] int32
            Local expert indices from DeepEP dispatch. -1 = masked.
        dispatched_probs : Tensor [N_recv, topk] float32
            Routing probabilities from DeepEP dispatch.

        Returns
        -------
        Tensor [N_recv, H] bf16
            Expert output, same token ordering as input.
        """
        x = dispatched_hidden_states
        T = x.shape[0]
        E = self._E
        H = self._H
        I = self._I

        # Topk path: real DeepEP dispatch with multi-expert routing.
        # Identity layout (K=1, pre-sorted) was removed — it had an
        # unfixable dx bug due to expert-sorted ↔ token-order mismatch
        # when total_pad_rows > 0.  All production callers use topk.
        assert dispatched_indices is not None, (
            "dispatched_indices is required (identity layout path removed)"
        )
        assert dispatched_probs is not None, (
            "dispatched_probs required when dispatched_indices is given"
        )
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores_data,
            TK_padded,
            total_pad_rows,
            N_recv,
            score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            dispatched_indices, dispatched_probs,
            tokens_per_expert, E, device=x.device,
        )

        # Build router_scores differentiably from dispatched_probs so that
        # ds gradients flow back to the caller (e.g. DeepEP's gate).
        # deepep_topk_to_sonic_metadata produces a non-differentiable copy;
        # we reconstruct the same values using autograd-tracked indexing.
        # When the CUDA metadata kernel is available, score_src_idx is
        # produced as a free side-output of scatter_and_fixup_kernel and we
        # skip the standalone Triton _build_score_src_idx_kernel entirely
        # (saves ~252 µs/iter at user shape).
        router_scores = _differentiable_router_scores(
            dispatched_probs, dispatched_indices,
            num_activated_expert_per_token_offset,
            TK_padded - total_pad_rows, TK_padded, E,
            score_src_idx=score_src_idx,
        )
        # T_down = N_recv for the topk path: _router_forward outputs [N_recv, H]
        T_down = N_recv

        # 2. NO x-padding needed — route-level padding handles alignment.
        #    Padding rows in x_gather_idx point to row 0 with score=0.

        # 3. Stack weights (cached) + fused FP8 prequantize on first microbatch.
        self.prequantize_weights()
        w1 = stack_ernie_w1(self._experts, H, I)
        w2 = stack_ernie_w2(self._experts)

        # 4. Prepare tensor inputs for PyLayer.
        #    x passes through WITHOUT detach — dx must flow back to the caller
        #    so that FusedDispatch.backward can do the reverse A2A with it.
        #    router_scores is differentiably derived from dispatched_probs,
        #    so ds gradients propagate back to the gate automatically.
        x.stop_gradient = False
        # router_scores is a non-leaf tensor (computed from dispatched_probs),
        # Paddle autograd tracks it automatically; explicit flag is defensive.
        router_scores.stop_gradient = False

        # Non-differentiable inputs (integer metadata + stacked weights whose
        # grads go through main_grad, not autograd)
        x_gather_idx.stop_gradient = True
        s_scatter_idx.stop_gradient = True
        w1.stop_gradient = True
        w2.stop_gradient = True

        # 5. Run _SonicMoEDeepEPFunc (FP8 fwd + manual bwd with main_grad)
        _SonicMoEDeepEPFunc._topk = dispatched_indices.shape[1]
        output = _SonicMoEDeepEPFunc.apply(
            x,
            router_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            w1, w2,
            self._experts,
            E, N_recv, T_down, TK_padded,
            self._activation_type,
            self._stream_id,
        )

        return output

    def step(self):
        """Call after optimizer.step(). Flushes wgrad and invalidates all caches.

        Training loop contract:
            for microbatch in microbatches:
                out = node(x, tpe, indices, probs)
                out.backward(grad)
            optimizer.step()
            node.step()         # ← here
            optimizer.zero_grad()
        """
        flush_native_grads()
        invalidate_weight_caches()
        self._warmed_for_step = False

    def __call__(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
        dispatched_indices: torch.Tensor | None = None,
        dispatched_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(
            dispatched_hidden_states, tokens_per_expert,
            dispatched_indices, dispatched_probs,
        )


# ── Internal PyLayer (topk DeepEP metadata, production path) ─────────────────


class _SonicMoEDeepEPFunc(paddle.autograd.PyLayer):
    """PyLayer for the topk DeepEP dispatch path (production path).

    Uses route-level padding (frontier design): x is NOT padded. Padding rows
    gather from row 0 with score=0, contributing nothing to output or grads.

    T_down = N_recv: _DownProjection outputs [N_recv, H] directly, no slicing.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,       # [T, H] bf16  (grad)
        router_scores: torch.Tensor,        # [TK_padded] float32
        expert_frequency_offset: torch.Tensor,  # [E+1] int32
        x_gather_idx: torch.Tensor,         # [TK_padded] int32
        s_scatter_idx: torch.Tensor,        # [TK_padded] int32
        s_reverse_scatter_idx: torch.Tensor, # [TK] int32
        num_activated_expert_per_token_offset: torch.Tensor,  # [N_recv+1]
        w1: torch.Tensor,                   # [2I, H, E] bf16
        w2: torch.Tensor,                   # [H, I, E] bf16
        experts: Sequence[Any] = None,
        n_experts: int = 0,
        N_recv: int = 0,
        T_down: int = 0,
        TK_padded: int = 0,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ) -> torch.Tensor:
        _refresh_fp8_config()
        topk = getattr(_SonicMoEDeepEPFunc, '_topk', n_experts)

        # ── UpProjection forward (via FakeCtx) ───────────────────────────
        up_ctx = _FakeCtx()
        with enable_fp8(True):
            y1, z = _UpProjection.forward(
                up_ctx,
                hidden_states, w1, None,        # x, w1, b1
                expert_frequency_offset, TK_padded, None, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                False,                          # is_inference_mode_enabled
                False,                          # use_low_precision_postact_buffer
            )

        # ── DownProjection forward (via FakeCtx) ─────────────────────────
        # Topk path: T_down = N_recv, output is [N_recv, H] directly.
        down_ctx = _FakeCtx()
        with enable_fp8(True):
            out = _DownProjection.forward(
                down_ctx,
                y1, z, w2, None,                # y1, z, w2, b2
                router_scores, s_scatter_idx, expert_frequency_offset,
                T_down, topk, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                None,                           # fp8_protocol
            )

        # ── Save everything for backward ─────────────────────────────────
        ctx._up_ctx = up_ctx
        ctx._down_ctx = down_ctx
        ctx._experts = experts
        ctx._I = w1.shape[0] // 2

        return out

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        up_ctx = ctx._up_ctx
        down_ctx = ctx._down_ctx

        # ── DownProjection backward ──────────────────────────────────────
        I = ctx._I
        # Native-layout views into per-expert main_grad storage (per-layer,
        # zero extra alloc). Caller MUST flush_native_grads() before reading
        # main_grad — see module-level comment.
        w1_native = _w1_native_view(ctx._experts)  # [E, 2I, H] fp32
        w2_native = _w2_native_view(ctx._experts)  # [E, H, I] fp32
        _mark_pending_flush(ctx._experts)

        down_ctx._wgrad_w2_accumulator = w2_native
        down_grads = _DownProjection.backward(down_ctx, output_grad)
        dz = down_grads[1]
        dw2 = down_grads[2]

        # ds: deterministic index based on _has_b2 (b2 is always None in this path,
        # so ds is always at index 3.  Previous code used a fragile float32 heuristic.)
        ds_idx = 4 if getattr(down_ctx, '_has_b2', False) else 3
        ds = down_grads[ds_idx]

        # ── UpProjection backward ────────────────────────────────────────
        up_ctx._wgrad_w1_accumulator = w1_native
        up_grads = _UpProjection.backward(up_ctx, None, dz)
        dx = up_grads[0]   # [T_orig, H] — matches x input shape
        dw1 = up_grads[1]

        # ── Weight grads → per-expert main_grad ──────────────────────────
        # FP8 path: CUTLASS accumulates directly into the native-layout view
        # of main_grad via the _wgrad_accumulator, returning dw1=dw2=None.
        # If non-None, we're on the BF16 fallback — accumulate with per-iter
        # transpose into the same native view.
        if dw1 is not None:
            _accumulate_w1(dw1, ctx._experts, ctx._I)
        if dw2 is not None:
            _accumulate_w2(dw2, ctx._experts)

        # ── Return grads for tensor inputs ───────────────────────────────
        # Tensor inputs: hidden_states, router_scores, efo, x_gather, s_scatter,
        #                s_reverse_scatter, naept_offset, w1, w2
        return dx, ds, None, None, None, None, None, None, None
