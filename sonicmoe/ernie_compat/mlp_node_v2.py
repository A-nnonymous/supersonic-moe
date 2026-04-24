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


# ── Weight layout cache ──────────────────────────────────────────────────────

_W_CACHE: dict[tuple, torch.Tensor] = {}


def _cache_key(*tensors: torch.Tensor) -> tuple:
    return tuple((t.data_ptr(), t._inplace_version) for t in tensors)


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

# ── Native-layout accumulators (zero-transpose path) ─────────────────────────
# Instead of transposing wgrad every iter to match ERNIE's per-expert layout,
# we accumulate in SonicMoE's native layout (dw1: [2I,H,E], dw2: [H,I,E])
# and defer the transpose to optimizer-step time via flush_native_grads().
# This eliminates 2 contiguous() kernels per iter from the profiled path.

_NATIVE_W1_GRAD: torch.Tensor | None = None  # [2I, H, E] fp32
_NATIVE_W2_GRAD: torch.Tensor | None = None  # [H, I, E] fp32
_NATIVE_GRAD_EXPERTS: list | None = None  # reference to experts for flush
_NATIVE_GRAD_I: int = 0


def _ensure_native_grads(experts, I: int, w1_shape: tuple, w2_shape: tuple, device):
    """Lazily allocate fp32 grad accumulators in physical layout [E, 2I, H] / [E, H, I].

    dw1 from backward: logical [2I, H, E], physical [E, 2I, H] (non-contiguous view).
    dw2 from backward: logical [H, I, E], physical [E, H, I] (non-contiguous view).

    We store in physical layout [E, 2I, H] / [E, H, I] (contiguous).
    accumulate does permute(2,0,1) on dw to get [E, ...] then add_.
    The permute+contiguous is 1 TilingSwapDim kernel — this is the inherent cost
    of main_grad accumulation (frontier avoids it by not accumulating at all).
    """
    global _NATIVE_W1_GRAD, _NATIVE_W2_GRAD, _NATIVE_GRAD_EXPERTS, _NATIVE_GRAD_I
    if _NATIVE_W1_GRAD is not None and _NATIVE_W2_GRAD is not None \
       and _NATIVE_GRAD_EXPERTS is experts:
        return

    two_I = w1_shape[0]; H = w1_shape[1]; E = w1_shape[2]
    if _NATIVE_W1_GRAD is None or _NATIVE_GRAD_EXPERTS is not experts:
        _NATIVE_W1_GRAD = torch.zeros(E, two_I, H, dtype=torch.float32, device=device)

    H2 = w2_shape[0]; I2 = w2_shape[1]; E2 = w2_shape[2]
    if _NATIVE_W2_GRAD is None or _NATIVE_GRAD_EXPERTS is not experts:
        _NATIVE_W2_GRAD = torch.zeros(E2, H2, I2, dtype=torch.float32, device=device)

    _NATIVE_GRAD_EXPERTS = experts
    _NATIVE_GRAD_I = I


def _accumulate_w1(dw1: torch.Tensor, experts: list, I: int):
    """Accumulate dw1 [2I, H, E] → physical [E, 2I, H] buffer.

    Cost: 1 TilingSwapDim (contiguous copy to [E,2I,H]) + 1 MultiPrecisionAdd.
    This is the inherent cost of main_grad accumulation in ERNIE training.
    The frontier benchmark avoids this entirely via zero_grad(set_to_none=True).
    """
    global _NATIVE_W1_GRAD
    _ensure_native_grads(
        experts, I, dw1.shape, (dw1.shape[1], I, dw1.shape[2]),
        dw1.device,
    )
    # dw1 logical [2I,H,E] → permute(2,0,1) → [E,2I,H] non-contiguous
    # .contiguous() materializes it (TilingSwapDim kernel) → then add_ (MultiPrecisionAdd)
    _NATIVE_W1_GRAD.add_(dw1.permute(2, 0, 1).contiguous())


def _accumulate_w2(dw2: torch.Tensor, experts: list):
    """Accumulate dw2 [H, I, E] → physical [E, H, I] buffer.

    Same cost structure as _accumulate_w1.
    """
    global _NATIVE_W2_GRAD
    I = dw2.shape[1]
    _ensure_native_grads(
        experts, I, (2 * I, dw2.shape[0], dw2.shape[2]), dw2.shape,
        dw2.device,
    )
    _NATIVE_W2_GRAD.add_(dw2.permute(2, 0, 1).contiguous())


def flush_native_grads():
    """Transpose native-layout grad accumulators into per-expert main_grad.

    Call this at optimizer-step time (NOT per-iter). Performs the layout
    conversion that was deferred from per-iter _accumulate_w1/_accumulate_w2.

    After flushing, the native buffers are zeroed for the next accumulation window.
    """
    global _NATIVE_W1_GRAD, _NATIVE_W2_GRAD, _NATIVE_GRAD_EXPERTS, _NATIVE_GRAD_I
    if _NATIVE_GRAD_EXPERTS is None:
        return

    experts = _NATIVE_GRAD_EXPERTS
    I = _NATIVE_GRAD_I

    # ── W1: physical [E, 2I, H] → per-expert main_grad [E, H, 2I] split-half ──
    if _NATIVE_W1_GRAD is not None:
        mg = _w1_mg_stacked(experts)  # [E, H, 2I] contig fp32
        E_val = mg.shape[0]
        H = mg.shape[1]
        # Buffer is already [E, 2I, H] contiguous
        # Interleave→split-half: [E,I,2,H] → [E,H,2,I] contiguous
        rhs = _NATIVE_W1_GRAD.view(E_val, I, 2, H).permute(0, 3, 2, 1).contiguous()
        mg.view(E_val, H, 2, I).add_(rhs)
        _NATIVE_W1_GRAD.zero_()

    # ── W2: physical [E, H, I] → per-expert main_grad [E, I, H] ───────────────
    if _NATIVE_W2_GRAD is not None:
        mg = _w2_mg_stacked(experts)  # [E, I, H] contig fp32
        # Buffer is [E, H, I], need [E, I, H] → permute(0, 2, 1)
        rhs = _NATIVE_W2_GRAD.permute(0, 2, 1).contiguous()
        mg.add_(rhs)
        _NATIVE_W2_GRAD.zero_()


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

def _differentiable_router_scores(
    dispatched_probs: torch.Tensor,    # [N_recv, topk] float32
    dispatched_indices: torch.Tensor,  # [N_recv, topk] int32, -1 = masked
    TK: int,                           # number of real (non-padding) entries
    TK_padded: int,                    # TK + padding for 128-alignment
    E: int,                            # number of experts
) -> torch.Tensor:
    """Reconstruct router_scores from dispatched_probs with autograd connection.

    The CUDA metadata kernel produces topk_scores as a non-differentiable data
    copy.  This function builds the same [TK_padded] tensor using differentiable
    fancy-indexing into dispatched_probs, so that ds gradients flow back through
    Paddle autograd to the caller's gate / dispatched_probs tensor.

    Within each token, entries are ordered by ascending expert ID, matching the
    CUDA kernel's rank-based ordering in ``scatter_and_fixup_kernel``.
    """
    N_recv, topk = dispatched_indices.shape
    device = dispatched_indices.device

    # Sort each token's expert IDs ascending.  Invalid slots (-1) sort to end
    # by replacing with E (larger than any valid expert 0..E-1).
    masked = dispatched_indices.int().clone()
    masked[masked < 0] = E
    sort_perm = masked.argsort(axis=1, stable=True)  # [N_recv, topk]

    # After sorting, valid entries (expert < E) come first within each token.
    sorted_valid = torch.gather(masked, 1, sort_perm) < E  # [N_recv, topk]

    # Flat source index into dispatched_probs.reshape(-1):
    #   dispatched_probs[t, sort_perm[t, j]] → flat = t * topk + sort_perm[t, j]
    row_offsets = torch.arange(N_recv, device=device, dtype=sort_perm.dtype)
    src_flat = row_offsets.unsqueeze(1) * topk + sort_perm  # [N_recv, topk]
    score_src_idx = src_flat[sorted_valid]  # [TK]

    # Differentiable gather from dispatched_probs
    gathered = dispatched_probs.reshape(-1)[score_src_idx]  # [TK]

    if TK_padded > TK:
        # Padding zeros (no gradient contribution)
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
        ) = deepep_topk_to_sonic_metadata(
            dispatched_indices, dispatched_probs,
            tokens_per_expert, E, device=x.device,
        )

        # Build router_scores differentiably from dispatched_probs so that
        # ds gradients flow back to the caller (e.g. DeepEP's gate).
        # deepep_topk_to_sonic_metadata produces a non-differentiable copy;
        # we reconstruct the same values using autograd-tracked indexing.
        router_scores = _differentiable_router_scores(
            dispatched_probs, dispatched_indices,
            TK_padded - total_pad_rows, TK_padded, E,
        )
        # T_down = N_recv for the topk path: _router_forward outputs [N_recv, H]
        T_down = N_recv

        # 2. NO x-padding needed — route-level padding handles alignment.
        #    Padding rows in x_gather_idx point to row 0 with score=0.

        # 3. Stack weights (cached)
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
                T_down, None, stream_id,
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
        E_val = len(ctx._experts)
        H_val = output_grad.shape[1]
        _ensure_native_grads(
            ctx._experts, I,
            (2 * I, H_val, E_val),  # w1_shape
            (H_val, I, E_val),       # w2_shape
            output_grad.device,
        )
        down_ctx._wgrad_w2_accumulator = _NATIVE_W2_GRAD
        down_grads = _DownProjection.backward(down_ctx, output_grad)
        dz = down_grads[1]
        dw2 = down_grads[2]

        # ds: deterministic index based on _has_b2 (b2 is always None in this path,
        # so ds is always at index 3.  Previous code used a fragile float32 heuristic.)
        ds_idx = 4 if getattr(down_ctx, '_has_b2', False) else 3
        ds = down_grads[ds_idx]

        # ── UpProjection backward ────────────────────────────────────────
        up_ctx._wgrad_w1_accumulator = _NATIVE_W1_GRAD
        up_grads = _UpProjection.backward(up_ctx, None, dz)
        dx = up_grads[0]   # [T_orig, H] — matches x input shape
        dw1 = up_grads[1]

        # ── Weight grads → per-expert main_grad ──────────────────────────
        if dw1 is not None:
            _accumulate_w1(dw1, ctx._experts, ctx._I)
        if dw2 is not None:
            _accumulate_w2(dw2, ctx._experts)

        # ── Return grads for tensor inputs ───────────────────────────────
        # Tensor inputs: hidden_states, router_scores, efo, x_gather, s_scatter,
        #                s_reverse_scatter, naept_offset, w1, w2
        return dx, ds, None, None, None, None, None, None, None
