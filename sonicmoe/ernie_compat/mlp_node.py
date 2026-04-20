"""SonicMoE ↔ ERNIE integration: PyLayer with fully manual forward/backward.

``SonicMoEFunc`` calls SonicMoE's ``_UpProjection`` / ``_DownProjection``
static forward/backward methods directly — **without** engaging their autograd.
All intermediate tensors, FP8 quantization state, and prequantized-scale
side-channels are managed explicitly.

This matches the ERNIE-core ``Fp8FusedMoeFunc`` contract:
  - Forward:  ``[T, H] bf16 → [T, H] bf16``
  - Backward: weight grads accumulated in-place to ``main_grad`` (float32),
              only ``(dx, d_router_scores, ...)`` returned to outer autograd.
"""

from __future__ import annotations

from typing import Any, Sequence

import paddle
import torch

from sonicmoe.enums import ActivationType
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


def stack_ernie_w1(experts: Sequence[Any], H: int, I: int) -> torch.Tensor:
    """Per-expert ``[H, 2I]`` split-half → logical ``[2I, H, E]`` interleaved.

    Physical layout ``[E, 2I, H]`` (contiguous), presented as ``[2I, H, E]``
    via ``.permute(1, 2, 0)`` — matching the stride order QuACK GEMM expects.
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
    _W_CACHE.clear()
    _W_CACHE[key] = w1
    return w1


def stack_ernie_w2(experts: Sequence[Any]) -> torch.Tensor:
    """Per-expert ``[I, H]`` → logical ``[H, I, E]``.

    Physical ``[E, H, I]`` (contiguous) via ``.permute(1, 2, 0)``.
    """
    key = ("w2",) + _cache_key(*(e.down_proj.weight for e in experts))
    if key in _W_CACHE:
        return _W_CACHE[key]
    physical = torch.stack([e.down_proj.weight for e in experts]) \
                    .permute(0, 2, 1).contiguous()  # [E, H, I]
    w2 = physical.permute(1, 2, 0)  # logical [H, I, E]
    w2.stop_gradient = False
    _W_CACHE.clear()
    _W_CACHE[key] = w2
    return w2


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

def _accumulate_w1(dw1: torch.Tensor, experts: Sequence[Any], I: int):
    """``dw1 [2I, H, E]`` interleaved → per-expert ``main_grad [H, 2I]`` split-half."""
    for e in range(len(experts)):
        w = experts[e].up_gate_proj.weight
        if not hasattr(w, "main_grad") or w.main_grad is None:
            w.main_grad = paddle.zeros(w.shape, dtype="float32")
        dw_e = dw1[:, :, e]              # [2I, H]
        gate, up = dw_e[0::2, :], dw_e[1::2, :]
        w.main_grad.add_(torch.cat([gate, up], dim=0).t().float())


def _accumulate_w2(dw2: torch.Tensor, experts: Sequence[Any]):
    """``dw2 [H, I, E]`` → per-expert ``main_grad [I, H]``."""
    for e in range(len(experts)):
        w = experts[e].down_proj.weight
        if not hasattr(w, "main_grad") or w.main_grad is None:
            w.main_grad = paddle.zeros(w.shape, dtype="float32")
        w.main_grad.add_(dw2[:, :, e].t().float())


# ── The PyLayer ──────────────────────────────────────────────────────────────

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

        # ds: at index 4 (after db2=None at index 3)
        has_b2 = getattr(down_ctx, '_has_b2', False)
        ds_idx = 4 if has_b2 else 3
        # Actually, backward always returns in fixed order:
        # [None(y1), dz, dw2, db2_or_skip, ds, None, None, None, None, None, None?, ...]
        # Find ds by checking which is float32
        ds = None
        for g in down_grads[3:]:
            if g is not None and g.dtype == torch.float32:
                ds = g
                break

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
