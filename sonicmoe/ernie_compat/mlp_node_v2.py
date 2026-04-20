"""Complete SonicMoE MlpNode wrapper for ERNIE integration.

``SonicMoEMlpNode`` packages unzip + FP8 FFN + zip into a single callable,
using the DeepEP zero-sync metadata path to eliminate argsort overhead.

Drop-in replacement for ERNIE's MlpNode:
    ``forward(dispatched_hidden_states, tokens_per_expert) → expert_output``

Performance notes (zero migration overhead via route-level padding):
  - x is NOT padded — padding is handled at the routing metadata level
    (gather indices for pad rows point to row 0, score=0 nullifies contribution).
    This matches the frontier's ``_pad_routing_metadata`` design exactly.
  - No grad padding needed in backward (output is T-sized, not T+pad).
  - Metadata caching: when tokens_per_expert hasn't changed, metadata is reused.
  - Native-layout grad accumulation: single add_() per weight, no transpose.
"""

from __future__ import annotations

from typing import Any, Sequence

import paddle
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat.deepep_metadata import deepep_to_sonic_metadata
from sonicmoe.ernie_compat.mlp_node import (
    SonicMoEFunc,
    invalidate_weight_caches,
    stack_ernie_w1,
    stack_ernie_w2,
)


class SonicMoEMlpNode:
    """Drop-in replacement for ERNIE's MlpNode.

    Packages unzip + FP8 FFN + zip into a single callable.
    Uses DeepEP's pre-sorted token layout for zero-sync metadata conversion.

    Performance optimizations (zero migration overhead):
      - Route-level padding: x is passed directly (no padding, no cat, no copy).
      - Metadata caching: reuses metadata when tokens_per_expert is unchanged.
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

        # ── Metadata cache (reuse when tokens_per_expert unchanged) ────────
        self._cached_tpe: list[int] | None = None
        self._cached_metadata: tuple | None = None

    def forward(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Run FP8 expert FFN on DeepEP-dispatched tokens.

        Parameters
        ----------
        dispatched_hidden_states : Tensor [T, H] bf16
            Already sorted by expert from DeepEP dispatch.
        tokens_per_expert : list[int] or Tensor [E]
            Per-expert token counts from DeepEP ``buffer.dispatch()``.

        Returns
        -------
        Tensor [T, H] bf16
            Expert output, same ordering as input.
        """
        x = dispatched_hidden_states
        T = x.shape[0]
        E = self._E
        H = self._H
        I = self._I

        # 1. Zero-sync metadata conversion (with caching)
        tpe_list = (
            tokens_per_expert.tolist()
            if isinstance(tokens_per_expert, torch.Tensor)
            else list(tokens_per_expert)
        )
        if tpe_list == self._cached_tpe and self._cached_metadata is not None:
            (
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                router_scores,
                TK_padded,
                total_pad_rows,
            ) = self._cached_metadata
        else:
            (
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                router_scores,
                TK_padded,
                total_pad_rows,
            ) = deepep_to_sonic_metadata(tpe_list, T, E, device=x.device)
            self._cached_tpe = tpe_list
            self._cached_metadata = (
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                router_scores,
                TK_padded,
                total_pad_rows,
            )

        # 2. NO x-padding needed — route-level padding handles alignment.
        #    Padding rows in x_gather_idx point to row 0 with score=0.

        # 3. Stack weights (cached)
        w1 = stack_ernie_w1(self._experts, H, I)
        w2 = stack_ernie_w2(self._experts)

        # 4. Prepare tensor inputs for PyLayer
        x = x.detach()
        x.stop_gradient = False
        router_scores = router_scores.detach()
        router_scores.stop_gradient = False

        # Non-differentiable inputs
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
            E, T, TK_padded,
            self._activation_type,
            self._stream_id,
        )

        return output

    def __call__(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(dispatched_hidden_states, tokens_per_expert)


# ── Internal PyLayer using pre-computed DeepEP metadata ─────────────────────

from sonicmoe.ernie_compat.mlp_node import (
    _FakeCtx,
    _accumulate_w1,
    _accumulate_w2,
    _ensure_native_grads,
)
# Import native grad buffers (module-level globals that get updated by _ensure_native_grads)
import sonicmoe.ernie_compat.mlp_node as _mlp_node_mod
from sonicmoe.functional import (
    _DownProjection,
    _UpProjection,
    _refresh_fp8_config,
)
from sonicmoe.functional.utils import enable_fp8

# ── Grad-padding buffer (module-level, reused across iters) ────────────────
# In the DeepEP identity layout, _DownProjection uses T=TK_padded so its output
# is [TK_padded, H]. We slice to [T_orig, H] in forward, which means backward
# receives [T_orig, H] grad that must be re-padded to [TK_padded, H].
# Pre-allocated buffer eliminates per-iter torch.cat.
_GRAD_PAD_BUF: torch.Tensor | None = None
_GRAD_PAD_BUF_ROWS: int = 0


class _SonicMoEDeepEPFunc(paddle.autograd.PyLayer):
    """PyLayer that takes pre-computed DeepEP metadata (skips argsort entirely).

    Uses route-level padding (frontier design): x is NOT padded. Padding rows
    gather from row 0 with score=0, contributing nothing to output or grads.

    In the DeepEP identity layout, TK_padded positions map 1:1 to "virtual tokens"
    (each assigned to 1 expert), so _DownProjection uses T=TK_padded to allocate
    its output, then we slice out[:T_orig] to return only real tokens.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,       # [T, H] bf16  (grad)
        router_scores: torch.Tensor,        # [TK_padded] float32 (grad)
        expert_frequency_offset: torch.Tensor,  # [E+1] int32
        x_gather_idx: torch.Tensor,         # [TK_padded] int32
        s_scatter_idx: torch.Tensor,        # [TK_padded] int32
        s_reverse_scatter_idx: torch.Tensor, # [TK_padded] int32
        num_activated_expert_per_token_offset: torch.Tensor,  # [TK_padded+1] int32
        w1: torch.Tensor,                   # [2I, H, E] bf16
        w2: torch.Tensor,                   # [H, I, E] bf16
        experts: Sequence[Any] = None,
        n_experts: int = 0,
        T_orig: int = 0,
        TK_padded: int = 0,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ) -> torch.Tensor:
        _refresh_fp8_config()

        # ── UpProjection forward (via FakeCtx) ───────────────────────────
        # x is [T_orig, H] — NOT padded.  Padding gather indices point to
        # row 0 (score=0 nullifies contribution).  This matches the frontier's
        # route-level padding where x is never modified.
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
        # T=TK_padded: in the DeepEP identity layout each TK position is a
        # "virtual token" (K=1 per position), so output is [TK_padded, H].
        # The _router_forward scatter uses num_activated_expert_per_token_offset
        # = arange(TK_padded+1) which means out[i] = y2[i] * score[i].
        # Real tokens get score=1, padding gets score=0 → zero contribution.
        down_ctx = _FakeCtx()
        with enable_fp8(True):
            out = _DownProjection.forward(
                down_ctx,
                y1, z, w2, None,                # y1, z, w2, b2
                router_scores, s_scatter_idx, expert_frequency_offset,
                TK_padded, None, stream_id,     # T = TK_padded (identity layout)
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
        ctx._T_orig = T_orig
        # Output is [TK_padded, H]; slice to [T_orig, H] (zero-copy view)
        return out[:T_orig]

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        global _GRAD_PAD_BUF, _GRAD_PAD_BUF_ROWS

        up_ctx = ctx._up_ctx
        down_ctx = ctx._down_ctx
        T_orig = ctx._T_orig
        # _DownProjection.backward needs output_grad matching its T dimension.
        # ctx.T was saved as TK_padded. output_grad is [T_orig, H].
        # Pad back to [TK_padded, H] for backward.
        T_padded = down_ctx.T if hasattr(down_ctx, 'T') else output_grad.shape[0]

        if T_padded > T_orig:
            # Pre-allocated buffer: eliminates per-iter torch.cat
            H = output_grad.shape[1]
            if _GRAD_PAD_BUF is None or _GRAD_PAD_BUF_ROWS < T_padded:
                _GRAD_PAD_BUF = torch.zeros(
                    T_padded, H, dtype=output_grad.dtype, device=output_grad.device
                )
                _GRAD_PAD_BUF_ROWS = T_padded
            grad_padded = _GRAD_PAD_BUF[:T_padded]
            grad_padded[:T_orig].copy_(output_grad)
            grad_padded[T_orig:].zero_()
            output_grad = grad_padded

        # ── DownProjection backward ──────────────────────────────────────
        # Inject fp32 accumulators into ctx for fused wgrad GEMM epilogue.
        # The backward will detect _wgrad_w2_accumulator and accumulate
        # directly in the GEMM epilogue (D = A@B + C, beta=1), eliminating
        # the separate transpose+cast+add kernels.
        #
        # Ensure native grad buffers are allocated before injection.
        I = ctx._I
        E_val = len(ctx._experts)
        H_val = output_grad.shape[1]
        _ensure_native_grads(
            ctx._experts, I,
            (2 * I, H_val, E_val),  # w1_shape
            (H_val, I, E_val),       # w2_shape
            output_grad.device,
        )
        down_ctx._wgrad_w2_accumulator = _mlp_node_mod._NATIVE_W2_GRAD
        down_grads = _DownProjection.backward(down_ctx, output_grad)
        dz = down_grads[1]
        dw2 = down_grads[2]

        ds = None
        for g in down_grads[3:]:
            if g is not None and g.dtype == torch.float32:
                ds = g
                break

        # ── UpProjection backward ────────────────────────────────────────
        up_ctx._wgrad_w1_accumulator = _mlp_node_mod._NATIVE_W1_GRAD
        up_grads = _UpProjection.backward(up_ctx, None, dz)
        dx = up_grads[0]   # [T_orig, H] — matches x input shape
        dw1 = up_grads[1]

        # ── Weight grads → per-expert main_grad ──────────────────────────
        # If accumulators were used, dw1/dw2 are None (already accumulated).
        # If fallback path was taken (non-FP8), accumulate normally.
        if dw1 is not None:
            _accumulate_w1(dw1, ctx._experts, ctx._I)
        if dw2 is not None:
            _accumulate_w2(dw2, ctx._experts)

        # ── Return grads for tensor inputs ───────────────────────────────
        # Tensor inputs: hidden_states, router_scores, efo, x_gather, s_scatter,
        #                s_reverse_scatter, naept_offset, w1, w2
        return dx, ds, None, None, None, None, None, None, None
