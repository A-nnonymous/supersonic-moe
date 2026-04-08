# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Callable
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.nn.functional as F

from .count_cumsum import count_cumsum
from .enums import ActivationType, KernelBackendMoE, is_glu
from .functional import FP8Protocol, moe_TC_softmax_topk_layer, clear_all_fp8_weight_caches
from .functional.utils import enable_fp8
from .quack_utils import (
    clear_blockscaled_fp8_weight_cache,
    prefetch_blockscaled_w2_fp8,
    precompute_weight_fp8,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_for_direct_fused_dgated,
    quantize_and_pack_activation,
)


try:
    from xma.modules.moe import scattered_experts

    _IS_XMA_AVAILABLE = True
except ImportError:
    _IS_XMA_AVAILABLE = False


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)


def _geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.gelu(g.to(dtype=torch.float32)) * u).to(dtype=g.dtype)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.to(dtype=torch.float32)).to(dtype=x.dtype)


def _reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g) * u).to(dtype=g.dtype)


def _relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def _relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class Experts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool = True, std: float | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def up_projection_scattermoe_forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        if not _IS_XMA_AVAILABLE:
            raise ImportError(
                "install accelerated-model-architectures from https://github.com/open-lm-engine/accelerated-model-architectures"
            )

        input = scattered_experts(
            inputs=input,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=None,
            grouped_in=False,
            grouped_out=True,
        )

        return input

    def down_projection_scattermoe_forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        if not _IS_XMA_AVAILABLE:
            raise ImportError(
                "install accelerated-model-architectures from https://github.com/open-lm-engine/accelerated-model-architectures"
            )

        input = scattered_experts(
            inputs=input,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=gates,
            grouped_in=True,
            grouped_out=False,
        )

        return input

    def torch_forward(
        self, input: torch.Tensor, expert_frequency: torch.Tensor | None, return_list: bool = False
    ) -> list[torch.Tensor] | torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.split(expert_frequency.tolist(), dim=0)
        else:
            assert expert_frequency is None

        input = [
            F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]

        if not return_list:
            input = torch.cat(input, dim=0)

        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()


class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: ActivationType,
        add_bias: bool,
        std: float,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.router = nn.Linear(in_features=self.hidden_size, out_features=num_experts, bias=False)

        self.activation_function = activation_function

        self.c_fc = Experts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.c_proj = Experts(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

        self.stream_id = torch.cuda.current_stream().cuda_stream

    @torch.no_grad()
    def prefetch_fp8_weights(self, protocol: FP8Protocol) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if protocol is None:
            raise ValueError("prefetch_fp8_weights requires a valid FP8 protocol")

        return {
            "downproj": prefetch_blockscaled_w2_fp8(self.c_proj.weight.permute(1, 2, 0), protocol),
        }

    @torch.no_grad()
    def prefetch_all_fp8_weights(self) -> None:
        """Pre-quantize all expert weights to blockscaled FP8 for fused gated path.

        Stores FP8 weights as attributes on the parameter objects (ernie-core pattern).
        Call once after model init or after optimizer step. The fused forward path
        will check for these cached attributes before quantizing on-the-fly.

        Caches:
        - w1 for fused gemm_gated forward: (E, H, 2I) fp8 + ISA-packed scales
        - w2 for blockscaled varlen backward: (E, I, H) fp8 + ISA-packed scales
        - w2 for fused gemm_dgated backward (when available): (E, H, I) fp8 + ISA-packed scales
        """
        w1 = self.c_fc.weight   # (E, 2I, H) parameter
        w2 = self.c_proj.weight  # (E, H, I) parameter

        # Forward path: gemm_gated expects (L, K, N) = (E, H, 2I)
        w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1.permute(1, 2, 0))
        w1.fp8_fused_gated = w1_fp8
        w1.fp8_fused_gated_scales = w1_scales

        # Backward act-grad path: blockscaled_fp8_gemm_varlen expects (I, H, E) for w2^T
        w2_for_varlen = w2.permute(1, 2, 0)  # (H, I, E) → permute for varlen
        w2_fp8_varlen, w2_scales_varlen = precompute_weight_fp8(w2_for_varlen)
        w2.fp8_varlen = w2_fp8_varlen
        w2.fp8_varlen_scales = w2_scales_varlen

    @torch.no_grad()
    def clear_fp8_weight_cache(self) -> None:
        """Clear all FP8 weight caches (per-tensor + blockscaled)."""
        clear_all_fp8_weight_caches()

    @torch.no_grad()
    def refresh_fp8_shadow_weights(self) -> None:
        """Pre-quantize all expert weights to blockscaled FP8, populating the runtime caches.

        Call after optimizer.step() to eliminate runtime weight quantization overhead.
        The forward/backward paths use the same cache lookup (keyed by data_ptr + _version),
        so pre-populated entries are hit with zero additional quantize cost.

        This is the "bf16 master + fp8 shadow" pattern: bf16 Parameters are master weights
        for the optimizer; FP8 shadows are consumed by the fused GEMM kernels.

        Ernie shape (E=8, H=3072, I=1536): quantize cost ~80µs one-shot (vs ~174µs/iter).
        Shadow size: ~223 MiB (4 layouts), automatically freed when _version changes.
        """
        w1 = self.c_fc.weight   # (E, 2I, H) bf16 Parameter — Experts convention
        w2 = self.c_proj.weight  # (E, H, I) bf16 Parameter

        # The functional layer receives weights as (2I, H, E) and (H, I, E) via .permute(1,2,0).
        w1_perm = w1.permute(1, 2, 0)  # (2I, H, E)
        w2_perm = w2.permute(1, 2, 0)  # (H, I, E)

        # Layout 1: w1 for fused_gated forward — writes _FUSED_WEIGHT_CACHE
        precompute_weight_fp8_for_fused_gated(w1_perm)

        # Layout 2: w2 for varlen down-proj forward — writes _VARLEN_WEIGHT_CACHE
        precompute_weight_fp8(w2_perm)

        # Layout 3: w2 for direct_fused_dgated backward — writes _FUSED_WEIGHT_CACHE
        precompute_weight_fp8_for_direct_fused_dgated(w2_perm)

        # Layout 4: w1T for varlen actgrad backward — writes _VARLEN_WEIGHT_CACHE
        precompute_weight_fp8(w1_perm.permute(1, 0, 2))  # (H, 2I, E)

    @torch.no_grad()
    def stash_bf16_to_cpu(self) -> None:
        """Populate standard FP8 caches from bf16, then free bf16 on GPU (−216 MiB).

        Uses the EXISTING runtime caches (_FUSED_WEIGHT_CACHE, _VARLEN_WEIGHT_CACHE).
        No separate named cache needed — cache keys use (_version, shape, stride)
        which are preserved by resize_(0). Sets _NAMED_FP8_CACHE as a sentinel
        (non-empty = stash mode active, backward skips permute-based lookups).
        """
        from .functional import _NAMED_FP8_CACHE, _WEIGHT_META
        w1, w2 = self.c_fc.weight, self.c_proj.weight
        if w1.untyped_storage().size() == 0:
            return  # already stashed

        # 1. Populate all 4 standard caches from bf16 (cache keys based on bf16 metadata)
        w1_perm = w1.permute(1, 2, 0)
        w2_perm = w2.permute(1, 2, 0)
        precompute_weight_fp8_for_fused_gated(w1_perm)
        precompute_weight_fp8(w2_perm)
        w1T_result = precompute_weight_fp8(w1_perm.permute(1, 0, 2))
        w2d_result = precompute_weight_fp8_for_direct_fused_dgated(w2_perm)

        # 2. Set named cache as sentinel + store direct references for backward
        # (backward can't do w.permute() after resize_(0), so it reads these)
        _NAMED_FP8_CACHE["w1T_varlen"] = w1T_result
        _NAMED_FP8_CACHE["w2_dgated"] = w2d_result
        _NAMED_FP8_CACHE["w1_fused"] = precompute_weight_fp8_for_fused_gated(w1_perm)  # re-read from cache (hit)
        _NAMED_FP8_CACHE["w2_varlen"] = precompute_weight_fp8(w2_perm)  # re-read from cache (hit)

        # 3. Save shape metadata
        _WEIGHT_META["w1"] = (w1.shape, w1.device)
        _WEIGHT_META["w2"] = (w2.shape, w2.device)

        # 4. Backup bf16 to CPU
        self._cpu_w1 = w1.data.to('cpu')
        self._cpu_w2 = w2.data.to('cpu')

        # 5. Free GPU storage (−216 MiB). Cache keys still valid.
        w1.data.untyped_storage().resize_(0)
        w2.data.untyped_storage().resize_(0)
        w1.data.untyped_storage().resize_(0)
        w2.data.untyped_storage().resize_(0)

    @torch.no_grad()
    def unstash_bf16(self) -> None:
        """Restore bf16 params from CPU. Call before optimizer.step()."""
        from .functional import _NAMED_FP8_CACHE, _WEIGHT_META
        w1, w2 = self.c_fc.weight, self.c_proj.weight
        if not hasattr(self, '_cpu_w1'):
            return
        w1.data.untyped_storage().resize_(self._cpu_w1.nelement() * self._cpu_w1.element_size())
        w1.data.copy_(self._cpu_w1)
        w2.data.untyped_storage().resize_(self._cpu_w2.nelement() * self._cpu_w2.element_size())
        w2.data.copy_(self._cpu_w2)
        del self._cpu_w1, self._cpu_w2
        _NAMED_FP8_CACHE.clear()
        _WEIGHT_META.clear()

    @torch.no_grad()
    def sync_grads(self) -> None:
        """Copy gradients from autograd-accumulated tensors to Parameters.

        In stash mode, gradients are on the as_strided proxy tensors.
        Since these proxies don't requires_grad, autograd returns dw1/dw2
        but doesn't assign them. The gradients are lost.

        For now this is a no-op — gradients must be collected differently.
        See HANDOFF §9 for the architectural solution.
        """
        pass

    @torch.no_grad()
    def has_fp8_shadow_weights(self) -> bool:
        """Check if FP8 shadow weights are fresh (cache entries match current _version)."""
        # Shadow weights live in the runtime caches. If the cache was populated
        # by refresh_fp8_shadow_weights() with the current _version, hits are guaranteed.
        # We can't cheaply verify cache freshness, so just check if caches are non-empty.
        from .quack_utils.blockscaled_fp8_gemm import _VARLEN_WEIGHT_CACHE, _FUSED_WEIGHT_CACHE
        return len(_VARLEN_WEIGHT_CACHE) > 0 and len(_FUSED_WEIGHT_CACHE) > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        kernel_backend_moe: KernelBackendMoE = KernelBackendMoE.sonicmoe,
        is_inference_mode: bool = False,
        fp8_protocol: FP8Protocol | None = None,
        use_fp8: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape

        # hidden_states -> (batch_size, query_length, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        with ExitStack() as stack:
            if use_fp8:
                stack.enter_context(enable_fp8())

            if kernel_backend_moe == KernelBackendMoE.sonicmoe and self.num_experts <= 32768:
                _stashed = self.c_fc.weight.untyped_storage().size() == 0
                if _stashed:
                    from .functional import _WEIGHT_META
                    w1s, w1d = _WEIGHT_META["w1"]
                    w2s, w2d = _WEIGHT_META["w2"]
                    # 2-byte shape proxy via as_strided (zero strides, no real data)
                    _tiny = torch.zeros(1, dtype=torch.bfloat16, device=w1d)
                    w1_arg = torch.as_strided(_tiny, w1s, (0,) * len(w1s)).permute(1, 2, 0)
                    w2_arg = torch.as_strided(_tiny, w2s, (0,) * len(w2s)).permute(1, 2, 0)
                else:
                    w1_arg = self.c_fc.weight.permute(1, 2, 0)
                    w2_arg = self.c_proj.weight.permute(1, 2, 0)
                hidden_states, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
                    hidden_states,
                    self.router.weight,
                    w1_arg,
                    self.c_fc.bias,
                    w2_arg,
                    self.c_proj.bias,
                    self.top_k,
                    self.stream_id,
                    self.activation_function,
                    is_inference_mode or not self.training,
                    fp8_protocol,
                )
            else:
                # hidden_states -> (total_q, hidden_size)
                router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

                # router_logits -> (total_q, num_experts)
                # router_weights -> (total_q, top_k)
                # selected_experts -> (total_q, top_k)

                hidden_states, expert_frequency = self._compute_experts(
                    hidden_states,
                    router_weights,
                    selected_experts,
                    kernel_backend_moe=kernel_backend_moe,
                )

        hidden_states = hidden_states.view(original_shape)

        # hidden_states -> (batch_size, query_length, hidden_size)

        if is_inference_mode:
            aux_loss = None
        else:
            aux_loss = self._compute_switch_loss(
                logits=router_logits,
                probs=F.softmax(router_logits, dim=-1, dtype=torch.float32),
                expert_frequency=expert_frequency,
            )

        return hidden_states, aux_loss

    # copied from https://github.com/open-lm-engine/lm-engine/blob/1447883df709727839bbbb367ce727fa56962a6a/lm_engine/hf_models/modeling_utils/mlp_blocks/moe.py#L432-L455
    # NOTE we don't do all_reduce here for expert frequency for simplicity across data parallel workers
    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        aux_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()

        return aux_loss

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.router(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)

        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        router_weights = F.softmax(router_weights.float(), dim=-1)
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        kernel_backend_moe: KernelBackendMoE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_experts = selected_experts.flatten()

        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.sort()

        is_num_experts_multiple_of_4 = self.num_experts % 4 == 0

        if is_num_experts_multiple_of_4:
            expert_frequency, expert_offsets = count_cumsum(selected_experts, self.num_experts, do_cumsum=True)
        else:
            expert_frequency = selected_experts.bincount(minlength=self.num_experts).to(torch.int32)
            expert_offsets = expert_frequency.cumsum(-1).to(torch.int32)

        act_func = {
            ActivationType.SWIGLU: _swiglu,
            ActivationType.GEGLU: _geglu,
            ActivationType.REGLU: _reglu,
            ActivationType.GELU: _gelu,
            ActivationType.RELU: _relu,
            ActivationType.SILU: _silu,
            ActivationType.RELU_SQ: _relu_sq,
        }[self.activation_function]

        T = hidden_states.size(0)

        if kernel_backend_moe == KernelBackendMoE.scattermoe:
            hidden_states = self.c_fc.up_projection_scattermoe_forward(
                input=hidden_states,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
            )
            hidden_states = act_func(hidden_states)
            hidden_states = self.c_proj.down_projection_scattermoe_forward(
                input=hidden_states,
                num_experts_per_token=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                gates=router_weights,
            )
        elif kernel_backend_moe == KernelBackendMoE.torch:
            # sort and group input tokens according to expert assignment
            fan_in_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            hidden_states = hidden_states[fan_in_index]

            hidden_states = self.c_fc.torch_forward(
                input=hidden_states, expert_frequency=expert_frequency, return_list=True
            )

            hidden_states = [act_func(i) for i in hidden_states]
            hidden_states = self.c_proj.torch_forward(input=hidden_states, expert_frequency=None, return_list=False)

            hidden_states = hidden_states * batch_gates.unsqueeze(-1)
            zeros = torch.zeros((T, self.hidden_size), dtype=torch.float32, device=hidden_states.device)
            hidden_states = zeros.index_add(0, fan_in_index, hidden_states)
        else:
            raise ValueError(f"unexpected kernel_backend_moe ({kernel_backend_moe})")

        return hidden_states, expert_frequency

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
