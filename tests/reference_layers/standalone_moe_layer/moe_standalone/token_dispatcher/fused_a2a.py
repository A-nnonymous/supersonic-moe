# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepEp communication functions"""

import warnings

import paddle
import paddle.distributed.fleet as fleet
from moe_standalone.compat import HAVE_DEEP_EP, deep_ep, kitchen_quant, manual_backward
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group


try:
    import kitchen
except:
    pass

import logging
import queue

from moe_standalone.compat import global_rr_queue_log


logger = logging.getLogger(__name__)


class DeepEPBuffer:
    """
    DeepEPBuffer
    """

    def __init__(self):
        """
        __init__
        """
        self._buffer = None

    def get_buffer(self, group, hidden_bytes):
        """Get or create a buffer for all-to-all communication.

        Args:
            group (paddle.distributed.ProcessGroup): Process group for communication
            hidden_bytes (int): Number of hidden bytes needed

        Returns:
            Buffer: Communication buffer
        """
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
            deep_ep.Buffer.get_dispatch_config(group.world_size),
            deep_ep.Buffer.get_combine_config(group.world_size),
        ):
            # Split long line for PEP8 compliance
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size), num_nvl_bytes)
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size), num_rdma_bytes)

        # Allocate buffer if not existed or not enough buffer
        # NOTES: the adaptive routing configuration of the network **must be off**
        if (
            self._buffer is None
            or self._buffer.group != group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
            self._buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
            logger.info("DeepEP buffer created.")
        return self._buffer

    def clear_buffer(self):
        """
        clear_buffer to remove memory allocation caused by deepep
        """
        if self._buffer is not None:
            del self._buffer
            self._buffer = None
            logger.info("DeepEP buffer cleared.")


deepep_buffer = DeepEPBuffer()


def barrier_ep():
    """barrier_ep"""
    hcg = fleet.get_hybrid_communicate_group()
    if hasattr(hcg, "get_expert_parallel_group"):
        ep_group = hcg.get_expert_parallel_group()
        paddle.distributed.barrier(ep_group)
    else:
        warnings.warn("No get_expert_parallel_group found, please check your environment.")


def wait_for_deepep(group_id):
    """wait_for_deepep"""
    comm_event = deep_ep.get_event_from_comm_stream(group_id)
    comm_event.calc_stream_wait(group_id)


def get_hidden_bytes(x: paddle.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (paddle.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.shape[1] * max(x.element_size(), 2)


def fused_dispatch_forward_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    do_barrier_ep=True,
):
    """Forward pass of fused dispatch."""
    if do_barrier_ep:
        barrier_ep()

    # Calculate layout before actual dispatch
    global deepep_buffer
    if isinstance(x, tuple):
        buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(x[0]))
    else:
        buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(x))
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event_,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    assert token_probs.dtype == paddle.float32
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive,
    # so this is not compatible with CUDA graph
    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    states = dict()
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    return recv_x, recv_token_probs, states, event


def fused_dispatch_backward_func(
    grad_output,
    grad_token_probs,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    do_barrier_ep=True,
):
    """Backward pass of fused dispatch."""
    if do_barrier_ep:
        barrier_ep()

    global deepep_buffer
    buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(grad_output))

    grad_x, grad_token_probs, event = buffer.combine(
        grad_output.contiguous(),
        handle,
        topk_weights=grad_token_probs.cast(paddle.float32),
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return grad_x, None, grad_token_probs


def fused_combine_forward_func(
    x,
    group,
    states,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    do_barrier_ep=True,
):
    """Forward pass of fused combine."""
    if do_barrier_ep:
        barrier_ep()

    global deepep_buffer
    handle = states["handle"]
    buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(x))
    combined_x, _, event = buffer.combine(
        x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return combined_x


def fused_combine_backward_func(
    grad_output,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    do_barrier_ep=True,
):
    """Backward pass of fused combine."""
    if do_barrier_ep:
        barrier_ep()

    global deepep_buffer
    if isinstance(grad_output, tuple):
        buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(grad_output[0]))
        grad_x, _, _, _, _, event = buffer.dispatch(
            (grad_output[0].contiguous(), grad_output[1].contiguous()),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    else:
        buffer = deepep_buffer.get_buffer(group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    return grad_x


def set_grad_in_dtype_non_consistent(ctx):
    """
    Make grad dtype not consistent with forward dtype
    """
    if hasattr(ctx, "set_grad_in_dtype_consistent"):
        ctx.set_grad_in_dtype_consistent(False)


class FusedDispatchAsync(PyLayer):
    """FusedDispatchAsync."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_dispatch_a2a=False,
        use_ue8m0=False,
    ):
        """Forward pass of fused dispatch."""
        set_grad_in_dtype_non_consistent(ctx)
        if fp8_dispatch_a2a:
            x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
                using_ue8m0_scale=use_ue8m0,
            )
            x = (x_fp8, scale)

        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish=True,
        )

        assert fn is not None, "use FusedDispatchAsync async, but fn is None."
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        ctx.group = group
        ctx.handle = states["handle"]
        ctx.event = event

        wait_for_deepep(group.id)

        if fp8_dispatch_a2a:
            recv_x, scale = recv_x
            return (
                recv_x,
                recv_token_probs,
                states,
                {"scale": scale},  # scale用dict传输, 是为了避免其进入计算图
            ) + fn_out
        else:
            return (
                recv_x,
                recv_token_probs,
                states,
                None,
            ) + fn_out

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs, *fn_out_grads):
        """Backward pass of fused dispatch."""
        grad_x, grad_token_indices, grad_token_probs = fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            ctx.group,
            ctx.handle,
            async_finish=True,
        )

        fn_args_grads = ctx.bwf(*fn_out_grads)

        wait_for_deepep(ctx.group.id)

        return (
            grad_x,
            grad_token_indices,
            grad_token_probs,
        ) + fn_args_grads


class FusedCombineAsync(PyLayer):
    """FusedCombineAsync."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        *fn_args,
        fn,
        is_first_fwd=False,
    ):
        """Forward pass of fused combine."""
        set_grad_in_dtype_non_consistent(ctx)
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            async_finish=True,
        )

        assert fn is not None, "use FusedCombineAsync async, but fn is None."
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)

        ctx.handle = states["handle"]
        ctx.group = group

        wait_for_deepep(group.id)

        return (combined_x,) + fn_out

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        """Backward pass of fused combine."""
        grad_x = fused_combine_backward_func(
            grad_output,
            ctx.group,
            ctx.handle,
            async_finish=True,
        )

        fn_args_grads = ctx.bwf(*fn_out_grads)

        wait_for_deepep(ctx.group.id)
        return (grad_x,) + fn_args_grads


class FusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        fp8_dispatch_a2a=False,
        do_barrier_ep=True,
        use_ue8m0=False,
    ):
        """Forward pass of fused dispatch."""
        set_grad_in_dtype_non_consistent(ctx)
        if fp8_dispatch_a2a:
            x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
                using_ue8m0_scale=use_ue8m0,
            )
            x = (x_fp8, scale)

        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            do_barrier_ep,
        )

        ctx.group = group
        ctx.handle = states["handle"]
        ctx.event = event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.do_barrier_ep = do_barrier_ep

        if fp8_dispatch_a2a:
            recv_x, scale = recv_x
            # scale用dict传输, 是为了避免其进入计算图
            return recv_x, recv_token_probs, states, {"scale": scale}
        else:
            return recv_x, recv_token_probs, states, None

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        return fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            ctx.group,
            ctx.handle,
            None,  # previous_event
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            ctx.do_barrier_ep,
        )


# fusedcombine
class FusedCombine(PyLayer):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        do_barrier_ep=True,
    ):
        """Forward pass of fused combine."""
        set_grad_in_dtype_non_consistent(ctx)
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            do_barrier_ep,
        )

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.previous_event = previous_event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.do_barrier_ep = do_barrier_ep

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        return fused_combine_backward_func(
            grad_output,
            ctx.group,
            ctx.handle,
            ctx.previous_event,
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            ctx.do_barrier_ep,
        )


class FusedCombineFunctor(PyLayer):
    """FusedCombineFunctor class for basic deepep combine"""

    @staticmethod
    def forward(
        ctx, hold_tensors, x, group, states, previous_event=None, async_finish=False, allocate_on_comm_stream=False
    ):
        """Forward pass of fused combine, do not need to do compute or comm, get output directly."""
        set_grad_in_dtype_non_consistent(ctx)
        fwd_output = hold_tensors["res_output"]

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.previous_event = previous_event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream

        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        return fused_combine_backward_func(
            grad_output,
            ctx.group,
            ctx.handle,
            ctx.previous_event,
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )


class FusedCombineRefinedRecompute(object):
    """RefinedRecompute class for deepep fused_combine."""

    def __init__(self):
        """__init__"""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "FusedCombine")

    def forward(self, x, group, states, previous_event=None, async_finish=False, allocate_on_comm_stream=False):
        """forward"""
        tracer = framework._dygraph_tracer()
        if not tracer._has_grad:
            fwd_output = self._first_fwd(x, group, states, previous_event, async_finish, allocate_on_comm_stream)
            self._hold_tensors_queue.put(fwd_output)
            return fwd_output
        else:
            assert not self._hold_tensors_queue.empty(), "Queue should not be empty for the second forward pass."
            fwd_output = self._hold_tensors_queue.get()
            output = self._second_fwd(
                fwd_output, x, group, states, previous_event, async_finish, allocate_on_comm_stream
            )
            return output

    @paddle.no_grad()
    def _first_fwd(self, x, group, states, previous_event, async_finish, allocate_on_comm_stream):
        """_first_fwd"""
        return fused_combine_forward_func(
            x,
            group,
            states,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
        )

    def _second_fwd(self, fwd_output, x, group, states, previous_event, async_finish, allocate_on_comm_stream):
        """_second_fwd"""
        return FusedCombineFunctor.apply(
            {"res_output": fwd_output}, x, group, states, previous_event, async_finish, allocate_on_comm_stream
        )

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.forward(*args, **kwargs)


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group: Group,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        fp8_dispatch_a2a=False,
        inner_layer_overlap_handle=None,
        do_barrier_ep=True,
        use_ue8m0=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        if inner_layer_overlap_handle is None:
            return FusedDispatch.apply(
                x.contiguous(),
                token_indices,
                token_probs,
                num_experts,
                group,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
                fp8_dispatch_a2a,
                do_barrier_ep,
                use_ue8m0,
            )
        else:
            assert previous_event is None
            assert not allocate_on_comm_stream
            assert isinstance(inner_layer_overlap_handle, dict)
            assert "fn" in inner_layer_overlap_handle
            assert "fn_args" in inner_layer_overlap_handle
            assert isinstance(inner_layer_overlap_handle["fn_args"], tuple)

            recv_x, recv_token_probs, states, fp8_dispatched_handle, *fn_out = FusedDispatchAsync.apply(
                x.contiguous(),
                token_indices,
                token_probs,
                num_experts,
                group,
                *(inner_layer_overlap_handle["fn_args"]),
                fn=inner_layer_overlap_handle["fn"],
                is_first_fwd=not framework._dygraph_tracer()._has_grad,
                fp8_dispatch_a2a=fp8_dispatch_a2a,
                use_ue8m0=use_ue8m0,
            )
            inner_layer_overlap_handle["fn_out"] = fn_out
            return recv_x, recv_token_probs, states, fp8_dispatched_handle

    def fused_combine(
        x,
        group,
        handle,
        _rr_fusedcombined=None,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        inner_layer_overlap_handle=None,
        use_rr_deepep_combine=False,
        do_barrier_ep=True,
    ):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        states = dict()
        states["handle"] = handle

        if inner_layer_overlap_handle is None:
            if not use_rr_deepep_combine:
                return FusedCombine.apply(
                    x,
                    group,
                    states,
                    previous_event,
                    async_finish,
                    allocate_on_comm_stream,
                    do_barrier_ep,
                )
            else:
                return _rr_fusedcombined(
                    x=x,
                    group=group,
                    states=states,
                    previous_event=previous_event,
                    async_finish=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
                )

        else:
            assert (
                not use_rr_deepep_combine
            ), "When inner_layer_overlap_handle is used, we do not support rr_deepep_combine."
            assert previous_event is None
            assert not allocate_on_comm_stream
            assert isinstance(inner_layer_overlap_handle, dict)
            assert "fn" in inner_layer_overlap_handle
            assert "fn_args" in inner_layer_overlap_handle
            assert isinstance(inner_layer_overlap_handle["fn_args"], tuple)

            combined_x, *fn_out = FusedCombineAsync.apply(
                x,
                group,
                states,
                *(inner_layer_overlap_handle["fn_args"]),
                fn=inner_layer_overlap_handle["fn"],
                is_first_fwd=not framework._dygraph_tracer()._has_grad,
            )
            inner_layer_overlap_handle["fn_out"] = fn_out
            return combined_x

else:
    fused_dispatch = None
    fused_combine = None
