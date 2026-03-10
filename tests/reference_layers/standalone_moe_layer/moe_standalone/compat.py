# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""
Compatibility shim layer.

Replaces all ``ernie_core.models.utils``, ``ernie_core.models.comm_utils``,
``ernie_core.models.fp8_linear``, ``ernie_core.models.sequence_parallel_utils``,
``ernie_core.models.refined_recompute.*`` and ``ernie_core.ops.triton_ops.*``
imports so that the standalone package has **zero** dependency on ernie-core.

Functions / classes that are non-trivial are faithfully copied here.
Optional C-extension / third-party imports are guarded with try/except.
"""

import importlib
import logging
import os
import queue
from functools import partial
from typing import Any, Callable, List

import numpy as np
import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# try_import — generic lazy importer (from ernie_core.models.utils)
# ---------------------------------------------------------------------------
def try_import(modules, name=None, fail_msg=None):
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for m in modules:
        assert isinstance(m, str), m
        try:
            m = importlib.import_module(m)
        except ImportError:
            m = None
        if m is not None:
            if name is None:
                return m
            elif hasattr(m, name):
                return getattr(m, name)
    if fail_msg is not None:
        logger.warning(fail_msg)


# ---------------------------------------------------------------------------
# External optional libraries (conditional)
# ---------------------------------------------------------------------------
moe_permutation = try_import(["moe_permutation"], fail_msg="moe_permutation is not installed.")

TDU = try_import(["paddlefleet.extensions.ops", "paddlefleet.ops", "TokenDispatcherUtils"])

deep_gemm = try_import(["paddlefleet.ops.deep_gemm", "deep_gemm"])
if deep_gemm is not None:
    if not hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
        setattr(deep_gemm, "gemm_fp8_fp8_bf16_nt", deep_gemm.fp8_gemm_nt)
    if not hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
        setattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous", deep_gemm.m_grouped_fp8_gemm_nt_contiguous)

deep_ep = try_import(["paddlefleet.ops.deep_ep", "paddle.distributed.communication.deep_ep"])
HAVE_DEEP_EP = deep_ep is not None

swiglu = try_import(["paddle.nn.functional", "paddle.incubate.nn.functional"], name="swiglu")

try:
    import kitchen
except ImportError:
    kitchen = None

try:
    import FusedQuantOps as FQO
except ImportError:
    FQO = None

try:
    from paddle.incubate.nn.functional import fused_transpose_wlch_split_quant
except ImportError:
    fused_transpose_wlch_split_quant = None

try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None

try:
    from bincount_ops import int_bincount
except ImportError:
    int_bincount = None

try:
    from custom_setup_ops import matmul_bwd
except ImportError:
    matmul_bwd = None

try:
    import fused_ln as fused
except ImportError:
    fused = None

try:
    import moe_ops
except ImportError:
    moe_ops = None

try:
    import moe_ops_fp8
except ImportError:
    moe_ops_fp8 = None

try:
    import FusedQuantOps
except ImportError:
    FusedQuantOps = None

try:
    from moe_combine import moe_combine_no_weight
except ImportError:
    moe_combine_no_weight = None


# ---------------------------------------------------------------------------
# kitchen_quant / kitchen_fp8_gemm (from ernie_core.models.fp8_linear)
# ---------------------------------------------------------------------------
def _kitchen_quant_stub(*args, **kwargs):
    raise NotImplementedError("kitchen_quant requires `kitchen` library")


def _kitchen_fp8_gemm_stub(*args, **kwargs):
    raise NotImplementedError("kitchen_fp8_gemm requires `kitchen` library")


if kitchen is not None and hasattr(kitchen, "quant"):
    kitchen_quant = kitchen.quant
else:
    kitchen_quant = _kitchen_quant_stub

if kitchen is not None and hasattr(kitchen, "fp8_gemm"):
    kitchen_fp8_gemm = kitchen.fp8_gemm
else:
    kitchen_fp8_gemm = _kitchen_fp8_gemm_stub


# ---------------------------------------------------------------------------
# global_training_logs helpers (from ernie_core.models.utils)
# ---------------------------------------------------------------------------
try:
    from src.utils.misc import global_training_logs
except (ImportError, ModuleNotFoundError):
    global_training_logs = {}


def get_global_training_logs():
    try:
        from src.utils.misc import global_training_logs as _gt

        return _gt
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from rl.utils.stat_utils import global_training_logs as _gt

        return _gt
    except (ImportError, ModuleNotFoundError):
        pass
    return {}


def global_training_logs_enabled():
    gtl = get_global_training_logs()
    return isinstance(gtl, dict) or gtl.is_enabled()


def global_moe_balance_training_logs_enabled():
    gtl = get_global_training_logs()
    return isinstance(gtl, dict) or gtl.is_moe_balance_logs_enabled()


# ---------------------------------------------------------------------------
# manual_backward and helpers (from ernie_core.models.utils)
# ---------------------------------------------------------------------------
def is_tensor(data):
    return isinstance(data, (paddle.Tensor, paddle.base.core.eager.Tensor))


def detach_and_requires_grad_(*args):
    ret = [a.detach() if is_tensor(a) else a for a in args]
    for r, a in zip(ret, args):
        if is_tensor(a):
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(PyLayer):
    @staticmethod
    def forward(ctx, input):
        if input.is_contiguous():
            fake_output = paddle.Tensor()
            fake_output.get_tensor()._share_data_nocheck_with(input.get_tensor())
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [FakeClone.apply(a) if is_tensor(a) else a for a in detached_args]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [FakeClone.apply(o) for o in out if o is not None]
    for o in out_cached:
        o._clear_dataptr()
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        grad, out_cached = zip(*[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient])
        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if is_tensor(t)])

    return bwd_f, out


# ---------------------------------------------------------------------------
# fake_scatter_add / FakeGather / FusedUnpermutation (from ernie_core.models.utils)
# ---------------------------------------------------------------------------
def fake_scatter_add(x, index, value):
    x_tmp = paddle.zeros_like(x)
    x_tmp.scatter_(index, value, overwrite=False)
    x.add_(x_tmp)
    return x


class FakeGather(PyLayer):
    @staticmethod
    def forward(ctx, input, indices):
        assert len(indices.shape) == 1
        ctx.save_for_backward(indices, input.shape)
        if indices.shape[0] == 0:
            out_shape = input.shape
            out_shape[0] = 0
            return paddle.zeros(out_shape, dtype=input.dtype)
        return paddle.index_select(input, axis=0, index=indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, input_shape = ctx.saved_tensor()
        grad_input = paddle.zeros(input_shape, dtype=grad_output.dtype)
        if indices.shape[0] != 0:
            grad_input = fake_scatter_add(grad_input, indices.unsqueeze(-1), grad_output)
        return grad_input, None


class FusedUnpermutation(PyLayer):
    @staticmethod
    def forward(ctx, output_tokens, permuted_tokens, token_permuted_indices, dispatched_probs, prob_permuted_indices):
        assert token_permuted_indices.stop_gradient
        if dispatched_probs is not None:
            assert prob_permuted_indices is not None and prob_permuted_indices.stop_gradient

        output_tokens.stop_gradient = False
        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            output_tokens = moe_permutation.unpermute(
                output_tokens,
                permuted_tokens,
                token_permuted_indices,
                dispatched_probs,
                prob_permuted_indices,
            )
        else:
            output_tokens = FakeClone.apply(output_tokens)

        ctx.save_for_backward(permuted_tokens, token_permuted_indices, dispatched_probs, prob_permuted_indices)
        return output_tokens

    @staticmethod
    def backward(ctx, output_tokens_grad):
        permuted_tokens, token_permuted_indices, dispatched_probs, prob_permuted_indices = ctx.saved_tensor()
        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            permuted_tokens_grad, dispatched_probs_grad = moe_permutation.unpermute_grad(
                output_tokens_grad,
                permuted_tokens,
                token_permuted_indices,
                dispatched_probs,
                prob_permuted_indices,
            )
        else:
            permuted_tokens_grad = paddle.zeros_like(permuted_tokens)
            if dispatched_probs is not None:
                dispatched_probs_grad = paddle.zeros_like(dispatched_probs)

        if dispatched_probs is None:
            return output_tokens_grad, permuted_tokens_grad, None
        else:
            return output_tokens_grad, permuted_tokens_grad, None, dispatched_probs_grad, None


# ---------------------------------------------------------------------------
# Sequence-parallel stubs (from ernie_core.models.sequence_parallel_utils)
# ---------------------------------------------------------------------------
class ScatterOp(PyLayer):
    """Scatter input across the model-parallel group (SP scatter)."""

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        if group is None or group.world_size <= 1:
            return input
        world_size = group.world_size
        rank = group.rank
        seq_len = input.shape[0]
        assert seq_len % world_size == 0
        chunk_size = seq_len // world_size
        return input[rank * chunk_size : (rank + 1) * chunk_size].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or ctx.group.world_size <= 1:
            return grad_output
        output_list = []
        paddle.distributed.all_gather(output_list, grad_output, group=ctx.group)
        return paddle.concat(output_list, axis=0)


class GatherOp(PyLayer):
    """Gather input across the model-parallel group (SP gather)."""

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        if group is None or group.world_size <= 1:
            return input
        output_list = []
        paddle.distributed.all_gather(output_list, input, group=ctx.group)
        return paddle.concat(output_list, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or ctx.group.world_size <= 1:
            return grad_output
        world_size = ctx.group.world_size
        rank = ctx.group.rank
        seq_len = grad_output.shape[0]
        assert seq_len % world_size == 0
        chunk_size = seq_len // world_size
        return grad_output[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


class AllGatherOp(PyLayer):
    """AllGather with no-op backward (used for gate)."""

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        if group is None or group.world_size <= 1:
            return input
        output_list = []
        paddle.distributed.all_gather(output_list, input, group=ctx.group)
        return paddle.concat(output_list, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ContextParallelAllGatherOp(PyLayer):
    """Context-parallel AllGather stub."""

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        if group is None or group.world_size <= 1:
            return input
        output_list = []
        paddle.distributed.all_gather(output_list, input, group=ctx.group)
        return paddle.concat(output_list, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ---------------------------------------------------------------------------
# comm_utils stubs (from ernie_core.models.comm_utils)
# ---------------------------------------------------------------------------
class PrintOp:
    """No-op print stub."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class _ProfileContextManager:
    """No-op profiling context manager."""

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def profile(name=""):
    return _ProfileContextManager(name)


# ---------------------------------------------------------------------------
# Refined-recompute stubs
# ---------------------------------------------------------------------------
class RefinedRcomputeMoEGateDispatch:
    """Stub for ernie_core.models.refined_recompute.moe_gate_dispatch"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("RefinedRcomputeMoEGateDispatch not available in standalone package")


class RefinedRcomputeMoECombine:
    """Stub for ernie_core.models.refined_recompute.moe_combine"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("RefinedRcomputeMoECombine not available in standalone package")


class _GlobalRRQueueLog:
    """Stub for ernie_core.models.refined_recompute.queue_check.global_rr_queue_log"""

    def update(self, queue_obj, name):
        pass


global_rr_queue_log = _GlobalRRQueueLog()


# ---------------------------------------------------------------------------
# dispatch_to decorator (from ernie_core.ops.triton_ops.utils)
# ---------------------------------------------------------------------------
def is_torch_compat_available() -> bool:
    return hasattr(paddle, "enable_compat")


def dispatch_to(dispatch_fn, *, cond=None):
    if cond is None:
        cond = lambda self, *args, **kwargs: True

    def decorator(fn):
        def wrapper(*args, **kwargs):
            if cond(*args, **kwargs) and is_torch_compat_available():
                return dispatch_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        wrapper.__original_fn__ = fn
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# FusedMoETopk / routing_map_forward stubs
# These are Triton-accelerated ops. In standalone mode, the Python fallback
# path in DeepEPMOELayer.calc_topk_probs_indices / routing_map_forward is
# always used unless the user has triton+paddle-compat installed.
# ---------------------------------------------------------------------------
try:
    from ernie_core.ops.triton_ops.fused_moe_topk import FusedMoETopk, routing_map_forward
except ImportError:
    # Stubs — the dispatch_to decorator will fall through to the Python impl.
    class FusedMoETopk:
        @staticmethod
        def apply(*args, **kwargs):
            raise NotImplementedError("FusedMoETopk requires triton")

    def routing_map_forward(*args, **kwargs):
        raise NotImplementedError("routing_map_forward requires triton")


# ---------------------------------------------------------------------------
# paddleformers timer stub
# ---------------------------------------------------------------------------
def get_timers():
    """Stub for paddleformers.trainer.plugins.timer.get_timers"""

    class _DummyTimers:
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

        def start(self, *a, **kw):
            pass

        def stop(self, *a, **kw):
            pass

    return _DummyTimers()


# ---------------------------------------------------------------------------
# paddleformers env-device helper
# ---------------------------------------------------------------------------
def get_env_device():
    """Stub for paddleformers.utils.tools.get_env_device"""
    return os.environ.get("PADDLE_DEVICE_TYPE", "gpu")
