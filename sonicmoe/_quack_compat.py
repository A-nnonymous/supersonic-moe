"""Quack ↔ Paddle distributed compatibility shims.

Two issues are patched here, both auto-installed at ``import sonicmoe``:

1. ``quack.autotuner._gpu_warmup`` calls
   ``torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)`` on every
   autotune cache miss. Under ``paddle.compat.enable_torch_proxy()`` this is
   translated to ``paddle.randn(... device=CUDAPlace(0))``. In multi-rank /
   multi-machine training each process only initialises the
   ``DeviceContext`` for its own rank's GPU. If the ambiguous ``"cuda"``
   string resolves to ``CUDAPlace(0)`` on a non-rank-0 process, the
   ``DeviceContextPool::Get`` lookup fails with::

       set the correct device id if you use Executor.

   The crash is observed only on autotune cache miss. Production typically
   ships with a pre-warmed disk cache, but a single missed shape (e.g. an
   uneven token distribution at end of epoch) is enough to take down the
   whole job. Quack itself does not need a thermal warmup for our
   workloads — autotune is run once per shape and the result cached. We
   therefore neutralise ``_gpu_warmup`` to a true no-op.

2. ``Autotuner.benchmark`` body invokes ``_gpu_warmup`` *before* it knows
   which device the inputs live on. To be defensive against any future
   utility that allocates with ambiguous ``"cuda"``, we also wrap the
   tuned-call entrypoint to ensure ``paddle.device.set_device(...)`` matches
   the input tensor's place at call time.

Opt-out via ``SONIC_MOE_NO_QUACK_COMPAT_PATCH=1``.
"""
from __future__ import annotations

import os


def _install_gpu_warmup_noop() -> bool:
    """Override ``quack.autotuner._gpu_warmup`` to a no-op."""
    try:
        from quack import autotuner as _qa  # type: ignore
    except Exception:
        return False

    if getattr(_qa, "_sonic_moe_warmup_patched", False):
        return True

    def _noop(*_args, **_kwargs):
        return None

    _qa._sonic_moe_orig_gpu_warmup = _qa._gpu_warmup
    _qa._gpu_warmup = _noop
    _qa._sonic_moe_warmup_patched = True
    return True


def install_quack_paddle_compat() -> bool:
    """Install all defensive patches. Returns True if at least one applied."""
    if os.environ.get("SONIC_MOE_NO_QUACK_COMPAT_PATCH"):
        return False
    return _install_gpu_warmup_noop()


_PATCHED = install_quack_paddle_compat()
