"""JIT warmup: pre-compile all CuTe + Triton kernels before training begins.

Usage::

    from sonicmoe.jit_warmup import warmup_jit

    # Dynamic-dim mode (recommended after compile_key fix):
    # A single warmup covers ALL seqlens.
    warmup_jit(E=8, H=3072, I=1536, device="cuda")

    # Explicit seqlen list (for legacy compile keys or perf testing):
    warmup_jit(E=8, H=3072, I=1536, device="cuda",
               total_K_list=[1024, 4096, 16384])

Environment variables:
    SONIC_MOE_CACHE_DIR : str
        Override cache root path.
    QUACK_COMPILE_WORKERS : int
        Number of parallel workers for CuTe compilation (default: CPU count).
    SONIC_MOE_JIT_VERBOSE : "1"
        Enable verbose compile logging.
"""

import logging
import os
import time

import torch

_log = logging.getLogger("sonicmoe.jit")


def warmup_jit(
    E: int,
    H: int,
    I: int,
    device: torch.device | str = "cuda",
    *,
    fp8: bool = True,
    total_K_list: list[int] | None = None,
    max_workers: int = 0,
    cache_dir: str | None = None,
):
    """Pre-compile all CuTe + Triton kernels with dummy data.

    After the compile_key fix (dynamic dims removed from cache key),
    a single warmup shape covers ALL future seqlens.  Pass
    ``total_K_list`` only if you need per-shape kernel variants.

    Parameters
    ----------
    E, H, I : int
        Model dimensions (num_experts, hidden_size, intermediate_size).
    device : str or torch.device
        CUDA device to warmup on.
    fp8 : bool
        Whether to warmup FP8 codepath (default True).
    total_K_list : list[int], optional
        Explicit total_K values to warmup.  Default: single shape
        ``E * 128`` (one tile per expert, minimal memory).
    max_workers : int
        Parallel CuTe compile workers (0 = auto).
    cache_dir : str, optional
        Override SONIC_MOE_CACHE_DIR.
    """
    from sonicmoe.cache_manager import setup_cache

    if cache_dir:
        setup_cache(cache_dir)
    else:
        setup_cache()

    if max_workers == 0:
        max_workers = min(os.cpu_count() or 8, 16)
    os.environ["QUACK_COMPILE_WORKERS"] = str(max_workers)

    if total_K_list is None:
        total_K_list = [E * 128]

    _log.info(
        f"[Warmup] E={E}, H={H}, I={I}, fp8={fp8}, "
        f"shapes={len(total_K_list)}, workers={max_workers}"
    )
    t0 = time.perf_counter()

    for i, total_K in enumerate(total_K_list):
        t1 = time.perf_counter()
        _warmup_single(E, H, I, total_K, device, fp8)
        dt = time.perf_counter() - t1
        _log.info(
            f"[Warmup] shape {i + 1}/{len(total_K_list)} "
            f"(total_K={total_K}) done in {dt:.1f}s"
        )

    total = time.perf_counter() - t0
    _log.info(f"[Warmup] All done in {total:.1f}s")


def _warmup_single(E: int, H: int, I: int, total_K: int, device, fp8: bool):
    """Trigger one fwd+bwd pass with dummy data to compile all kernels."""
    from sonicmoe.functional import _refresh_fp8_config

    device = torch.device(device)

    # Ensure FP8 config is set before we touch any kernels
    _refresh_fp8_config()

    # Build minimal routing metadata: uniform tokens per expert
    tokens_per_expert = total_K // E
    remainder = total_K - tokens_per_expert * E

    efo = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        efo[i + 1] = efo[i] + tokens_per_expert + (1 if i < remainder else 0)
    TK = int(efo[-1].item())
    assert TK == total_K

    # T = number of unique tokens (before expert duplication)
    # Use T = TK for simplicity (1:1 mapping, as if topk=1)
    T = TK

    # Dummy tensors
    x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device=device)

    # Identity routing: token i -> expert i%E
    x_gather_idx = torch.arange(TK, device=device, dtype=torch.int32)
    s_scatter_idx = torch.arange(TK, device=device, dtype=torch.int32)
    s_reverse_scatter_idx = torch.arange(TK, device=device, dtype=torch.int32)
    naept_off = torch.arange(T + 1, device=device, dtype=torch.int32).clamp(max=TK)
    scores = torch.ones(TK, dtype=torch.float32, device=device)

    # Use the autograd Function directly (avoids Paddle dependency)
    from sonicmoe.functional import _UpProjection, _DownProjection
    from sonicmoe.functional import ActivationType

    x.requires_grad_(True)

    # Forward
    up_out = _UpProjection.apply(
        x, w1, None,
        efo, TK, None, 0,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, naept_off,
        True,  # is_varlen_K
        ActivationType.SWIGLU,
        False,  # inference
        False,  # low_prec
    )

    down_out = _DownProjection.apply(
        up_out, None, w2, None,
        scores, s_scatter_idx,
        efo, T, None, 0,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, naept_off,
        True,  # is_varlen_K
        ActivationType.SWIGLU,
        None,  # pre_quant_activations
    )

    # Backward — triggers wgrad compilation
    loss = down_out.sum()
    loss.backward()

    torch.cuda.synchronize(device)
