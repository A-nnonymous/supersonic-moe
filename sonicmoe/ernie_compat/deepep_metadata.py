"""DeepEP → SonicMoE metadata conversion (zero argsort, zero sync).

DeepEP dispatch produces tokens already sorted by expert, so routing metadata
is trivially identity permutations + cumsum over ``tokens_per_expert``.

Padding strategy: **route-level padding** (matching the FP8 frontier design).
  - Padding rows use gather index 0 (arbitrary valid row) with score=0.
  - x is NOT modified — no sentinel rows appended.
  - This is identical to ``_pad_routing_metadata`` in functional/__init__.py.

Two implementations:
  1. **CUDA kernel** (V2): host computes prefix-sum (trivial on CPU for E ≤ 384),
     then launches 1-block-per-expert fill kernel with vectorized int4/float4
     stores.  No ``item()`` DtoH, no barriers, no over-allocation.
  2. **Python fallback**: pure PyTorch ops, used when CUDA kernel is not compiled.

``deepep_to_sonic_metadata`` dispatches to CUDA if available.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Try to import the JIT-compiled CUDA kernel
_HAS_CUDA_KERNEL = False
try:
    from sonicmoe.ernie_compat.deepep_metadata_cuda import deepep_metadata_cuda
    _HAS_CUDA_KERNEL = True
except Exception:
    pass


# ── Identity-arange cache (per-device) ──────────────────────────────────────
# s_scatter_idx, s_reverse_scatter_idx, num_activated_expert_per_token_offset
# are pure identity sequences. They depend only on (TK_padded, device, dtype)
# and never on routing — caching them eliminates 3 arange allocations per iter.
_ARANGE_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_arange(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Return a cached identity arange of length n.

    The returned tensor is a read-only handle — callers must NOT mutate it.
    """
    key = (n, dtype, str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None or t.shape[0] < n:
        t = torch.arange(max(n, 1), dtype=dtype, device=device)
        _ARANGE_CACHE[key] = t
    if t.shape[0] == n:
        return t
    return t[:n]


def deepep_to_sonic_metadata(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Zero-sync metadata conversion: DeepEP format → SonicMoE routing tensors.

    Uses **route-level padding** (frontier design): padding rows use gather
    index 0 with score=0.  x is NOT modified (no sentinel rows needed).

    Dispatches to CUDA kernel if available, otherwise falls back to Python.

    Parameters
    ----------
    tokens_per_expert : list[int] or Tensor of shape [E]
        Per-expert token counts from ``DeepEP buffer.dispatch()``.
    T : int
        Total number of real tokens (``sum(tokens_per_expert)``).
    E : int
        Number of local experts.
    device : str or torch.device
        Target device for output tensors.
    block : int
        Alignment block size (128 for FP8 GEMM).

    Returns
    -------
    expert_frequency_offset : Tensor [E+1] int32
    x_gather_idx : Tensor [TK_padded] int32
    s_scatter_idx : Tensor [TK_padded] int32
    s_reverse_scatter_idx : Tensor [TK_padded] int32
    num_activated_expert_per_token_offset : Tensor [T+1] int32
    router_scores : Tensor [TK_padded] float32
    TK_padded : int
    total_pad_rows : int
        Number of padding slots added (for reference; x is NOT padded).
    """
    if _HAS_CUDA_KERNEL:
        return _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, device, block)
    return _deepep_to_sonic_metadata_python(tokens_per_expert, T, E, device, block)


# ── Host-side prefix-sum (shared by CUDA and Python paths) ──────────────────

def _host_prefix_sum(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    E: int,
    T: int,
    block: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], int, int]:
    """Compute all per-expert metadata on CPU.  O(E), trivial for E ≤ 384."""
    if isinstance(tokens_per_expert, torch.Tensor):
        tpe = tokens_per_expert.tolist()
    else:
        tpe = list(tokens_per_expert)

    # Per-expert arrays
    offsets = [0]       # [E+1] padded cumulative offsets
    seg_starts = []     # [E] = offsets[e]
    seg_lens = []       # [E] = offsets[e+1] - offsets[e]
    real_counts = []    # [E] = tpe[e]
    real_bases = []     # [E] cumulative real token offset
    pad_bases = []      # [E] = T + cumulative padding offset

    real_cum = 0
    pad_cum = 0

    for i in range(E):
        count = tpe[i]
        padded = ((count + block - 1) // block * block) if count > 0 else 0
        pad = padded - count

        seg_starts.append(offsets[-1])
        seg_lens.append(padded)
        real_counts.append(count)
        real_bases.append(real_cum)
        pad_bases.append(T + pad_cum)

        offsets.append(offsets[-1] + padded)
        real_cum += count
        pad_cum += pad

    TK_padded = offsets[-1]
    total_pad_rows = pad_cum

    return offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases, TK_padded, total_pad_rows


# ── CUDA implementation (V2: host prefix-sum + multi-block fill) ─────────────

def _deepep_to_sonic_metadata_cuda(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """CUDA V2: host prefix-sum → 1-block-per-expert fill kernel.

    Eliminates: item() DtoH sync, __syncthreads barrier, over-allocation.
    """
    (offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases,
     TK_padded, total_pad_rows) = _host_prefix_sum(tokens_per_expert, E, T, block)

    # Transfer pre-computed metadata to device in one fused tensor to minimize
    # HtoD launches.  Layout: [offsets(E+1) | seg_starts(E) | seg_lens(E) |
    #                           real_counts(E) | real_bases(E) | pad_bases(E)]
    # Total: 6E+1 int32s — fits in a single 128-byte cacheline for E≤20.
    fused_host = offsets + seg_starts + seg_lens + real_counts + real_bases + pad_bases
    fused_dev = torch.tensor(fused_host, dtype=torch.int32, device=device)

    # Slice views (no copy — shares storage)
    o = E + 1
    expert_freq_offset = fused_dev[:o]
    seg_starts_d  = fused_dev[o:o + E];          o += E
    seg_lens_d    = fused_dev[o:o + E];           o += E
    real_counts_d = fused_dev[o:o + E];           o += E
    real_bases_d  = fused_dev[o:o + E];           o += E
    pad_bases_d   = fused_dev[o:o + E]

    # Allocate exact-size output tensors
    x_gather_idx = torch.empty(TK_padded, dtype=torch.int32, device=device)
    router_scores = torch.empty(TK_padded, dtype=torch.float32, device=device)

    # Launch fill kernel: E blocks × 256 threads
    if TK_padded > 0:
        stream = torch.cuda.current_stream(device).stream_base.raw_stream
        deepep_metadata_cuda(
            expert_freq_offset=expert_freq_offset,
            x_gather_idx=x_gather_idx,
            router_scores=router_scores,
            seg_starts=seg_starts_d,
            seg_lens=seg_lens_d,
            real_counts=real_counts_d,
            real_bases=real_bases_d,
            pad_bases=pad_bases_d,
            E=E,
            stream=stream,
        )

    # Identity scatter indices (cached — never depends on routing)
    s_scatter_idx = _cached_arange(TK_padded, torch.int32, device)
    s_reverse_scatter_idx = s_scatter_idx

    # num_activated_expert_per_token_offset: arange(T_padded+1)
    # Must cover T_padded positions (not T_orig) because _DownProjection
    # uses T_padded for the output tensor and indexes naept[0..T_padded].
    # Each padded position maps to exactly 1 expert (padding has score=0).
    T_for_naept = TK_padded  # T_padded = TK_padded in the DeepEP identity layout
    num_activated_expert_per_token_offset = _cached_arange(
        T_for_naept + 1, torch.int32, device,
    )

    return (
        expert_freq_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        router_scores,
        TK_padded,
        total_pad_rows,
    )


# ── Python fallback ──────────────────────────────────────────────────────────

def _deepep_to_sonic_metadata_python(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pure-Python fallback using PyTorch ops."""
    (offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases,
     TK_padded, total_pad_rows) = _host_prefix_sum(tokens_per_expert, E, T, block)

    expert_frequency_offset = torch.tensor(offsets, dtype=torch.int32, device=device)

    if isinstance(tokens_per_expert, torch.Tensor):
        tpe = tokens_per_expert.tolist()
    else:
        tpe = list(tokens_per_expert)

    # Build x_gather_idx and router_scores
    gather_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []

    for i in range(E):
        rc = real_counts[i]
        pad_count = seg_lens[i] - rc
        if seg_lens[i] == 0:
            continue

        if rc > 0:
            gather_parts.append(
                torch.arange(real_bases[i], real_bases[i] + rc,
                             dtype=torch.int32, device=device)
            )
            score_parts.append(
                torch.ones(rc, dtype=torch.float32, device=device)
            )

        if pad_count > 0:
            # Route-level padding: padding rows gather from row 0 (score=0
            # nullifies contribution).  Matches frontier _pad_routing_metadata.
            gather_parts.append(
                torch.zeros(pad_count, dtype=torch.int32, device=device)
            )
            score_parts.append(
                torch.zeros(pad_count, dtype=torch.float32, device=device)
            )

    if TK_padded == 0:
        x_gather_idx = torch.empty(0, dtype=torch.int32, device=device)
        router_scores = torch.empty(0, dtype=torch.float32, device=device)
    else:
        x_gather_idx = torch.cat(gather_parts)
        router_scores = torch.cat(score_parts)

    s_scatter_idx = _cached_arange(TK_padded, torch.int32, device)
    s_reverse_scatter_idx = s_scatter_idx
    num_activated_expert_per_token_offset = _cached_arange(
        TK_padded + 1, torch.int32, device,
    )

    return (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        router_scores,
        TK_padded,
        total_pad_rows,
    )
