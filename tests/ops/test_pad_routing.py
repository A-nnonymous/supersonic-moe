"""Axiomatic correctness test for route-level padding.

Verifies that _pad_routing_metadata preserves every token's computation:
no token dropped, no token misdirected, no phantom contribution from padding.

Gold reference: pure-Python per-expert matmul + interleaved SwiGLU in float64.
The gold operates on the ORIGINAL (unpadded) routing metadata.  The padded
path must produce near-identical results for every real token (float64
machine-epsilon tolerance, since GPU matmul tiling differs across shapes).

Shape: T=10, H=16, I=8, E=4, K=2 — small enough to enumerate every value.
Every expert gets ~5 tokens, all non-128-aligned → padding is triggered.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _interleaved_swiglu(z: torch.Tensor) -> torch.Tensor:
    """SwiGLU on interleaved layout: z(N, 2I) → y1(N, I).
    Columns 0,2,4,...=gate; 1,3,5,...=up."""
    gate = z[:, 0::2]
    up = z[:, 1::2]
    return F.silu(gate) * up


def _gold_moe_forward(
    x: torch.Tensor,       # (T, H) float64
    w1: torch.Tensor,      # (2I, H, E) float64 interleaved
    w2: torch.Tensor,      # (H, I, E) float64
    efo: list[int],        # (E+1,) expert_frequency_offset
    x_gather_idx: list[int],       # (TK,)
    s_reverse_scatter_idx: list[int],  # (TK,)
    topk_scores_flat: list[float],     # (TK,)
    T: int, E: int, K: int,
) -> torch.Tensor:
    """Pure-Python gold MoE forward on ORIGINAL routing.

    Pipeline for each token t, expert assignment k:
      1. Gather: x_expert_sorted[p] = x[x_gather_idx[p]]
      2. Per-expert up-proj: z = x_gathered @ w1_e.T  (interleaved)
      3. Interleaved SwiGLU: y1 = swiglu(z)
      4. Per-expert down-proj: y2 = y1 @ w2_e.T
      5. Score-weighted scatter: o[t] += score * y2[sorted_pos_of(t,k)]
    """
    TK = len(x_gather_idx)
    I = w1.shape[0] // 2
    H = x.shape[1]

    # Gather
    x_gathered = x[x_gather_idx]  # (TK, H)

    # Per-expert up-proj + SwiGLU
    y1 = torch.zeros(TK, I, dtype=x.dtype, device=x.device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1[:, :, e].T  # (n, 2I)
        y1[s:end] = _interleaved_swiglu(z_e)

    # Per-expert down-proj
    y2 = torch.zeros(TK, H, dtype=x.dtype, device=x.device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        y2[s:end] = y1[s:end] @ w2[:, :, e].T  # (n, H)

    # Score-weighted scatter
    o = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    for flat_idx in range(TK):
        sorted_pos = s_reverse_scatter_idx[flat_idx]
        t = flat_idx // K
        o[t] += topk_scores_flat[flat_idx] * y2[sorted_pos]

    return o


def _gold_moe_forward_padded(
    x, w1, w2, padded_efo, padded_x_gather, padded_s_reverse,
    padded_scores_flat, T, E, K, TK_original,
) -> torch.Tensor:
    """Same gold but on PADDED routing.  If padding is correct,
    result must equal the unpadded gold exactly (float64)."""
    padded_total = len(padded_x_gather)
    I = w1.shape[0] // 2
    H = x.shape[1]

    x_gathered = x[padded_x_gather]  # (padded_total, H)

    y1 = torch.zeros(padded_total, I, dtype=x.dtype, device=x.device)
    for e in range(E):
        s, end = padded_efo[e], padded_efo[e + 1]
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1[:, :, e].T
        y1[s:end] = _interleaved_swiglu(z_e)

    y2 = torch.zeros(padded_total, H, dtype=x.dtype, device=x.device)
    for e in range(E):
        s, end = padded_efo[e], padded_efo[e + 1]
        if s >= end:
            continue
        y2[s:end] = y1[s:end] @ w2[:, :, e].T

    # Score-weighted scatter — only first T*K entries of s_reverse/scores matter
    o = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    for flat_idx in range(T * K):
        sorted_pos = padded_s_reverse[flat_idx]
        t = flat_idx // K
        o[t] += padded_scores_flat[flat_idx] * y2[sorted_pos]

    return o


def test_pad_routing_axiomatic():
    """Axiomatic: padded gold == unpadded gold, token by token, in float64."""
    T, H, I, E, K = 10, 16, 8, 4, 2
    device = "cuda"

    torch.manual_seed(42)
    x = torch.randn(T, H, device=device, dtype=torch.float64)
    # w1 interleaved (2I, H, E), w2 (H, I, E)
    w1 = torch.randn(2 * I, H, E, device=device, dtype=torch.float64) * 0.1
    w2 = torch.randn(H, I, E, device=device, dtype=torch.float64) * 0.1

    # Deterministic routing: token t selects experts (t*K+k) % E
    topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        for k in range(K):
            topk_indices[t, k] = (t * K + k) % E
    logits = torch.randn(T, E, device=device, dtype=torch.float64)
    gathered = logits.gather(1, topk_indices.long())
    topk_scores = F.softmax(gathered, dim=-1).float()

    # Compute routing metadata (CPU, exact)
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    TK = T * K
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    efo = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, efo,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    efo_list = efo.cpu().tolist()
    xg_list = x_gather_idx.cpu().tolist()
    sr_list = s_reverse_scatter_idx.cpu().tolist()
    scores_flat = topk_scores.flatten().cpu().tolist()

    # ── Axiom 1: verify segment lengths are non-128-aligned ──
    seg_lens = [efo_list[i + 1] - efo_list[i] for i in range(E)]
    assert any(s % 128 != 0 for s in seg_lens), \
        f"Expected non-aligned segments, got {seg_lens}"

    # ── Gold on original routing ──
    o_gold = _gold_moe_forward(
        x, w1, w2, efo_list, xg_list, sr_list, scores_flat, T, E, K,
    )

    # ── Apply _pad_routing_metadata ──
    from sonicmoe.functional import _pad_routing_metadata
    (pefo, pxg, pss, psr, pscores, pt, padded) = _pad_routing_metadata(
        efo, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
        topk_scores.flatten().to(device), TK, T, E, K,
    )
    assert padded, "Expected padding to be triggered"

    pefo_list = pefo.cpu().tolist()
    pxg_list = pxg.cpu().tolist()
    psr_list = psr.cpu().tolist()
    pscores_list = pscores.cpu().tolist()

    # ── Axiom 2: padded segments are all 128-aligned ──
    padded_segs = [pefo_list[i + 1] - pefo_list[i] for i in range(E)]
    for e, s in enumerate(padded_segs):
        assert s % 128 == 0, f"Expert {e} segment {s} not 128-aligned"

    # ── Axiom 3: all TK original tokens are preserved ──
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import _get_padding_plan
    _, _, _, dst_idx = _get_padding_plan(efo, TK)
    reconstructed_xg = pxg[dst_idx].cpu().tolist()
    assert reconstructed_xg == xg_list, \
        f"x_gather_idx not preserved at dst_idx positions"

    # ── Axiom 4: original scores preserved, padding scores = 0 ──
    assert pscores_list[:TK] == scores_flat, "Original scores not preserved"
    assert all(s == 0.0 for s in pscores_list[TK:]), "Padding scores not zero"

    # ── Axiom 5: s_reverse_scatter_idx preserved (remapped) ──
    # For each flat_idx in [0, TK), padded_s_reverse[flat_idx] should be
    # dst_idx[original_s_reverse[flat_idx]] — the remapped position.
    dst_idx_list = dst_idx.cpu().tolist()
    for flat_idx in range(TK):
        expected = dst_idx_list[sr_list[flat_idx]]
        actual = psr_list[flat_idx]
        assert actual == expected, \
            f"s_reverse_scatter_idx[{flat_idx}]: expected {expected}, got {actual}"

    # ── Axiom 6 (THE KEY): padded gold == original gold, near-exact in float64 ──
    #     GPU matmul tiling/reduction order depends on shape; padded expert
    #     segments are longer, producing ULP-level float64 differences.
    _F64_ATOL = 1e-13
    o_padded = _gold_moe_forward_padded(
        x, w1, w2, pefo_list, pxg_list, psr_list,
        pscores_list, T, E, K, TK,
    )

    # Per-token comparison
    max_diff = 0.0
    for t in range(T):
        diff = (o_gold[t] - o_padded[t]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < _F64_ATOL, \
            f"Token {t}: gold vs padded diff = {diff} (threshold {_F64_ATOL})"

    print(f"PASS: {T} tokens × {K} experts/tok, E={E}, "
          f"segments={seg_lens} → padded={padded_segs}")
    print(f"  All {TK} real tokens preserved, 0 dropped, 0 phantom")
    print(f"  Padded gold == original gold: max diff = {max_diff:.2e} (float64)")
    print(f"  N_pad={pt - TK} padding rows, all with score=0")


if __name__ == "__main__":
    test_pad_routing_axiomatic()
