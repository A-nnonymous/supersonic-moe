"""Axiomatic backward-correctness test for route-level padding.

Proves that route-level padding introduces ZERO gradient error:
  - No token misrouted (routing decisions unchanged)
  - No token lost (all T*K contributions present)
  - dz[pad] == 0 exactly (score-gating before dSwiGLU)
  - dw1, dw2, dx negligible diff between padded and unpadded (float64)

Gold reference: pure-Python per-expert matmul + interleaved SwiGLU backward
in float64.  No CUTLASS kernels, no FP8 — tests the padding MATH only.

NOTE: GPU matmul tiling/reduction order depends on problem shape.  Padded
expert segments are longer (e.g. 7→128), so the intermediate reductions may
differ at ULP level.  We use float64 to minimize this, and assert:
  - EXACT zero where score-gating guarantees it (dz[pad], score[pad])
  - Machine-epsilon tolerance (~1e-14) for matmul-dependent values

Shape: T=10, H=64, I=32, E=3, K=2 — every expert gets ~6-7 tokens,
all non-128-aligned, so padding is always triggered.
"""
import os
import sys
from typing import NamedTuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Pure-Python helpers (float64, no kernels)
# ---------------------------------------------------------------------------

def _interleaved_swiglu(z: torch.Tensor) -> torch.Tensor:
    """SwiGLU on interleaved layout: z(N, 2I) -> y1(N, I).
    Columns 0,2,4,...=gate; 1,3,5,...=up."""
    gate = z[:, 0::2]
    up = z[:, 1::2]
    return F.silu(gate) * up


def _d_interleaved_swiglu(z: torch.Tensor, dy1_s: torch.Tensor) -> torch.Tensor:
    """Backward SwiGLU on interleaved layout.

    Args:
        z:     (N, 2I) pre-activation, interleaved gate/up
        dy1_s: (N, I) score-gated gradient of y1 (= dy1 * score)

    Returns:
        dz: (N, 2I) gradient of z, interleaved gate/up
    """
    gate = z[:, 0::2]  # (N, I)
    up = z[:, 1::2]    # (N, I)
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    # d(SiLU(gate)*up)/d_gate = up * sig * (1 + gate*(1-sig))
    d_gate = dy1_s * up * sig * (1.0 + gate * (1.0 - sig))
    # d(SiLU(gate)*up)/d_up = SiLU(gate)
    d_up = dy1_s * silu_gate
    dz = torch.zeros_like(z)
    dz[:, 0::2] = d_gate
    dz[:, 1::2] = d_up
    return dz


# ---------------------------------------------------------------------------
# Gold MoE forward (from test_pad_routing.py)
# ---------------------------------------------------------------------------

def _gold_moe_forward(x, w1, w2, efo, xg, sr, scores_flat, T, E, K):
    """Pure-Python gold forward on ORIGINAL (unpadded) routing."""
    TK = len(xg)
    I = w1.shape[0] // 2
    H = x.shape[1]
    dtype, device = x.dtype, x.device

    x_gathered = x[xg]
    z = torch.zeros(TK, 2 * I, dtype=dtype, device=device)
    y1 = torch.zeros(TK, I, dtype=dtype, device=device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        z[s:end] = x_gathered[s:end] @ w1[:, :, e].T
        y1[s:end] = _interleaved_swiglu(z[s:end])

    y2 = torch.zeros(TK, H, dtype=dtype, device=device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        y2[s:end] = y1[s:end] @ w2[:, :, e].T

    o = torch.zeros(T, H, dtype=dtype, device=device)
    for flat_idx in range(TK):
        sorted_pos = sr[flat_idx]
        t = flat_idx // K
        o[t] += scores_flat[flat_idx] * y2[sorted_pos]
    return o


def _gold_moe_forward_padded(x, w1, w2, pefo, pxg, psr, pscores, T, E, K, TK_orig):
    """Same gold forward but on PADDED routing."""
    padded_total = len(pxg)
    I = w1.shape[0] // 2
    H = x.shape[1]
    dtype, device = x.dtype, x.device

    x_gathered = x[pxg]
    z = torch.zeros(padded_total, 2 * I, dtype=dtype, device=device)
    y1 = torch.zeros(padded_total, I, dtype=dtype, device=device)
    for e in range(E):
        s, end = pefo[e], pefo[e + 1]
        if s >= end:
            continue
        z[s:end] = x_gathered[s:end] @ w1[:, :, e].T
        y1[s:end] = _interleaved_swiglu(z[s:end])

    y2 = torch.zeros(padded_total, H, dtype=dtype, device=device)
    for e in range(E):
        s, end = pefo[e], pefo[e + 1]
        if s >= end:
            continue
        y2[s:end] = y1[s:end] @ w2[:, :, e].T

    o = torch.zeros(T, H, dtype=dtype, device=device)
    for flat_idx in range(T * K):
        sorted_pos = psr[flat_idx]
        t = flat_idx // K
        o[t] += pscores[flat_idx] * y2[sorted_pos]
    return o


# ---------------------------------------------------------------------------
# Gold MoE backward
# ---------------------------------------------------------------------------

def _gold_moe_backward(x, w1, w2, grad_out, efo, xg, ss, scores_flat, T, E, K):
    """Pure-Python gold backward on routing tensors (unpadded OR padded).

    Returns: (dx, dw1, dw2, dz, z)
    All in float64 for bit-exact comparison.
    """
    TK = len(xg)
    I = w1.shape[0] // 2
    H = x.shape[1]
    dtype, device = x.dtype, x.device

    # --- Forward recomputation ---
    x_gathered = x[xg]  # (TK, H)
    z = torch.zeros(TK, 2 * I, dtype=dtype, device=device)
    y1 = torch.zeros(TK, I, dtype=dtype, device=device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        z[s:end] = x_gathered[s:end] @ w1[:, :, e].T
        y1[s:end] = _interleaved_swiglu(z[s:end])

    # --- Backward ---
    # Step 1: Gather grad_out to expert-sorted space
    dout_gathered = grad_out[xg]  # (TK, H)

    # Step 2: Per-expert dy1 = dout_gathered @ w2_e  (w2 is H,I,E)
    dy1 = torch.zeros(TK, I, dtype=dtype, device=device)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        dy1[s:end] = dout_gathered[s:end] @ w2[:, :, e]  # (n,H)@(H,I)

    # Step 3: Score gating BEFORE dSwiGLU (the key invariant)
    s_scores = torch.zeros(TK, dtype=dtype, device=device)
    for i in range(TK):
        s_scores[i] = scores_flat[ss[i]]
    dy1_s = dy1 * s_scores.unsqueeze(1)

    # Step 4: dSwiGLU
    dz = _d_interleaved_swiglu(z, dy1_s)

    # Step 5: y1s = y1 * score (for dw2 weight grad)
    y1s = y1 * s_scores.unsqueeze(1)

    # Step 6: dw2 per expert = dout_gathered[s:e].T @ y1s[s:e]
    dw2 = torch.zeros_like(w2)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        dw2[:, :, e] = dout_gathered[s:end].T @ y1s[s:end]

    # Step 7: dw1 per expert = dz[s:e].T @ x_gathered[s:e]
    dw1 = torch.zeros_like(w1)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        dw1[:, :, e] = dz[s:end].T @ x_gathered[s:end]

    # Step 8: dx scatter-add
    dx = torch.zeros_like(x)
    for e in range(E):
        s, end = efo[e], efo[e + 1]
        if s >= end:
            continue
        for i in range(s, end):
            t = xg[i]
            dx[t] += dz[i] @ w1[:, :, e]  # (2I,) @ (2I, H) -> (H,)

    return dx, dw1, dw2, dz, z


# ---------------------------------------------------------------------------
# Shared test setup
# ---------------------------------------------------------------------------

class Setup(NamedTuple):
    T: int; H: int; I: int; E: int; K: int; TK: int
    x: torch.Tensor; w1: torch.Tensor; w2: torch.Tensor; grad_out: torch.Tensor
    topk_indices: torch.Tensor; topk_scores: torch.Tensor
    # Original routing (CPU lists for gold functions)
    efo_list: list; xg_list: list; ss_list: list; sr_list: list; scores_flat: list
    # Padded routing (CPU lists)
    pefo_list: list; pxg_list: list; pss_list: list; psr_list: list; pscores_list: list
    # GPU tensors for padding analysis
    efo: torch.Tensor; dst_idx: torch.Tensor; padded_total: int
    pad_mask: torch.Tensor  # bool (padded_total,), True = padding position


def _setup() -> Setup:
    T, H, I, E, K = 10, 64, 32, 3, 2
    TK = T * K
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(T, H, device=device, dtype=torch.float64)
    w1 = torch.randn(2 * I, H, E, device=device, dtype=torch.float64) * 0.1
    w2 = torch.randn(H, I, E, device=device, dtype=torch.float64) * 0.1
    grad_out = torch.randn(T, H, device=device, dtype=torch.float64)

    # Deterministic routing
    topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        for k in range(K):
            topk_indices[t, k] = (t * K + k) % E
    logits = torch.randn(T, E, device=device, dtype=torch.float64)
    gathered = logits.gather(1, topk_indices.long())
    topk_scores = F.softmax(gathered, dim=-1).float()

    # Compute routing metadata
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
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
    ss_list = s_scatter_idx.cpu().tolist()
    sr_list = s_reverse_scatter_idx.cpu().tolist()
    scores_flat = topk_scores.flatten().double().cpu().tolist()

    # Apply padding
    from sonicmoe.functional import _pad_routing_metadata
    (pefo, pxg, pss, psr, pscores, pt, padded) = _pad_routing_metadata(
        efo, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
        topk_scores.flatten().to(device), TK, T, E, K,
    )
    assert padded, "Expected padding to be triggered"

    pefo_list = pefo.cpu().tolist()
    pxg_list = pxg.cpu().tolist()
    pss_list = pss.cpu().tolist()
    psr_list = psr.cpu().tolist()
    pscores_list = pscores.double().cpu().tolist()

    # Compute dst_idx and pad_mask
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import _get_padding_plan
    _, _, padded_total, dst_idx = _get_padding_plan(efo, TK)
    is_real = torch.zeros(padded_total, dtype=torch.bool, device=device)
    is_real[dst_idx] = True
    pad_mask = ~is_real

    return Setup(
        T=T, H=H, I=I, E=E, K=K, TK=TK,
        x=x, w1=w1, w2=w2, grad_out=grad_out,
        topk_indices=topk_indices, topk_scores=topk_scores,
        efo_list=efo_list, xg_list=xg_list, ss_list=ss_list,
        sr_list=sr_list, scores_flat=scores_flat,
        pefo_list=pefo_list, pxg_list=pxg_list, pss_list=pss_list,
        psr_list=psr_list, pscores_list=pscores_list,
        efo=efo, dst_idx=dst_idx, padded_total=padded_total,
        pad_mask=pad_mask,
    )


# ===================================================================
# Axiom 1: Token conservation — dst_idx is injective, no collisions
# ===================================================================

def test_token_conservation():
    s = _setup()
    dst = s.dst_idx.cpu().tolist()
    assert len(set(dst)) == s.TK, \
        f"dst_idx not injective: {len(set(dst))} unique vs {s.TK} expected"
    assert all(0 <= d < s.padded_total for d in dst), \
        f"dst_idx out of range [0, {s.padded_total})"
    print(f"PASS: token_conservation — {s.TK} tokens, {len(dst)} mapped, "
          f"all unique in [0, {s.padded_total})")


# ===================================================================
# Axiom 2: Score invariant — original scores preserved, padding = 0
# ===================================================================

def test_score_invariant():
    s = _setup()
    # Original scores preserved (exact, not allclose)
    for i in range(s.TK):
        assert s.pscores_list[i] == s.scores_flat[i], \
            f"Score mismatch at {i}: {s.pscores_list[i]} != {s.scores_flat[i]}"
    # Padding scores exactly zero
    for i in range(s.TK, len(s.pscores_list)):
        assert s.pscores_list[i] == 0.0, \
            f"Padding score at {i} is {s.pscores_list[i]}, expected exact 0.0"
    n_pad = len(s.pscores_list) - s.TK
    print(f"PASS: score_invariant — {s.TK} original scores preserved, "
          f"{n_pad} padding scores == 0.0")


# Float64 machine epsilon tolerance for matmul-derived comparisons.
# GPU matmul tiling differs between shapes (e.g. 7-row vs 128-row expert
# segments), causing ULP-level differences in float64.
_F64_ATOL = 1e-13


# ===================================================================
# Axiom 3: Forward near-exact — padded output == unpadded output
# ===================================================================

def test_forward_bit_exact():
    s = _setup()
    o_gold = _gold_moe_forward(
        s.x, s.w1, s.w2, s.efo_list, s.xg_list, s.sr_list,
        s.scores_flat, s.T, s.E, s.K,
    )
    o_padded = _gold_moe_forward_padded(
        s.x, s.w1, s.w2, s.pefo_list, s.pxg_list, s.psr_list,
        s.pscores_list, s.T, s.E, s.K, s.TK,
    )
    max_diff = 0.0
    for t in range(s.T):
        diff = (o_gold[t] - o_padded[t]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < _F64_ATOL, \
            f"Token {t}: forward diff = {diff} (threshold {_F64_ATOL})"
    print(f"PASS: forward_bit_exact — {s.T} tokens, max diff = {max_diff:.2e} "
          f"(threshold {_F64_ATOL})")


# ===================================================================
# Axiom 4: dz[pad] == 0 exactly (score-gating before dSwiGLU)
# ===================================================================

def test_backward_dz_zero_at_padding():
    s = _setup()
    _, _, _, dz_padded, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.pefo_list, s.pxg_list, s.pss_list, s.pscores_list,
        s.T, s.E, s.K,
    )
    # dz has shape (padded_total, 2I) — check padding rows are exactly zero
    dz_at_pad = dz_padded[s.pad_mask]
    max_val = dz_at_pad.abs().max().item() if dz_at_pad.numel() > 0 else 0.0
    assert max_val == 0.0, \
        f"dz at padding positions: max abs = {max_val} (expected exact 0.0)"
    n_pad = s.pad_mask.sum().item()
    n_real = (~s.pad_mask).sum().item()
    print(f"PASS: dz_zero_at_padding — {n_pad} padding rows all == 0.0, "
          f"{n_real} real rows computed normally")


# ===================================================================
# Axiom 5: dw1 bit-exact between padded and unpadded
# ===================================================================

def test_backward_dw1_bit_exact():
    s = _setup()
    _, dw1_gold, _, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.efo_list, s.xg_list, s.ss_list, s.scores_flat,
        s.T, s.E, s.K,
    )
    _, dw1_padded, _, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.pefo_list, s.pxg_list, s.pss_list, s.pscores_list,
        s.T, s.E, s.K,
    )
    diff = (dw1_gold - dw1_padded).abs().max().item()
    assert diff < _F64_ATOL, \
        f"dw1 diff = {diff} (threshold {_F64_ATOL})"
    print(f"PASS: dw1_bit_exact — shape {tuple(dw1_gold.shape)}, max diff = {diff:.2e}")


# ===================================================================
# Axiom 6: dw2 bit-exact between padded and unpadded
# ===================================================================

def test_backward_dw2_bit_exact():
    s = _setup()
    _, _, dw2_gold, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.efo_list, s.xg_list, s.ss_list, s.scores_flat,
        s.T, s.E, s.K,
    )
    _, _, dw2_padded, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.pefo_list, s.pxg_list, s.pss_list, s.pscores_list,
        s.T, s.E, s.K,
    )
    diff = (dw2_gold - dw2_padded).abs().max().item()
    assert diff < _F64_ATOL, \
        f"dw2 diff = {diff} (threshold {_F64_ATOL})"
    print(f"PASS: dw2_bit_exact — shape {tuple(dw2_gold.shape)}, max diff = {diff:.2e}")


# ===================================================================
# Axiom 7: dx bit-exact, especially dx[0] (padding gather target)
# ===================================================================

def test_backward_dx_bit_exact():
    s = _setup()
    dx_gold, _, _, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.efo_list, s.xg_list, s.ss_list, s.scores_flat,
        s.T, s.E, s.K,
    )
    dx_padded, _, _, _, _ = _gold_moe_backward(
        s.x, s.w1, s.w2, s.grad_out,
        s.pefo_list, s.pxg_list, s.pss_list, s.pscores_list,
        s.T, s.E, s.K,
    )
    # Check ALL tokens
    diff = (dx_gold - dx_padded).abs().max().item()
    assert diff < _F64_ATOL, \
        f"dx diff = {diff} (threshold {_F64_ATOL})"
    # Specifically check dx[0] — the row padding gathers from
    diff0 = (dx_gold[0] - dx_padded[0]).abs().max().item()
    assert diff0 < _F64_ATOL, \
        f"dx[0] diff = {diff0} (padding gather target, threshold {_F64_ATOL})"
    print(f"PASS: dx_bit_exact — shape {tuple(dx_gold.shape)}, max diff = {diff:.2e}, "
          f"dx[0] diff = {diff0:.2e}")


# ===================================================================
# Axiom 8: No token misrouting — expert sets unchanged by padding
# ===================================================================

def test_no_token_misrouting():
    s = _setup()
    dst = s.dst_idx.cpu().tolist()
    # For each real token in original space, check that the same x row
    # is gathered and assigned to the same expert in padded space.
    for orig_pos in range(s.TK):
        padded_pos = dst[orig_pos]
        # Same x row gathered
        assert s.pxg_list[padded_pos] == s.xg_list[orig_pos], \
            f"Position {orig_pos}: x_gather mismatch after padding"
        # Same expert (find which expert segment the position belongs to)
        orig_expert = None
        for e in range(s.E):
            if s.efo_list[e] <= orig_pos < s.efo_list[e + 1]:
                orig_expert = e
                break
        padded_expert = None
        for e in range(s.E):
            if s.pefo_list[e] <= padded_pos < s.pefo_list[e + 1]:
                padded_expert = e
                break
        assert orig_expert == padded_expert, \
            f"Position {orig_pos}: expert {orig_expert} -> {padded_expert} after padding"
    # Also verify per-token expert sets via topk_indices (unchanged by padding)
    for t in range(s.T):
        experts = set(s.topk_indices[t].cpu().tolist())
        assert len(experts) == s.K, f"Token {t}: duplicate experts {experts}"
    print(f"PASS: no_token_misrouting — all {s.TK} positions keep same "
          f"(x_row, expert) assignment")


# ===================================================================
# Main
# ===================================================================

_ALL_TESTS = [
    test_token_conservation,
    test_score_invariant,
    test_forward_bit_exact,
    test_backward_dz_zero_at_padding,
    test_backward_dw1_bit_exact,
    test_backward_dw2_bit_exact,
    test_backward_dx_bit_exact,
    test_no_token_misrouting,
]

if __name__ == "__main__":
    for fn in _ALL_TESTS:
        fn()
    print(f"\nAll {len(_ALL_TESTS)} axioms passed.")
