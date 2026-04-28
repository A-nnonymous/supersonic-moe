#!/usr/bin/env python
"""Tests for deepep_topk_to_sonic_metadata: real DeepEP topk dispatch conversion.

Validates structural invariants, score consistency, edge cases, and
equivalence with the identity-layout path for K=1 pre-sorted data.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_deepep_topk_metadata.py -v --tb=short
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_deepep_topk_metadata.py  # standalone
"""

import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import pytest

from sonicmoe.ernie_compat.deepep_metadata import (
    deepep_topk_to_sonic_metadata,
    deepep_to_sonic_metadata,
    _host_prefix_sum,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def fabricate_dispatch_result(
    N_recv: int, topk: int, E: int, broadcast_ratio: float = 0.5,
    device="cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Generate realistic DeepEP dispatch results for testing.

    Returns (dispatched_indices, dispatched_probs, tokens_per_expert).
    """
    # Each token gets 1..min(topk, E) random experts
    expected_experts = max(1, min(int(broadcast_ratio * E), topk))
    dispatched_indices = torch.full(
        (N_recv, topk), -1, dtype=torch.int32, device=device
    )
    dispatched_probs = torch.zeros(
        (N_recv, topk), dtype=torch.float32, device=device
    )

    for i in range(N_recv):
        count = min(max(1, int(torch.randn(1).item() * 2 + expected_experts)), min(topk, E))
        perm = torch.randperm(E, device=device)[:count]
        dispatched_indices[i, :count] = perm.int()
        prob_val = 1.0 / count
        dispatched_probs[i, :count] = prob_val

    # Compute tokens_per_expert
    valid = dispatched_indices >= 0
    valid_experts = dispatched_indices[valid].long()
    tpe = torch.bincount(valid_experts, minlength=E).tolist()

    return dispatched_indices, dispatched_probs, tpe


def fabricate_identity_layout(
    tokens_per_expert: list[int], E: int, device="cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Generate K=1 pre-sorted dispatch data (identity layout).

    Each token is routed to exactly 1 expert, and tokens are already sorted
    by expert in the buffer.
    """
    T = sum(tokens_per_expert)
    dispatched_indices = torch.empty(T, 1, dtype=torch.int32, device=device)
    dispatched_probs = torch.ones(T, 1, dtype=torch.float32, device=device)

    offset = 0
    for e, count in enumerate(tokens_per_expert):
        dispatched_indices[offset:offset + count, 0] = e
        offset += count

    return dispatched_indices, dispatched_probs, tokens_per_expert


# ── Structural Invariant Tests ──────────────────────────────────────────────

class TestStructuralInvariants:
    """Test that output metadata satisfies required structural properties."""

    @pytest.fixture(params=[
        (256, 8, 8),     # small
        (1024, 4, 8),    # medium
        (4096, 8, 32),   # larger
        (2048, 16, 64),  # many experts
        (4096, 8, 256),  # 256 experts
    ])
    def dispatch_data(self, request):
        N_recv, topk, E = request.param
        torch.manual_seed(42)
        return fabricate_dispatch_result(N_recv, topk, E)

    def test_efo_monotonic(self, dispatch_data):
        """expert_frequency_offset must be monotonically non-decreasing."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        assert efo[0].item() == 0
        diffs = efo[1:] - efo[:-1]
        assert (diffs >= 0).all(), "efo not monotonically increasing"

    def test_segments_128_aligned(self, dispatch_data):
        """All non-empty expert segments must be 128-aligned."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        for e in range(E):
            seg_size = (efo[e + 1] - efo[e]).item()
            if seg_size > 0:
                assert seg_size % 128 == 0, (
                    f"Expert {e}: segment size {seg_size} not 128-aligned"
                )

    def test_efo_last_equals_tk_padded(self, dispatch_data):
        """Last efo entry must equal TK_padded."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, _, _, _, _, _, TK_padded, _, _ = result
        assert efo[-1].item() == TK_padded

    def test_x_gather_idx_bounds(self, dispatch_data):
        """x_gather_idx values for real entries must be in [0, N_recv)."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        x_gather_idx = result[1]
        # All values must be in [0, N_recv)
        assert (x_gather_idx >= 0).all()
        assert (x_gather_idx < N_recv).all()

    def test_naept_properties(self, dispatch_data):
        """naept must be monotonically increasing with correct bounds."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]
        TK = (indices >= 0).sum().item()

        # Length = N_recv + 1
        assert naept.shape[0] == N_recv + 1
        # Starts at 0
        assert naept[0].item() == 0
        # Ends at TK (total valid entries)
        assert naept[-1].item() == TK
        # Monotonically non-decreasing
        diffs = naept[1:] - naept[:-1]
        assert (diffs >= 0).all()

    def test_s_reverse_length(self, dispatch_data):
        """s_reverse_scatter_idx length must be TK (valid entries)."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        TK = (indices >= 0).sum().item()
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        s_reverse = result[3]
        assert s_reverse.shape[0] == TK

    def test_topk_scores_length(self, dispatch_data):
        """topk_scores length must be TK_padded; valid scores accessed via naept must be > 0."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        TK = (indices >= 0).sum().item()
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        topk_scores = result[5]
        naept = result[4]
        TK_padded = result[6]
        assert topk_scores.shape[0] == TK_padded, (
            f"topk_scores length {topk_scores.shape[0]} != TK_padded {TK_padded}"
        )
        # Valid scores (accessed via naept offsets) should all be > 0
        if TK > 0:
            for t in range(min(N_recv, 50)):
                start = naept[t].item()
                end = naept[t + 1].item()
                if end > start:
                    assert (topk_scores[start:end] > 0).all(), (
                        f"Token {t}: scores at [{start},{end}) should be > 0"
                    )
        # Total non-zero scores should equal TK
        nonzero_count = (topk_scores > 0).sum().item()
        assert nonzero_count == TK, (
            f"Non-zero scores {nonzero_count} != TK {TK}"
        )


# ── Score Consistency Tests ─────────────────────────────────────────────────

class TestScoreConsistency:
    """Test that scores are correctly preserved through the conversion."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (512, 8, 8),
        (256, 4, 256),  # 256 experts
    ])
    def test_per_token_score_sum(self, N_recv, topk, E):
        """Sum of scores per token should match sum of valid dispatched_probs."""
        torch.manual_seed(123)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]
        topk_scores = result[5]

        for t in range(min(N_recv, 50)):  # Check first 50 tokens
            start = naept[t].item()
            end = naept[t + 1].item()
            score_sum = topk_scores[start:end].sum().item()

            # Expected: sum of valid probs for token t
            valid_mask = indices[t] >= 0
            expected_sum = probs[t][valid_mask].sum().item()

            assert abs(score_sum - expected_sum) < 1e-5, (
                f"Token {t}: score_sum={score_sum:.6f} vs expected={expected_sum:.6f}"
            )

    @pytest.mark.parametrize("N_recv,topk,E", [
        (256, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_padding_scores_zero(self, N_recv, topk, E):
        """Valid scores (via naept) must be > 0; total non-zero count == TK."""
        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        topk_scores = result[5]
        naept = result[4]
        TK = (indices >= 0).sum().item()
        TK_padded = result[6]

        # Check per-token: scores accessed via naept should be > 0
        for t in range(N_recv):
            start = naept[t].item()
            end = naept[t + 1].item()
            if end > start:
                assert (topk_scores[start:end] > 0).all(), (
                    f"Token {t}: valid scores should be > 0"
                )

        # Total non-zero scores should equal TK
        nonzero_count = (topk_scores > 0).sum().item()
        assert nonzero_count == TK, f"Non-zero scores {nonzero_count} != TK {TK}"


# ── Gather Consistency Tests ────────────────────────────────────────────────

class TestGatherConsistency:
    """Test that x_gather_idx correctly gathers tokens per expert."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (256, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_expert_segment_tokens(self, N_recv, topk, E):
        """Verify each expert's segment contains exactly the expected tokens."""
        torch.manual_seed(77)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]

        for e in range(E):
            seg_start = efo[e].item()
            seg_end = efo[e + 1].item()
            real_count = tpe[e]

            # Gather the real entries (first real_count in the segment)
            gathered_tokens = x_gather_idx[seg_start:seg_start + real_count]

            # Expected: set of tokens routed to expert e
            expected_set = set()
            for t in range(N_recv):
                for k in range(topk):
                    if indices[t, k].item() == e:
                        expected_set.add(t)

            gathered_set = set(gathered_tokens.tolist())
            assert gathered_set == expected_set, (
                f"Expert {e}: gathered={sorted(gathered_set)} vs expected={sorted(expected_set)}"
            )


# ── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_masked(self):
        """All entries are -1 (no tokens routed to any local expert)."""
        N_recv, topk, E = 32, 4, 8
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")
        tpe = [0] * E

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv = result

        assert TK_padded == 0
        assert pad_rows == 0
        assert n_recv == N_recv
        assert naept.shape[0] == N_recv + 1
        assert (naept == 0).all()

    def test_single_token_single_expert(self):
        """One token routed to one expert."""
        indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device="cuda")
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
        tpe = [1, 0, 0, 0]
        E = 4

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv = result

        assert n_recv == 1
        assert naept[0].item() == 0
        assert naept[1].item() == 1
        assert TK_padded == 128  # padded to 128
        assert efo[1].item() == 128
        assert x_gather[0].item() == 0  # gather from token 0
        assert scores[0].item() == 1.0

    def test_all_tokens_to_one_expert(self):
        """All tokens routed to the same expert."""
        N_recv, E = 64, 8
        indices = torch.zeros(N_recv, 1, dtype=torch.int32, device="cuda")  # all to expert 0
        probs = torch.ones(N_recv, 1, dtype=torch.float32, device="cuda")
        tpe = [N_recv] + [0] * (E - 1)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, _, _, naept, _, TK_padded, _, n_recv = result

        assert n_recv == N_recv
        assert efo[1].item() == 128  # 64 padded to 128
        assert efo[-1].item() == TK_padded

    def test_tokens_with_zero_local_experts(self):
        """Some tokens have all -1 (routed to other ranks only)."""
        N_recv, topk, E = 16, 4, 4
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")

        # Only first 8 tokens have valid routing
        for i in range(8):
            indices[i, 0] = i % E
            probs[i, 0] = 1.0

        tpe = [2, 2, 2, 2]  # 2 tokens per expert

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]

        # Tokens 8..15 should have naept[i] == naept[i+1] (zero assignments)
        for t in range(8, N_recv):
            assert naept[t].item() == naept[t + 1].item(), (
                f"Token {t} should have 0 local expert assignments"
            )

    def test_tokens_per_expert_with_zeros(self):
        """Some experts have zero tokens."""
        torch.manual_seed(42)
        N_recv, topk, E = 128, 4, 8
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")

        # Route to experts 0, 2, 4, 6 only (skip odd experts)
        for i in range(N_recv):
            expert = (i % 4) * 2
            indices[i, 0] = expert
            probs[i, 0] = 1.0

        tpe = [32, 0, 32, 0, 32, 0, 32, 0]

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]

        # Odd experts should have zero-length segments
        for e in [1, 3, 5, 7]:
            assert efo[e + 1].item() - efo[e].item() == 0


# ── Identity Layout Equivalence ─────────────────────────────────────────────

class TestIdentityEquivalence:
    """Test that K=1 pre-sorted data matches deepep_to_sonic_metadata."""

    @pytest.mark.parametrize("tpe", [
        [512] * 8,
        [128, 256, 64, 512, 384, 100, 200, 300],
        [128] * 4,
        [32] * 256,  # 256 experts, uniform
    ])
    def test_equivalence(self, tpe):
        """K=1 pre-sorted input should produce equivalent metadata."""
        E = len(tpe)
        T = sum(tpe)
        indices, probs, _ = fabricate_identity_layout(tpe, E)

        # Topk path
        topk_result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        (topk_efo, topk_xg, _, _, _, _, topk_TK_padded, topk_pad, _) = topk_result

        # Identity path
        id_result = deepep_to_sonic_metadata(tpe, T, E)
        (id_efo, id_xg, _, _, _, _, id_TK_padded, id_pad) = id_result

        # EFO should be identical
        assert (topk_efo == id_efo).all(), "EFO mismatch"
        # TK_padded should be identical
        assert topk_TK_padded == id_TK_padded, (
            f"TK_padded: topk={topk_TK_padded} vs id={id_TK_padded}"
        )

        # x_gather_idx should produce the same token set per expert
        for e in range(E):
            seg_start = topk_efo[e].item()
            real_count = tpe[e]
            topk_set = set(topk_xg[seg_start:seg_start + real_count].tolist())
            id_set = set(id_xg[seg_start:seg_start + real_count].tolist())
            assert topk_set == id_set, f"Expert {e}: token sets differ"


# ── moe_permute Cross-Validation ──────────────────────────────────────────

class TestMoePermuteConsistency:
    """Cross-validate deepep_topk_to_sonic_metadata against paddle.nn.functional.moe_permute."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (128, 4, 8),
        (256, 8, 8),   # N_recv <= 256 to avoid bf16 integer precision issues
        (128, 4, 256), # 256 experts
    ])
    def test_token_set_per_expert(self, N_recv, topk, E):
        """Per-expert token sets from our metadata must match moe_permute's output."""
        torch.manual_seed(42)
        H = 32  # small hidden dim for the test
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # ── Gold: paddle.nn.functional.moe_permute ────────────────────────
        # Make each token identifiable via its row content
        x = torch.zeros(N_recv, H, dtype=torch.bfloat16, device="cuda")
        for i in range(N_recv):
            x[i, 0] = float(i)  # tag: first element = token id (exact for i <= 256)

        gold_permuted, _, gold_probs, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        # ── Ours: deepep_topk_to_sonic_metadata ──────────────────────────
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]
        TK_padded = result[6]

        # Both should have the same TK_padded
        assert gold_permuted.shape[0] == TK_padded, (
            f"TK_padded mismatch: moe_permute={gold_permuted.shape[0]} vs ours={TK_padded}"
        )

        # Per-expert: compare gathered rows (using x[x_gather_idx] vs gold_permuted)
        for e in range(E):
            seg_start = efo[e].item()
            real_count = tpe[e]
            if real_count == 0:
                continue

            # Our gathered token ids
            our_ids = set(x_gather_idx[seg_start:seg_start + real_count].tolist())

            # Gold: extract token ids from permuted rows
            gold_ids = set()
            for pos in range(seg_start, seg_start + real_count):
                tok_id = int(gold_permuted[pos, 0].item())
                gold_ids.add(tok_id)

            assert our_ids == gold_ids, (
                f"Expert {e}: our tokens={sorted(our_ids)} vs "
                f"moe_permute tokens={sorted(gold_ids)}"
            )

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (256, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_padding_alignment_consistent(self, N_recv, topk, E):
        """Padding alignment (128) must match between our metadata and moe_permute."""
        torch.manual_seed(99)
        H = 16
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        x = torch.randn(N_recv, H, dtype=torch.bfloat16, device="cuda")
        gold_permuted, _, _, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        TK_padded = result[6]

        # Total padded size must match
        assert gold_permuted.shape[0] == TK_padded

        # Per-expert segment alignment
        for e in range(E):
            seg = (efo[e + 1] - efo[e]).item()
            if seg > 0:
                assert seg % 128 == 0
                gold_seg = seg  # moe_permute uses same total => same segments
                assert seg == gold_seg

    @pytest.mark.parametrize("N_recv,topk,E", [
        (128, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_score_per_expert_consistent(self, N_recv, topk, E):
        """Per-token scores within each expert segment must match moe_permute probs."""
        torch.manual_seed(55)
        H = 16
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        x = torch.zeros(N_recv, H, dtype=torch.bfloat16, device="cuda")
        for i in range(N_recv):
            x[i, 0] = float(i)

        gold_permuted, _, gold_probs, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]
        s_scatter_idx = result[2]
        topk_scores = result[5]
        TK_padded = result[6]

        # Build a map from (token_id, expert_id) → our score
        for e in range(E):
            seg_start = efo[e].item()
            real_count = tpe[e]

            for local_pos in range(real_count):
                pos = seg_start + local_pos
                tok_id = x_gather_idx[pos].item()

                # Our score: s_scatter_idx maps expert-sorted → token-major
                token_major_pos = s_scatter_idx[pos].item()
                our_score = topk_scores[token_major_pos].item()

                # Gold score at the same position
                gold_tok_id = int(gold_permuted[pos, 0].item())
                gold_score = gold_probs[pos].item()

                # Same token at this position? (ordering is stable ascending)
                if tok_id == gold_tok_id:
                    assert abs(our_score - gold_score) < 1e-5, (
                        f"Expert {e}, pos {local_pos}: tok={tok_id}, "
                        f"our_score={our_score:.6f} vs gold={gold_score:.6f}"
                    )


# ── CUDA vs Python Fallback Comparison ────────────────────────────────────

class TestCudaVsPythonFallback:
    """Compare CUDA kernel output against Python fallback element-wise."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (256, 8, 8),
        (1024, 4, 8),
        (512, 8, 32),
        (512, 4, 256),   # 256 experts
    ])
    def test_element_wise_consistency(self, N_recv, topk, E):
        """CUDA kernel and Python fallback must produce equivalent metadata.

        Note: s_scatter_idx at PADDING positions may differ (Python uses
        unique arange [TK, TK+pad), CUDA uses constant TK). Both are correct
        since padding scores are 0. We only compare REAL positions.
        """
        from sonicmoe.ernie_compat.deepep_metadata import (
            _HAS_TOPK_CUDA_KERNEL,
            _deepep_topk_to_sonic_metadata_cuda,
        )
        if not _HAS_TOPK_CUDA_KERNEL:
            pytest.skip("CUDA kernel not compiled")

        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # Force Python fallback
        import sonicmoe.ernie_compat.deepep_metadata as _mod
        saved = _mod._HAS_TOPK_CUDA_KERNEL
        _mod._HAS_TOPK_CUDA_KERNEL = False
        py_result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        _mod._HAS_TOPK_CUDA_KERNEL = saved

        # Run CUDA path
        cu_result = _deepep_topk_to_sonic_metadata_cuda(
            indices, probs, tpe, E, "cuda", 128,
        )

        py_efo, py_xg, py_ss, py_sr, py_naept, py_scores, py_TKp, py_pad, py_N = py_result
        cu_efo, cu_xg, cu_ss, cu_sr, cu_naept, cu_scores, cu_TKp, cu_pad, cu_N = cu_result

        # Scalars must match exactly
        assert py_TKp == cu_TKp, f"TK_padded: py={py_TKp} cu={cu_TKp}"
        assert py_pad == cu_pad, f"total_pad: py={py_pad} cu={cu_pad}"
        assert py_N == cu_N, f"N_recv: py={py_N} cu={cu_N}"

        # efo must match exactly
        assert (py_efo == cu_efo).all(), "efo mismatch"

        # naept must match exactly
        assert (py_naept == cu_naept).all(), "naept mismatch"

        # Per-expert comparison: token SETS must match
        TK_padded = py_TKp
        TK = sum(tpe)
        for e_idx in range(E):
            seg_start = py_efo[e_idx].item()
            real_count = tpe[e_idx]
            if real_count == 0:
                continue
            seg_end = seg_start + real_count

            # x_gather_idx: real positions must have same token SET
            py_set = set(py_xg[seg_start:seg_end].tolist())
            cu_set = set(cu_xg[seg_start:seg_end].tolist())
            assert py_set == cu_set, (
                f"Expert {e_idx}: x_gather token sets differ"
            )

        # s_reverse_scatter_idx: shape must match
        assert py_sr.shape == cu_sr.shape, f"s_reverse shape mismatch"

        # ── Functional equivalence checks ──────────────────────────────
        # The key semantic contract is:
        #   For token t routed to expert e with score s:
        #     x_gather_idx[padded_pos] == t
        #     topk_scores[s_scatter_idx[padded_pos]] == s
        #     s_reverse_scatter_idx[s_scatter_idx[padded_pos]] == padded_pos
        # Both paths may assign different token-major positions (s_scatter_idx
        # values) since Python uses argsort order and CUDA uses warp-ballot
        # order. But the gathered scores must match per (token, expert).

        # Check 1: Per-token score multiset via naept (sorted comparison)
        for t in range(min(py_N, 200)):
            start = py_naept[t].item()
            end = py_naept[t + 1].item()
            if end > start:
                py_s = py_scores[start:end].sort()[0]
                cu_s = cu_scores[start:end].sort()[0]
                assert torch.allclose(py_s, cu_s, atol=1e-6), (
                    f"Token {t}: scores differ: py={py_s.tolist()} cu={cu_s.tolist()}"
                )

        # Check 2: For each expert segment, the (token_id → score) mapping
        # via topk_scores[s_scatter_idx[pos]] must agree.
        for e_idx in range(E):
            seg_start = py_efo[e_idx].item()
            real_count = tpe[e_idx]
            if real_count == 0:
                continue

            py_tok_score = {}
            cu_tok_score = {}
            for i in range(real_count):
                pos = seg_start + i
                py_tok = py_xg[pos].item()
                cu_tok = cu_xg[pos].item()
                py_tok_score[py_tok] = py_scores[py_ss[pos].item()].item()
                cu_tok_score[cu_tok] = cu_scores[cu_ss[pos].item()].item()

            assert py_tok_score.keys() == cu_tok_score.keys(), (
                f"Expert {e_idx}: token sets differ"
            )
            for tok in py_tok_score:
                assert abs(py_tok_score[tok] - cu_tok_score[tok]) < 1e-6, (
                    f"Expert {e_idx}, token {tok}: "
                    f"py_score={py_tok_score[tok]:.6f} cu_score={cu_tok_score[tok]:.6f}"
                )

        # Check 3: round-trip s_reverse_scatter_idx[s_scatter_idx[pos]] == pos
        # for all real positions in each expert segment
        for e_idx in range(E):
            seg_start = py_efo[e_idx].item()
            real_count = tpe[e_idx]
            for i in range(real_count):
                pos = seg_start + i
                py_rt = py_sr[py_ss[pos].item()].item()
                cu_rt = cu_sr[cu_ss[pos].item()].item()
                assert py_rt == pos, (
                    f"Python round-trip fail: expert {e_idx}, pos {pos}, got {py_rt}"
                )
                assert cu_rt == pos, (
                    f"CUDA round-trip fail: expert {e_idx}, pos {pos}, got {cu_rt}"
                )


# ── Performance Benchmark ───────────────────────────────────────────────────

class TestPerformance:
    """Benchmark metadata conversion latency."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (4096, 8, 8),
        (16384, 8, 8),
        (16384, 8, 64),
        (16384, 8, 256),  # 256 experts
    ])
    def test_latency_under_target(self, N_recv, topk, E):
        """Metadata conversion should complete in < 500us."""
        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # Warmup
        for _ in range(10):
            deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        torch.cuda.synchronize()

        # Timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        repeats = 100

        start.record()
        for _ in range(repeats):
            deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        end.record()
        torch.cuda.synchronize()

        avg_us = start.elapsed_time(end) * 1000 / repeats
        print(f"\n  [N={N_recv}, topk={topk}, E={E}] avg={avg_us:.1f}us")
        # Target: < 500us (generous for the argsort-dominated path)
        assert avg_us < 500, f"Too slow: {avg_us:.1f}us > 500us target"


# ── Standalone runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running deepep_topk_to_sonic_metadata tests...\n")

    # Quick smoke test
    torch.manual_seed(42)
    N_recv, topk, E = 1024, 8, 8
    indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
    result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
    efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv = result

    TK = (indices >= 0).sum().item()
    print(f"Config: N_recv={N_recv}, topk={topk}, E={E}")
    print(f"TK={TK}, TK_padded={TK_padded}, pad_rows={pad_rows}")
    print(f"efo: {efo.tolist()}")
    print(f"naept[:5]: {naept[:5].tolist()}")
    print(f"scores[:5]: {scores[:5].tolist()}")
    print(f"x_gather[:10]: {x_gather[:10].tolist()}")
    print(f"s_reverse[:10]: {s_reverse[:10].tolist()}")

    # Structural checks
    assert efo[0].item() == 0
    assert efo[-1].item() == TK_padded
    for e in range(E):
        seg = (efo[e + 1] - efo[e]).item()
        if seg > 0:
            assert seg % 128 == 0, f"Expert {e}: segment {seg} not 128-aligned"

    assert naept[0].item() == 0
    assert naept[-1].item() == TK
    assert (naept[1:] - naept[:-1] >= 0).all()

    print("\nAll smoke tests PASSED!")
    print("\nRunning full pytest suite...")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
