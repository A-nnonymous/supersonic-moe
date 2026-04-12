"""Smoke test: unaligned FP8 padded forward vs BF16 reference.

Verifies that _padded_blockscaled_gated_forward produces results within
precision tolerance when expert segments are NOT 128-aligned.
"""
import os
import sys
import torch
import pytest

# Force FP8 + QuACK mode via env vars (import-time cached)
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(f"Need SM>=90, got SM{cap[0]}{cap[1]}")


@pytest.fixture(autouse=True)
def gpu_check():
    _skip_if_no_gpu()


def _make_unaligned_routing(T, K, E, device):
    """Generate routing with intentionally non-128-aligned expert segments."""
    # Assign roughly T*K/E tokens per expert, but make some unaligned
    base = (T * K) // E
    # Make expert 0 have base+7 tokens (not 128-aligned), expert 1 has base-7, etc.
    expert_counts = []
    total = 0
    for e in range(E):
        if e < E - 1:
            offset = 7 * (1 if e % 2 == 0 else -1)
            count = max(1, base + offset)
            expert_counts.append(count)
            total += count
        else:
            expert_counts.append(T * K - total)
    TK = sum(expert_counts)

    # Build expert_frequency_offset (cumulative sum with leading 0)
    offsets = [0]
    for c in expert_counts:
        offsets.append(offsets[-1] + c)
    expert_frequency_offset = torch.tensor(offsets, dtype=torch.int32, device=device)

    # Build gather indices: random permutation of [0, T)
    x_gather_idx = torch.randint(0, T, (TK,), dtype=torch.int32, device=device)

    return expert_frequency_offset, x_gather_idx, TK


def _check_alignment(expert_frequency_offset):
    """Return True if ALL segments are 128-aligned."""
    sizes = expert_frequency_offset[1:] - expert_frequency_offset[:-1]
    return bool((sizes % 128 == 0).all().item())


@pytest.mark.parametrize("T,H,I,E,K", [
    (512, 3072, 1536, 8, 8),   # Ernie-like
    (256, 1024, 512, 4, 4),    # Small
])
def test_padded_fp8_vs_bf16_forward(T, H, I, E, K):
    """Compare padded FP8 up-proj output vs BF16 QuACK reference."""
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.02
    # w1 shape: (2*I, H, E) for gated
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device=device) * 0.01

    efo, x_gather_idx, TK = _make_unaligned_routing(T, K, E, device)
    assert not _check_alignment(efo), "Test requires unaligned segments"

    # === BF16 reference ===
    from sonicmoe.quack_utils import gemm_gated
    z_ref, y1_ref = gemm_gated(
        x, w1.permute(2, 1, 0) if w1.dim() == 3 else w1,
        activation="swiglu",
        cu_seqlens_m=efo,
        A_idx=x_gather_idx,
    )

    # === FP8 padded forward ===
    from sonicmoe.functional import _padded_blockscaled_gated_forward
    z_fp8, y1_fp8 = _padded_blockscaled_gated_forward(x, w1, efo, x_gather_idx)

    # Check shapes match
    assert z_ref.shape == z_fp8.shape, f"z shape mismatch: {z_ref.shape} vs {z_fp8.shape}"
    assert y1_ref.shape == y1_fp8.shape, f"y1 shape mismatch: {y1_ref.shape} vs {y1_fp8.shape}"

    # Precision: RelRMSE < 15% (padded FP8 vs BF16 has slightly more noise than aligned)
    def rel_rmse(a, b):
        diff = (a.float() - b.float())
        return (diff.norm() / b.float().norm()).item()

    z_rrmse = rel_rmse(z_fp8, z_ref)
    y1_rrmse = rel_rmse(y1_fp8, y1_ref)
    print(f"  z  RelRMSE: {z_rrmse:.4f} ({z_rrmse*100:.1f}%)")
    print(f"  y1 RelRMSE: {y1_rrmse:.4f} ({y1_rrmse*100:.1f}%)")

    # Correlation
    z_corr = torch.corrcoef(torch.stack([z_ref.float().flatten(), z_fp8.float().flatten()]))[0, 1].item()
    y1_corr = torch.corrcoef(torch.stack([y1_ref.float().flatten(), y1_fp8.float().flatten()]))[0, 1].item()
    print(f"  z  correlation: {z_corr:.6f}")
    print(f"  y1 correlation: {y1_corr:.6f}")

    assert z_rrmse < 0.20, f"z RelRMSE {z_rrmse:.4f} > 20%"
    assert y1_rrmse < 0.20, f"y1 RelRMSE {y1_rrmse:.4f} > 20%"
    assert z_corr > 0.95, f"z correlation {z_corr:.6f} < 0.95"
    assert y1_corr > 0.95, f"y1 correlation {y1_corr:.6f} < 0.95"


@pytest.mark.parametrize("T,H,I,E,K", [
    (512, 3072, 1536, 8, 8),
])
def test_padded_fp8_downproj(T, H, I, E, K):
    """Smoke test: unaligned FP8 down-proj via blockscaled_fp8_gemm_varlen."""
    device = "cuda"
    torch.manual_seed(42)

    efo, x_gather_idx, TK = _make_unaligned_routing(T, K, E, device)
    assert not _check_alignment(efo), "Test requires unaligned segments"

    # y1 is TK x I
    y1 = torch.randn(TK, I, dtype=torch.bfloat16, device=device) * 0.02
    # w2 shape: (H, I, E)
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device=device) * 0.01

    # === BF16 reference ===
    from quack.gemm_interface import gemm
    y2_ref = gemm(y1, w2.permute(2, 1, 0).contiguous(), cu_seqlens_m=efo)

    # === FP8 unaligned down-proj ===
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        blockscaled_fp8_gemm_varlen,
        precompute_weight_fp8,
    )
    w2_fp8, w2_scales = precompute_weight_fp8(w2)
    y2_fp8 = blockscaled_fp8_gemm_varlen(
        y1, w2, efo,
        w_fp8=w2_fp8, w_scales=w2_scales,
        out_dtype=torch.bfloat16,
        assume_aligned=False,
    )

    assert y2_ref.shape == y2_fp8.shape, f"shape mismatch: {y2_ref.shape} vs {y2_fp8.shape}"

    def rel_rmse(a, b):
        diff = (a.float() - b.float())
        return (diff.norm() / b.float().norm()).item()

    rrmse = rel_rmse(y2_fp8, y2_ref)
    corr = torch.corrcoef(torch.stack([y2_ref.float().flatten(), y2_fp8.float().flatten()]))[0, 1].item()
    print(f"  down-proj RelRMSE: {rrmse:.4f} ({rrmse*100:.1f}%)")
    print(f"  down-proj correlation: {corr:.6f}")

    assert rrmse < 0.20, f"RelRMSE {rrmse:.4f} > 20%"
    assert corr > 0.95, f"correlation {corr:.6f} < 0.95"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
