"""Test precompute_weight_fp8_warmup is bit-exact vs. the 4-call sequence
and substantially faster.  Guards the fused pair-quantize Triton kernel.
"""
import time
import pytest
import torch

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    precompute_weight_fp8,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_warmup,
    _FUSED_WEIGHT_CACHE,
    _VARLEN_WEIGHT_CACHE,
)


def _eq_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two same-shape tensors as raw bytes (handles fp8/uint8 views).

    Note: avoids ``torch.equal`` because the paddle/torch interop venv
    monkey-patches it into an elementwise ``==`` returning a tensor.
    """
    av = a.contiguous().view(torch.uint8) if not a.is_contiguous() else a.view(torch.uint8)
    bv = b.contiguous().view(torch.uint8) if not b.is_contiguous() else b.view(torch.uint8)
    if av.shape != bv.shape:
        return False
    return bool((av == bv).all().item())


@pytest.mark.parametrize("shape", [
    (3072, 1536, 8),   # Production reference shape (T8192_H3072_I1536_E8)
    (1024, 512, 4),    # Smaller for fast CI
])
def test_fused_warmup_bit_exact(shape):
    """Verify fused warmup populates caches bit-identically to the 4-call sequence."""
    H, I, E = shape
    torch.manual_seed(0)
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.05

    # Unfused reference
    _FUSED_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE.clear()
    ref_w1f = precompute_weight_fp8_for_fused_gated(w1)
    ref_w2v = precompute_weight_fp8(w2)
    ref_w2d = precompute_weight_fp8_for_direct_fused_dgated(w2)
    ref_w1t = precompute_weight_fp8(w1.permute(1, 0, 2))

    # Fused warmup
    _FUSED_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE.clear()
    precompute_weight_fp8_warmup(w1, w2)
    new_w1f = precompute_weight_fp8_for_fused_gated(w1)
    new_w2v = precompute_weight_fp8(w2)
    new_w2d = precompute_weight_fp8_for_direct_fused_dgated(w2)
    new_w1t = precompute_weight_fp8(w1.permute(1, 0, 2))

    for name, (rfp8, rsc), (nfp8, nsc) in [
        ("w1_fused", ref_w1f, new_w1f),
        ("w2_varlen", ref_w2v, new_w2v),
        ("w2_dgated", ref_w2d, new_w2d),
        ("w1T_varlen", ref_w1t, new_w1t),
    ]:
        assert _eq_bytes(rfp8, nfp8), f"{name}: fp8 data differs"
        assert _eq_bytes(rsc, nsc), f"{name}: ISA-packed scales differ"


def test_fused_warmup_speedup():
    """Sanity-check that the fused warmup is meaningfully faster than 4 calls.

    Targets >=1.5x at the production reference shape.  We do not assert
    a tighter ratio because GPU contention from concurrent processes can
    affect kernel timings.
    """
    H, I, E = 3072, 1536, 8
    torch.manual_seed(0)
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.05

    def unfused():
        _FUSED_WEIGHT_CACHE.clear(); _VARLEN_WEIGHT_CACHE.clear()
        precompute_weight_fp8_for_fused_gated(w1)
        precompute_weight_fp8(w2)
        precompute_weight_fp8_for_direct_fused_dgated(w2)
        precompute_weight_fp8(w1.permute(1, 0, 2))

    def fused():
        _FUSED_WEIGHT_CACHE.clear(); _VARLEN_WEIGHT_CACHE.clear()
        precompute_weight_fp8_warmup(w1, w2)

    # warm
    for _ in range(5):
        unfused(); fused()
    torch.cuda.synchronize()

    n = 30
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): unfused()
    torch.cuda.synchronize(); t_unf = (time.perf_counter() - t0) / n * 1e6

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fused()
    torch.cuda.synchronize(); t_fus = (time.perf_counter() - t0) / n * 1e6

    speedup = t_unf / t_fus
    print(f"\nunfused 4-call: {t_unf:.1f} µs/iter")
    print(f"fused warmup:   {t_fus:.1f} µs/iter")
    print(f"speedup:        {speedup:.2f}x")
    assert speedup >= 1.5, f"expected >=1.5x, got {speedup:.2f}x"


if __name__ == "__main__":
    test_fused_warmup_bit_exact((3072, 1536, 8))
    test_fused_warmup_bit_exact((1024, 512, 4))
    test_fused_warmup_speedup()
    print("\nALL TESTS PASSED")
