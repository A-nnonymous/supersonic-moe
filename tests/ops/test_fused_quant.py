#!/usr/bin/env python3
"""Correctness + performance test for fused_dual_colwise_quantize.

Verifies bit-exact output vs sequential dual_quantize_varlen + colwise_quantize_and_pack.
"""
import os, sys, time
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch
torch.set_default_device("cuda")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    dual_quantize_varlen,
    colwise_quantize_and_pack,
)
from sonicmoe.quack_utils.fused_quant_kernels import fused_dual_colwise_quantize


def test_correctness(T=8192, TK=65536, dz_dim=3072, dout_dim=3072, E=8, K=8):
    """Verify bit-exact match with sequential kernels."""
    print(f"\n=== Correctness: T={T} TK={TK} dz_dim={dz_dim} dout_dim={dout_dim} ===")

    torch.manual_seed(42)
    dz = torch.randn(TK, dz_dim, dtype=torch.bfloat16)
    dout = torch.randn(T, dout_dim, dtype=torch.bfloat16)
    gather_idx = torch.randint(0, T, (TK,), dtype=torch.int32)

    # Reference: sequential
    ref_dz_row_fp8, ref_dz_row_sc, ref_dz_col_fp8, ref_dz_col_sc = \
        dual_quantize_varlen(dz, TK, dz_dim)
    ref_dout_col_fp8, ref_dout_col_sc = colwise_quantize_and_pack(
        dout, logical_rows=dout_dim, logical_cols=TK, gather_idx=gather_idx,
    )
    torch.cuda.synchronize()

    # Fused
    fused_dz_row_fp8, fused_dz_row_sc, fused_dz_col_fp8, fused_dz_col_sc, \
        fused_dout_col_fp8, fused_dout_col_sc = fused_dual_colwise_quantize(
            dz, dout, gather_idx, TK, dz_dim, dout_dim,
        )
    torch.cuda.synchronize()

    # Compare
    checks = [
        ("dz_row_fp8",  ref_dz_row_fp8.view(torch.uint8),  fused_dz_row_fp8.view(torch.uint8)),
        ("dz_row_sc",   ref_dz_row_sc.view(torch.uint8),   fused_dz_row_sc.view(torch.uint8)),
        ("dz_col_fp8",  ref_dz_col_fp8.view(torch.uint8),  fused_dz_col_fp8.view(torch.uint8)),
        ("dz_col_sc",   ref_dz_col_sc.view(torch.uint8),   fused_dz_col_sc.view(torch.uint8)),
        ("dout_col_fp8", ref_dout_col_fp8.view(torch.uint8), fused_dout_col_fp8.view(torch.uint8)),
        ("dout_col_sc",  ref_dout_col_sc.view(torch.uint8),  fused_dout_col_sc.view(torch.uint8)),
    ]
    all_pass = True
    for name, ref, fused in checks:
        if ref.shape != fused.shape:
            print(f"  {name}: SHAPE MISMATCH ref={ref.shape} fused={fused.shape}")
            all_pass = False
            continue
        match = (ref == fused).all().item()
        if match:
            print(f"  {name}: BIT-EXACT ✓  (shape {tuple(ref.shape)})")
        else:
            diff_count = (ref != fused).sum().item()
            total = ref.numel()
            print(f"  {name}: MISMATCH ✗  {diff_count}/{total} bytes differ ({diff_count/total*100:.4f}%)")
            # Show first few differences
            diff_idx = (ref != fused).nonzero(as_tuple=False)[:5]
            for idx in diff_idx:
                idx_t = tuple(idx.tolist())
                print(f"    [{idx_t}] ref={ref[idx_t].item()} fused={fused[idx_t].item()}")
            all_pass = False

    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")
    return all_pass


def bench(T=8192, TK=65536, dz_dim=3072, dout_dim=3072, warmup=10, iters=50):
    """Benchmark fused vs sequential."""
    print(f"\n=== Benchmark: T={T} TK={TK} dz_dim={dz_dim} dout_dim={dout_dim} ===")

    torch.manual_seed(42)
    dz = torch.randn(TK, dz_dim, dtype=torch.bfloat16)
    dout = torch.randn(T, dout_dim, dtype=torch.bfloat16)
    gather_idx = torch.randint(0, T, (TK,), dtype=torch.int32)

    # Warmup
    for _ in range(warmup):
        dual_quantize_varlen(dz, TK, dz_dim)
        colwise_quantize_and_pack(dout, logical_rows=dout_dim, logical_cols=TK, gather_idx=gather_idx)
        fused_dual_colwise_quantize(dz, dout, gather_idx, TK, dz_dim, dout_dim)
    torch.cuda.synchronize()

    # Sequential benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        dual_quantize_varlen(dz, TK, dz_dim)
        colwise_quantize_and_pack(dout, logical_rows=dout_dim, logical_cols=TK, gather_idx=gather_idx)
        end_events[i].record()
    torch.cuda.synchronize()
    seq_times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    seq_median = sorted(seq_times)[len(seq_times) // 2]

    # Fused benchmark
    start_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events2[i].record()
        fused_dual_colwise_quantize(dz, dout, gather_idx, TK, dz_dim, dout_dim)
        end_events2[i].record()
    torch.cuda.synchronize()
    fused_times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events2, end_events2)]
    fused_median = sorted(fused_times)[len(fused_times) // 2]

    speedup = seq_median / fused_median if fused_median > 0 else 0
    print(f"  Sequential (dual + colwise): {seq_median:.1f} µs (median of {iters})")
    print(f"  Fused:                       {fused_median:.1f} µs (median of {iters})")
    print(f"  Speedup:                     {speedup:.2f}×")
    print(f"  Saved:                       {seq_median - fused_median:.1f} µs")
    return seq_median, fused_median


if __name__ == "__main__":
    passed = test_correctness()
    if passed:
        bench()
    else:
        print("\nSkipping benchmark due to correctness failure.")
        sys.exit(1)
