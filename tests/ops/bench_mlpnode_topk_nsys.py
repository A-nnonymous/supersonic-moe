#!/usr/bin/env python3
"""nsys GPU-projection benchmark for SonicMoEMlpNode topk path.

Measures FP8 fwd+bwd wall-clock via nsys sqlite GPU-projection (merged
overlapping kernel intervals on the same SM — the gold-standard method
matching tools/introspect.py).

Usage:
    # Profile and extract GPU-projection µs/iter:
    source .runenv.sh
    CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda \
        --output=/tmp/mlpnode_topk \
        python tests/ops/bench_mlpnode_topk_nsys.py \
            --T 8192 --E 8 --I 1536 --topk 8 \
            --warmup 8 --iters 12

    # Then extract:
    nsys export --type=sqlite --output=/tmp/mlpnode_topk.sqlite \
        /tmp/mlpnode_topk.nsys-rep
    python tests/ops/bench_mlpnode_topk_nsys.py --extract /tmp/mlpnode_topk.sqlite --iters 12

README baseline (Session 53, PyTorch native, Ernie shape T=8192 E=8 I=1536):
    BF16: 3644 µs/iter    FP8: 2715 µs/iter    Speedup: 1.34×
"""

import argparse
import math
import os
import sqlite3
import sys
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def gpu_projection_us(sqlite_path: str, n_iters: int) -> float:
    """Compute GPU-projection µs/iter from nsys sqlite export.

    Only counts kernels inside the NVTX "BENCH" range (excludes warmup/JIT).
    Merges overlapping CUDA kernel intervals, then divides by n_iters.
    """
    conn = sqlite3.connect(sqlite_path)

    # Find NVTX range "BENCH" start/end timestamps
    try:
        nvtx_rows = conn.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = 'BENCH'"
        ).fetchall()
    except Exception:
        nvtx_rows = []

    if nvtx_rows:
        bench_start, bench_end = nvtx_rows[0]
        rows = conn.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ? ORDER BY start",
            (bench_start, bench_end),
        ).fetchall()
    else:
        # Fallback: use all kernels
        rows = conn.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ).fetchall()

    conn.close()

    if not rows:
        raise RuntimeError("No kernel records found in sqlite")

    # Merge overlapping intervals
    merged = []
    cur_start, cur_end = rows[0]
    for s, e in rows[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    total_ns = sum(e - s for s, e in merged)
    return total_ns / 1000.0 / n_iters  # ns → µs, per iter


def run_benchmark(T, E, I, topk, n_warmup, n_iters):
    """Run FP8 topk MlpNode benchmark (forward + backward)."""
    import paddle
    paddle.enable_compat()
    import torch

    from sonicmoe.enums import ActivationType
    from sonicmoe.ernie_compat import (
        SonicMoEMlpNode,
        flush_native_grads,
        invalidate_weight_caches,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100

    H = 3072
    N_recv = T  # in topk dispatch, N_recv tokens each routed to topk experts
    device = "cuda"

    class MockExpert:
        def __init__(self, h, i, seed):
            paddle.seed(seed)
            self.up_gate_proj = type("P", (), {
                "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
            })()
            self.down_proj = type("P", (), {
                "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
            })()
            self.up_gate_proj.weight.stop_gradient = False
            self.down_proj.weight.stop_gradient = False

    experts = [MockExpert(H, I, e) for e in range(E)]
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(
        experts=experts, n_experts=E, hidden_size=H, intermediate_size=I,
    )

    # Build deterministic topk dispatch
    torch.manual_seed(42)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = (dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)).float()
    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    paddle.seed(0)
    x = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
    grad_out = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01

    # Warmup
    print(f"Warmup ({n_warmup} iters)...")
    for _ in range(n_warmup):
        out = node.forward(x, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    flush_native_grads()
    torch.cuda.synchronize()

    # Also capture CUDA event timing for immediate feedback
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    print(f"Benchmark ({n_iters} iters)...")
    torch.cuda.nvtx.range_push("BENCH")
    start_ev.record()
    for _ in range(n_iters):
        out = node.forward(x, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    end_ev.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    flush_native_grads()

    cuda_ms = start_ev.elapsed_time(end_ev)
    cuda_us_per_iter = cuda_ms / n_iters * 1000

    print(f"\n{'='*60}")
    print(f"SonicMoEMlpNode topk FP8 benchmark")
    print(f"  Shape: N_recv={N_recv} H={H} I={I} E={E} topk={topk}")
    print(f"  TK (total token-expert pairs): {sum(tpe)}")
    print(f"  CUDA events: {cuda_us_per_iter:.1f} µs/iter")
    print(f"  README baseline (PyTorch native FP8): 2715 µs/iter")
    print(f"{'='*60}")

    return cuda_us_per_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192,
                        help="N_recv tokens (Ernie shape: 8192)")
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--I", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--extract", type=str, default=None,
                        help="Path to nsys sqlite file to extract GPU-projection")
    args = parser.parse_args()

    if args.extract:
        us = gpu_projection_us(args.extract, args.iters)
        print(f"\nGPU-projection: {us:.1f} µs/iter ({args.iters} iters)")
        print(f"README baseline (PyTorch native FP8): 2715 µs/iter")
        return

    run_benchmark(args.T, args.E, args.I, args.topk, args.warmup, args.iters)


if __name__ == "__main__":
    main()
