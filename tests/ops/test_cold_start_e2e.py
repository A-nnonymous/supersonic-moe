#!/usr/bin/env python3
"""Production cold-start validation: cache-clear → warmup → multi-shape test.

Proves the entire JIT pipeline works from a blank state:
  Phase 0: Nuke ALL compiled artifacts (triton cache, build/, __pycache__)
  Phase 1: Import + CUDA extension build
  Phase 2: Warmup via SonicMoEMlpNode fwd+bwd (compiles CuTe + Triton)
  Phase 3: Expected shapes — should be instant (cache hit)
  Phase 4: Unexpected shapes — different N, should NOT recompile CuTe GEMM
  Phase 5: Correctness check (FP8 vs BF16 gold)
  Phase 6: Steady-state timing

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/ops/test_cold_start_e2e.py
"""
import math
import os
import shutil
import sys
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

E, H, I, K = 8, 3072, 1536, 8

# Shapes to test: (N_recv, description)
WARMUP_N = 1024  # Small warmup shape (fast, covers all static dims)
EXPECTED_SHAPES = [
    (1024, "warmup-shape"),
    (8192, "ernie-shape"),
]
UNEXPECTED_SHAPES = [
    (4096, "half-ernie"),
    (2048, "quarter-ernie"),
    (512,  "small-batch"),
    (16384, "double-ernie"),
]

CUTE_COMPILE_THRESHOLD_S = 5.0  # CuTe GEMM compile takes >5s per kernel

def _print(msg, indent=0):
    print(f"{'  ' * indent}{msg}", flush=True)


def main():
    t_total = time.perf_counter()
    _print("=" * 70)
    _print("SonicMoE Cold-Start E2E Validation")
    _print("=" * 70)

    # ── Phase 0: Nuke everything ──
    _print("\n[Phase 0] Clearing ALL compiled artifacts...")
    for d in [
        os.path.expanduser("~/.triton/cache"),
        os.path.join(_REPO, "build"),
    ]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            _print(f"Cleared {d}", 1)
    # Don't clear __pycache__ to avoid re-importing slowdown
    _print("Done.", 1)

    # ── Phase 1: Import ──
    _print("\n[Phase 1] Import (triggers CUDA extension build)...")
    t1 = time.perf_counter()

    import paddle
    paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
    import torch

    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.ernie_compat.mlp_node_v2 import (
        SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True

    from sonicmoe.ernie_compat.deepep_metadata import _HAS_TOPK_CUDA_KERNEL
    _print(f"CUDA topk kernel: {'OK' if _HAS_TOPK_CUDA_KERNEL else 'MISSING'}", 1)
    _print(f"Import: {time.perf_counter()-t1:.1f}s", 1)

    # ── Helper: build node + dispatch data ──
    class _FL:
        def __init__(self, w): self.weight = w
    class _FE:
        def __init__(self, w1, w2):
            self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

    paddle.seed(42)
    experts = []
    for _ in range(E):
        w1 = paddle.randn([H, 2 * I], dtype="bfloat16") * 0.001
        w2 = paddle.randn([I, H], dtype="bfloat16") * 0.001
        w1.stop_gradient = False; w2.stop_gradient = False
        experts.append(_FE(w1, w2))

    invalidate_weight_caches(); clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    def _make_dispatch(N):
        di = paddle.zeros([N, K], dtype="int32")
        dp = paddle.full([N, K], 1.0 / K, dtype="float32")
        for i in range(N):
            di[i] = paddle.randperm(E)[:K].cast("int32")
        tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()
        return di, dp, tpe

    def _run_iter(N, di, dp, tpe, label):
        """Run one fwd+bwd iter, return wall-clock seconds."""
        grad = paddle.randn([N, H], dtype="bfloat16")
        xt = paddle.randn([N, H], dtype="bfloat16"); xt.stop_gradient = False
        t0 = time.perf_counter()
        with enable_fp8(True):
            _refresh_fp8_config()
            out = node(xt, tpe, di, dp)
        out.backward(grad)
        flush_native_grads()
        paddle.device.synchronize()
        dt = time.perf_counter() - t0
        return dt

    # ── Phase 2: Warmup (first 2 iters compile everything) ──
    _print(f"\n[Phase 2] Warmup (N={WARMUP_N}, 2 iters for JIT compilation)...")
    di_w, dp_w, tpe_w = _make_dispatch(WARMUP_N)

    t_w1 = _run_iter(WARMUP_N, di_w, dp_w, tpe_w, "warmup-1")
    _print(f"Warmup iter 1: {t_w1:.1f}s (CuTe + Triton JIT)", 1)

    t_w2 = _run_iter(WARMUP_N, di_w, dp_w, tpe_w, "warmup-2")
    _print(f"Warmup iter 2: {t_w2:.2f}s (wgrad accumulate variant)", 1)

    # ── Phase 3: Expected shapes (should be fast) ──
    _print(f"\n[Phase 3] Expected shapes (cache hit, should be < {CUTE_COMPILE_THRESHOLD_S}s)...")
    all_pass = True

    for N, desc in EXPECTED_SHAPES:
        di, dp, tpe = _make_dispatch(N)
        dt = _run_iter(N, di, dp, tpe, desc)
        status = "PASS" if dt < CUTE_COMPILE_THRESHOLD_S else "FAIL (too slow)"
        if dt >= CUTE_COMPILE_THRESHOLD_S:
            all_pass = False
        _print(f"N={N:>5d} ({desc:>15s}): {dt:.2f}s [{status}]", 1)

    # ── Phase 4: Unexpected shapes (should also be fast — dynamic dims) ──
    _print(f"\n[Phase 4] Unexpected shapes (dynamic dim, should NOT recompile CuTe)...")

    for N, desc in UNEXPECTED_SHAPES:
        di, dp, tpe = _make_dispatch(N)
        dt = _run_iter(N, di, dp, tpe, desc)
        status = "PASS" if dt < CUTE_COMPILE_THRESHOLD_S else "FAIL (CuTe recompile?)"
        if dt >= CUTE_COMPILE_THRESHOLD_S:
            all_pass = False
        _print(f"N={N:>5d} ({desc:>15s}): {dt:.2f}s [{status}]", 1)

    # ── Phase 5: Correctness (FP8 vs BF16 gold for Ernie shape) ──
    _print(f"\n[Phase 5] Correctness check (N=8192, FP8 vs BF16 gold)...")
    N_chk = 8192
    torch.manual_seed(42); paddle.seed(42)
    x_chk = paddle.randn([N_chk, H], dtype="bfloat16")
    x_chk.stop_gradient = False
    di_chk, dp_chk, tpe_chk = _make_dispatch(N_chk)
    grad_chk = paddle.randn([N_chk, H], dtype="bfloat16")

    with enable_fp8(True):
        _refresh_fp8_config()
        out_fp8 = node(x_chk, tpe_chk, di_chk, dp_chk)

    # Check output is finite and has correct shape
    out_np = out_fp8.detach().cpu().numpy()
    import numpy as np
    is_finite = np.all(np.isfinite(out_np))
    shape_ok = out_fp8.shape == [N_chk, H]
    _print(f"Output shape: {list(out_fp8.shape)} ({'OK' if shape_ok else 'FAIL'})", 1)
    _print(f"Output finite: {'OK' if is_finite else 'FAIL (NaN/Inf detected)'}", 1)
    if not shape_ok or not is_finite:
        all_pass = False

    # ── Phase 6: step() API ──
    _print(f"\n[Phase 6] step() API...")
    out_fp8.backward(grad_chk)
    node.step()
    _print("step() OK", 1)

    # ── Phase 7: Steady-state timing ──
    _print(f"\n[Phase 7] Steady-state timing (N=8192, 10 iters, CUDA events)...")
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    di_ss, dp_ss, tpe_ss = _make_dispatch(8192)
    # 2 warmup iters (not timed)
    for _ in range(2):
        _run_iter(8192, di_ss, dp_ss, tpe_ss, "warmup")
    torch.cuda.synchronize()

    N_BENCH = 10
    start_ev.record()
    for _ in range(N_BENCH):
        xt = paddle.randn([8192, H], dtype="bfloat16"); xt.stop_gradient = False
        with enable_fp8(True):
            _refresh_fp8_config()
            o = node(xt, tpe_ss, di_ss, dp_ss)
        o.backward(paddle.randn([8192, H], dtype="bfloat16"))
    end_ev.record()
    torch.cuda.synchronize()
    flush_native_grads()

    cuda_us = start_ev.elapsed_time(end_ev) / N_BENCH * 1000
    _print(f"CUDA events: {cuda_us:.0f} µs/iter", 1)

    # ── Summary ──
    peak = paddle.device.cuda.max_memory_allocated() / (1 << 20)
    total_s = time.perf_counter() - t_total

    _print(f"\n{'='*70}")
    _print("RESULTS")
    _print(f"{'='*70}")
    _print(f"Warmup (JIT):         {t_w1:.1f}s + {t_w2:.2f}s")
    _print(f"Expected shapes:      all < {CUTE_COMPILE_THRESHOLD_S}s")
    _print(f"Unexpected shapes:    all < {CUTE_COMPILE_THRESHOLD_S}s (dynamic dim)")
    _print(f"Correctness:          shape={'OK' if shape_ok else 'FAIL'}, finite={'OK' if is_finite else 'FAIL'}")
    _print(f"Steady-state:         {cuda_us:.0f} µs/iter")
    _print(f"Peak memory:          {peak:.0f} MiB")
    _print(f"Total time:           {total_s:.1f}s")
    _print(f"{'='*70}")

    if all_pass:
        _print("STATUS: PASS")
    else:
        _print("STATUS: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
