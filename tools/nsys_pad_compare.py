#!/usr/bin/env python3
"""4-way nsys timeline: bf16_raw, bf16_rounding, fp8_rounding, fp8_padding.

Measures GPU-projection time for each FP8 alignment strategy and BF16
baselines using nsys profiling.  Validates that route-level padding
(fp8_padding) performs comparably to token-rounding (fp8_rounding).

Usage:
    # Quick smoke test (1 shape, 5 measured iters)
    CUDA_VISIBLE_DEVICES=0 python tools/nsys_pad_compare.py --quick

    # Full benchmark (default shapes, 20 measured iters)
    CUDA_VISIBLE_DEVICES=0 python tools/nsys_pad_compare.py

    # Custom shape
    CUDA_VISIBLE_DEVICES=0 python tools/nsys_pad_compare.py --T 300 --E 32 --K 8
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ═══════════════════════════════════════════════════════════════════════════════
# nsys GPU-projection parser
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_projection_us(sqlite_path: str) -> float:
    """Compute GPU-projection time (sweep-line interval merge) from nsys sqlite."""
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0.0

    # Sweep-line merge overlapping intervals
    merged_ns = 0
    cur_start, cur_end = rows[0]
    for s, e in rows[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged_ns += cur_end - cur_start
            cur_start, cur_end = s, e
    merged_ns += cur_end - cur_start
    return merged_ns / 1000.0  # ns → µs


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess script templates
# ═══════════════════════════════════════════════════════════════════════════════

_WORKER_TEMPLATE = textwrap.dedent("""\
    import os, sys, json
    os.environ["USE_QUACK_GEMM"] = "1"
    {env_extras}
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, {project_root!r})

    from sonicmoe.functional import (
        moe_TC_softmax_topk_layer,
        _refresh_fp8_config,
        clear_all_fp8_weight_caches,
    )
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.enums import ActivationType

    T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
    warmup_iters = {warmup}
    measured_iters = {measured}

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)

    x = (torch.randn(T, H, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16) * 0.01
    w1_param = torch.randn(E, 2 * (I // 1), H, device="cuda", dtype=torch.bfloat16) * 0.02
    w2_param = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02
    w1_f = w1_param.permute(1, 2, 0)
    w2_f = w2_param.permute(1, 2, 0)

    fp8_mode = {fp8_mode!r}
    use_fp8 = fp8_mode in ("rounding", "padding")

    if use_fp8:
        clear_all_fp8_weight_caches()

    ctx = enable_fp8(True) if use_fp8 else torch.no_grad()
    with ctx:
        if use_fp8:
            _refresh_fp8_config()
        # Warmup — caches must persist
        for _ in range(warmup_iters):
            o, _, _ = moe_TC_softmax_topk_layer(
                x.detach().requires_grad_(True), router_w, w1_f, None, w2_f, None,
                K=K, stream_id=0, activation_type=ActivationType.SWIGLU,
            )
            o.backward(torch.randn_like(o))
        torch.cuda.synchronize()

        # Profiled region
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(measured_iters):
            o, _, _ = moe_TC_softmax_topk_layer(
                x.detach().requires_grad_(True), router_w, w1_f, None, w2_f, None,
                K=K, stream_id=0, activation_type=ActivationType.SWIGLU,
            )
            o.backward(torch.randn_like(o))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    print(json.dumps({{"status": "ok", "fp8_mode": fp8_mode, "measured_iters": measured_iters}}))
""")


def _get_env_extras(mode: str) -> str:
    """Return extra os.environ lines for each mode."""
    if mode == "bf16_raw":
        return 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    elif mode == "bf16_rounding":
        return 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    elif mode == "fp8_rounding":
        return 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"'
    elif mode == "fp8_padding":
        return 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"'
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _get_fp8_mode(mode: str) -> str:
    if mode.startswith("bf16"):
        return "off"
    elif mode == "fp8_rounding":
        return "rounding"
    elif mode == "fp8_padding":
        return "padding"
    return "off"


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_nsys_profile(
    mode: str,
    T: int, H: int, I: int, E: int, K: int,
    warmup: int, measured: int,
    output_dir: str,
    python: str,
) -> dict:
    """Run one nsys-profiled subprocess and parse GPU-projection time."""
    script = _WORKER_TEMPLATE.format(
        project_root=PROJECT_ROOT,
        env_extras=_get_env_extras(mode),
        T=T, H=H, I=I, E=E, K=K,
        warmup=warmup, measured=measured,
        fp8_mode=_get_fp8_mode(mode),
    )

    nsys_prefix = os.path.join(output_dir, f"nsys_{mode}")
    nsys_rep = nsys_prefix + ".nsys-rep"
    nsys_sqlite = nsys_prefix + ".sqlite"

    # Write script to temp file
    script_path = os.path.join(output_dir, f"worker_{mode}.py")
    with open(script_path, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"

    # Step 1: nsys profile
    nsys_cmd = [
        "nsys", "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        f"--output={nsys_prefix}",
        "--force-overwrite=true",
        "--stats=false",
        python, script_path,
    ]

    print(f"  [{mode}] Running nsys profile...")
    t0 = time.time()
    result = subprocess.run(
        nsys_cmd, capture_output=True, text=True, env=env, timeout=600,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [{mode}] FAILED (exit {result.returncode})")
        print(f"    stderr: {result.stderr[-300:]}")
        return {"mode": mode, "status": "error", "error": result.stderr[-300:]}

    # Step 2: Export to sqlite
    export_cmd = [
        "nsys", "export", "--type=sqlite",
        f"--output={nsys_sqlite}",
        nsys_rep,
    ]
    subprocess.run(export_cmd, capture_output=True, timeout=120)

    if not os.path.exists(nsys_sqlite):
        return {"mode": mode, "status": "error", "error": "sqlite export failed"}

    # Step 3: Parse GPU-projection
    total_us = gpu_projection_us(nsys_sqlite)
    per_iter_us = total_us / measured if measured > 0 else 0

    print(f"  [{mode}] GPU-proj: {total_us:.0f} µs total, {per_iter_us:.0f} µs/iter ({elapsed:.1f}s wall)")

    return {
        "mode": mode,
        "status": "ok",
        "total_us": round(total_us, 1),
        "per_iter_us": round(per_iter_us, 1),
        "measured_iters": measured,
        "wall_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="4-way nsys pad comparison")
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--H", type=int, default=3072)
    parser.add_argument("--I", type=int, default=1536)
    parser.add_argument("--E", type=int, default=32)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 warmup, 5 measured")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--modes", nargs="+",
                        default=["bf16_raw", "fp8_rounding", "fp8_padding"],
                        help="Modes to benchmark")
    args = parser.parse_args()

    if args.quick:
        args.warmup = 3
        args.measured = 5

    python = sys.executable
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "output", "nsys_pad_compare",
        f"T{args.T}_E{args.E}_K{args.K}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"nsys pad comparison: T={args.T}, H={args.H}, I={args.I}, E={args.E}, K={args.K}")
    print(f"  warmup={args.warmup}, measured={args.measured}")
    print(f"  output: {output_dir}")
    print(f"  modes: {args.modes}")
    print()

    results = []
    for mode in args.modes:
        r = run_nsys_profile(
            mode, args.T, args.H, args.I, args.E, args.K,
            args.warmup, args.measured, output_dir, python,
        )
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['mode']:20s}  {r['per_iter_us']:8.0f} µs/iter")
        else:
            print(f"  {r['mode']:20s}  ERROR: {r.get('error', 'unknown')[:50]}")

    # Compare fp8_padding vs fp8_rounding if both present
    rounding = next((r for r in results if r["mode"] == "fp8_rounding" and r["status"] == "ok"), None)
    padding = next((r for r in results if r["mode"] == "fp8_padding" and r["status"] == "ok"), None)
    if rounding and padding:
        delta_pct = (padding["per_iter_us"] - rounding["per_iter_us"]) / rounding["per_iter_us"] * 100
        print(f"\n  fp8_padding vs fp8_rounding: {delta_pct:+.1f}%")

    # Save JSON
    report_path = os.path.join(output_dir, "pad_compare_results.json")
    with open(report_path, "w") as f:
        json.dump({
            "shape": {"T": args.T, "H": args.H, "I": args.I, "E": args.E, "K": args.K},
            "config": {"warmup": args.warmup, "measured": args.measured},
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
