"""Analyze NSYS sqlite exports for three-way comparison.

Usage:
    python tools/analyze_nsys_three_way.py \
        --official /path/to/official_bf16.sqlite \
        --fork /path/to/fork_fp8.sqlite \
        --ernie /path/to/ernie_moe.sqlite

Produces per-kernel breakdown and GPU projection totals.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def gpu_projection(db_path, label, iter_pattern="iter_2", exclude_elementwise=True):
    """Compute GPU projection from NSYS sqlite export.
    
    Returns (total_us, phase_details) where phase_details is a list of
    (phase_name, gpu_us, [(kernel_name, dur_us), ...]) tuples.
    """
    conn = sqlite3.connect(db_path)

    # Find the target iteration
    iters = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE ?",
        (f"%{iter_pattern}",)
    ).fetchall()
    if not iters:
        print(f"  [WARN] No NVTX event matching '*{iter_pattern}' in {db_path}")
        conn.close()
        return 0, []

    iter_start = min(r[1] for r in iters)
    iter_end = max(r[2] for r in iters)

    # Find forward/backward phases
    phases = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE start >= ? AND end <= ? AND text IN ('forward','backward') "
        "ORDER BY start",
        (iter_start, iter_end)
    ).fetchall()

    if not phases:
        print(f"  [WARN] No forward/backward NVTX ranges in {db_path}")
        conn.close()
        return 0, []

    total = 0
    phase_details = []

    for pn, ps, pe in phases:
        filter_clause = ""
        if exclude_elementwise:
            filter_clause = "AND s.value NOT LIKE '%elementwise_kernel%'"

        kernels = conn.execute(
            f"SELECT s.value, k.start, k.end, (k.end-k.start)/1000.0 "
            f"FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds s ON k.shortName=s.id "
            f"WHERE k.start<? AND k.end>? {filter_clause} "
            f"ORDER BY k.start",
            (pe, ps)
        ).fetchall()

        # Merge overlapping intervals for GPU projection
        intervals = sorted([(max(ks, ps), min(ke, pe)) for _, ks, ke, _ in kernels])
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        gpu_us = sum((e - s) for s, e in merged) / 1000.0
        total += gpu_us

        # Collect kernel details (aggregate by name)
        kernel_agg = defaultdict(float)
        for name, ks, ke, dur in kernels:
            short = name[:100]
            kernel_agg[short] += dur

        kernel_list = sorted(kernel_agg.items(), key=lambda x: -x[1])
        phase_details.append((pn, gpu_us, kernel_list))

    conn.close()
    return total, phase_details


def print_breakdown(label, total, phase_details, top_n=15):
    """Print formatted breakdown."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    for pn, gpu_us, kernels in phase_details:
        print(f"\n  {pn}: {gpu_us:.1f} µs")
        print(f"  {'─'*60}")
        for i, (name, dur) in enumerate(kernels):
            if i >= top_n and dur < 5:
                remaining = sum(d for _, d in kernels[i:])
                print(f"    ... +{len(kernels)-i} more kernels: {remaining:.1f} µs")
                break
            pct = dur / gpu_us * 100 if gpu_us > 0 else 0
            print(f"    {dur:8.1f} µs ({pct:5.1f}%)  {name}")

    print(f"\n  ▶ TOTAL GPU PROJECTION: {total:.1f} µs")
    return total


def main():
    parser = argparse.ArgumentParser(description="Three-way NSYS analysis")
    parser.add_argument("--official", type=str, help="Official BF16 NSYS sqlite")
    parser.add_argument("--fork", type=str, help="Fork FP8 NSYS sqlite")
    parser.add_argument("--ernie", type=str, help="Ernie MoE NSYS sqlite")
    parser.add_argument("--iter", type=str, default="iter_2", help="Iteration to analyze")
    parser.add_argument("--top-n", type=int, default=15, help="Top N kernels per phase")
    args = parser.parse_args()

    if not any([args.official, args.fork, args.ernie]):
        parser.print_help()
        return

    results = {}

    if args.official:
        total, details = gpu_projection(args.official, "Official BF16", args.iter)
        print_breakdown("Official SonicMoE BF16", total, details, args.top_n)
        results["official_bf16"] = total

    if args.fork:
        total, details = gpu_projection(args.fork, "Fork FP8", args.iter)
        print_breakdown("Fork SonicMoE FP8", total, details, args.top_n)
        results["fork_fp8"] = total

    if args.ernie:
        # Ernie uses different NVTX naming
        total, details = gpu_projection(args.ernie, "Ernie MoE", args.iter,
                                        exclude_elementwise=False)
        print_breakdown("Ernie DeepEPMOELayer", total, details, args.top_n)
        results["ernie"] = total

    # Summary comparison
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*80}")
        baseline = results.get("official_bf16", 0)
        for name, total in results.items():
            speedup = ""
            if baseline > 0 and name != "official_bf16":
                ratio = baseline / total
                pct = (1 - total / baseline) * 100
                speedup = f" → {pct:+.1f}% vs BF16 ({ratio:.2f}×)"
            print(f"  {name:20s}: {total:8.1f} µs{speedup}")


if __name__ == "__main__":
    main()
