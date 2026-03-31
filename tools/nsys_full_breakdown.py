#!/usr/bin/env python3
"""Comprehensive nsys analysis: kernel breakdown by NVTX range.

Reads nsys SQLite export and produces a detailed breakdown of:
- Kernels (sorted by total GPU time)
- Memory operations (malloc, memset, memcpy)
- NVTX GPU projection (time per phase)
- Summary per forward/backward

Usage:
    # Export nsys report to SQLite
    nsys export --type=sqlite --output=/tmp/sonic_bf16.sqlite /tmp/sonic_bf16.nsys-rep
    nsys export --type=sqlite --output=/tmp/sonic_fp8.sqlite /tmp/sonic_fp8.nsys-rep

    # Analyze
    python tools/nsys_full_breakdown.py /tmp/sonic_bf16.sqlite /tmp/sonic_fp8.sqlite --labels bf16 fp8
    python tools/nsys_full_breakdown.py /tmp/sonic_fp8.sqlite --labels fp8
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def ns_to_us(ns):
    return ns / 1000.0

def ns_to_ms(ns):
    return ns / 1e6

def query_nvtx_gpu_projection(db_path, iter_prefix=None):
    """Get NVTX ranges with their GPU-projected durations using kernel overlap.
    
    Returns dict mapping range_name -> list of {start, end, gpu_kernels: [{name, start, end, dur}]}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all NVTX ranges
    try:
        nvtx_rows = conn.execute("""
            SELECT text, start, end, (end - start) as dur
            FROM NVTX_EVENTS 
            WHERE eventType = 59 OR eventType = 1
            ORDER BY start
        """).fetchall()
    except Exception:
        try:
            nvtx_rows = conn.execute("""
                SELECT text, start, end, (end - start) as dur
                FROM NVTX_EVENTS 
                ORDER BY start
            """).fetchall()
        except Exception:
            nvtx_rows = []
    
    # Get all GPU kernels (join with StringIds for actual names)
    try:
        kernel_rows = conn.execute("""
            SELECT s.value as name, k.start, k.end, (k.end - k.start) as dur
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        """).fetchall()
    except Exception:
        kernel_rows = []
    
    # Get all memory ops
    try:
        memset_rows = conn.execute("""
            SELECT 'memset' as name, start, end, (end - start) as dur
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            ORDER BY start
        """).fetchall()
    except Exception:
        memset_rows = []
    
    try:
        memcpy_rows = conn.execute("""
            SELECT 
                CASE copyKind 
                    WHEN 1 THEN 'memcpy_HtoD'
                    WHEN 2 THEN 'memcpy_DtoH'
                    WHEN 8 THEN 'memcpy_DtoD'
                    ELSE 'memcpy_other'
                END as name,
                start, end, (end - start) as dur
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            ORDER BY start
        """).fetchall()
    except Exception:
        memcpy_rows = []
    
    conn.close()
    
    # Combine all GPU activities
    all_gpu_ops = []
    for r in kernel_rows:
        name = r["name"] if isinstance(r["name"], str) else str(r["name"])
        all_gpu_ops.append({"name": name, "start": r["start"], "end": r["end"], "dur": r["dur"], "type": "kernel"})
    for r in memset_rows:
        all_gpu_ops.append({"name": r["name"], "start": r["start"], "end": r["end"], "dur": r["dur"], "type": "memset"})
    for r in memcpy_rows:
        all_gpu_ops.append({"name": r["name"], "start": r["start"], "end": r["end"], "dur": r["dur"], "type": "memcpy"})
    
    all_gpu_ops.sort(key=lambda x: x["start"])
    
    return nvtx_rows, all_gpu_ops


def compute_gpu_time_in_range(gpu_ops, range_start, range_end):
    """Compute total GPU time for ops overlapping with [range_start, range_end].
    Returns (total_gpu_ns, list of (name, dur_in_range))."""
    ops_in_range = []
    for op in gpu_ops:
        if op["end"] <= range_start or op["start"] >= range_end:
            continue
        clipped_start = max(op["start"], range_start)
        clipped_end = min(op["end"], range_end)
        dur = clipped_end - clipped_start
        ops_in_range.append((op["name"], dur, op["type"]))
    
    total = sum(d for _, d, _ in ops_in_range)
    return total, ops_in_range


def analyze_report(db_path, label=""):
    """Full analysis of a single nsys SQLite report."""
    nvtx_rows, gpu_ops = query_nvtx_gpu_projection(db_path)
    
    print(f"\n{'='*80}")
    print(f"  ANALYSIS: {label} ({db_path})")
    print(f"{'='*80}")
    
    if not nvtx_rows:
        print("  WARNING: No NVTX events found. Falling back to raw kernel analysis.")
    
    # ---------- NVTX GPU Projection ----------
    print(f"\n--- NVTX GPU Projection ---")
    
    # Group by range name (aggregate across iterations)
    range_stats = defaultdict(lambda: {"count": 0, "wall_ns": 0, "gpu_ns": 0, "kernels": defaultdict(lambda: {"count": 0, "gpu_ns": 0})})
    
    for row in nvtx_rows:
        text = row["text"]
        if text is None:
            continue
        text = str(text)
        rs = row["start"]
        re = row["end"]
        gpu_total, ops = compute_gpu_time_in_range(gpu_ops, rs, re)
        
        stats = range_stats[text]
        stats["count"] += 1
        stats["wall_ns"] += (re - rs)
        stats["gpu_ns"] += gpu_total
        
        for name, dur, typ in ops:
            short = name if len(name) < 80 else name[:77] + "..."
            stats["kernels"][short]["count"] += 1
            stats["kernels"][short]["gpu_ns"] += dur
    
    # Print top-level NVTX ranges
    for rname in ["forward", "backward", "zero_grad", "clone_input"]:
        if rname in range_stats:
            s = range_stats[rname]
            n = s["count"]
            avg_wall = ns_to_us(s["wall_ns"] / n) if n else 0
            avg_gpu = ns_to_us(s["gpu_ns"] / n) if n else 0
            print(f"\n  [{rname}] x{n}  avg_wall={avg_wall:.1f}µs  avg_gpu={avg_gpu:.1f}µs")
            
            # Top kernels within this range
            sorted_k = sorted(s["kernels"].items(), key=lambda x: -x[1]["gpu_ns"])
            for kname, kstat in sorted_k[:20]:
                kavg = ns_to_us(kstat["gpu_ns"] / n)
                pct = 100.0 * kstat["gpu_ns"] / s["gpu_ns"] if s["gpu_ns"] > 0 else 0
                kcnt = kstat["count"] // n
                print(f"    {kavg:8.1f}µs ({pct:5.1f}%) x{kcnt:2d}  {kname}")
    
    # ---------- Global Kernel Breakdown ----------
    print(f"\n--- Global Kernel Breakdown (all GPU ops, sorted by total time) ---")
    
    kernel_agg = defaultdict(lambda: {"count": 0, "total_ns": 0, "min_ns": float("inf"), "max_ns": 0})
    for op in gpu_ops:
        short = op["name"] if len(op["name"]) < 100 else op["name"][:97] + "..."
        ka = kernel_agg[short]
        ka["count"] += 1
        ka["total_ns"] += op["dur"]
        ka["min_ns"] = min(ka["min_ns"], op["dur"])
        ka["max_ns"] = max(ka["max_ns"], op["dur"])
    
    sorted_kernels = sorted(kernel_agg.items(), key=lambda x: -x[1]["total_ns"])
    total_gpu_ns = sum(v["total_ns"] for _, v in sorted_kernels)
    
    print(f"  Total GPU time: {ns_to_ms(total_gpu_ns):.3f}ms across {len(sorted_kernels)} unique kernels")
    print(f"  {'Avg(µs)':>10s} {'Tot(ms)':>10s} {'%':>6s} {'Cnt':>5s} {'Min(µs)':>10s} {'Max(µs)':>10s}  Kernel")
    print(f"  {'-'*10} {'-'*10} {'-'*6} {'-'*5} {'-'*10} {'-'*10}  {'-'*40}")
    
    cumulative_pct = 0.0
    for kname, kstat in sorted_kernels[:40]:
        avg = ns_to_us(kstat["total_ns"] / kstat["count"])
        tot = ns_to_ms(kstat["total_ns"])
        pct = 100.0 * kstat["total_ns"] / total_gpu_ns if total_gpu_ns > 0 else 0
        cumulative_pct += pct
        mn = ns_to_us(kstat["min_ns"])
        mx = ns_to_us(kstat["max_ns"])
        print(f"  {avg:10.1f} {tot:10.3f} {pct:5.1f}% {kstat['count']:5d} {mn:10.1f} {mx:10.1f}  {kname}")
    
    print(f"\n  Top-40 cumulative: {cumulative_pct:.1f}% of total GPU time")
    
    return range_stats, kernel_agg, total_gpu_ns


def compare_reports(results, labels):
    """Compare two or more nsys reports side-by-side."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {' vs '.join(labels)}")
    print(f"{'='*80}")
    
    for phase in ["forward", "backward"]:
        print(f"\n  --- {phase} ---")
        vals = []
        for i, (range_stats, _, _) in enumerate(results):
            if phase in range_stats:
                s = range_stats[phase]
                n = s["count"]
                avg_gpu = ns_to_us(s["gpu_ns"] / n) if n else 0
                vals.append(avg_gpu)
                print(f"    {labels[i]:>15s}: avg_gpu = {avg_gpu:.1f}µs")
            else:
                vals.append(None)
                print(f"    {labels[i]:>15s}: N/A")
        
        if len(vals) >= 2 and vals[0] and vals[1]:
            speedup = vals[0] / vals[1] if vals[1] > 0 else float("inf")
            print(f"    {'speedup':>15s}: {speedup:.2f}x ({labels[1]} over {labels[0]})")
    
    # Total iteration
    print(f"\n  --- Total iteration (forward + backward) ---")
    for i, (range_stats, _, total_gpu_ns) in enumerate(results):
        fwd = range_stats.get("forward", {}).get("gpu_ns", 0)
        bwd = range_stats.get("backward", {}).get("gpu_ns", 0)
        n_fwd = range_stats.get("forward", {}).get("count", 1)
        n_bwd = range_stats.get("backward", {}).get("count", 1)
        avg_fwd = ns_to_us(fwd / n_fwd) if n_fwd else 0
        avg_bwd = ns_to_us(bwd / n_bwd) if n_bwd else 0
        total = avg_fwd + avg_bwd
        print(f"    {labels[i]:>15s}: fwd={avg_fwd:.1f}µs + bwd={avg_bwd:.1f}µs = {total:.1f}µs ({ns_to_ms(total * 1000):.3f}ms)")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive nsys analysis")
    parser.add_argument("reports", nargs="+", help="SQLite report file(s)")
    parser.add_argument("--labels", nargs="+", help="Labels for each report")
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.reports):
        print("Error: number of labels must match number of reports")
        sys.exit(1)
    
    labels = args.labels or [f"report_{i}" for i in range(len(args.reports))]
    
    results = []
    for rpath, label in zip(args.reports, labels):
        r = analyze_report(rpath, label)
        results.append(r)
    
    if len(results) >= 2:
        compare_reports(results, labels)


if __name__ == "__main__":
    main()
