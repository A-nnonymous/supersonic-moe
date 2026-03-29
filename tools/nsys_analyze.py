"""
Analyze nsys SQLite export to compare FP8 vs BF16 kernel timelines.
Identifies launch overhead, gaps between kernels, and kernel counts.

Usage:
    python tools/nsys_analyze.py /tmp/sonicmoe_fp8.sqlite /tmp/sonicmoe_bf16.sqlite
    python tools/nsys_analyze.py /tmp/sonicmoe_fp8.sqlite  # single mode
"""
import sqlite3
import sys
from collections import defaultdict


def query_kernels(db_path: str):
    """Extract CUDA kernel launches with timing info."""
    conn = sqlite3.connect(db_path)
    
    # Get CUDA kernel activity
    try:
        rows = conn.execute("""
            SELECT
                k.shortName as kernel_name,
                k.start as start_ns,
                k.end as end_ns,
                (k.end - k.start) as duration_ns,
                k.gridX, k.gridY, k.gridZ,
                k.blockX, k.blockY, k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            ORDER BY k.start
        """).fetchall()
    except Exception:
        # Try alternate table name
        rows = conn.execute("""
            SELECT
                demangledName as kernel_name,
                start as start_ns,
                end as end_ns,
                (end - start) as duration_ns,
                gridX, gridY, gridZ,
                blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start
        """).fetchall()

    conn.close()
    return rows


def query_nvtx_ranges(db_path: str):
    """Extract NVTX ranges to identify iteration boundaries."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT text, start, end, (end - start) as duration_ns
            FROM NVTX_EVENTS
            WHERE eventType = 59 OR eventType = 60 OR rangeId IS NOT NULL
            ORDER BY start
        """).fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows


def query_memops(db_path: str):
    """Extract memory operations (malloc, free, memcpy, memset)."""
    conn = sqlite3.connect(db_path)
    results = {}
    
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        try:
            rows = conn.execute(f"""
                SELECT start, end, (end - start) as duration_ns, bytes
                FROM {table}
                ORDER BY start
            """).fetchall()
            results[table] = rows
        except Exception:
            results[table] = []
    
    conn.close()
    return results


def query_cuda_api(db_path: str):
    """Extract CUDA API calls (launch, malloc, sync, etc.)."""
    conn = sqlite3.connect(db_path)
    try:
        # Get string IDs for API names
        string_map = {}
        try:
            for row in conn.execute("SELECT id, value FROM StringIds"):
                string_map[row[0]] = row[1]
        except Exception:
            pass

        rows = conn.execute("""
            SELECT nameId, start, end, (end - start) as duration_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            ORDER BY start
        """).fetchall()
        
        # Resolve names
        resolved = []
        for nameId, start, end, dur in rows:
            name = string_map.get(nameId, f"api_{nameId}")
            resolved.append((name, start, end, dur))
        
        conn.close()
        return resolved
    except Exception:
        conn.close()
        return []


def list_tables(db_path: str):
    """List all tables in the SQLite database."""
    conn = sqlite3.connect(db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    conn.close()
    return [t[0] for t in tables]


def analyze_single(db_path: str, label: str):
    """Analyze a single nsys SQLite export."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label} ({db_path})")
    print(f"{'='*80}")

    # List tables
    tables = list_tables(db_path)
    print(f"\nTables available: {len(tables)}")
    for t in tables:
        print(f"  - {t}")

    # Kernel analysis
    kernels = query_kernels(db_path)
    print(f"\n--- CUDA Kernels: {len(kernels)} total ---")

    if not kernels:
        print("No kernel data found!")
        return {}

    # Group by kernel name
    by_name = defaultdict(list)
    for k in kernels:
        name = k[0]
        dur = k[3]
        by_name[name].append(dur)

    print(f"\nUnique kernel types: {len(by_name)}")
    print(f"\n{'Kernel Name':<70} {'Count':>6} {'Total(µs)':>12} {'Avg(µs)':>10} {'Min(µs)':>10} {'Max(µs)':>10}")
    print("-" * 120)

    sorted_kernels = sorted(by_name.items(), key=lambda x: sum(x[1]), reverse=True)
    for name, durs in sorted_kernels:
        total = sum(durs) / 1000
        avg = (sum(durs) / len(durs)) / 1000
        mn = min(durs) / 1000
        mx = max(durs) / 1000
        short_name = name[:69]
        print(f"{short_name:<70} {len(durs):>6} {total:>12.1f} {avg:>10.1f} {mn:>10.1f} {mx:>10.1f}")

    # Gap analysis (time between consecutive kernel ends and next kernel starts)
    gaps = []
    for i in range(1, len(kernels)):
        prev_end = kernels[i-1][2]  # end_ns
        curr_start = kernels[i][1]  # start_ns
        gap = curr_start - prev_end
        if gap > 0:
            gaps.append((gap, kernels[i-1][0], kernels[i][0]))

    if gaps:
        gaps.sort(key=lambda x: x[0], reverse=True)
        total_gap = sum(g[0] for g in gaps)
        total_kernel = sum(k[3] for k in kernels)
        total_wall = kernels[-1][2] - kernels[0][1] if len(kernels) > 1 else 0

        print(f"\n--- Gap Analysis ---")
        print(f"Total kernel time: {total_kernel/1e6:.3f} ms")
        print(f"Total gap time: {total_gap/1e6:.3f} ms")
        print(f"Total wall time: {total_wall/1e6:.3f} ms")
        print(f"Gap fraction: {total_gap/total_wall*100:.1f}%" if total_wall > 0 else "")
        print(f"Kernel fraction: {total_kernel/total_wall*100:.1f}%" if total_wall > 0 else "")

        print(f"\nTop 20 largest gaps:")
        print(f"{'Gap(µs)':>10} {'From Kernel':<40} {'To Kernel':<40}")
        print("-" * 92)
        for gap_ns, from_k, to_k in gaps[:20]:
            print(f"{gap_ns/1000:>10.1f} {from_k[:39]:<40} {to_k[:39]:<40}")

    # CUDA API analysis
    api_calls = query_cuda_api(db_path)
    if api_calls:
        api_by_name = defaultdict(list)
        for name, start, end, dur in api_calls:
            api_by_name[name].append(dur)
        
        print(f"\n--- CUDA API Calls: {len(api_calls)} total ---")
        print(f"{'API Name':<50} {'Count':>8} {'Total(µs)':>12} {'Avg(µs)':>10}")
        print("-" * 82)
        sorted_api = sorted(api_by_name.items(), key=lambda x: sum(x[1]), reverse=True)
        for name, durs in sorted_api[:20]:
            total = sum(durs) / 1000
            avg = (sum(durs) / len(durs)) / 1000
            print(f"{name[:49]:<50} {len(durs):>8} {total:>12.1f} {avg:>10.1f}")

    # Memory ops
    memops = query_memops(db_path)
    for table, rows in memops.items():
        if rows:
            total_bytes = sum(r[3] for r in rows if r[3])
            total_time = sum(r[2] for r in rows)
            print(f"\n--- {table}: {len(rows)} ops, {total_bytes/1e6:.1f} MB, {total_time/1e6:.3f} ms ---")

    # NVTX ranges
    nvtx = query_nvtx_ranges(db_path)
    if nvtx:
        print(f"\n--- NVTX Ranges: {len(nvtx)} ---")
        for text, start, end, dur in nvtx[:30]:
            if dur and dur > 0:
                print(f"  {text}: {dur/1e6:.3f} ms")

    return {
        "kernels": kernels,
        "by_name": dict(by_name),
        "gaps": gaps,
        "api_calls": api_calls,
    }


def compare(fp8_data, bf16_data):
    """Compare FP8 vs BF16 results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: FP8 vs BF16")
    print(f"{'='*80}")

    fp8_kernels = fp8_data.get("kernels", [])
    bf16_kernels = bf16_data.get("kernels", [])

    fp8_total = sum(k[3] for k in fp8_kernels) / 1e6
    bf16_total = sum(k[3] for k in bf16_kernels) / 1e6
    fp8_count = len(fp8_kernels)
    bf16_count = len(bf16_kernels)

    fp8_wall = (fp8_kernels[-1][2] - fp8_kernels[0][1]) / 1e6 if fp8_kernels else 0
    bf16_wall = (bf16_kernels[-1][2] - bf16_kernels[0][1]) / 1e6 if bf16_kernels else 0

    fp8_gaps = sum(g[0] for g in fp8_data.get("gaps", [])) / 1e6
    bf16_gaps = sum(g[0] for g in bf16_data.get("gaps", [])) / 1e6

    print(f"\n{'Metric':<30} {'FP8':>15} {'BF16':>15} {'Ratio':>10}")
    print("-" * 72)
    print(f"{'Kernel launches':<30} {fp8_count:>15} {bf16_count:>15} {fp8_count/bf16_count if bf16_count else 0:>10.2f}x")
    print(f"{'Total kernel time (ms)':<30} {fp8_total:>15.3f} {bf16_total:>15.3f} {fp8_total/bf16_total if bf16_total else 0:>10.2f}x")
    print(f"{'Total gap time (ms)':<30} {fp8_gaps:>15.3f} {bf16_gaps:>15.3f} {fp8_gaps/bf16_gaps if bf16_gaps else 0:>10.2f}x")
    print(f"{'Wall time (ms)':<30} {fp8_wall:>15.3f} {bf16_wall:>15.3f} {fp8_wall/bf16_wall if bf16_wall else 0:>10.2f}x")
    print(f"{'GPU utilization':<30} {fp8_total/fp8_wall*100 if fp8_wall else 0:>14.1f}% {bf16_total/bf16_wall*100 if bf16_wall else 0:>14.1f}%")

    # Compare unique kernel types
    fp8_types = set(fp8_data.get("by_name", {}).keys())
    bf16_types = set(bf16_data.get("by_name", {}).keys())
    only_fp8 = fp8_types - bf16_types
    only_bf16 = bf16_types - fp8_types

    if only_fp8:
        print(f"\nKernels ONLY in FP8 ({len(only_fp8)}):")
        for k in sorted(only_fp8):
            durs = fp8_data["by_name"][k]
            print(f"  {k[:70]}: {len(durs)} calls, {sum(durs)/1000:.1f} µs total")

    if only_bf16:
        print(f"\nKernels ONLY in BF16 ({len(only_bf16)}):")
        for k in sorted(only_bf16):
            durs = bf16_data["by_name"][k]
            print(f"  {k[:70]}: {len(durs)} calls, {sum(durs)/1000:.1f} µs total")


def main():
    if len(sys.argv) < 2:
        print("Usage: python nsys_analyze.py <fp8.sqlite> [bf16.sqlite]")
        sys.exit(1)

    fp8_path = sys.argv[1]
    fp8_data = analyze_single(fp8_path, "FP8" if "fp8" in fp8_path.lower() else "Primary")

    if len(sys.argv) >= 3:
        bf16_path = sys.argv[2]
        bf16_data = analyze_single(bf16_path, "BF16" if "bf16" in bf16_path.lower() else "Secondary")
        compare(fp8_data, bf16_data)


if __name__ == "__main__":
    main()
