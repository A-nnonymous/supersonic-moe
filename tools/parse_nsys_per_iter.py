"""Parse nsys sqlite, compute per-iter GPU-projection µs for each BENCH range."""

from __future__ import annotations

import argparse
import sqlite3
import sys


def parse(db: str) -> None:
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get NVTX ranges. Schema: NVTX_EVENTS(eventType,start,end,text,domainId,...)
    bench_rows = cur.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'BENCH_%'"
    ).fetchall()
    iter_rows = cur.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'ITER%'"
    ).fetchall()
    kernel_rows = cur.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()

    print(f"{'Bench':<25s}  {'iters':>5s}  {'mean_µs':>10s}  {'p50_µs':>10s}  {'min_µs':>10s}  {'max_µs':>10s}")
    print("-" * 80)

    for label, b_start, b_end in bench_rows:
        # iter ranges that fall inside this bench
        iters = [(s, e) for _, s, e in iter_rows if b_start <= s and e <= b_end]
        per_iter_us = []
        for s, e in iters:
            # union of kernel intervals overlapping [s, e]
            ks = [(max(ks_, s), min(ke_, e)) for ks_, ke_ in kernel_rows if ke_ > s and ks_ < e]
            if not ks:
                per_iter_us.append(0.0)
                continue
            ks.sort()
            merged = [ks[0]]
            for a, b in ks[1:]:
                if a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            total_ns = sum(b - a for a, b in merged)
            per_iter_us.append(total_ns / 1000.0)
        per_iter_us.sort()
        n = len(per_iter_us)
        if n == 0:
            print(f"{label:<25s}  no iter data")
            continue
        mean = sum(per_iter_us) / n
        p50 = per_iter_us[n // 2]
        print(f"{label:<25s}  {n:>5d}  {mean:>10.2f}  {p50:>10.2f}  {per_iter_us[0]:>10.2f}  {per_iter_us[-1]:>10.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("db")
    a = p.parse_args()
    parse(a.db)
