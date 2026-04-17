# Directory Index: `/benchmarks/`

> One-off and repeatable benchmark entrypoints for FP8, BF16, and routing experiments.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Stable child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `clean_results/` | Saved clean-run benchmark text outputs used as reference snapshots. | — |
| `nsys_run/` | Minimal benchmark entrypoints tailored for nsys profiling runs. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `nsys_clean/` | Volatile / generated subtree. | Raw nsys capture artifacts (`.sqlite`, `.nsys-rep`); high churn and not useful for durable indexing. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `_cold_mem.sh` | Shell helper for cold mem. | — |
| `bench_triton_moe.py` | Benchmark entrypoint for bench triton moe. | — |
| `e2e_fp8_vs_bf16.py` | End-to-end FP8 vs BF16 benchmark for SonicMoE. | — |
| `mem_precision_remote.sh` | Shell helper for mem precision remote. | — |
| `moe-cute.py` | Benchmark entrypoint for moe cute. | — |
| `moe-token-rounding.py` | Benchmark entrypoint for moe token rounding. | — |
| `nsys_remote_bench.sh` | Shell helper: Self-contained nsys benchmark script for remote execution on idle node.. | — |
