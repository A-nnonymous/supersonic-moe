# Directory Index: `/reports/`

> Collected benchmark outputs, summaries, and historical experiment artifacts.
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
| `fp8_upgrade/` | Historical FP8-upgrade engineering log and benchmark reports. | Engineering log is canonical history (Sessions 1-65, Phases 1-25). |
| `grid_session53/` | Session 53 grid benchmark shards and consolidated JSON output. | — |
| `nsys_final/` | Final consolidated nsys-derived breakdowns for Session 53. | — |
| `wgrad_tma_add_nsys/` | Session 65 TMA reduce-add nsys profiles + RESULTS.json. | 4 nsys-rep files for Nsight Systems GUI |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `CONTRIBUTION_SUMMARY.md` | Publication-style summary of paddle_compat branch contribution (Session 53-58). | — |
| `cross_framework_report.md` | 4-way Paddle/SonicMoE BF16/FP8 precision comparison. | — |
| `fp8_frontier_path_analysis.json` | Compiled BF16-vs-FP8 path-comparison report consumed by the visualization module. | — |
| `quant_bench_final.json` | Quant benchmark snapshot emitted by `tools/introspect.py`. | — |
| `session53_breakdown.md` | Session 53 — Performance, Memory & Precision Breakdown (Final). | canonical 27-shape data |
| `session53_nsys_consolidated.json` | Consolidated nsys data for Session 53. | — |
| `wgrad_bench.json` | Structured wgrad benchmark report. | — |
| `wgrad_fp8_benchmark_legacy.json` | Legacy wgrad benchmark snapshot kept for history. | — |

## Redundancy and cleanup watchlist
- `reports/README.md` should stay aligned with `docs/HANDOFF.md` whenever the authoritative handoff changes.
- `quant_bench.json` and `quant_bench_final.json` look like structured-vs-legacy variants of the same benchmark family; verify the intended canonical file before adding new results.
- `wgrad_fp8_benchmark_legacy.json` is historical only; new wgrad report outputs should stay structured and live under `reports/`.
