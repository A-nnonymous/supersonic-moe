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
| `fp8_upgrade/` | Historical FP8-upgrade notes; partly superseded by newer docs and reports. | `reports/fp8_upgrade/HANDOFF.md` is stale and explicitly superseded by `docs/HANDOFF.md`. |
| `grid_session53/` | Session 53 grid benchmark shards and consolidated JSON output. | — |
| `nsys_final/` | Final consolidated nsys-derived breakdowns for Session 53. | — |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `fp8_frontier_path_analysis.json` | Compiled BF16-vs-FP8 path-comparison report consumed by the new visualization module. | untracked in git |
| `quant_bench.json` | Structured quant benchmark report with per-kernel summaries and metadata. | — |
| `quant_bench_final.json` | Legacy flat quant benchmark snapshot still emitted by `tools/introspect.py`. | legacy snapshot, compare with `reports/quant_bench.json` before reusing |
| `README.md` | High-level map of report outputs and profiling artifacts. | keep aligned with `docs/HANDOFF.md` |
| `session53_breakdown.md` | Markdown note: Session 53 — Performance, Memory & Precision Breakdown (Final). | — |
| `session53_full_report.json` | JSON artifact with top-level keys: `metadata`, `shapes`. | — |
| `session53_nsys_consolidated.json` | JSON artifact with top-level keys: `session`, `device`, `method`, `common`. | — |
| `wgrad_bench.json` | Structured wgrad benchmark report under `reports/`. | — |
| `wgrad_fp8_benchmark_legacy.json` | Legacy full-replacement wgrad benchmark snapshot kept for history. | legacy benchmark snapshot, kept for historical comparison with `reports/wgrad_bench.json`, untracked in git |

## Redundancy and cleanup watchlist
- `reports/README.md` should stay aligned with `docs/HANDOFF.md` whenever the authoritative handoff changes.
- `quant_bench.json` and `quant_bench_final.json` look like structured-vs-legacy variants of the same benchmark family; verify the intended canonical file before adding new results.
- `wgrad_fp8_benchmark_legacy.json` is historical only; new wgrad report outputs should stay structured and live under `reports/`.
