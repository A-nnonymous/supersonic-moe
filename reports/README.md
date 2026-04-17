# SonicMoE Reports Map

## Canonical references

| File | Role |
|------|------|
| [`../docs/HANDOFF.md`](../docs/HANDOFF.md) | **Authoritative project handoff** — current state, architecture, measurements, and validation guidance |
| [`../INDEX.md`](../INDEX.md) | Root directory map with redundancy markers and shallow overview |
| [`INDEX.md`](INDEX.md) | Reports subtree index for quick navigation inside this directory |

## Historical FP8-upgrade documents

| File | Role |
|------|------|
| [`fp8_upgrade/HANDOFF.md`](fp8_upgrade/HANDOFF.md) | **Stale historical handoff** retained for reference only; superseded by `docs/HANDOFF.md` |
| [`fp8_upgrade/engineering_log.md`](fp8_upgrade/engineering_log.md) | Chronological engineering log and milestone history |
| [`fp8_upgrade/BLOCKSCALED_ALIGNMENT.md`](fp8_upgrade/BLOCKSCALED_ALIGNMENT.md) | Alignment and ISA scale-packing reference |
| [`fp8_upgrade/FP8_BENCHMARK_REPORT.md`](fp8_upgrade/FP8_BENCHMARK_REPORT.md) | Older benchmark narrative for the FP8-upgrade phase |

## Current result bundles

| File / Directory | Role |
|------------------|------|
| [`session53_breakdown.md`](session53_breakdown.md) | Human-readable Session 53 performance / memory / precision summary |
| [`session53_full_report.json`](session53_full_report.json) | Structured aggregate Session 53 report |
| [`session53_nsys_consolidated.json`](session53_nsys_consolidated.json) | Consolidated nsys-derived data |
| [`grid_session53/`](grid_session53/) | Per-GPU grid shards plus merged Session 53 grid output |
| [`nsys_final/`](nsys_final/) | Final nsys-derived kernel breakdown artifacts |
| [`wgrad_bench.json`](wgrad_bench.json) | Structured wgrad benchmark report |
| [`wgrad_fp8_benchmark_legacy.json`](wgrad_fp8_benchmark_legacy.json) | Legacy full-replacement wgrad benchmark snapshot moved out of the repo root |

## Benchmark data variants to treat carefully

- `quant_bench.json` and `quant_bench_final.json` appear to be structured-vs-legacy variants of the same benchmark family; confirm which one is canonical before extending either.
- `wgrad_bench.json` is the report-local structured wgrad benchmark artifact.
- `wgrad_fp8_benchmark_legacy.json` is historical only; new benchmark outputs should stay under `reports/` and avoid recreating a root-level duplicate.
