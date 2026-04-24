# Directory Index: `/docs/`

> Canonical architecture, handoff, and design documentation.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `KNOWLEDGE_BASE.md` | Self-contained expert knowledge base: architecture, padding proof, dead ends, config, file map. | Session 60, canonical reference |
| `HANDOFF.md` | Redirect to root `HANDOFF.md` (Session 65). | redirect only |
| `pad_audit_methodology.md` | Route-level padding mathematical proof and correctness analysis. | Session 57, valid |
| `session60_lessons.md` | Session 60 engineering lessons (#67-#72): gate-MLP gradient chain, torch-proxy compat. | Session 60, current |
| `cute_dsl_optimization_guide.md` | CuTe DSL FP8 Quantization — Architecture & Optimization Guide. | Session 43, architecture valid |
| `FP8_ARCH_SPEC.md` | SonicMoE FP8 Frontier — Architecture Specification. | Session 42, architecture valid, perf numbers stale |
| `fp8_architecture_comparison.md` | FP8 MoE 三方架构对比：DeepEP · SonicMoE BF16 · SonicMoE FP8 Frontier. | Session 33-35, architecture valid, perf numbers stale |
| `wgrad_fp8_dual_quant_design.md` | Wgrad FP8 Optimization — Zero-Copy Dual Quantization Design. | Historical design doc, superseded by split strategy |
| `setup_dev_b.sh` | Blackwell (dev_b) environment setup script. | dev_b env has cutlass mismatch, not usable for quack |
