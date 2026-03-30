# SonicMoE Agent Quick Context

## Status (2025-03-30)

**1.58x E2E over BF16** at production shape (E=128, tpe=256, T=4096, H=4096, I=1024).
8/8 contract tests PASS.

| Phase | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| Forward | 1.134 | 1.143 | 0.99x |
| Backward | 3.624 | 1.870 | **1.94x** |
| Total | 4.758 | 3.014 | **1.58x** |

Precision: RelRMSE 5.3-6.6%, Correlation 0.998 (all within <10% / >0.99 thresholds).

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Complete status, kernel breakdown, profiling methodology, lessons, next steps
2. `AGENTS.md` — Scope, non-negotiables, architecture, validation commands

## Current Blocker

FP8 forward can't beat BF16 because `GemmGatedSm100` (fused GEMM+SwiGLU) crashes with blockscaled FP8 — `cute.recast_layout(2, 1, ...)` incompatible with `MmaMXF8Op` accumulator TMEM layout. Not fixable via monkey-patch. Needs CUTLASS C++ standalone kernel or quack ≥ 0.4.0.

## Key Architecture

- **BF16 forward**: 2 fused kernels (`GemmGatedSm100` + `GemmDefaultSm100`)
- **FP8 forward**: 4 kernels (gather_quant + GEMM + SwiGLU_quant + GEMM) — decomposed because fused crashes
- **BF16 backward**: `GemmDGatedSm100` (fused act-grad + dSwiGLU) + 2 weight-grad GEMMs
- **FP8 backward**: 2 FP8 act-grad GEMMs + Triton SwiGLU bwd + 2 BF16 weight-grad GEMMs
- **Weight-grad stays BF16**: per-expert M=256 too small, FP8 overhead > bandwidth savings (verified 3.6x slower)

## Environment

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
# Official unmodified repo: /root/.../official/sonic-moe (578 lines functional/__init__.py)
# See /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md for cluster details
```

## Key API Facts (quack 0.3.7)

- `blockscaled_fp8_gemm_varlen(a, w, cu_seqlens, ...)` — core FP8 GEMM
- `gather_quantize_and_pack_activation(x, idx)` — fused gather+quant+ISA-pack
- `swiglu_forward_quant_pack_zsave_triton(z)` — fused SwiGLU+quant+z-save
- `precompute_weight_fp8(w)` — weight cache (limit 8, covers fwd+bwd)
- `_ALIGNMENT_ASSUMED` — global flag, gates FP8 vs BF16 fallback
- `enable_quack_gemm()` — context manager (bare call is no-op; set `USE_QUACK_GEMM=1` env var before import instead)

## Next Priority

1. **Forward speedup**: CUTLASS C++ fused GEMM+SwiGLU+FP8 kernel (bypass DSL) — only way to beat BF16 forward
2. **Backward overlap**: multi-stream act-grad ∥ weight-grad
3. **Token rounding**: routing-layer 128-alignment guarantee
