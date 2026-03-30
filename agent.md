# SonicMoE Agent Quick Context

## Status (2025-03-30)

**1.57x E2E over BF16** at production shape (E=128, tpe=256). 8/8 contract tests PASS.

- Forward: 1.142ms FP8 vs 1.130ms BF16 = **0.99x** (bottleneck: 4 kernels vs 2 fused)
- Backward: 1.894ms FP8 vs 3.650ms BF16 = **1.93x** (act-grad GEMM weight bandwidth halved)

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Complete status, kernel breakdown, lessons, next steps
2. `AGENTS.md` — Scope, non-negotiables, architecture

## Current Blocker

FP8 forward can't beat BF16 because `GemmGatedSm100` (fused GEMM+SwiGLU) crashes with blockscaled FP8 inputs — `cute.recast_layout(2, 1, ...)` incompatible with `MmaMXF8Op` accumulator TMEM layout. Not fixable via monkey-patch. Needs CUTLASS C++ standalone kernel or quack ≥ 0.4.0.

## Environment

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
# See /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md for cluster details
```

## Key API Facts (quack 0.3.7)

- `blockscaled_fp8_gemm_varlen(a, w, cu_seqlens, ...)` — core FP8 GEMM
- `gather_quantize_and_pack_activation(x, idx)` — fused gather+quant+ISA-pack
- `swiglu_forward_quant_pack_zsave_triton(z)` — fused SwiGLU+quant+z-save
- `precompute_weight_fp8(w)` — weight cache (limit 8, covers fwd+bwd)
- `_ALIGNMENT_ASSUMED` — global flag, gate FP8 paths on alignment

## Next Priority

1. **Forward speedup**: CUTLASS C++ fused GEMM+SwiGLU+FP8 kernel (bypass DSL)
2. **Backward overlap**: multi-stream act-grad ∥ weight-grad
3. **Token rounding**: routing-layer 128-alignment guarantee
