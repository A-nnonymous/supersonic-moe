# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For detailed handoff context and debugging history, read `agent.md` and `reports/fp8_upgrade/HANDOFF.md`.

## Current Status (2025-03-30)

**1.58x E2E over BF16** at production shape (E=128, tpe=256, T=4096, H=4096, I=1024).
Forward: 0.99x (bottleneck), Backward: 1.94x. 8/8 contract tests PASS.
Precision: RelRMSE 5.3-6.6%, Correlation 0.998.

## Scope

- Full-chain blockscaled FP8 (1×32 UE8M0) MoE training on Blackwell (sm_100a)
- Goal: maximize E2E speedup over BF16, currently 1.57x. Forward fusion is the key blocker.
- Key files: `sonicmoe/functional/__init__.py`, `blockscaled_fp8_gemm.py`, `swiglu_triton.py`

## Non-Negotiable

- All FP8 paths must maintain <10% RelRMSE, >0.99 correlation vs BF16
- Use native CUTLASS/QuACK GEMM path, not Triton `tl.dot_scaled` (broken on sm_100a)
- Non-aligned routing must auto-fallback to BF16 fused path (no performance penalty)
- Benchmark with `tools/bench_aligned_e2e.py` (E2E, zero_grad) — never trust kernel-only numbers

## Architecture

- QuACK (quack-kernels 0.3.7) wraps CUTLASS DSL (nvidia-cutlass-dsl 4.4.2) for Blackwell SM100
- `GemmGatedSm100` = fused GEMM+SwiGLU (BF16 only — crashes with blockscaled FP8)
- `GemmDefaultSm100` = plain GEMM (works with blockscaled FP8, used in decomposed path)
- Blockscaled FP8 uses ISA-packed E8M0 scales with `sf_vec_size=32`
- Weight layout: interleaved gate/value columns `[g0, v0, g1, v1, ...]`
- `_ALIGNMENT_ASSUMED` gates FP8 vs BF16 fallback in `functional/__init__.py`

## Environment

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
# See /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md for cluster details
```

## Validation

```bash
# Contract tests (8/8 small pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# E2E benchmark (production shape)
CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py
```

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Complete status, kernel timings, lessons, next steps
2. `agent.md` — Quick handoff context
3. `sonicmoe/functional/__init__.py` — FP8/BF16 dispatch, alignment gating
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — Core FP8 GEMM + weight cache
5. `sonicmoe/quack_utils/swiglu_triton.py` — Fused SwiGLU+quant Triton kernels