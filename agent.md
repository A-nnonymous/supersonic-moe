# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For detailed handoff, debugging history, and kernel breakdowns, read `reports/fp8_upgrade/HANDOFF.md`.

## Current Status (2026-03-31, Session 21)

**⚠️ FP8 is 18% SLOWER than official BF16 baseline (2930µs vs 2475µs).**

Previous reports of "1.59x speedup" were misleading — they compared against fork BF16 (4581µs)
which has 2101µs of spurious contiguous copy overhead from quack 0.3.7 layout differences.
The TRUE baseline is official BF16 (quack 0.2.5) at 2475µs.

8/8 contract tests PASS. Precision: RelRMSE 5.3-6.6%, Correlation 0.998.

## The Core Problem

FP8's Triton quant/SwiGLU overhead (~850µs total) greatly exceeds FP8 GEMM savings (~218µs).
The forward also loses ~160µs because fused GemmGated (BF16-only) can't be used with FP8.

**To beat official BF16, the next agent must close a 455µs gap.**

## Priority 1: Triton Kernel Optimization

| Kernel | Current | Theoretical Min | Headroom |
|--------|---------|-----------------|----------|
| `_swiglu_fwd_quant_pack_zsave_kernel` | 229µs | 44µs | 5.2x |
| `_swiglu_bwd_quant_pack_kernel` | 384µs | 49µs | 7.8x |
| `_gather_quantize_and_pack_kernel` ×2 | 196µs | 107µs | 1.8x |

Biggest wins: increase BLOCK_ROWS, reduce group-loop iterations, sigmoid approximation in bwd.
Use `tools/ncu_profile_kernels.py` + ncu for roofline analysis before optimizing.

## Priority 2: Restore fused forward GEMM+SwiGLU

Official uses `GemmGatedSm100` (fused GEMM+SwiGLU, 418µs).
Fork FP8 decomposes into GEMM (263µs) + SwiGLU (229µs) = 492µs.
Fixing CUTLASS DSL `recast_layout` bug would unlock ~74µs savings.

## Non-Negotiable

- All FP8 paths must maintain <10% RelRMSE, >0.99 correlation vs BF16
- Use native CUTLASS/QuACK GEMM path, not Triton `tl.dot_scaled` (broken on sm_100a)
- Non-aligned routing must auto-fallback to BF16 fused path (no performance penalty)
- **Always profile with nsys NVTX GPU Projection + sync barriers, never trust CUDA events alone**
- **Always compare against official BF16 (2475µs), never against fork BF16**

## Architecture

- QuACK (quack-kernels 0.3.7) wraps CUTLASS DSL for Blackwell SM100
- `GemmGatedSm100` = fused GEMM+SwiGLU (BF16 only — crashes with blockscaled FP8)
- `GemmDefaultSm100` = plain GEMM (works with blockscaled FP8)
- Blockscaled FP8 uses ISA-packed E8M0 scales with `sf_vec_size=32`
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

# nsys breakdown analysis
python tools/nsys_full_breakdown.py reports/sonic_fork_fp8_v4.sqlite
```

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Complete status, three-way kernel breakdown, gap analysis
2. `sonicmoe/functional/__init__.py` — FP8/BF16 dispatch, alignment gating
3. `sonicmoe/quack_utils/swiglu_triton.py` — Triton SwiGLU kernels (optimization target)
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — Core FP8 GEMM + quant kernels
