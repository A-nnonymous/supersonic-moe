# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For the complete project state, read `reports/fp8_upgrade/HANDOFF.md`.

## Current Status (2026-04-01)

- **FP8 significantly beats BF16** across all tested shapes
- NSYS GPU projection: **14.9%** (I=1024), **42.5%** (I=2048), **49.4%** (I=4096) faster
- Wall-clock: **1.66×** to **2.37×** faster
- Precision: **44/44** tests pass across 4 seeds (RelRMSE <10%, correlation >0.99)
- Memory: FP8 uses 1.38–1.48× more (weight caches, not yet a win)
- Branch: `fork-main-sync` at commit `e5d3ca8`

## Best training path

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0
```

## Scope

- Full-chain blockscaled FP8 (1×32 UE8M0) MoE training on Blackwell (sm_100a)
- Goal: FP8 E2E faster than official BF16
- Key files:
  - `sonicmoe/functional/__init__.py` — main FP8 logic
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant kernels, weight caches
  - `sonicmoe/quack_utils/gemm_dgated.py` — CUTLASS GemmDGated

## Non-Negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback behavior intact
- Use NSYS NVTX GPU projection + sync barriers for authoritative comparisons
- Compare against official BF16, never fork BF16
- Exclude `elementwise_kernel` from BF16 baseline (QuACK 0.3.7 bug)

## Validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

CUDA_VISIBLE_DEVICES=7 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

CUDA_VISIBLE_DEVICES=7 python tools/measure_aligned_perf_memory.py
```

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, all numbers, lessons, next steps
2. `reports/fp8_upgrade/engineering_log.md` — what happened and why
3. `sonicmoe/functional/__init__.py` — main code
