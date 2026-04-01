# SonicMoE Agent Context

Use this as the quick cold-start summary. For full state, measurements, and lessons, read `reports/fp8_upgrade/HANDOFF.md` first.

## Current Status (2026-04-01)

- **FP8 significantly beats BF16** across all tested shapes
- NSYS GPU projection: **14.9%** (I=1024), **42.5%** (I=2048), **49.4%** (I=4096) faster
- Wall-clock: **1.66×** to **2.37×** faster
- Precision: **44/44** tests pass across 4 seeds (RelRMSE <10%, correlation >0.99)
- Memory: FP8 uses 1.38–1.48× more (not yet a win)
- Branch: `fork-main-sync` at `e5d3ca8`

## Best training path

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0
```

## Key files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all selective FP8 decisions |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, FP8 GEMM wrappers, weight caches |
| `sonicmoe/quack_utils/gemm_dgated.py` | Low-level CUTLASS GemmDGated |
| `reports/fp8_upgrade/HANDOFF.md` | **Complete project state, measurements, lessons, next steps** |
| `reports/fp8_upgrade/engineering_log.md` | Cleaned milestone log |

## Non-negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback intact
- Use NSYS NVTX GPU projection for performance claims (exclude `elementwise_kernel` from BF16)
- Compare against official BF16, never fork BF16

## Quick validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

CUDA_VISIBLE_DEVICES=7 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

## What NOT to waste time on

- FP8 wgrad at T=4096/E=128 (K_per_expert=256 too small, proven dead-end)
- Eager FP8 weight cache eviction (makes memory worse)
- FP8 down-proj at I=1024 (quant cost > GEMM savings)
- See HANDOFF.md §3 for full dead-end list

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, all numbers, lessons, next steps
2. `reports/fp8_upgrade/engineering_log.md` — what happened and why
3. `sonicmoe/functional/__init__.py` — main code
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant + GEMM wrappers
