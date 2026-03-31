# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For detailed handoff context, read `agent.md`, `reports/fp8_upgrade/HANDOFF.md`, and `reports/fp8_upgrade/engineering_log.md`.

## Current Status (2026-03-31)

- Current best training path: aligned fused FP8 with `SONIC_MOE_FP8_FUSED_GATED=1` and `SONIC_MOE_FP8_WGRAD=0`
- Authoritative training NSYS GPU projection:
  - official BF16: `2475.2us`
  - current fused FP8 + BF16 wgrad: `2600.3us`
  - current fused FP8 + FP8 wgrad: `5650.4us`
- 8/8 FP8 contract tests pass
- Current local aligned inference and peak-memory measurements still lose to BF16
- The old `2930us / 455us gap` story is stale for the current fused branch

**Key task:** redesign the FP8 weight-grad path. The main frontier is no longer forward SwiGLU trimming; it is full-chain FP8 wgrad.

## Scope

- Full-chain blockscaled FP8 (1×32 UE8M0) MoE training on Blackwell (sm_100a)
- Goal: FP8 E2E faster than official BF16 and eventually better on memory / inference too
- Key files:
  - `sonicmoe/functional/__init__.py`
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - `sonicmoe/functional/backward.py`
  - `sonicmoe/functional/moe_config.py`
  - `sonicmoe/functional/grouped_gemm.py`

## Non-Negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback behavior intact
- Use NSYS NVTX GPU projection + sync barriers for authoritative comparisons
- Compare against official BF16, never fork BF16
- Do not claim memory or inference wins without measurement

## Validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

CUDA_VISIBLE_DEVICES=0 python tools/measure_aligned_perf_memory.py
```
