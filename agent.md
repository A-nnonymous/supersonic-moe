# SonicMoE Agent Context

Use this as the quick cold-start summary for the next agent. For the full state, measurements, and lessons, read `reports/fp8_upgrade/HANDOFF.md` and `reports/fp8_upgrade/engineering_log.md` first.

## Current Status (2026-03-31)

- The **best current training path** is aligned fused FP8 with **BF16 wgrad**:
  - `SONIC_MOE_FP8_FUSED_GATED=1`
  - `SONIC_MOE_FP8_WGRAD=0`
- **Authoritative training NSYS GPU projection**:
  - official BF16: `777.3us` fwd + `1697.9us` bwd = `2475.2us`
  - current fused FP8 + BF16 wgrad: `812.1us` fwd + `1788.2us` bwd = `2600.3us`
  - current fused FP8 + FP8 wgrad: `812.0us` fwd + `4838.4us` bwd = `5650.4us`
- **Meaning:** the aligned fused act-grad path is mostly landed; the remaining training blocker is **FP8 wgrad**.
- `tests/fp8_large_project_contract_test.py`: **8 passed, 3 deselected** with `SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1`.
- Current local aligned inference and peak-memory measurements still lose to BF16, so do not claim an inference or memory win.

## What not to waste time on first

- Do **not** start by repeating the old “trim Triton SwiGLU first” loop.
- Do **not** use fork BF16 as the baseline.
- Do **not** enable `SONIC_MOE_FP8_WGRAD=1` by default and assume the job is done.

## Best next direction

Focus on **FP8 weight-grad redesign**.

The strongest concrete path is to study and potentially adapt the specialized weight-grad kernels already in-tree:

- `sonicmoe/functional/backward.py`
- `sonicmoe/functional/moe_config.py`
- `sonicmoe/functional/grouped_gemm.py`

Key symbols:

- `HopperWgmma_MoE_Up_proj_WeightGrad_Bwd`
- `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd`
- `HopperWgmma_MoE_kernel(..., compute_weight_gradient=True)`

## Read order

1. `reports/fp8_upgrade/HANDOFF.md`
2. `reports/fp8_upgrade/engineering_log.md`
3. `sonicmoe/functional/__init__.py`
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
5. `sonicmoe/functional/backward.py`
6. `sonicmoe/functional/moe_config.py`
7. `sonicmoe/functional/grouped_gemm.py`
