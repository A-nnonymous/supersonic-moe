# SonicMoE Work Reports

This directory is the live work log for the FP8 upgrade effort. It is not meant to hold speculative essays; it should hold the current state, the validated commands, and the next handoff targets.

> 从当前阶段开始，工程记录统一使用中文；每次记录都必须先写清楚指标注释、基线和收益来源。

## Current status

- authoritative Python environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- Blackwell path: QuACK-enabled (`USE_QUACK_GEMM=1`)
- latest validated fork state: the current `fork-main-sync` working tree carrying the Blackwell FP8 protocol changes
- latest targeted validation:
  - stable Blackwell fp8 regression: `USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - stable result: `10 passed`
  - env-on blockscaled regression: `USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=128 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - env-on blockscaled result: `10 passed`
  - serial: `python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py`
  - serial result: `18 passed, 91 skipped`
  - opt-in parallel: `make test-blackwell-parallel PYTEST_WORKERS=2`
  - parallel result: `18 passed, 91 skipped in 168.14s`
  - multi-GPU dry-run: `python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run`
  - fp8 metric probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics`
  - latest fused-path validation:
    - `USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'preact_cutely_fused_path_matches_reference_boundary or boundary_keeps_finite_forward_backward or blackwell_fp8_protocol_runtime_and_reference_quant'`
    - `USE_QUACK_GEMM=1 python -m pytest -q tests/moe_blackwell_test.py`

## What is already done

- upstream `main` was merged into the local branch before the fork push
- `tests/moe_test.py` was made Blackwell-aware
- `tests/moe_blackwell_test.py` was added as a dedicated QuACK smoke/regression test
- `Makefile` now exposes `make test-blackwell`
- `README.md` now carries the active TODO list and the FP8 roadmap instead of stale control-plane material
- a Blackwell-only FP8 protocol layer now exists in:
  - `sonicmoe/functional/fp8_protocol.py`
  - `sonicmoe/functional/fp8_quant.py`
  - `sonicmoe/functional/fp8_reference.py`
- the protocol is wired through `MoE.forward(..., fp8_protocol=...)` and `moe_TC_softmax_topk_layer(..., fp8_protocol=...)`
- a gated adapter landing point now exists in `sonicmoe/functional/fp8_cutely_fused.py`
- the first real pre-SwiGLU fused forward step is now landed in the same file
- current protocol scope is intentionally constrained to:
  - activation dtype: `torch.float8_e4m3fn`
  - scale encoding: `torch.float8_e8m0fnu`
  - stable mainline scale granularity: `1x128`
  - runtime target: Blackwell + QuACK enabled
- `1x32 ue8m0` blockscaled down-proj scaffolding is now in-tree:
- `1x32 ue8m0` blockscaled down-proj experimental path is now runnable end-to-end with static aligned capacity:
  - file: `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - current implementation: static `expert capacity` + GPU pack/unpack transition path
  - default status: disabled
  - enable flag: `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1`
  - required contract: `SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=<128 的整数倍>`
  - current blocker: GPU pack/unpack overhead still dominates
- current FP8 boundary path is still slower than baseline; see `reports/fp8_upgrade/ENGINEERING_LOG.md` for exact numbers
- 最新 reciprocal 小里程碑已落地：
  - `sonicmoe/functional/fp8_quant.py` 改为 reciprocal-multiply
  - `cutify/ops/cute/fused_weighted_swiglu_act_quant.py` 的 4 组 scale 路径改为 `cute.arch.rcp_approx(...)`
  - 局部量化微基准：`5.80%` 提速，`RMSE=0`
- 当前最接近可交付的真实路径是稳定 `fp8-mainline`，不是 env-on blockscaled：
  - 中等 shape `4096,4096,1024,128,8`
  - 稳定 `fp8-mainline`：
    - output RMSE：`0.01073638`
    - bf16 peak / fp8 peak：`7049.88 / 6867.00 MiB`
    - bf16 e2e / fp8 e2e：`7.338 / 7.693 ms`
  - env-on blockscaled：
    - output RMSE：`0.01073363`
    - bf16 peak / fp8 peak：`7049.88 / 7396.13 MiB`
    - bf16 e2e / fp8 e2e：`7.338 / 11.668 ms`
- 因此当前下一优先级已经明确：
  1. 继续消掉 blockscaled 的 `grouped_out` / router 聚合过渡层；
  2. 修掉更大 shape `8192,4096,1024,128,8` 在 preact fused quant kernel 上的 runtime crash；
  3. 然后再继续 backward/mainloop 迁移。
- 最新进展补充：
  - blockscaled 已经吃掉 `grouped_a` 物化，改成 `pack+quant` 融合；
  - 中等 shape `4096,4096,1024,128,8`：
    - output RMSE：`0.01073363`
    - bf16 peak / blockscaled peak：`7049.88 / 7396.13 MiB`
    - `pack+quant` 融合后 e2e：`8.414 ms`
    - grouped router 直连后 e2e：`8.196 ms`
    - 相对 `11.668 ms` 的累计收益：`29.75%`
  - 结论：blockscaled 主矛盾已进一步切换为 `grouped_out` 本体，不再是前半段 `pack+quant` 或 flat unpack。

## What the next agent should do first

1. Read `reports/fp8_upgrade/HANDOFF.md`
2. Confirm the environment with `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
3. Re-run `USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
4. Continue from `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`, and first remove GPU pack/unpack / make routing metadata directly produce static blockscaled layout
5. 如果要快速复现实测差距，优先跑中等 shape `4096,4096,1024,128,8`，不要直接上当前会 crash 的 `8192,4096,1024,128,8`

## Working rule

Whenever the active plan changes, update this directory first so the next agent does not need to re-discover state from commit history or chat logs.
