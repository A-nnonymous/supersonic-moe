# SonicMoE Work Reports

This directory is the live work log for the FP8 upgrade effort. It is not meant to hold speculative essays; it should hold the current state, the validated commands, and the next handoff targets.

> 从当前阶段开始，工程记录统一使用中文；每次记录都必须先写清楚指标注释、基线和收益来源。

## Current status

- authoritative Python environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- Blackwell path: QuACK-enabled (`USE_QUACK_GEMM=1`)
- latest validated fork state: `80271c3` on `fork-main-sync`, plus local vec4 fallback / chunked-launch changes in `operator-incubator`
- latest targeted validation:
  - stable Blackwell fp8 regression: `USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - stable result: `15 passed`
  - env-on blockscaled regression: `USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=128 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - env-on blockscaled result: `13 passed`
  - serial: `python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py`
  - serial result: `18 passed, 91 skipped`
  - opt-in parallel: `make test-blackwell-parallel PYTEST_WORKERS=2`
  - parallel result: `18 passed, 91 skipped in 168.14s`
  - multi-GPU dry-run: `python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run`
  - fp8 metric probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics`
  - fp8 theory probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_fp8_analysis`
  - stagewise memory probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_stage_memory`
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
    - bf16 e2e / fp8 e2e：`7.338 / 8.196 ms`
- 阶段显存新结论（shape `4096,4096,1024,128,8`）：
  - 稳定 `fp8-mainline`：`forward:down-proj-router` peak alloc `4012.00 MiB`，final peak `7050.88 MiB`
  - env-on blockscaled：`forward:down-proj-router` peak alloc `5309.38 MiB`，final peak `7580.13 MiB`
  - 结论：blockscaled 当前真正的显存墙已经是 `grouped_out` / `down-proj-router`，不是前半段 `pack+quant`
- 更大真实 shape 已可跑通：
  - shape `8192,4096,1024,128,8`
  - 官方 bf16：peak `7690.63 MiB`，e2e `12.202 ms`
  - 稳定 `fp8-mainline`：output RMSE `0.01074074`，loss RMSE `0.00000025`，peak `7572.75 MiB`，inf `3.933 ms`，e2e `12.888 ms`
  - 理论账：stable fp8 边界流量下界 `516.00 MiB`，真正节省 payload 仅 `62.00 MiB`
  - 结论：大 shape crash 已通过本地 fallback / 分块 vec4 路径消掉；现在 stable fp8 的 inference 已领先 bf16，但 e2e 仍慢，主因还是 `quant -> dequant -> bf16` 边界搬运
- 中等真实 shape `4096,4096,1024,128,8`：
  - 官方 bf16：inf `2.344 ms`，e2e `7.338 ms`
  - 稳定 `fp8-mainline`：output RMSE `0.01073638`，loss RMSE `0.00000020`，peak `6867.00 MiB`，inf `2.204 ms`，e2e `8.022 ms`
  - 结论：稳定 fp8 inference 已领先 bf16 `5.97%`
- blockscaled 的理论账也已明确：
  - shape `4096,4096,1024,128,8`，`capacity=1024`
  - `grouped_out_bf16` 理论大小 `1024.00 MiB`
  - 结论：这就是当前 blockscaled 很难在显存 / e2e 上赢过 bf16 的核心原因
- 因此当前下一优先级已经明确：
  1. 优先优化稳定 `fp8-mainline`，让量化后激活直接进入 down-proj mainloop，去掉反量化回 bf16；
  2. 同时把激活 / 权重精度路径逐步做成外部可控开关，朝全流程 FP8 推进；
  3. blockscaled 只在能直接吃掉 `grouped_out` / router 聚合过渡层时继续重投入。
- 本轮新增的运行时精度开关：
  - `SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION={bf16,fp8}`
  - `SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION={bf16,fp8-blockscaled}`
  - `SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION={bf16,fp8}`
  - 默认行为不变；旧开关 `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1` 仍兼容
- 本轮明确不落地主线的实验：
  - `SONIC_MOE_FP8_PREACT_UE8M0_SCALE`
  - 原因：capture-safe 后虽然安全，但 graph capture 下会自动回退，暂未形成可重复、可归因的确定收益
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
4. For hotspot diagnosis, first run `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_stage_memory`
5. 优先处理稳定 `fp8-mainline` 的大 row-count vec4 fast path；只有在能直接去掉 `grouped_out` 时，再继续重投 blockscaled

## Working rule

Whenever the active plan changes, update this directory first so the next agent does not need to re-discover state from commit history or chat logs.
