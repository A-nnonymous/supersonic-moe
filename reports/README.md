# SonicMoE Work Reports

This directory is the live work log for the FP8 upgrade effort. It is not meant to hold speculative essays; it should hold the current state, the validated commands, and the next handoff targets.

> 从当前阶段开始，工程记录统一使用中文；每次记录都必须先写清楚指标注释、基线和收益来源。

## Current status

- authoritative Python environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- Blackwell path: QuACK-enabled (`USE_QUACK_GEMM=1`)
- latest validated local milestone:
  - fused preact dequant now supports `restored_out=...` and directly reuses the QuACK `y1` output buffer on the aligned stable path
  - QuACK preact FP8 boundary now runs under `torch.no_grad()` so the mainline no longer builds a dead gradient path there
  - latest large-shape state:
    - `8192,4096,1024,128,8`
    - bf16: peak `7690.63 MiB`, inf / train fwd / e2e / bwd `4.081 / 3.942 / 5.728 / 1.647 ms`
    - fp8: peak `7700.75 MiB`, output RMSE `0.01074111`, loss RMSE `0.00000025`, inf / train fwd / e2e / bwd `1.842 / 2.111 / 5.586 / 3.743 ms`
    - relative to bf16: inference `+121.55%`, train fwd `+86.74%`, e2e `+2.54%`
    - stagewise raw probe on the same shape still reports final peak `7690.63 -> 7572.75 MiB`, so the real blocker is backward transient overhead rather than a settled final-peak regression
- latest targeted validation:
  - stable Blackwell fp8 regression: `USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - stable result: `20 passed`
  - env-on blockscaled regression: `USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=128 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py`
  - env-on blockscaled result: `13 passed`
  - serial: `python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py`
  - serial result: `18 passed, 91 skipped`
  - opt-in parallel: `make test-blackwell-parallel PYTEST_WORKERS=2`
  - parallel result: `18 passed, 91 skipped in 168.14s`
  - multi-GPU dry-run: `python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run`
  - fp8 metric probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics`
  - fp8 theory probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_fp8_analysis`
  - combined theory + stagewise probe: `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_fp8_analysis --report_stage_memory`
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
  - 官方 bf16：peak `7690.63 MiB`，inf / train fwd / e2e / bwd `4.081 / 3.942 / 5.728 / 1.647 ms`
  - 稳定 `fp8-mainline`：output RMSE `0.01074111`，loss RMSE `0.00000025`，peak `7700.75 MiB`，inf / train fwd / e2e / bwd `1.842 / 2.111 / 5.586 / 3.743 ms`
  - 相对 bf16：inference `+121.55%`，train fwd `+86.74%`，e2e `+2.54%`，但 peak `+10.12 MiB`
  - 单独 `--report_stage_memory` 原始对排：
    - `forward:down-proj-router` peak delta `-245.88 MiB`
    - backward 三段 `delta_alloc=+138.12 MiB`
    - final peak `7690.63 -> 7572.75 MiB`
  - 理论账：stable fp8 边界流量下界 `516.00 MiB`，真正节省 payload 仅 `62.00 MiB`
  - 结论：大 shape crash 已通过本地 fallback / 分块 vec4 路径消掉；现在 stable fp8 的前向与 e2e 已经被拉回可竞争区间，但 backward transient overhead 仍是下一步主要矛盾
- 中等真实 shape `4096,4096,1024,128,8`：
  - 官方 bf16：peak `7049.88 MiB`，inf / train fwd / e2e / bwd `1.141 / 1.210 / 3.437 / 2.296 ms`
  - 稳定 `fp8-mainline`：output RMSE `0.01073675`，loss RMSE `0.00000021`，peak `6931.00 MiB`，inf / train fwd / e2e / bwd `1.125 / 1.697 / 3.733 / 2.608 ms`
  - 相对 bf16：inference `+1.42%`，e2e `-7.93%`，peak `-118.88 MiB`
- blockscaled 的理论账也已明确：
  - shape `4096,4096,1024,128,8`，`capacity=1024`
  - `grouped_out_bf16` 理论大小 `1024.00 MiB`
  - 结论：这就是当前 blockscaled 很难在显存 / e2e 上赢过 bf16 的核心原因
- 因此当前下一优先级已经明确：
  1. 优先继续压缩 `8192` backward transient overhead，保持 large-shape e2e 领先同时把最终 peak 稳定压在 bf16 下；
  2. 在不丢失这次前向收益的前提下，继续把量化后激活直喂 down-proj mainloop，减少 `bf16` 回退；
  3. 同时把激活 / 权重精度路径逐步做成外部可控开关，朝全流程 FP8 推进；
  4. blockscaled 只在能直接吃掉 `grouped_out` / router 聚合过渡层时继续重投入。
- 最新账本补充（shape `4096,4096,1024,128,8`）：
  - 当前稳定主线：
    - `stable_fp8_boundary_lower_bound_mib=258.00`
    - `stable_fp8_saved_payload_mib=31.00`
  - 若做到不破坏 `varlen/gather-A` 合同的 direct FP8 mainloop：
    - `direct_fp8_boundary_floor_mib=160.25`
    - `direct_fp8_boundary_saved_mib=97.75`
  - 若进一步把 `w1/w2` 做成 FP8 存储：
    - `aggressive_weight_fp8_storage_mib=1548.00`
    - `aggressive_weight_saved_mib=1524.00`
    - `aggressive_total_saved_mib=1555.75`
  - 结论：
    - 真正值得追的大头已经很明确：先打掉 `bf16` 回退边界，再逐步推进权重 FP8 存储；
    - 小于这个量级的局部优化，不值得为之破坏 SonicMoE 的核心内存合同。
- 本轮新增的 cudagraph-safe 基础设施：
  - `round_scale_to_e8m0(..., out=...)`
  - `quantize_activation_blockwise(..., out=..., scale_out=...)`
  - `dequantize_activation_blockwise(..., out=...)`
  - `apply_activation_fp8_protocol_cutely_fused(..., scale_out=...)`
  - `apply_preact_activation_fp8_protocol_cutely_fused(..., scale_out=...)`
  - 作用：
    - 允许后续把 scale / quant 输出写入预分配 buffer
    - 为恢复/推进 cudagraph-compatible FP8 主线打基础
  - 对应验证：
    - 定向回归 `4 passed`
    - Blackwell 主回归 `19 passed`
- 最新主线收口（shape `4096,4096,1024,128,8`）：
  - 主线现在默认去掉 preact fused FP8 boundary 里的 `STE` 混合
  - 原因：
    - `y1` 本身在 `_UpProjection.forward(...)` 后已经 `mark_non_differentiable`
    - `_DownProjection.backward(...)` 也不是从前向 `y1` 回梯度
  - 最新默认稳定结果：
    - output RMSE：`0.01073675`
    - loss RMSE：`0.00000021`
    - bf16 peak / fp8 peak：`7049.88 / 6867.00 MiB`
    - inf / train fwd / e2e / bwd：`2.430 / 2.824 / 7.272 / 4.842 ms`
  - 这说明：
    - 主线里确实存在可以安全删掉的语义冗余
    - 但收益仍主要体现在训练/e2e，不代表所有局部“更小 buffer”方案都会赢
- 本轮新增实验开关（默认关闭）：
  - `SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER=1`
  - 作用：
    - 让 QuACK up-proj 把 `postact_out` 申请成 `float8_e4m3fn` dummy buffer
    - 后续 fused preact 边界只靠 `z` 重建 `bf16 y1`
  - 当前真实 benchmark 结论：
    - 这条路可跑、也已测试锁住
    - 但 `4096,4096,1024,128,8` 下更慢：
      - inf / train fwd / e2e / bwd：`2.484 / 2.742 / 7.535 / 5.051 ms`
    - 所以默认不启用，只保留为后续更深层 epilogue/mainloop 实验能力
- 2026-03-24 新证伪结论：
  - grouped `fp8-direct-downproj` 原型虽然数值正确，但真实 `4096,4096,1024,128,8` 下显著退化：
    - stable `fp8-mainline`：peak `6867.00 MiB`，inf `2.111 ms`，e2e `7.256 ms`
    - grouped `fp8-direct` 原型：peak `7395.25 MiB`，inf `2.824 ms`，e2e `8.256 ms`
  - 这说明 stable 主线真正要守的是 SonicMoE 的 `varlen/gather-A` 内存合同；
  - 下一实现方向应切换为：
    1. `gemm_gated` / up-proj epilogue 直接产出 `varlen FP8 postact + scales`
    2. 后续再让 down-proj 在不引入 grouped/static layout 的前提下消费这些 FP8 激活
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
4. For hotspot diagnosis, first run `USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_fp8_analysis --report_stage_memory`
5. 优先处理稳定 `fp8-mainline` 的 `varlen FP8 epilogue -> direct mainloop` 与 cudagraph-safe buffer 复用；`SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER` 目前只是实验开关，不要当成默认优化路径；只有在能直接去掉 `grouped_out` 时，再继续重投 blockscaled

## Working rule

Whenever the active plan changes, update this directory first so the next agent does not need to re-discover state from commit history or chat logs.
