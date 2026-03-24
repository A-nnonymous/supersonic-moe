# Next-Agent Handoff

This is the minimum context a new agent needs to continue work without replaying the entire history.

## 0. 最新中文结论

- 新结论（这一轮最重要）：
  - 已经补上真实路径的阶段显存 probe：
    - benchmark 开关：`--report_stage_memory`
    - 底层 env 开关：`SONIC_MOE_STAGEWISE_MEMORY=1`
  - `4096,4096,1024,128,8` 的阶段显存已经证明：
    - 稳定 `fp8-mainline` 的 `forward:down-proj-router` peak alloc：`4012.00 MiB`
    - env-on `blockscaled` 的同阶段 peak alloc：`5309.38 MiB`
    - 二者差值：`1297.38 MiB`
    - final peak：稳定主线 `7050.88 MiB`，blockscaled `7580.13 MiB`
  - 这说明 blockscaled 当前真正的墙就是：
    - `grouped_out`
    - `down-proj/router` 聚合边界
    - 而不是前半段 pack+quant
- 大 shape 稳定性状态已更新：
  - 本地 `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py` 已加 row-count fallback
  - `8192,4096,1024,128,8` 不再 runtime crash
  - 但这只是保底稳定性补丁，不是最终性能解
- 更大真实 shape 结果：
  - shape `6144,4096,1024,128,8`
    - bf16：peak `7370.25 MiB`，e2e `9.550 ms`
    - 稳定 `fp8-mainline`：output RMSE `0.01074142`，loss RMSE `0.00000034`，peak `7220.63 MiB`，e2e `9.954 ms`
    - env-on `blockscaled`：output RMSE `0.01073865`，loss RMSE `0.00000035`，peak `7749.88 MiB`，e2e `12.207 ms`
  - shape `8192,4096,1024,128,8`
    - bf16：peak `7690.63 MiB`，e2e `12.202 ms`
    - 稳定 `fp8-mainline`：output RMSE `0.01074074`，loss RMSE `0.00000025`，peak `7572.75 MiB`，e2e `12.972 ms`
  - 当前结论：
    - 稳定 `fp8-mainline` 继续保持显存优势，但 e2e 仍慢 `4%~6%`
    - env-on `blockscaled` 依旧不是可交付主线
- 下一任 agent 的第一优先级已经变化：
  1. 优先优化稳定 `fp8-mainline`，而不是 blockscaled；
  2. 目标是恢复/重建大 row-count 下的高性能 vec4 preact fused quant path；
  3. blockscaled 仅在能直接吃掉 `grouped_out` / router 边界时再继续重投入。

- 最新里程碑：
  - blockscaled 下行前向已经不再把 `grouped_out` unpack 成 flat `y2`；
  - router 聚合现在直接消费 grouped/static layout；
  - grouped reverse scatter index 直接复用 `selected_experts`，不再靠 `searchsorted` 反推 expert id。
- 中等真实 shape `4096,4096,1024,128,8` 的 env-on blockscaled 最新结果：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak：`7396.13 MiB`
  - inf fwd / train fwd / e2e / bwd：`2.918 / 3.818 / 8.196 / 5.279 ms`
- 相对上一版 env-on blockscaled（已做 `pack+quant` 融合，但还会 flat unpack）：
  - inf fwd 提升 `5.72%`
  - e2e 提升 `2.59%`
  - bwd 提升 `0.75%`
  - peak 不变
- 相对稳定 `fp8-mainline`：
  - e2e 仍慢 `6.54%`
  - bwd 反而快 `0.45%`
- 这说明：
  - blockscaled 路径的“后半段额外开销”已经又去掉一层；
  - 但真正剩下的显存/性能墙已经收敛为 `grouped_out` 本体，而不再是 unpack。
- 最新里程碑：
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 已经吃掉了 blockscaled 前半段的 `grouped_a` 物化；
  - 新主路径是：
    - `flat sorted activation`
    - `-> _pack_quantize_expert_segments_kernel`
    - `-> grouped fp8 activation + grouped scale`
    - `-> pack_blockscaled_1x32_scales(...)`
- 中等真实 shape `4096,4096,1024,128,8` 的 env-on blockscaled 最新结果：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak：`7396.13 MiB`
  - inf fwd / train fwd / e2e / bwd：`3.095 / 3.791 / 8.414 / 5.319 ms`
- 相对上一版 env-on blockscaled：
  - inf fwd 提升 `29.05%`
  - train fwd 提升 `43.54%`
  - e2e 提升 `27.89%`
  - bwd 提升 `27.21%`
  - peak memory 不变
- 这说明当前 blockscaled 的第一堵墙已经不再是前半段 pack/quant，而是：
  - `grouped_out`
  - `flat unpack`
  - router 聚合边界
- 因此下一步第一优先级再次变化：
  1. 不再继续抠 `grouped_a` 前半段；
  2. 直接处理 `grouped_out -> flat out`；
  3. 尽量让 router 聚合直接消费 grouped/static layout。
- 最新小里程碑已经落地：
  - `sonicmoe/functional/fp8_quant.py` 的 e8m0 blockwise quant 现在使用 reciprocal-multiply；
  - `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py` 的 4 组 scale 路径也已经切到 `cute.arch.rcp_approx(...)`。
- 工程化注意：
  - `operator-incubator` 当前不是 git repo；
  - 所以上面这条 fused kernel reciprocal 改动只存在于本地 workspace，**不会**自动跟随 SonicMoE 的 git commit/push 传播。
  - 本轮的“大 row-count vec4 fallback”也同样只存在于本地 workspace。
  - 下一任 agent 如果换了机器或重置了该目录，需要手工同步这处改动。
- 这一步已经验证：
  - SonicMoE blockwise quant 对旧 divide reference **完全等价**；
  - fused preact quant kernel 在 `using_pow2_scaling=True` 下对手写 golden **bitwise 等价**；
  - 局部量化微基准（`shape=(128,512,1024)`）提速 `5.80%`，`RMSE=0`。
- 第一条真正有用的中等真实 shape 对照现在是：
  - shape：`4096,4096,1024,128,8`
  - 官方 bf16：
    - peak：`7049.88 MiB`
    - inf fwd / train fwd / e2e / bwd：`2.344 / 2.236 / 7.338 / 4.994 ms`
  - 稳定 fp8-mainline（本轮 reciprocal 后）：
    - output RMSE：`0.01073638`
    - loss RMSE：`0.00000020`
    - peak：`6867.00 MiB`
    - inf fwd / train fwd / e2e / bwd：`2.390 / 2.890 / 7.693 / 5.303 ms`
  - 当前 env-on blockscaled：
    - output RMSE：`0.01073363`
    - loss RMSE：`0.00000019`
    - peak：`7396.13 MiB`
    - inf fwd / train fwd / e2e / bwd：`4.362 / 6.715 / 11.668 / 7.307 ms`
- 这组数据说明：
  - **最接近可交付的是稳定 fp8-mainline，而不是当前 blockscaled 路径**；
  - 稳定 fp8-mainline 已经在这档 shape 上显存领先 bf16 `182.88 MiB`，但 e2e 仍慢 `4.84%`；
  - env-on blockscaled 目前显存和性能都明显落后，主因仍是 grouped bridge / static capacity buffer。
- 当前第一优先级已经进一步收敛：
  1. 不再纠缠 reciprocal 这种小收益点；
  2. 直接处理 blockscaled 的 `grouped_out` 本体；
  3. 目标是让 routing metadata / router 聚合直接接受静态 expert layout；
  4. 同时修掉更大 shape `8192,4096,1024,128,8` 在 preact fused quant kernel 上的 runtime crash。
- `1x32 ue8m0` blockscaled down-proj 已经收敛到一个**可交付的静态对齐合同版**：
  - 显式开关：

```bash
SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1
SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=<128 的整数倍>
```

  - 当前完整 Blackwell 回归：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=128 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

  - 当前结果：`10 passed`
- 当前 blockscaled 路径的真实合同已经变成：
  - 底层 kernel **只接受静态对齐后的 expert capacity**
  - padding/capacity 规划上移为外部/上层逻辑责任
  - SonicMoE 当前仍有一个 GPU pack/unpack 过渡层，但 host-side grouped metadata 已经彻底去掉
- 与上一版 3D grouped bridge 相比：
  - cudagraph inference 已恢复
  - training / e2e 开销显著下降
- 但它仍不是默认主线，因为：
  - GPU pack/unpack 还在
  - 性能仍慢于稳定 `fp8-mainline`

- Blackwell + QuACK 的 `fp8_protocol` 默认前向路径已经切到 pre-SwiGLU 融合量化：

```text
z(pre-SwiGLU) -> fused_weighted_swiglu_act_quant_best -> fused_act_dequant_best -> y1
```

- 这一步已经通过：
  - `tests/fp8_protocol_test.py` 的 targeted fused-path 回归
  - `tests/moe_blackwell_test.py` smoke 回归
- 这一步的收益来源：
  - 替换掉了旧的 post-SwiGLU torch-side quant/dequant。
  - 还没有改 backward 主 kernel，所以收益主要来自 forward。
- 这一步的遗留问题：
  - 精度相对旧 reference path 有回退。
  - 共享机小 shape 数据显示仍慢于 bf16。
  - `prob/topk_scores` 还没有被真正前移进 fused epilogue。
- 新增了一条 `1x32 ue8m0` blockscaled down-proj 主循环试接线：
  - 代码位置：`sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - scale pack helper 已落地：`pack_blockscaled_1x32_scales(...)`
  - `w2` 的 fp8 cache 也已落地
- 这条路径当前通过的关键手段是：
  - `float8` runtime tensor 采用 `uint8 storage view + element_type override`
  - 不再走上一版的 host-side `3D grouped pad/unpad`
  - 改为：
    - 显式静态 `expert capacity`
    - GPU 侧 `searchsorted + index_copy_/index_select` pack/unpack
- 但这条路径当前**不是默认主线**，必须显式打开：

```bash
SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1
SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=<128 的整数倍>
```

- 不默认打开的原因已经进一步收敛：
  - host-side blocker 已去掉
  - inference cudagraph 已恢复
  - 当前剩余问题是 GPU pack/unpack 仍有额外开销，小 shape 上仍慢于稳定 `fp8-mainline`
- 所以下一任 agent 的第一优先级，已经从“重啃 float8 runtime 参数契约”切换为：
  - **去掉 GPU pack/unpack**
  - **把 routing metadata 直接产生成 blockscaled mainloop 所需的静态布局**
  - **把同样的静态对齐合同扩展到剩余 GEMM 相关路径**

## 1. Environment

- activate: `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
- Python: `3.13.x`
- GPU target here: Blackwell (`sm_100a`)
- Blackwell route in SonicMoE: QuACK (`USE_QUACK_GEMM=1`)

## 2. Repository state

- fork remote contains the latest pushed work on `main`
- use the current branch head as the source of truth for protocol work; the last pushed pre-protocol doc commit was `57d7faa`
- key files changed so far:
  - `Makefile`
  - `README.md`
  - `tests/moe_test.py`
  - `tests/moe_blackwell_test.py`
  - `sonicmoe/functional/fp8_protocol.py`
  - `sonicmoe/functional/fp8_quant.py`
  - `sonicmoe/functional/fp8_reference.py`
  - `tests/fp8_protocol_test.py`

## 3. Validated command

Run this before doing new FP8 work:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Expected result at handoff time:

```text
18 passed, 91 skipped
```

Faster opt-in command on this machine:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

Observed runtime for the parallel entry:

```text
18 passed, 91 skipped in 168.14s
```

Multi-GPU shard launcher:

```bash
make test-blackwell-multigpu BLACKWELL_TEST_GPUS=0,1,2
python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run
```

Metric-reporting command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

共享机小 shape 对照命令（本轮已经跑过）：

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

本轮 blockscaled 小 shape 结论（请补上 capacity）：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

- output RMSE vs bf16：`0.00131902`
- loss RMSE vs bf16：`0.00000013`
- bf16 peak memory：`380.25 MiB`
- blockscaled fp8 peak memory：`142.77 MiB`
- Fwd inference：`0.476 ms`
- Fwd training：`2.046 ms`
- Fwd+Bwd：`4.004 ms`
- Bwd：`3.528 ms`

本轮 reciprocal 相关验证命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'runtime_and_reference_quant or blockwise_quant_matches_divide_reference_after_e8m0_encoding or preact_cutely_fused_path_matches_reference_boundary'
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=512 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果摘要：

- `3 passed, 8 deselected`
- `12 passed`
- `12 passed`
- 稳定 `fp8-mainline` 中等 shape：
  - output RMSE：`0.01073638`
  - peak：`6867.00 MiB`
  - inf fwd / train fwd / e2e / bwd：`2.390 / 2.890 / 7.693 / 5.303 ms`
- env-on blockscaled 中等 shape：
  - output RMSE：`0.01073363`
  - peak：`7396.13 MiB`
  - inf fwd / train fwd / e2e / bwd：`2.918 / 3.818 / 8.196 / 5.279 ms`

## 4. What has already been settled

- the Blackwell smoke path is real and regression-tested
- the main MoE test matrix no longer tries to force unsupported non-QuACK SonicMoE execution on Blackwell
- the FP8 protocol is now code, not just planning
- the current protocol scope is fixed to `e4m3` activations + `e8m0` scales
- `1x128` is still the default stable mainline granularity
- `1x32` blockscaled scaffolding is now present, but env-gated and not stable enough for default enablement
- `1x32` blockscaled down-proj path has now been run end-to-end under the static-capacity env contract and passes Blackwell regression
- the protocol is now threaded through `MoE.forward(..., fp8_protocol=...)` and `moe_TC_softmax_topk_layer(..., fp8_protocol=...)`
- the current functional-boundary implementation is intentionally correctness-first and is still slower than the baseline
- a first memory optimization pass already landed:
  - fp8 boundary peak memory: `15312.85 MiB -> 13017.85 MiB`
  - fp8 boundary Fwd+Bwd: `119.803 ms -> 109.117 ms`
- a new adapter landing point now exists in `sonicmoe/functional/fp8_cutely_fused.py`
- the first real high-performance forward step is now landed in the same file:
  - default Blackwell/QuACK forward path consumes `z` instead of `y1`
  - quant path uses `cutify.fused_weighted_swiglu_act_quant_best`
  - dequant path uses `cutify.fused_act_dequant_best`
- CUDA graph capture compatibility had to be fixed:
  - direct `ue8m0` packing inside the incubator path broke capture
  - current workaround is: kernel emits float32 dequant scale, SonicMoE re-encodes it to `e8m0`
- the incubator fused quant kernel does **not** match the current SonicMoE boundary 1:1:
  - incubator input contract: pre-SwiGLU `(T, 2H)`
  - current SonicMoE boundary: post-SwiGLU `(TK, I)`
  - this mismatch is now handled by the pre-SwiGLU bridge logic in `fp8_cutely_fused.py`
- the next kernel target is the Hopper FP8 up-projection epilogue, not a standalone gather kernel and not a monolithic full-graph rewrite

## 5. The next concrete edits

### Stage 1: 继续消掉 blockscaled 的输出本体

当前最该做的，不再是额外的 reciprocal 微调，而是：

- 把 `blockscaled_fp8_gemm.py` 里的
  - `grouped_out`
  这几段过渡存储继续吃掉。
- 优先顺序：
  1. 现在 `grouped_out -> flat out` 已经去掉；
  2. 下一步要么让 down-proj 直接写 router 可聚合布局；
  3. 要么让 router 聚合原生接受 `grouped_out` contract 且不再受这个中间 buffer 约束。

### Stage 2: 把 prob/topk_scores 真正吃进融合 epilogue

The protocol/reference modules are already wired through:

- `sonicmoe/moe.py`
- `sonicmoe/functional/__init__.py`
- `sonicmoe/functional/fp8_reference.py`
- `sonicmoe/functional/fp8_cutely_fused.py`

The next step should remove the remaining semantic gap for:

- optional router probability weighting
- post-router scaling placement
- backward cache consumption
- backward fused kernel contract

### Stage 3: first fused kernel

Implement the Hopper-side fused up-projection epilogue equivalent to:

```text
grouped_gemm(varlen/gather-A) -> SwiGLU -> optional prob -> 1x128 quant
```

Reuse:

- SonicMoE routing metadata and grouped GEMM structure
- Paddle fused op semantics
- operator-incubator CuTe prototypes

### Stage 4: 去掉静态合同版的 GPU pack/unpack

这一步的重点已经继续前移。当前 runtime wrapping 与 cudagraph 都够用了，真正要解的是剩余搬运开销。

Before enabling the new down-proj mainloop path by default, solve these blockers:

- remove the current GPU `searchsorted + index_copy_/index_select` pack/unpack
- make routing metadata produce the blockscaled static layout directly, or an equivalent zero-copy view contract
- after that, re-measure inference-cg / e2e / bwd under the same small shape harness

The new code already gives you:

- the scale packing formula
- the `w2` fp8 cache position
- the `_DownProjection.forward` insertion point
- a running static-capacity blockscaled path
- benchmark harness support for env-on blockscaled runs with cudagraph capture re-enabled

So the next edit should focus on removing GPU transitional overhead, not on re-deriving the FP8 math.

## 6. File map for fast navigation

- SonicMoE execution path:
  - `sonicmoe/moe.py`
  - `sonicmoe/functional/__init__.py`
  - `sonicmoe/functional/forward.py`
  - `sonicmoe/functional/backward.py`
- Blackwell adapter:
  - `sonicmoe/quack_utils/gemm_gated.py`
  - `sonicmoe/quack_utils/gemm_dgated.py`
- operator incubator kernels:
  - `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py`
  - `operator-incubator/cutify/ops/cute/fused_act_dequant.py`
  - `operator-incubator/cutify/ops/cute/fused_swiglu_weighted_bwd.py`

## 7. Do not re-discover these points

- `dev_b` is not the authoritative environment for SonicMoE because it is on Python 3.10
- `xfer` is the environment to use for SonicMoE work
- Blackwell currently relies on QuACK; do not expect the default Hopper-only path to compile on `sm_100a`
- current torch already provides both `torch.float8_e4m3fn` and `torch.float8_e8m0fnu`; a nightly upgrade is not currently required
- the benchmark gate for performance-facing work is now documented in `reports/fp8_upgrade/ENGINEERING_LOG.md`
- `pytest-xdist` is now installed in `xfer`; keep the worker count conservative (`2`) for the single-GPU Blackwell regression path
- the reporting policy is fixed:
  - accuracy baseline: official bf16
  - memory baseline: official bf16
  - performance baselines: previous commit and official bf16
- from this turn onward, engineering records should be written in Chinese and the metric annotations must come first
- keep `reports/` up to date whenever the branch, validation command, or next target changes
