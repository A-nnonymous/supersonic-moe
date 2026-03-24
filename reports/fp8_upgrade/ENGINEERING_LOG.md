# FP8 Engineering Log

This log records concrete code changes plus their immediate validation and performance numbers.

## 2026-03-24 - 收口 ue8m0 运行时实验，并补上外部精度开关骨架

### 指标注释（先看这个）

- 精度基线：当前稳定 `fp8-mainline` 默认行为。
- 稳定性基线：上一轮 Blackwell 主回归 `13 passed`。
- 性能基线：上一轮稳定 `fp8-mainline@8192,4096,1024,128,8`：
  - inf `3.933 ms`
  - e2e `12.888 ms`
  - peak `7572.75 MiB`
- 收益来源：
  - 把“每一步精度由外部开关控制”的合同显式化，避免后续全流程 FP8 演进继续依赖隐式 env；
  - 对 `ue8m0` 运行时实验做真实回读，确认其在 graph capture 下会自动回退，因此本轮不把这条收益不确定的实验路径带进主线。

### 改动

- 在 `sonicmoe/functional/__init__.py` 中新增运行时精度开关解析：
  - `SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION={bf16,fp8}`
  - `SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION={bf16,fp8-blockscaled}`
  - `SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION={bf16,fp8}`
- 保持与旧环境变量兼容：
  - 如果未显式设置新开关，仍会继续识别 `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1`
- 新增运行时约束：
  - `SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION=fp8` 目前只允许与 `SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION=fp8-blockscaled` 组合使用
- 在 `tests/fp8_protocol_test.py` 中新增两条回归：
  - 可显式关闭 up-proj epilogue FP8 boundary，且前反向保持有限值
  - 非法组合（fp8 weight + bf16 mainloop）会显式报错
- 在 `sonicmoe/functional/fp8_cutely_fused.py` 中收回未准备落地主线的 `SONIC_MOE_FP8_PREACT_UE8M0_SCALE` 实验逻辑，避免后续 handoff 误把它当成稳定收益点

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
15 passed
```

### ue8m0 运行时实验回读（仅记录，不落主线）

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_PREACT_UE8M0_SCALE=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
CUDA_VISIBLE_DEVICES=6 USE_QUACK_GEMM=1 SONIC_MOE_FP8_PREACT_UE8M0_SCALE=1 python benchmarks/moe-cute.py --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_fp8_analysis
```

结果：

- 回归：`13 passed`
- `8192,4096,1024,128,8`：
  - output RMSE：`0.01074074`
  - loss RMSE：`0.00000025`
  - peak：`7571.25 MiB`
  - inf：`3.908 ms`
  - e2e：`12.672 ms`

结论：

- 这条实验在 capture-safe guard 下是**安全**的；
- 但 inference 路径在 cudagraph capture 下本来就会自动回退，不会稳定命中新量化分支；
- 当前观测到的时间变化仍在共享机器噪声范围内，**没有形成可重复、可归因的确定收益**；
- 因此本轮选择把它从主线代码中收回，只在工程记录里保留实验结论，避免后续 agent 误把它当成已验证优化点。

## 2026-03-24 - 去掉无用 scale decode，并在 inference 下跳过 STE 混合

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一版稳定 `fp8-mainline`（已接入理论分析，但仍会做无用 scale decode，且 inference 仍做 STE 混合）。
- 性能基线 2：同 shape 官方 `bf16`。
- 收益来源：
  - 运行时不再做**没有消费者**的 `round_scale_to_e8m0(...)`；
  - inference 模式直接返回 `restored`，不再做 `postact + (restored - postact).detach()`；
  - 这两项都只缩减 FP8 boundary 的无效边界开销，不改变训练语义。

### 改动

- 在 `sonicmoe/functional/fp8_reference.py` 中：
  - `apply_activation_fp8_protocol(...)` 新增：
    - `return_scales: bool = True`
    - `use_ste: bool = True`
- 在 `sonicmoe/functional/fp8_cutely_fused.py` 中：
  - `apply_activation_fp8_protocol_cutely_fused(...)`
  - `apply_preact_activation_fp8_protocol_cutely_fused(...)`
  - 新增：
    - `return_scales: bool = True`
    - `use_ste: bool = True`
  - 当 `return_scales=False` 时，直接跳过 scale decode；
  - 当 `use_ste=False` 时，直接返回 dequant 后结果，不再做 STE 混合。
- 在 `sonicmoe/functional/__init__.py` 中：
  - SonicMoE 真实运行主线调用 FP8 boundary 时改为：
    - `return_scales=False`
    - `use_ste=not is_inference_mode_enabled`

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
13 passed
```

### 真实 shape 数据

#### shape `4096,4096,1024,128,8`

- 官方 bf16：
  - peak：`7049.88 MiB`
  - inf fwd：`2.344 ms`
  - e2e：`7.338 ms`
- 上一版稳定 fp8-mainline：
  - output RMSE：`0.01073638`
  - loss RMSE：`0.00000020`
  - peak：`6867.00 MiB`
  - inf fwd：`2.587 ms`（复测正常值）
  - train fwd：`2.912 ms`
  - e2e：`7.903 ms`
  - bwd：`5.316 ms`
- 本轮稳定 fp8-mainline：
  - output RMSE：`0.01073638`
  - loss RMSE：`0.00000020`
  - peak：`6867.00 MiB`
  - inf fwd：`2.204 ms`
  - train fwd：`2.907 ms`
  - e2e：`8.022 ms`
  - bwd：`5.819 ms`

结论：

- 相对 bf16：
  - inference **领先** `5.97%`
  - peak memory 仍领先 `182.88 MiB`
- 相对上一版稳定 fp8-mainline：
  - inference 提升 `14.82%`
  - train fwd 基本持平（`0.17%`）
  - e2e / bwd 噪声较大，本轮不据此下强结论

#### shape `8192,4096,1024,128,8`

- 官方 bf16：
  - peak：`7690.63 MiB`
  - inf fwd：`4.147 ms`
  - e2e：`12.202 ms`
- 上一版稳定 fp8-mainline（去掉无用 scale decode 后、但 inference 仍做 STE）：
  - output RMSE：`0.01074074`
  - loss RMSE：`0.00000025`
  - peak：`7572.75 MiB`
  - inf fwd：`4.372 ms`
  - train fwd：`4.557 ms`
  - e2e：`12.751 ms`
  - bwd：`8.379 ms`
- 本轮稳定 fp8-mainline：
  - output RMSE：`0.01074074`
  - loss RMSE：`0.00000025`
  - peak：`7572.75 MiB`
  - inf fwd：`3.933 ms`
  - train fwd：`4.421 ms`
  - e2e：`12.888 ms`
  - bwd：`8.955 ms`

结论：

- 相对 bf16：
  - inference **领先** `5.16%`
  - peak memory 领先 `117.88 MiB`
  - e2e 仍慢 `5.62%`
- 相对上一版稳定 fp8-mainline：
  - inference 提升 `10.04%`
  - train fwd 提升 `2.98%`
  - e2e / bwd 在共享机器上仍有噪声，但 forward 侧收益是确定的

### 当前判断

- 这一步证明：
  - 稳定 `fp8-mainline` 的 forward 路径里仍有不少“非必要边界工作”可以继续吃掉；
  - 仅靠削减无用边界工作，已经能让真实大 shape 的 inference 领先 bf16。
- 但训练/e2e 还没追平 bf16，说明真正剩余的大头仍是：
  - `quant -> dequant -> bf16` 这层回退本身；
  - 要继续前进，就必须把量化后的激活直接送入 down-proj mainloop。

## 2026-03-24 - 用理论显存 / FLOPs 账解释为什么当前 FP8 还会慢

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线：同 shape 官方 `bf16` 与当前稳定 `fp8-mainline` / env-on `blockscaled`。
- 本轮收益来源：
  - 不是直接性能提升；
  - 而是把“FP8 为什么还没赢”从猜测变成可重复打印的理论账 + 实测账。
- 重要说明：
  - 新增 benchmark 开关：`--report_fp8_analysis`
  - 它会打印：
    - 权重大小
    - 关键中间结果大小
    - 稳定 `fp8-mainline` 边界流量的理论下界
    - blockscaled 静态布局 buffer 的理论大小

### 改动

- 在 `benchmarks/moe-cute.py` 中新增：
  - `--report_fp8_analysis`
  - 输出内容包括：
    - `w1/w2` 理论显存
    - `x / z / postact_bf16 / postact_fp8 / scales_f32 / scales_e8m0`
    - `stable_fp8_boundary_lower_bound_mib`
    - `stable_fp8_saved_payload_mib`
    - 如果 env-on `blockscaled`：
      - `grouped_out_bf16_mib`
      - `grouped_a_fp8_mib`
      - `grouped_a_scale_e8m0_mib`

### 为什么稳定 `fp8-mainline` 还没赢

#### shape `8192,4096,1024,128,8`

- 理论账：
  - `w1/w2` 权重总大小：`3072.00 MiB`
  - `x`：`64.00 MiB`
  - `z`（pre-SwiGLU）：`256.00 MiB`
  - `postact_bf16`：`128.00 MiB`
  - `postact_fp8`：`64.00 MiB`
  - `scales_f32`：`2.00 MiB`
  - `scales_e8m0`：`0.50 MiB`
  - 稳定 `fp8` 边界理论流量下界：`516.00 MiB`
  - 真正节省下来的 payload：`62.00 MiB`
- 实测账：
  - bf16：peak `7690.63 MiB`，inf `4.147 ms`，e2e `12.202 ms`
  - 稳定 `fp8-mainline`：peak `7572.75 MiB`，inf `4.404 ms`，e2e `13.096 ms`
  - output RMSE：`0.01074074`
  - loss RMSE：`0.00000025`

结论：

- 当前稳定 `fp8-mainline` 的问题，不是权重太大：
  - `w1/w2` 在 bf16 和当前 fp8 路径里都还是 bf16，理论上没有变小；
- 也不是 FP8 计算本身慢：
  - benchmark 打出来的 GEMM TFLOPS 仍然很高；
- 真正的问题是：
  - 现在路径是 `z -> fused quant -> fused dequant -> bf16 y1 -> down-proj`
  - 这条边界引入了大量额外搬运；
  - 但真正省下来的只是 `postact_bf16` 到 `postact_fp8+scale` 这一点点 payload
  - 因此 **“节省量 < 额外边界流量”**，性能自然很难超过 bf16
- 这说明稳定主线下一步真正应该做的是：
  - 不是继续优化 quant/dequant 小细节；
  - 而是让量化后的激活 **直接进入 down-proj mainloop**，去掉 `dequant -> bf16` 这层回退

### 为什么当前 blockscaled 还没赢

#### shape `4096,4096,1024,128,8`，`capacity=1024`

- 理论账：
  - `grouped_out_bf16`：`1024.00 MiB`
  - `grouped_a_fp8`：`128.00 MiB`
  - `grouped_a_scale_e8m0`：`4.00 MiB`
  - 同 shape 稳定 `fp8` 边界理论流量下界：`258.00 MiB`
  - 同 shape 稳定 `fp8` 真正节省下来的 payload：`31.00 MiB`
- 实测账：
  - env-on `blockscaled`：peak `7396.25 MiB`，inf `3.776 ms`，e2e `11.211 ms`
  - 同 shape bf16：peak `7049.88 MiB`，e2e `7.338 ms`
  - 同 shape 稳定 `fp8-mainline`：final peak `7050.88 MiB`（stagewise），e2e `7.562~7.693 ms`

结论：

- blockscaled 当前输，不是前半段 pack+quant 还不够好；
- 而是 `grouped_out` 这种静态对齐输出 buffer 本身太大；
- 只要 `grouped_out` 还在，显存和 e2e 都很难赢；
- 所以后续 blockscaled 方向只有一个值得继续投入的点：
  - 让 GEMM 直接写 router 可消费布局
  - 或者让 router 原生消费当前 grouped 布局

### 当前最高优先级（已根据理论账更新）

1. 稳定 `fp8-mainline`：
   - 目标是量化后激活直接进入 down-proj mainloop
   - 同时引入外部开关，逐步控制激活 / 权重的精度路径
2. env-on `blockscaled`：
   - 只有在能直接吃掉 `grouped_out` / router 边界时才继续重投入
3. 长期目标：
   - 最激进情况下实现前后向 GEMM 全流程 FP8
   - 但短期必须先把“量化后立刻反量化”的浪费拿掉

## 2026-03-24 - 用阶段显存 probe 锁定热点，并修掉 `8192` preact fused quant crash

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：同 shape 官方 `bf16`。
- 性能基线 2：同 shape 的稳定 `fp8-mainline` / env-on `blockscaled` 旧结果。
- 本轮收益来源分两类：
  - 工程收益：把真实路径的阶段显存观测补齐，后续可以有的放矢优化；
  - 稳定性收益：给大 row-count 的 vec4 preact fused quant 加保底 fallback，先让真实大 shape 跑通。
- 重要说明：
  - `stage-memory` 是默认关闭的，只在 `SONIC_MOE_STAGEWISE_MEMORY=1` 或 benchmark `--report_stage_memory` 时启用；
  - `8192` crash 修复当前落在本地 `operator-incubator` workspace，不在 SonicMoE git 仓库里。

### 改动

- 在 `sonicmoe/functional/__init__.py` 中新增阶段显存 probe：
  - 开关：`SONIC_MOE_STAGEWISE_MEMORY=1`
  - 使用的 torch 观测接口：
    - `torch.cuda.memory_allocated()`
    - `torch.cuda.memory_reserved()`
    - `torch.cuda.max_memory_allocated()`
    - `torch.cuda.max_memory_reserved()`
    - `torch.cuda.reset_peak_memory_stats()`
  - 当前 forward / backward 打点位置：
    - `forward:router-metadata`
    - `forward:up-proj`
    - `forward:fp8-boundary`
    - `forward:down-proj-router`
    - `backward:down-proj-core`
    - `backward:up-proj-core`
    - `backward:token-reduce`
- 在 `benchmarks/moe-cute.py` 中新增：
  - `--report_stage_memory`
  - 会在正式计时前先跑一遍独立的 training-mode fwd+bwd，并打印阶段显存
  - debug pass 改成使用独立 clone，避免污染后续计时图
- 在本地 `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py` 中：
  - 给 vec4 fast path 增加 row-count guard
  - 当 `x.shape[0] >= 4096` 时，回退到已验证可用的 generic path
  - 目的不是极致性能，而是先消掉 `8192,4096,1024,128,8` 的 runtime crash

### 阶段显存结论

统一 shape：`4096,4096,1024,128,8`

- 稳定 `fp8-mainline`：
  - `forward:down-proj-router` peak alloc：`4012.00 MiB`
  - `backward:token-reduce` final peak：`7050.88 MiB`
  - e2e：`7.562 ms`
- env-on `blockscaled`：
  - `forward:down-proj-router` peak alloc：`5309.38 MiB`
  - `backward:token-reduce` final peak：`7580.13 MiB`
  - e2e：`10.390 ms`

收益 / 诊断：

- `router-metadata` / `up-proj` / `fp8-boundary` 三段两条路径几乎重合；
- 真正拉开差距的是：
  - `forward:down-proj-router`：blockscaled 比稳定主线多出 `1297.38 MiB`
  - `final peak`：blockscaled 比稳定主线多出 `529.25 MiB`
- 这说明：
  - blockscaled 当前主矛盾已经不是前半段 pack+quant；
  - 而是 `grouped_out` 本体及其下游 router 聚合边界；
  - 后续如果继续抠 blockscaled，必须直接对准 `down-proj/router` 合同，而不是继续磨前半段。

### 大一点真实 shape 结果

#### shape `6144,4096,1024,128,8`

- 官方 bf16：
  - peak memory：`7370.25 MiB`
  - Fwd inference：`3.254 ms`
  - Fwd training：`3.214 ms`
  - Fwd+Bwd：`9.550 ms`
  - Bwd：`6.297 ms`
- 稳定 `fp8-mainline`：
  - output RMSE：`0.01074142`
  - loss RMSE：`0.00000034`
  - peak memory：`7220.63 MiB`
  - Fwd inference：`3.189 ms`
  - Fwd training：`3.606 ms`
  - Fwd+Bwd：`9.954 ms`
  - Bwd：`6.764 ms`
- env-on `blockscaled`：
  - output RMSE：`0.01073865`
  - loss RMSE：`0.00000035`
  - peak memory：`7749.88 MiB`
  - Fwd inference：`4.163 ms`
  - Fwd training：`6.729 ms`
  - Fwd+Bwd：`12.207 ms`
  - Bwd：`8.044 ms`

结论：

- 稳定 `fp8-mainline` 继续保持显存领先：
  - 相对 bf16 节省 `149.62 MiB`
- 但性能仍未超过 bf16：
  - e2e 慢 `4.23%`
  - bwd 慢 `7.42%`
- env-on `blockscaled` 仍不是当前可交付主线：
  - 显存反而比 bf16 多 `379.63 MiB`
  - e2e 慢 `27.82%`

#### shape `8192,4096,1024,128,8`

- 官方 bf16：
  - peak memory：`7690.63 MiB`
  - Fwd inference：`4.147 ms`
  - Fwd training：`4.174 ms`
  - Fwd+Bwd：`12.202 ms`
  - Bwd：`8.054 ms`
- 稳定 `fp8-mainline`（本地 vec4 fallback 生效后）：
  - output RMSE：`0.01074074`
  - loss RMSE：`0.00000025`
  - peak memory：`7572.75 MiB`
  - Fwd inference：`4.360 ms`
  - Fwd training：`4.464 ms`
  - Fwd+Bwd：`12.972 ms`
  - Bwd：`8.612 ms`

结论：

- 之前的 `CUresult 9` runtime crash 已经被保底 fallback 消掉；
- 稳定 `fp8-mainline` 继续保有显存优势：
  - 相对 bf16 节省 `117.88 MiB`
- 但性能仍落后：
  - inference 慢 `5.14%`
  - e2e 慢 `6.31%`
  - bwd 慢 `6.93%`
- 这说明当前稳定主线已经具备继续放大 shape 的可测性；
- 下一步性能攻关方向应优先考虑：
  - 恢复/重建大 row-count 下的高性能 vec4 path，而不是长期依赖 generic fallback。

### 验证命令

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
CUDA_VISIBLE_DEVICES=2 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_stage_memory
CUDA_VISIBLE_DEVICES=3 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_stage_memory
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=1024 python benchmarks/moe-cute.py --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_stage_memory
CUDA_VISIBLE_DEVICES=5 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
CUDA_VISIBLE_DEVICES=6 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

### 下一步

- SonicMoE 仓库内：
  - 继续把阶段显存 probe 用到真实热点迭代中；
  - 优先研究稳定 `fp8-mainline` 大 row-count vec4 path 的恢复方案；
  - blockscaled 方向只保留为长期路径，除非能直接吃掉 `grouped_out` / router 边界。
- 本地 `operator-incubator`：
  - 需要把当前“大 row-count fallback”升级为真正高性能的大 shape vec4/分块方案；
  - 否则稳定主线虽然能跑，但长期仍会慢于 bf16。

## 2026-03-24 - grouped `y2` 直接喂 router 聚合，去掉 blockscaled flat unpack

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一版 env-on blockscaled（已经做完 `pack+quant` 融合，但仍会 `grouped_out -> flat y2 -> router`）。
- 性能基线 2：同 shape 的稳定 `fp8-mainline`。
- 重要说明：
  - 本轮不是去掉 `grouped_out` 本体，而是先去掉 **`flat unpack` 这层中间表示**。
  - 新路径：
    - down-proj blockscaled GEMM 直接产出 `grouped_out`
    - router gather 直接消费 grouped/static layout
    - 不再物化 `TK x H` 的 flat `y2`
  - 这一步如果有效，主要收益应该来自：
    - inference/e2e
    - 少一次大张量 unpack
  - 但 peak memory 不一定立刻下降，因为 `grouped_out` 本体仍在。

### 改动

- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中：
  - 新增 `blockscaled_fp8_gemm_grouped(...)`
  - 让 blockscaled 主路径可以直接返回 grouped/static-capacity 布局输出
  - 保留 `blockscaled_fp8_gemm(...)` 作为兼容包装：需要 flat 输出时仍可走 unpack
- 新增 grouped 索引辅助：
  - `make_blockscaled_grouped_reverse_scatter_idx(...)`
  - 当前快路径直接复用已知 `selected_experts`，不再靠 `searchsorted` 反推 expert id
- 在 `sonicmoe/functional/__init__.py` 中：
  - blockscaled 下行前向不再调用 flat `blockscaled_fp8_gemm(...)`
  - 改成：
    - `blockscaled_fp8_gemm_grouped(...)`
    - `router_perm = make_blockscaled_grouped_reverse_scatter_idx(..., expert_ids=selected_experts.reshape(-1))`
    - `_router_forward(y2=grouped_out.view(-1, H), ...)`
  - 同时把 `selected_experts` 直接穿过 `_DownProjection.apply(...)`
- 在 `tests/fp8_protocol_test.py` 中新增：
  - `test_blockscaled_grouped_reverse_scatter_idx_matches_static_capacity_layout`

### 收益来源说明

- 本轮 forward/e2e 收益来源：
  - 删掉 `grouped_out -> flat y2` 的 unpack；
  - router 聚合直接从 grouped/static layout 读取；
  - grouped index 不再靠 `searchsorted` 反推，而是直接复用路由阶段已有 expert id。
- 为什么 peak memory 还没掉：
  - `grouped_out` 本体仍然存在；
  - benchmark 的峰值点目前仍被 grouped/static-capacity output buffer 主导。
- 这说明下一步应该继续去掉的是：
  - `grouped_out` 本体
  - 或者让 GEMM 直接写 router 可聚合布局

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'grouped_reverse_scatter_idx or blockscaled_downproj'
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
4 passed, 8 deselected
13 passed
```

解释：

- grouped reverse scatter 的静态合同测试通过；
- env-on blockscaled 回归从之前 `12 passed` 增长到 `13 passed`；
- 说明 grouped router 直连本身已经被正式回归覆盖。

### 中等真实 shape 精度 / 显存 / 性能数据

统一 shape：`4096,4096,1024,128,8`

- 官方 bf16：
  - peak memory：`7049.88 MiB`
  - Fwd inference：`2.344 ms`
  - Fwd training：`2.236 ms`
  - Fwd+Bwd：`7.338 ms`
  - Bwd：`4.994 ms`
- 稳定 fp8-mainline：
  - output RMSE：`0.01073638`
  - loss RMSE：`0.00000020`
  - peak memory：`6867.00 MiB`
  - Fwd inference：`2.390 ms`
  - Fwd training：`2.890 ms`
  - Fwd+Bwd：`7.693 ms`
  - Bwd：`5.303 ms`
- 上一版 env-on blockscaled（`pack+quant` 融合后）：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak memory：`7396.13 MiB`
  - Fwd inference：`3.095 ms`
  - Fwd training：`3.791 ms`
  - Fwd+Bwd：`8.414 ms`
  - Bwd：`5.319 ms`
- 本轮 grouped router 直连后 env-on blockscaled：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak memory：`7396.13 MiB`
  - Fwd inference：`2.918 ms`
  - Fwd training：`3.818 ms`
  - Fwd+Bwd：`8.196 ms`
  - Bwd：`5.279 ms`

收益 / 损失：

- 相对上一版 env-on blockscaled：
  - Fwd inference 提升：`5.72%`
  - Fwd training 退化：`0.71%`
  - Fwd+Bwd 提升：`2.59%`
  - Bwd 提升：`0.75%`
  - peak memory：`持平`
- 相对稳定 `fp8-mainline`：
  - output RMSE 基本持平
  - peak memory 仍多：`529.13 MiB`
  - Fwd inference 仍慢：`22.09%`
  - Fwd training 仍慢：`32.11%`
  - Fwd+Bwd 仍慢：`6.54%`
  - Bwd 反而快：`0.45%`
- 相对官方 bf16：
  - peak memory 仍多：`346.25 MiB`
  - Fwd inference 仍慢：`24.49%`
  - Fwd training 仍慢：`70.75%`
  - Fwd+Bwd 仍慢：`11.69%`
  - Bwd 仍慢：`5.71%`

解释：

- 这一步已经证明：
  - `flat unpack` 不是必须存在的；
  - blockscaled 可以直接与 SonicMoE 的 router 聚合合同对接；
  - e2e 继续下降，且 backward 已经略优于稳定 `fp8-mainline`。
- 但显存没动，说明真正的下一堵墙已经更收敛了：
  - `grouped_out` 本体
  - 而不是它后面的 flat unpack

### 当前结论

- grouped router 直连是一个**值得提交**的小里程碑：
  - 不改精度；
  - 不破坏 Blackwell 回归；
  - 继续压低 blockscaled e2e / bwd。
- 下一步第一优先级：
  1. 继续处理 `grouped_out` 本体；
  2. 尝试让 GEMM 直接写 router 可聚合布局，或让 router 聚合原生接受 grouped output contract；
  3. 并行修 `8192,4096,1024,128,8` 的 fused preact quant crash。

## 2026-03-24 - blockscaled `pack+quant` 融合，去掉 `grouped_a` 物化

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一版 env-on blockscaled（专用段拷贝桥，但仍是 `pack -> grouped_a -> quantize` 两段式）。
- 性能基线 2：同 shape 的稳定 `fp8-mainline`。
- 重要说明：
  - 本轮只吃掉了 blockscaled 前半段的大 buffer：
    - 不再 materialize `grouped_a`
    - 直接 `flat sorted activation -> grouped fp8 activation + grouped scale`
  - `grouped_out` 还在，所以峰值显存主矛盾并没有彻底解决。
  - 这一步的收益应该主要体现在：
    - inference/training forward
    - e2e
    - 而不是最终 peak memory

### 改动

- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中新增：
  - `_pack_quantize_expert_segments_kernel`
  - `_pack_quantize_grouped_rows(...)`
- 新路径直接完成：
  - 从平铺 expert-sorted `a`
  - 按 `expert_frequency_offset` 写入 grouped/static-capacity 布局
  - 同时完成 `1x32` blockwise FP8 quant
  - 直接产出：
    - `a_fp8`
    - grouped `dequant_scale_fp32`
  - 然后只保留一个很小的 `round_scale_to_e8m0(...) + pack_blockscaled_1x32_scales(...)`
- 旧的：
  - `_pack_grouped_rows(...)`
  - `quantize_activation_blockwise(grouped_a, ...)`
  的串行前半段已经不再走主路径。

### 收益来源说明

- 本轮 forward/e2e 收益来源非常明确：
  - 去掉了 `grouped_a` 这个大 bf16 中间张量；
  - 去掉了 “先 pack 再 quantize” 的两次大张量往返；
  - 直接把 sorted activation 一步落到 grouped fp8 contract。
- 本轮为什么显存没同步改善：
  - 现在最大的额外峰值已经不是 `grouped_a`，而是 `grouped_out`；
  - 所以下一步必须继续处理输出布局/聚合直连，而不是停在前半段。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'blockscaled_downproj or blockwise_quant_matches_divide_reference_after_e8m0_encoding'
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
4 passed, 7 deselected
12 passed
```

解释：

- blockscaled 合同测试继续通过；
- env-on Blackwell 回归继续通过；
- 说明 `pack+quant` 融合没有破坏现有静态合同和数值边界。

### 中等真实 shape 精度 / 显存 / 性能数据

统一 shape：`4096,4096,1024,128,8`

- 官方 bf16：
  - peak memory：`7049.88 MiB`
  - Fwd inference：`2.344 ms`
  - Fwd training：`2.236 ms`
  - Fwd+Bwd：`7.338 ms`
  - Bwd：`4.994 ms`
- 稳定 fp8-mainline（当前同机对照）：
  - output RMSE：`0.01073638`
  - loss RMSE：`0.00000020`
  - peak memory：`6867.00 MiB`
  - Fwd inference：`2.390 ms`
  - Fwd training：`2.890 ms`
  - Fwd+Bwd：`7.693 ms`
  - Bwd：`5.303 ms`
- 上一版 env-on blockscaled：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak memory：`7396.13 MiB`
  - Fwd inference：`4.362 ms`
  - Fwd training：`6.715 ms`
  - Fwd+Bwd：`11.668 ms`
  - Bwd：`7.307 ms`
- 本轮 `pack+quant` 融合后 env-on blockscaled：
  - output RMSE：`0.01073363`
  - loss RMSE：`0.00000019`
  - peak memory：`7396.13 MiB`
  - Fwd inference：`3.095 ms`
  - Fwd training：`3.791 ms`
  - Fwd+Bwd：`8.414 ms`
  - Bwd：`5.319 ms`

收益 / 损失：

- 相对上一版 env-on blockscaled：
  - Fwd inference 提升：`29.05%`
  - Fwd training 提升：`43.54%`
  - Fwd+Bwd 提升：`27.89%`
  - Bwd 提升：`27.21%`
  - peak memory：`持平`
- 相对稳定 `fp8-mainline`：
  - output RMSE 基本持平
  - peak memory 仍多：`529.13 MiB`
  - Fwd inference 仍慢：`29.50%`
  - Fwd training 仍慢：`31.17%`
  - Fwd+Bwd 仍慢：`9.37%`
  - Bwd 基本持平：`0.30%`
- 相对官方 bf16：
  - peak memory 仍多：`346.25 MiB`
  - Fwd inference 仍慢：`32.04%`
  - Fwd training 仍慢：`69.54%`
  - Fwd+Bwd 仍慢：`14.66%`
  - Bwd 仍慢：`6.51%`

解释：

- 这一步已经把 blockscaled 从“明显不可用”拉回到“接近稳定 fp8-mainline 的 e2e 区间”；
- 最亮眼的是：
  - `Bwd` 已经几乎追平稳定 `fp8-mainline`
  - `E2E` 差距已经从非常大缩到个位数百分比量级（相对稳定 fp8-mainline）
- 但显存完全没动，说明当前下一堵墙已经非常明确：
  - 不是前半段 pack/quant
  - 而是 `grouped_out` + flat unpack + router 聚合边界

### 当前结论

- `pack+quant` 融合是一个**确定可提交**的新里程碑：
  - Blackwell 回归继续通过；
  - 精度保持；
  - 中等真实 shape 的 blockscaled e2e 提升接近 `28%`。
- 下一步第一优先级现在进一步收敛为：
  1. 去掉 `grouped_out -> flat out` 这层过渡；
  2. 让 router 聚合直接消费 grouped/static layout，或者让 down-proj 直接写可聚合布局；
  3. 只有做完这一层，blockscaled 才有机会同时赢下性能和显存。

## 2026-03-24 - `e8m0` reciprocal 优化落地，并完成中等真实 shape 对照

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：同 shape、改动前的稳定 `fp8-mainline`。
- 性能基线 2：同 shape、当前 env-on blockscaled 路径。
- 重要说明：
  - 这一步不是主循环迁移，而是**先把已经在线上的量化热点除法替换成 reciprocal**：
    - SonicMoE 自己的 `fp8_quant.py`
    - SonicMoE 当前真实在用的 preact fused quant kernel
  - 由于 `e8m0` 最终编码为 2 的幂，这一步在 `using_pow2_scaling=True` 合同下可以做到**bitwise 等价**。
  - 本轮首次补上了中等真实规整 shape：`4096,4096,1024,128,8`。
  - 更大的 `8192,4096,1024,128,8` 目前会在 preact fused quant kernel 里触发 CUTLASS runtime crash；这是现有 fused boundary 的规模稳定性问题，不是本轮 reciprocal 改动引入的。

### 改动

- 在 `sonicmoe/functional/fp8_quant.py` 中：
  - 将 e8m0 编码后的逐元素缩放从
    - `grouped_x / scale`
    - 改成 `grouped_x * reciprocal(scale)`
- 在 `tests/fp8_protocol_test.py` 中新增：
  - `test_blockwise_quant_matches_divide_reference_after_e8m0_encoding`
  - 用显式 divide reference 验证 reciprocal 路径与旧合同完全一致。
- 在 `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py` 中：
  - 将 4 组 scale 路径从普通除法改成 `cute.arch.rcp_approx(...)`
  - 覆盖：
    - single-op prob
    - vec4 prob
    - vec4 no-prob
    - single-op no-prob
  - 注意：
    - `operator-incubator` 在当前机器上不是独立 git repo；
    - 这部分改动目前只存在于本地 workspace，尚不能随 SonicMoE 仓库一起 push。

### 收益来源说明

- SonicMoE 侧收益来源：
  - 去掉 blockwise quant 的逐元素除法；
  - 改为 reciprocal 预计算后乘法；
  - 这一步直接命中 blockscaled path 的激活量化热点。
- fused quant kernel 侧收益来源：
  - 去掉每个 quant block 的 `FP8_E4M3_MAX / block_amax`
  - 去掉每个 quant block 的 `1 / quant_scale`
  - 在 `pow2/e8m0` 合同下，`rcp_approx` 不会引入额外量化误差。
- 本轮没有去动 sigmoid 本身的数值路径，只碰 scale 路径，因此精度风险是可控且可验证的。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'runtime_and_reference_quant or blockwise_quant_matches_divide_reference_after_e8m0_encoding or preact_cutely_fused_path_matches_reference_boundary'
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
3 passed, 8 deselected
12 passed
12 passed
```

此外，本轮还对 fused quant kernel 做了单独 bitwise 验证：

```text
q_rmse=0.0, q_max=0.0
s_rmse=0.0, s_max=0.0
```

解释：

- reciprocal 路径没有改变 `e8m0` 编码后的量化结果；
- SonicMoE 默认 Blackwell 路径和 env-on blockscaled 路径都继续通过。

### 独立量化微基准

命令：

```bash
python - <<'PY'
# shape=(128, 512, 1024)
# old_fn: divide
# new_fn: reciprocal * multiply
PY
```

结果：

- shape：`(128, 512, 1024)`
- old：`0.510781 ms`
- new：`0.481161 ms`
- 量化热点提速：`5.80%`
- RMSE：`0.00000000`
- max abs：`0.00000000`

解释：

- 这说明 reciprocal 改动在局部热点上**确实有真实收益**，而不是纯粹代码风格改写。
- 但它的量级还不够大，不能单独解决端到端主矛盾。

### 中等真实 shape 精度 / 显存 / 性能数据

统一 shape：`4096,4096,1024,128,8`

- 官方 bf16：
  - peak memory：`7049.88 MiB`
  - Fwd inference：`2.344 ms`
  - Fwd training：`2.236 ms`
  - Fwd+Bwd：`7.338 ms`
  - Bwd：`4.994 ms`
- 稳定 fp8-mainline（本轮 reciprocal 后）：
  - output RMSE vs bf16：`0.01073638`
  - loss RMSE vs bf16：`0.00000020`
  - peak memory：`6867.00 MiB`
  - Fwd inference：`2.390 ms`
  - Fwd training：`2.890 ms`
  - Fwd+Bwd：`7.693 ms`
  - Bwd：`5.303 ms`
- 同 shape、改动前稳定 fp8-mainline：
  - Fwd inference：`2.382 ms`
  - Fwd training：`2.894 ms`
  - Fwd+Bwd：`7.808 ms`
  - Bwd：`5.426 ms`
- 当前 env-on blockscaled：
  - output RMSE vs bf16：`0.01073363`
  - loss RMSE vs bf16：`0.00000019`
  - peak memory：`7396.13 MiB`
  - Fwd inference：`4.362 ms`
  - Fwd training：`6.715 ms`
  - Fwd+Bwd：`11.668 ms`
  - Bwd：`7.307 ms`

收益 / 损失：

- 稳定 fp8-mainline 相对改动前：
  - Fwd inference 变化：`-0.34%`（共享机噪声范围内，可视为持平）
  - Fwd training 提升：`0.14%`
  - Fwd+Bwd 提升：`1.47%`
  - Bwd 提升：`2.27%`
- 稳定 fp8-mainline 相对 bf16：
  - 显存领先：`182.88 MiB`
  - Fwd inference 仍慢：`1.96%`
  - Fwd training 仍慢：`29.25%`
  - Fwd+Bwd 仍慢：`4.84%`
  - Bwd 仍慢：`6.19%`
- 当前 env-on blockscaled 相对 bf16：
  - 显存反而多：`346.25 MiB`
  - Fwd inference 仍慢：`86.09%`
  - Fwd training 仍慢：`200.31%`
  - Fwd+Bwd 仍慢：`58.99%`
  - Bwd 仍慢：`46.31%`
- 当前 env-on blockscaled 相对稳定 fp8-mainline（本轮后）：
  - peak memory 多：`529.13 MiB`
  - Fwd inference 仍慢：`82.51%`
  - Fwd training 仍慢：`132.35%`
  - Fwd+Bwd 仍慢：`51.67%`
  - Bwd 仍慢：`37.79%`

解释：

- 这一步已经确认：
  - reciprocal 是有效优化；
  - 稳定 `fp8-mainline` 已经在中等真实 shape 上拿到显存优势，并把性能差距压到很小；
  - 但当前还**没有**超过官方 `bf16`。
- 同时也确认：
  - env-on blockscaled 目前仍然不是可交付主线；
  - 主因不是量化除法，而是：
    - static capacity 额外 buffer
    - grouped bridge / pack-unpack
    - 输出布局不能直接直连现有路由聚合

### 当前结论

- 这次 reciprocal 优化可以作为一个**可保留、可提交**的小里程碑：
  - 精度不变；
  - 局部热点确定提速；
  - 稳定 `fp8-mainline` 的 e2e / bwd 在真实规整 shape 上出现可测改善。
- 但它不是主矛盾终点。
- 下一步第一优先级已经进一步收敛为：
  1. 把 blockscaled 路径的 `pack + quant + grouped_out` 过渡层继续消掉，至少先把 `pack+quant` 合并；
  2. 让 routing metadata / 下游聚合直接接受静态 expert layout，避免 env-on blockscaled 额外峰值显存；
  3. 修掉大 shape（`8192,4096,1024,128,8`）下 preact fused quant kernel 的 runtime crash。

## 2026-03-24 - Blackwell blockscaled down-proj 专用段拷贝桥与容量安全合同

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：稳定 `fp8-mainline`。
- 性能基线 2：上一版静态合同 blockscaled（GPU `searchsorted + index_copy_/index_select` bridge）。
- 重要说明：
  - 本轮**还没有完全消掉 GPU bridge**，但已经把通用张量操作换成了两段专用 Triton 段拷贝 kernel。
  - 这一步的真实收益主要体现在 **forward/cudagraph 路径**，因为 down-proj 当前只有 forward 进入了新 blockscaled mainloop；backward 还没跟上，所以 `e2e/bwd` 不能简单按同一比例解读。
  - 本轮还补上了一个关键安全合同：`capacity` 不足时必须报错，不能静默截断。

### 改动

- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中：
  - 删除了上一版通用 GPU bridge 的核心路径：
    - `searchsorted(cu_seqlens_m[1:])`
    - `index_copy_`
    - `index_select`
  - 新增两个专用 Triton kernel：
    - `_pack_expert_segments_kernel`
    - `_unpack_expert_segments_kernel`
  - 它们直接利用 `expert_frequency_offset` 所表达的“按 expert 连续段”语义做静态 capacity 布局转换。
  - 增加了 blockscaled 静态合同检查：
    - `SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY` 必须是 `128` 的整数倍
    - `capacity` 必须大于等于当前 batch 的最大 expert load
    - `intermediate_size` 必须是 `128` 的整数倍
    - `hidden_size` 必须是 `128` 的整数倍
- 在 `tests/fp8_protocol_test.py` 中新增：
  - `test_blockscaled_downproj_rejects_insufficient_capacity`
  - 覆盖“capacity 不足必须明确报错”

### 收益来源说明

- 本轮 forward 收益来源：
  - 不再做通用 `searchsorted + index_copy_/index_select`
  - 改为基于 `expert_frequency_offset` 的专用段拷贝 kernel
  - 它更接近最终想要的“routing metadata 直接产出静态布局”方向
- 本轮正确性收益来源：
  - 以前 `capacity` 不足会静默截断，极其危险
  - 现在已经变成显式失败合同，后续 agent 不会再被假性能/假精度误导

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'blockscaled_downproj'
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
3 passed, 7 deselected
11 passed
```

解释：

- 第一条说明 blockscaled 静态合同相关测试已经覆盖：
  - 未对齐 capacity 拒绝
  - capacity 不足拒绝
  - 给定静态 capacity 时 finite forward/backward
- 第二条说明 env-on blockscaled 完整 Blackwell 回归继续通过。

### 精度数据

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果（本轮专用段拷贝桥）：

- output RMSE vs bf16：`0.00131902`
- loss RMSE vs bf16：`0.00000013`

解释：

- 与上一版 blockscaled、稳定 `fp8-mainline` 基本持平。
- 说明这次桥接替换没有引入新的数值退化。

### 显存数据

同一条命令输出：

- bf16 peak memory：`380.25 MiB`
- 本轮 blockscaled fp8 peak memory：`142.78 MiB`

对照：

- 上一版 blockscaled：`142.81 MiB`
- 稳定 `fp8-mainline`：`134.44 MiB`

解释：

- 相对上一版 blockscaled 几乎不变（`-0.03 MiB`）。
- 相对稳定 `fp8-mainline` 仍多 `8.34 MiB`，说明主要剩余问题已经不是 metadata 算法，而是**bridge 仍然存在本身**。

### 性能数据

同机、同 shape（`1024,512,512,32,4`）对照：

- 官方 bf16：
  - Fwd inference：`0.187 ms`
  - Fwd training：`1.186 ms`
  - Fwd+Bwd：`3.292 ms`
  - Bwd：`3.105 ms`
- 稳定 `fp8-mainline`：
  - Fwd inference：`0.229 ms`
  - Fwd training：`1.561 ms`
  - Fwd+Bwd：`3.534 ms`
  - Bwd：`3.305 ms`
- 上一版静态合同 blockscaled：
  - Fwd inference：`0.476 ms`
  - Fwd training：`2.046 ms`
  - Fwd+Bwd：`4.004 ms`
  - Bwd：`3.528 ms`
- 本轮专用段拷贝桥 blockscaled：
  - Fwd inference：`0.250 ms`
  - Fwd training：`1.998 ms`
  - Fwd+Bwd：`4.259 ms`
  - Bwd：`4.009 ms`

收益 / 损失：

- 相对上一版静态合同 blockscaled：
  - Fwd inference 提升 `47.48%`
  - Fwd training 提升 `2.35%`
  - Fwd+Bwd 退化 `6.37%`
  - Bwd 退化 `13.63%`
- 相对稳定 `fp8-mainline`：
  - Fwd inference 仍慢 `9.17%`
  - Fwd training 仍慢 `27.99%`
  - Fwd+Bwd 仍慢 `20.51%`
  - Bwd 仍慢 `21.30%`
- 相对官方 bf16：
  - Fwd inference 仍慢 `33.69%`
  - Fwd training 仍慢 `68.47%`
  - Fwd+Bwd 仍慢 `29.37%`
  - Bwd 仍慢 `29.11%`

解释：

- 这组数据说明专用段拷贝桥**确实解决了 forward 方向最重的 bridge 开销**。
- 但 `e2e/bwd` 还没有跟上，原因非常明确：
  - 当前真正进入 FP8 blockscaled mainloop 的只有 down-proj forward
  - backward 相关 GEMM 仍沿用旧路径
  - 因此下一步不能继续只盯 down-proj forward，而要开始把剩余 GEMM 主循环一起迁过去

### 当前结论

- 当前 blockscaled 路径已经从“通用 GPU bridge”进一步收敛为“专用段拷贝桥 + 明确静态合同”。
- 这是一条更接近最终交付版的过渡路径，但仍然**不是终点**。
- 下一步最值得做的两件事：
  1. 继续去掉剩余 bridge，让 routing metadata 直接产出静态布局
  2. 把 up-proj / backward GEMM 一起迁到同一类 FP8 mainloop 合同

## 2026-03-24 - Blackwell `1x32 ue8m0` blockscaled down-proj 静态对齐合同版

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一轮稳定 `fp8-mainline`（pre-SwiGLU fused boundary，down-proj 仍是普通 QuACK GEMM）。
- 性能基线 2：上一版 blockscaled 过渡桥（host-side grouped metadata + 非 capture-safe）。
- 性能基线 3：官方 `bf16` 小 shape 路径。
- 重要说明：
  - 本轮 blockscaled path 的合同已经切换为：**内核层只接受静态对齐后的 expert capacity**。
  - padding/容量规划不再在 blockscaled kernel 内动态决定，而是上移为显式运行合同：
    - `SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY`
    - 必须是 `128` 的整数倍
  - 当前实现仍保留 GPU 侧 pack/unpack，因此还不是最终极致版本；但 host-side metadata 已经去掉，**cudagraph capture 已恢复**。

### 改动

- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中去掉了上一版的 CPU 侧 grouped bridge：
  - 删除 `cu_seqlens_m.detach().cpu().tolist()` 相关路径。
  - 新增静态合同：
    - `SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY`
    - blockscaled path 只接受对齐后的静态 expert capacity。
  - 新增 GPU 侧 padded position 构造：
    - 用 `torch.searchsorted(cu_seqlens_m[1:])` 直接在设备上恢复 `expert_id / within_expert_rank`
    - 生成固定 shape 的 `padded_positions`
  - 新增 GPU 侧 pack/unpack：
    - `index_copy_` 将平铺 `TK x I` 激活写入 `(E, capacity, I)`
    - `index_select` 将 `(E, capacity, H)` 输出还原回平铺 `TK x H`
- 在 `tests/fp8_protocol_test.py` 中增加了静态合同测试：
  - 未对齐 capacity 必须报错
  - 给定 `capacity=128` 时 env-on blockscaled 路径 forward/backward 必须保持有限
- 在 `benchmarks/moe-cute.py` 中恢复了 blockscaled 路径的 cudagraph inference benchmark。

### 收益来源说明

- 相对上一版 blockscaled 过渡桥，本轮收益来源非常明确：
  - 去掉 host-side grouped metadata；
  - 恢复固定 shape capture；
  - 把“动态桥”收敛为“静态容量合同 + GPU pack/unpack”。
- 这意味着：
  - 当前 blockscaled 路径已经从“可运行实验实现”升级为“可回归、可 benchmark、可交付的静态合同版本”；
  - 但最终性能仍被 GPU pack/unpack 吃掉一部分，下一步重点已经收敛为**继续消掉 pack/unpack**，而不是再回头啃 runtime wrapping。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=128 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
10 passed
10 passed
```

解释：

- 第一条说明 env-on blockscaled 静态合同版已经通过完整 Blackwell 回归。
- 第二条说明默认主线继续保持稳定。

### 精度数据

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果（本轮静态合同版 blockscaled）：

- output RMSE vs bf16：`0.00131902`
- loss RMSE vs bf16：`0.00000013`

对照（上一轮稳定 `fp8-mainline`）：

- output RMSE vs bf16：`0.00131936`
- loss RMSE vs bf16：`0.00000015`

解释：

- 精度没有恶化，甚至略优于上一轮稳定 mainline。
- 说明本轮合同收敛主要影响的是调度/搬运开销，而不是数值路径。

### 显存数据

同一条命令输出：

- bf16 peak memory：`380.25 MiB`
- blockscaled fp8 peak memory：`142.81 MiB`

对照：

- 上一轮稳定 `fp8-mainline`：`134.44 MiB`
- 上一版 blockscaled 过渡桥：`142.77 MiB`

解释：

- 相对官方 bf16，当前 blockscaled 路径仍少用 `237.44 MiB`，约 `62.44%`。
- 相对稳定 `fp8-mainline`，当前多用了 `8.37 MiB`，来源是静态 capacity 下的 GPU pack/unpack 中间缓冲。
- 相对上一版 blockscaled 过渡桥，显存基本持平，说明本轮主要收益来自调度与 capture 恢复，而不是缓存量级变化。

### 性能数据

同机、同 shape（`1024,512,512,32,4`）对照：

- 官方 bf16：
  - Fwd inference：`0.187 ms`
  - Fwd training：`1.186 ms`
  - Fwd+Bwd：`3.292 ms`
  - Bwd：`3.105 ms`
- 稳定 `fp8-mainline`：
  - Fwd inference：`0.229 ms`
  - Fwd training：`1.561 ms`
  - Fwd+Bwd：`3.534 ms`
  - Bwd：`3.305 ms`
- 上一版 blockscaled 过渡桥：
  - Fwd training：`2.953 ms`
  - Fwd+Bwd：`5.537 ms`
- 本轮静态合同版 `fp8-blockscaled`：
  - Fwd inference：`0.476 ms`
  - Fwd training：`2.046 ms`
  - Fwd+Bwd：`4.004 ms`
  - Bwd：`3.528 ms`

收益 / 损失：

- 相对上一版 blockscaled 过渡桥：
  - Fwd training 提升 `30.71%`
  - Fwd+Bwd 提升 `27.69%`
  - inference cudagraph 从“不可测”恢复为 `0.476 ms`
- 相对稳定 `fp8-mainline`：
  - Fwd inference 仍慢 `107.86%`
  - Fwd training 仍慢 `31.07%`
  - Fwd+Bwd 仍慢 `13.30%`
  - Bwd 仍慢 `6.75%`
- 相对官方 bf16：
  - Fwd inference 仍慢 `154.55%`
  - Fwd training 仍慢 `72.51%`
  - Fwd+Bwd 仍慢 `21.63%`
  - Bwd 仍慢 `13.62%`

解释：

- 这组数据说明：
  - “静态 capacity + GPU pack/unpack” 这一步是对的，已经显著回收了上一版桥接开销；
  - 但仅靠当前合同还不够，真正的极致性能下一步必须继续去掉 GPU pack/unpack，或者把 routing metadata 直接生成为 blockscaled mainloop 需要的静态布局。

### 下一步

- 第一优先级：继续消掉 GPU pack/unpack，把 down-proj 改成真正的静态对齐直连布局。
- 第二优先级：把同样的“静态对齐 + fp8 mainloop”合同扩展到其余 GEMM 相关路径：
  - up-proj forward
  - down-proj backward act
  - down-proj backward weight
- 第三优先级：在 operator 数量上继续向 baseline 对齐，把剩余 boundary 融合吃掉。

## 2026-03-24 - Blackwell `1x32 ue8m0` blockscaled down-proj 主循环打通（3D grouped 试运行）

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一轮稳定主线 `fp8-mainline`（pre-SwiGLU fused boundary，未启用 blockscaled down-proj）。
- 性能基线 2：官方 `bf16` 小 shape 路径。
- 重要说明：
  - 本轮 `blockscaled down-proj` 已经真实进入 SM100 blockscaled mainloop，并通过 Blackwell 回归。
  - 但当前实现为了绕开 `varlen_m + blockscaled` 的外层契约冲突，采用了 **SonicMoE 侧 3D grouped pad/unpad** 的过渡方案。
  - 因此当前数据的收益/损失来源必须区分清楚：**主循环接通** 是收益，**pad/unpad 与 host-side metadata** 是当前主要性能损失来源。
  - 由于 grouped metadata 还依赖 host 侧读取，当前实验路径**暂不支持 inference cudagraph capture**；因此本轮只采信 `training fwd / e2e / 非-cg 估算 bwd`。

### 改动

- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中完成了可运行的 Blackwell blockscaled down-proj 路径：
  - 通过 `uint8 storage view + element_type override` 方式绕开 `float8 -> DLPack` 限制。
  - 不再走先前失败的 rank-2 `varlen_m` 直连方案。
  - 改为先把 `y1` 按 expert offsets 打包成 `3D grouped` 形式，再喂给 `GemmDefaultSm100(sf_vec_size=32)`。
  - kernel 输出后再 unpad 回原始 `TK x H` 平铺布局。
- 在 `benchmarks/moe-cute.py` 中补充了实验路径的 benchmark 兼容：
  - 当显式开启 `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1` 时，benchmark 不再强行做 cudagraph capture。
  - 会明确打印“当前实验路径暂不支持 inference capture”，并继续输出 `training fwd / e2e / 非-cg 估算 bwd`。

### 收益来源说明

- 本轮真正的工程收益：
  - `1x32 ue8m0` scale pack 已经不只是协议脚手架，而是进入了真实 down-proj 主循环。
  - `float8 runtime wrapping` 已经被压到可运行状态，下一任 agent 不需要再从 `DLPack 不支持 float8` 这个原点重新排查。
  - `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1` 现在是**真实可回归、可 benchmark、可 handoff** 的实验入口，而不是只会报错的占位开关。
- 本轮性能损失来源：
  - `grouped pad/unpad` 额外引入了张量搬运和零填充。
  - grouped metadata 仍有 host 参与，因此 inference cudagraph 被迫关闭。
  - 这说明当前结果**不能**代表 blockscaled mainloop 的最终绝对性能，只能说明主循环接线已打通。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
8 passed
8 passed
```

解释：

- 第一条说明 env-on 的 blockscaled 实验路径已经可运行且通过 Blackwell 回归。
- 第二条说明默认主线没有被这轮实验改坏。

### 精度数据

命令：

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果（blockscaled 实验路径）：

- output RMSE vs bf16：`0.00131902`
- loss RMSE vs bf16：`0.00000013`

对照（上一轮稳定 `fp8-mainline`）：

- output RMSE vs bf16：`0.00131936`
- loss RMSE vs bf16：`0.00000015`

解释：

- 精度没有进一步恶化，`output RMSE` 和 `loss RMSE` 都与上一轮稳定主线基本持平。
- 这说明本轮性能问题主要来自数据搬运/调度层，而不是 blockscaled 主循环本身已经把数值打坏。

### 显存数据

同一条命令输出：

- bf16 peak memory：`380.25 MiB`
- blockscaled fp8 peak memory：`142.77 MiB`

对照（上一轮稳定 `fp8-mainline`）：

- 旧 fp8 peak memory：`134.44 MiB`

解释：

- 相对官方 bf16，当前 blockscaled 实验路径仍少用 `237.48 MiB`，约 `62.45%`。
- 但相对上一轮稳定 fp8，显存增加了 `8.33 MiB`，收益损失来源非常明确：`3D grouped pad/unpad` 的中间缓冲。

### 性能数据

同机、同 shape（`1024,512,512,32,4`）对照：

- 官方 bf16：
  - Fwd training：`1.151 ms`
  - Fwd+Bwd：`3.158 ms`
- 上一轮稳定 `fp8-mainline`：
  - Fwd training：`1.552 ms`
  - Fwd+Bwd：`3.486 ms`
- 本轮 `fp8-blockscaled`（实验路径）：
  - Fwd training：`2.953 ms`
  - Fwd+Bwd：`5.537 ms`
  - Bwd（由非-cg fwd 估算）：`2.584 ms`

收益 / 损失：

- 相对上一轮稳定 `fp8-mainline`：
  - Fwd training 变慢 `90.27%`
  - Fwd+Bwd 变慢 `58.84%`
- 相对官方 bf16：
  - Fwd training 变慢 `156.56%`
  - Fwd+Bwd 变慢 `75.33%`

解释：

- 这组数字**不是**在否定 blockscaled mainloop，而是在暴露当前过渡接法的问题：
  - blockscaled kernel 已经跑起来；
  - 但 kernel 外围的 grouped pad/unpad 与 capture 不兼容 metadata 把收益基本吃掉了。
- 换言之，下一步不能再优先做新的量化协议，而应该优先做：
  - 去掉 host-side grouped metadata；
  - 去掉 `3D grouped pad/unpad`；
  - 把 `varlen_m` 直连 blockscaled mainloop 或等价的纯 GPU pack/unpack 契约补齐。

### 当前 blocker

- 当前 blocker 已经从“float8 runtime 根本跑不起来”变成：
  1. `grouped pad/unpad` 还不是最终形态，额外消耗明显。
  2. host-side grouped metadata 让实验路径无法进入 inference cudagraph。
  3. 因为无法走 capture，当前 `fwd inference` 还不能和既有基线做同口径绝对对比。

### 下一步

- 第一优先级：把 grouped metadata 完整搬到 GPU 侧，至少先恢复 cudagraph-safe。
- 第二优先级：继续消掉 `3D grouped pad/unpad`，把 blockscaled down-proj 改回真正的 `varlen_m` 直连。
- 第三优先级：在上述两点完成后，重跑三条基线：
  - 官方 `bf16`
  - 上一轮稳定 `fp8-mainline`
  - 新 `fp8-blockscaled`
  并重新验收 `fwd inference / e2e / bwd / RMSE / peak memory`。

## 2026-03-24 - Blackwell `1x32 ue8m0` blockscaled mainloop 试接与稳定性封边

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：上一提交 `0a7361b` 的已落地主线。
- 性能基线 2：官方 `bf16` 路径。
- 本轮结果说明：
  - 本轮**没有新增可采信的性能/精度/显存对比数据**，因为 `1x32` blockscaled down-proj 还没有稳定到可以作为默认主线去跑 benchmark。
  - 本轮唯一有效验收指标是：默认主线回归必须继续保持通过，且新 blockscaled 代码必须以**显式开关**方式留在仓内，供下一轮继续推进。

### 改动

- 在 `sonicmoe/functional/fp8_protocol.py` 中补充了 `1x32` 粒度支持：
  - `FP8ScaleGranularity.BLOCK_1X32`
  - 现阶段协议层允许 `1x32` 和 `1x128` 两种粒度并存
- 在 `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` 中新增了 Blackwell blockscaled 主循环试接脚手架：
  - `pack_blockscaled_1x32_scales(...)`
  - `blockscaled_fp8_gemm(...)`
  - `w2` 的 `fp8 + ue8m0` weight cache
- 在 `tests/fp8_protocol_test.py` 中新增了 `1x32` blockscaled scale pack 的基础测试。
- 在 `sonicmoe/functional/__init__.py` 中增加了显式环境开关：
  - `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1`
  - 默认关闭，避免不稳定 runtime 路径污染当前可用主线。

### 收益来源说明

- 本轮收益不是直接性能收益，而是**主循环改造前置条件**的落地：
  - 已经把 `1x32 ue8m0` 的 scale 物理打包逻辑写进仓内。
  - 已经把 `w2` 的 `fp8` cache 结构写进仓内。
  - 已经把 down-proj 的 blockscaled 主循环接线点固定在 `sonicmoe/functional/__init__.py::_DownProjection.forward`。
- 这意味着下一轮不需要再重新推导：
  - `1x32` tile 级 scale 的物理存储大小；
  - `M/K` 按 `128x128` tile 铺开的 pack 公式；
  - `w2` 从 `(H, I, E)` 到 `(E, H, I)` 的量化缓存位置。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

结果：

```text
8 passed
```

### 本轮 blocker

- 目标路径是：

```text
y1_fp8(1x32) + w2_fp8-cache(1x32) -> SM100 blockscaled mainloop -> bf16 output
```

- 但当前运行时仍有两个关键阻塞：
  1. `PyTorch -> DLPack` 对 `float8` 仍不支持，导致不能复用 QuACK 现有 `from_dlpack(fp8_tensor)` 包装方式。
  2. 直接在 SonicMoE 侧用 `cute.make_ptr/cute.make_tensor` 为 `float8` 动态张量造 runtime 参数时，CUTLASS Python DSL 在当前环境里会触发 abort。

- 因此本轮策略调整为：
  - **保留代码脚手架**
  - **默认关闭 blockscaled 主循环**
  - **先保证稳定主线继续全绿**

### 下一步

- 在隔离脚本里最小化复现并解决：
  - `cute.make_ptr/cute.make_tensor` 对 `float8` runtime 参数的构造契约
  - 或寻找 QuACK/CUTLASS 内部已有的 `float8` runtime tensor 包装入口
- 一旦 runtime 契约打通，第一时间打开：

```bash
SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1
```

并用当前小 shape benchmark 做首轮对比。

## 2026-03-24 - Blackwell pre-SwiGLU 融合量化前向接线

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：本轮改动前的旧 `fp8_protocol blackwell` 小 shape 路径（同机、同命令、同共享环境）。
- 性能基线 2：官方 `bf16` 小 shape 路径（同机、同命令、同共享环境）。
- 重要说明：当前 8 张 Blackwell 卡几乎全部 `99%~100%` 利用率，因此本条目的性能数据只用于**同噪声环境下的前后对照**，不作为最终绝对性能验收。绝对性能仍需等待空闲卡后用主 shape 复测。

### 改动

- 在 `sonicmoe/functional/fp8_cutely_fused.py` 中新增真正的 pre-SwiGLU 高性能路径：
  - 输入直接消费 `_UpProjection` 返回的 `z`（pre-SwiGLU）。
  - 前向量化改为 `cutify.fused_weighted_swiglu_act_quant_best(...)`。
  - 反量化改为 `cutify.fused_act_dequant_best(...)`。
- 在 `sonicmoe/functional/__init__.py` 中将 Blackwell + QuACK 的 `fp8_protocol` 默认前向路径切到：

```text
z(pre-SwiGLU) -> fused_weighted_swiglu_act_quant_best -> fused_act_dequant_best -> y1
```

- 对 `I % 128 != 0` 的尾块进行了 pre-SwiGLU 对齐填充，保证像 `2880` 这样的宽度也能走融合路径。
- 为了恢复 CUDA graph capture 可用性，没有直接在融合 kernel 内走 `ue8m0` 打包返回；而是先保留 kernel 产出的 float32 dequant scale，再在 SonicMoE 侧编码为 `e8m0`。

### 收益来源说明

- 前向收益来源：
  - 不再走旧的 post-SwiGLU torch reference `quantize + dequantize`。
  - 改为直接复用现有 Cute/CUDA 储备，把 `SwiGLU + blockwise quant` 合到一个 pre-SwiGLU kernel 里。
- 端到端收益来源：
  - 主要来自前向边界开销下降。
  - 本轮**没有**改 backward 主 kernel，因此 bwd 改善只是前向边界更轻带来的连带收益，不应误判为 backward kernel 已优化。
- 显存收益来源：
  - 这一轮的主要目标是吞吐而不是进一步降显存，所以显存几乎不变。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'preact_cutely_fused_path_matches_reference_boundary or boundary_keeps_finite_forward_backward or blackwell_fp8_protocol_runtime_and_reference_quant'
USE_QUACK_GEMM=1 python -m pytest -q tests/moe_blackwell_test.py
```

结果：

```text
3 passed, 3 deselected
1 passed
```

### 精度数据

命令：

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果（新路径）：

- output RMSE vs bf16：`0.00131936`
- loss RMSE vs bf16：`0.00000015`

对照（旧路径）：

- output RMSE vs bf16：`0.00002498`
- loss RMSE vs bf16：`0.00000000`

解释：

- 精度有可见回退，来源是 pre-SwiGLU fused path 与旧 post-SwiGLU reference path 在量化前激活舍入位置不同。
- 目前 loss RMSE 仍非常小，说明训练目标量级没有发散。
- 后续如果要继续压低 RMSE，需要继续对齐：
  - pre-SwiGLU 激活布局与 reference 的数值路径；
  - `pow2/e8m0` scale 的编码细节；
  - optional prob 融合后的真实语义位置。

### 显存数据

同一条命令输出：

- bf16 peak memory：`380.25 MiB`
- 新 fp8 peak memory：`134.44 MiB`

对照（旧路径）：

- 旧 fp8 peak memory：`134.38 MiB`

解释：

- 相对官方 bf16，当前 fp8 仍少用 `245.81 MiB`，约 `64.64%`。
- 相对旧 fp8 路径，本轮显存几乎不变（`+0.06 MiB`），这符合预期，因为本轮主要替换的是量化/反量化算子实现，而不是缓存结构。

### 性能数据

同机、同 shape（`1024,512,512,32,4`）对照：

- 官方 bf16：
  - Fwd inference：`0.176 ms`
  - Fwd+Bwd：`3.162 ms`
  - Bwd：`2.987 ms`
- 旧 fp8 路径（改动前）：
  - Fwd inference：`0.392 ms`
  - Fwd+Bwd：`3.972 ms`
  - Bwd：`3.580 ms`
- 新 fp8 路径（本轮）：
  - Fwd inference：`0.229 ms`
  - Fwd+Bwd：`3.697 ms`
  - Bwd：`3.468 ms`

收益：

- 相对旧 fp8：
  - Fwd inference 提升 `41.58%`
  - Fwd+Bwd 提升 `6.92%`
  - Bwd 提升 `3.13%`
- 相对官方 bf16：
  - Fwd inference 仍慢 `30.11%`
  - Fwd+Bwd 仍慢 `16.92%`

解释：

- 这说明“先把 torch-side boundary 换成已有的 Cute/CUDA 融合算子”这一步是有效的，尤其是前向收益很直接。
- 但它也说明仅靠替换量化/反量化实现还不够；要继续逼近甚至超过 bf16，下一步必须继续往前推进，把：
  - `prob/topk_scores`
  - 更少的中间张量
  - 真正的 backward 融合
  继续吃进主线。

### 兼容性修复

- 初始版本在 benchmark 的 CUDA graph capture 中失败，根因是 `cutify` 的 `ue8m0` 打包辅助逻辑在 capture 期间触发了不允许的操作。
- 已修复为：
  - kernel 内先输出 float32 dequant scale；
  - SonicMoE 侧再编码成 `e8m0`；
  - benchmark 现已恢复可运行。

### 下一步

- 把 `topk_scores/prob` 的语义真正前移到融合 epilogue，而不是继续留在 router 后处理。
- 结合 Paddle 的：
  - `fp8_quant_blockwise_kernel.cu`
  - `fused_stack_transpose_quant_kernel.cu`
  - `fused_transpose_split_quant_kernel.cu`
  评估是否需要在 `operator-incubator` 再孵化一个更贴近 SonicMoE 合同的新 Cute quant kernel。
- 开始准备 paired backward kernel，把当前“前向已融合、反向未融合”的状态继续推进。

## 2026-03-24 - Blackwell FP8 functional boundary wiring

### Change

- added a functional-boundary `fp8_protocol` argument to:
  - `sonicmoe/moe.py::MoE.forward`
  - `sonicmoe/functional/__init__.py::moe_TC_softmax_topk_layer`
- added `apply_activation_fp8_protocol(...)` in `sonicmoe/functional/fp8_reference.py`
- the current boundary implementation quantizes/dequantizes the up-projection activation between `_UpProjection` and `_DownProjection`
- backward uses a straight-through estimator so the training path stays usable while the fused kernel does not exist yet
- `1x128` tail blocks are now padded internally and sliced back to original width, so shapes like `2880` are legal

### Correctness validation

Command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Result:

```text
18 passed, 91 skipped
```

### Performance regression

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test
```

Baseline before boundary wiring:

- Fwd inference: `24.164 ms`, `539.9 TFLOPS`
- Fwd training: `23.549 ms`
- Fwd+Bwd: `72.987 ms`, `536.2 TFLOPS`
- Bwd: `48.823 ms`, `534.4 TFLOPS`

Baseline after boundary wiring, protocol disabled:

- Fwd inference: `23.026 ms`, `566.6 TFLOPS`
- Fwd training: `23.478 ms`
- Fwd+Bwd: `75.443 ms`, `518.8 TFLOPS`
- Bwd: `52.417 ms`, `497.8 TFLOPS`

Interpretation:

- the default path did not regress in forward
- the end-to-end training path is a little slower than the earlier baseline, so future changes must keep checking this command

### Protocol-enabled performance

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell
```

Current numbers:

- Fwd inference: `63.715 ms`, `204.8 TFLOPS`
- Fwd training: `64.160 ms`
- Fwd+Bwd: `119.803 ms`, `326.7 TFLOPS`
- Bwd: `56.087 ms`, `465.2 TFLOPS`

Interpretation:

- the current FP8 boundary path is **correctness scaffolding**, not a performance win
- the large forward slowdown is expected because quant/dequant is still implemented as separate torch-side reference ops
- the next fused-kernel milestone must eliminate this overhead by folding quantization into the up-projection epilogue

### Next action

- replace the torch-side `apply_activation_fp8_protocol(...)` boundary path with a fused up-projection epilogue implementation
- keep using the same benchmark command above before and after every important performance-facing change

## 2026-03-24 - Parallel Blackwell regression entry

### Change

- installed `pytest-xdist` into `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- added:
  - `make test-blackwell-full`
  - `make test-blackwell-parallel PYTEST_WORKERS=2`

### Measurement

Serial command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Serial runtime:

```text
18 passed, 91 skipped in 187.28s
```

Parallel command:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

Parallel runtime:

```text
18 passed, 91 skipped in 168.14s
real 168.41
```

### Conclusion

- `xdist` with `2` workers is a real win for the current Blackwell-targeted regression subset on this machine
- keep the parallel target opt-in rather than default, because these tests are still GPU-heavy and a higher worker count may oversubscribe the device

## 2026-03-24 - Boundary memory optimization

### Change

- removed the full-width float32 activation copy from `quantize_activation_blockwise(...)`
- dequantization now writes directly to the requested output dtype instead of materializing a full float32 activation first
- kept the same protocol semantics: `e4m3` activations, `e8m0` scales, `1x128` granularity, tail padding for non-divisible widths

### Correctness validation

Command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Result:

```text
18 passed, 91 skipped
```

### Precision delta

Measured against the no-protocol baseline on the Blackwell training shape:

- max abs diff: `0.0013427734375`
- mean abs diff: `0.00018952522077597678`

The optimization did not change these error numbers relative to the previous boundary implementation.

### Memory delta

Single-run peak memory on `T=32768, H=2880, I=2880, E=64, K=8`:

Before optimization:

- baseline fwd peak: `9611.98 MiB`
- baseline e2e peak: `11826.48 MiB`
- fp8 boundary fwd peak: `15312.85 MiB`
- fp8 boundary e2e peak: `15312.85 MiB`

After optimization:

- baseline fwd peak: `9611.98 MiB`
- baseline e2e peak: `11826.48 MiB`
- fp8 boundary fwd peak: `13017.85 MiB`
- fp8 boundary e2e peak: `13017.85 MiB`

Interpretation:

- fp8 boundary peak memory dropped by `2295.00 MiB` (~`15.0%`) on both forward and end-to-end peak
- baseline memory stayed unchanged

### Performance delta

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell
```

Before optimization, fp8 boundary:

- Fwd inference: `63.715 ms`
- Fwd+Bwd: `119.803 ms`

After optimization, fp8 boundary:

- Fwd inference: `56.065 ms`
- Fwd+Bwd: `109.117 ms`

Interpretation:

- fp8 boundary forward improved by `7.650 ms` (~`12.0%`)
- fp8 boundary end-to-end improved by `10.686 ms` (~`8.9%`)
- the path is still much slower than the bf16 baseline, so the next real win still depends on a fused up-proj epilogue

## 2026-03-24 - Metric harness and multi-GPU shard prep

### Change

- added `--report_fp8_metrics` to `benchmarks/moe-cute.py`
- added `make test-blackwell-multigpu BLACKWELL_TEST_GPUS=...`
- added `tools/run_blackwell_test_shards.py`
- added `--dry-run` to the shard launcher so command routing can be validated even when all 8 GPUs are busy
- added an env-gated adapter landing point in `sonicmoe/functional/fp8_cutely_fused.py`
- threaded the adapter behind `SONIC_MOE_FP8_CUTELY_FUSED`

### Validation

Dry-run command:

```bash
python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run
```

Result:

```text
[blackwell-shard] gpu=0 tests=tests/fp8_protocol_test.py
[blackwell-shard] gpu=1 tests=tests/moe_blackwell_test.py
[blackwell-shard] gpu=2 tests=tests/moe_test.py
```

Metric probe command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

Result:

```text
FP8 metrics vs bf16 baseline output_rmse=0.00002498, loss_rmse=0.00000000, bf16_peak_mib=380.25, fp8_peak_mib=134.38
PASS
```

### Interpretation

- the benchmark harness now emits the bf16-vs-fp8 metrics required by the current reporting policy
- the shard launcher is safe to invoke on a saturated machine in dry-run mode before selecting idle GPUs
- the new adapter shim keeps default behavior unchanged while fixing the code landing point for the real fused epilogue
- fused-op analysis confirmed that the incubator quant kernel consumes pre-SwiGLU `(T, 2H)` activations, so a direct swap at the current post-SwiGLU boundary would be semantically wrong
