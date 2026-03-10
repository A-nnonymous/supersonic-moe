# SonicMoE FP8 集成计划

## 1. 目标

最终交付物只有一个：torch 侧增强版 MoE 层。

项目治理和多 agent 执行状态统一落在 `control_plane/fp8/` 下。计划负责说明做什么，控制面负责说明现在做到哪一步、谁在做、是否允许进入下一门禁。

Paddle 相关资产只承担三件事：

- 提供已验证的 FP8 kernel 语义参考
- 提供 `tests/reference_layers/standalone_moe_layer` 作为正确性和训练可用性的 vendored reference baseline
- 提供 compatibility 接入方案

不做的事：

- 不把 SonicMoE 重新做成 Paddle-first 组网
- 不让 Paddle 的 router 设计反向决定 SonicMoE 主路径
- 不把 cuBLASLt baseline 当成最终 fused 路线

## 2. 当前结论

### 2.1 仓库内 reference baseline

- `tests/reference_layers/standalone_moe_layer` 是当前正确性金标准。
- 关键兼容入口位于 `tests/reference_layers/standalone_moe_layer/moe_standalone/compat.py`。
- 它是普通 FP8-MoE 训练/推理组网，不追求 gather + gemm + act 深融合。
- 它包含通信逻辑，但这不是 SonicMoE 要复刻的目标。测试时可以忽略通信，只保留张量语义和数值对齐。

### 2.2 Paddle 技术储备

Paddle 当前真正成熟的是以下四类能力：

1. gate + act + quant 融合
2. blockwise / subchannel quant-dequant
3. 基于 cuBLASLt 的 FP8 GEMM
4. DeepEP 风格的 permute / unpermute / 组网支撑

其中真正可以直接沉淀为 SonicMoE 工程规划输入的是前 3 类的语义和数值口径。第 4 类更多是设计参考，不适合直接迁移成 SonicMoE 主路径。

### 2.3 对 SonicMoE 的直接影响

- SonicMoE 缺的是 torch 侧端到端 FP8 主路径，而不是单个量化算子。
- Paddle 可以提供 Hopper/Blackwell 两种 scale 语义的清晰参考。
- Paddle 现有 FP8 GEMM 主体是 cuBLASLt blockwise 路线，不是 SonicMoE 最终需要的 grouped/fused 路线。
- 所以后续工作必须是“两条线并行”：
  1. 先基于 baseline 跑通 torch 版 reference path
  2. 再做 Hopper / Blackwell fused path 替换

## 3. Paddle 储备审查结论

详细映射见 `control_plane/fp8/strategy/baseline_trace.md`。这里只保留工程上最重要的结论。

### 3.1 已有强项

- `fused_weighted_swiglu_act_quant_kernel.cu`
  已有 BF16 -> SwiGLU -> optional prob -> FP8 quant 融合前向语义。
- `fused_swiglu_weighted_bwd_kernel.cu`
  已有与上述前向匹配的反向语义，包括 `do1`、`probs_grad`、`o2_s`。
- `fused_act_dequant_kernel.cu`
  已有 FP8 -> BF16 反量化，且同时覆盖 `float32 scale` 和 `e8m0/ue8m0 packed scale`。
- `fp8_quant_blockwise_kernel.cu`
  已有成熟的 `128x128` blockwise quant、`1x128` blockwise quant、pow2 scale、ue8m0 scale 编码口径。
- `fp8_gemm_blockwise_kernel.cu`
  已有基于 cuBLASLt 的 subchannel/blockwise FP8 GEMM 参考实现。

### 3.2 不应直接迁移的部分

- `moe_permute_kernel.cu`
- `moe_unpermute_kernel.cu`
- `fused_transpose_split_quant_kernel.cu`
- `fused_stack_transpose_quant_kernel.cu`

这些实现都很有价值，但更偏 Paddle / DeepEP 的 dispatch 与 dense-blockwise 数据重整思路。它们可以指导优化，不应决定 SonicMoE 的上层接口和主路径结构。

### 3.3 统一认识

后续规划必须建立在这三个事实之上：

1. Paddle 已经解决了很多 FP8 语义问题。
2. Paddle 没有直接给出 SonicMoE 需要的 grouped fused GEMM 最终形态。
3. `tests/reference_layers/standalone_moe_layer` 才是当前最可靠的对齐基线。

## 4. v1 范围

v1 只做最有价值、最能闭环的部分：

- torch 侧增强版 MoE 层
- Hopper 和 Blackwell 两条路径
- forward + backward + training step 闭环
- 与 `tests/reference_layers/standalone_moe_layer` baseline 对齐
- 与 SonicMoE BF16 baseline 交叉对比
- Paddle compatibility 最小接入面

v1 约束：

- FP8 格式优先 `torch.float8_e4m3fn`
- 累加统一 FP32
- Router 仍然保持 FP32
- 先支持 per-tensor 协议；若某 backend 需要 subchannel scale，则在 adapter 层转换
- 输出对外优先 BF16
- 激活优先 `SWIGLU`

v1 不做：

- Router FP8 化
- 全激活覆盖
- learnable scale
- 通用 Paddle 组网抽象
- 没有测试支撑的极致 kernel 调优

## 5. 设计原则

### 5.1 参考优先级

1. SonicMoE 的算法和公开 API 第一优先。
2. `tests/reference_layers/standalone_moe_layer` 的数值和训练行为第二优先。
3. Paddle kernel 的量化、反量化、gate、scale 语义第三优先。
4. compatibility 只负责接入，不反向影响内部实现。

### 5.2 先基线，再融合

顺序固定：

1. reference path 对齐 baseline
2. fused path 替换 reference path
3. 对每个 diff 做归因

### 5.3 scale 统一在协议层

必须单独抽象 scale，不允许不同后端自己定义接口：

- `scale_x`
- `scale_w1`
- `scale_w2`
- scale encoding: `fp32` 或 `e8m0/ue8m0 packed`

### 5.4 黑白分流在 adapter 层

Hopper 和 Blackwell 的差异只允许留在 backend adapter 层，不允许扩散到 `moe.py` 的公开接口。

## 6. Agent 分工

| Agent | 工作内容 | 产出 |
|------|----------|------|
| A0 规划集成 | 维护计划、依赖、阶段门禁 | 本文档、集成清单 |
| A1 API 协议 | dtype、backend、scale、公开 API | 接口草案、配置对象 |
| A2 Hopper | grouped_gemm / moe_config / Hopper kernel 接线 | Hopper FP8 path |
| A3 Blackwell | QuACK / scale 映射 / Blackwell 分流 | Blackwell FP8 path |
| A4 主路径 | functional 入口、autograd、dispatch | torch 主实现 |
| A5 测试 | baseline 对齐、梯度、训练、容差 | 测试矩阵 |
| A6 Baseline / Compat | baseline trace、Paddle 语义映射、compat 设计 | `fp8_baseline_trace.md`、compat 说明 |
| A7 性能交付 | benchmark、profiling、使用说明 | perf 报告、usage 文档 |

执行规则：

- A1 先冻结协议，其他 agent 才能并行。
- A6 持续维护 baseline traceability。
- A5 不得用放宽容差代替归因。
- A7 在功能闭环前不推动大规模重构。

## 7. 实施路线

### Phase 0: 冻结协议

负责人：A0 + A1 + A6

输出：

- torch 主实现边界
- baseline 对齐口径
- Paddle compatibility 最小适配面
- Hopper / Blackwell 的 scale 协议

门禁：

- `tests/reference_layers/standalone_moe_layer`、Paddle、SonicMoE 三者职责边界明确

### Phase 1: Reference Path

负责人：A1 + A4 + A6

目标：

- 在 torch 侧先做可训练的 FP8 reference path
- 以 `tests/reference_layers/standalone_moe_layer` 为主基线
- 不要求这一步就有深融合

门禁：

- forward / backward / training step 跑通
- 与 baseline 对齐

### Phase 2: Hopper / Blackwell 后端接入

负责人：A2 + A3 + A4

目标：

- Hopper: 接通 grouped_gemm 现有 float8 能力
- Blackwell: 接通 QuACK 路线并处理 scale 映射

门禁：

- forward backend 可独立运行
- backend 差异不泄漏到公开 API

### Phase 3: 融合替换

负责人：A2 + A3 + A4 + A5

目标：

- 用 fused path 替换 reference path 的热点子路径
- 建立 diff 归因

门禁：

- 每次替换后都有 baseline diff 报告
- 没有无法解释的精度漂移

### Phase 4: 测试与交付

负责人：A5 + A6 + A7

目标：

- 测试矩阵完整
- compatibility 说明完整
- benchmark 完整

门禁：

- 测试、性能、文档均可复现

## 8. 文件级 ownership

| 文件 | Owner | 说明 |
|------|-------|------|
| `sonicmoe/enums.py` | A1 | precision / backend / scale 抽象 |
| `sonicmoe/moe.py` | A1 / A4 | 公开 API 和 runtime dispatch |
| `sonicmoe/functional/__init__.py` | A4 | 统一入口和 autograd |
| `sonicmoe/functional/forward.py` | A2 / A4 | forward adapter |
| `sonicmoe/functional/backward.py` | A2 / A3 / A4 | backward adapter |
| `sonicmoe/functional/moe_config.py` | A1 / A2 | Hopper config |
| `sonicmoe/functional/grouped_gemm.py` | A2 | Hopper kernel path |
| `sonicmoe/quack_utils/gemm_gated.py` | A3 | Blackwell forward |
| `sonicmoe/quack_utils/gemm_dgated.py` | A3 | Blackwell backward |
| `sonicmoe/quack_utils/gemm_interface.py` | A3 | QuACK adapter |
| `tests/moe_fp8_test.py` | A5 | FP8 专项测试 |
| `tests/moe_test.py` | A5 | BF16 回归 |
| `tests/test_commons.py` | A5 | 容差、比较助手 |
| `tests/reference_layers/standalone_moe_layer/moe_standalone/compat.py` | A6 | reference baseline compatibility 入口 |
| `control_plane/fp8/strategy/baseline_trace.md` | A6 | baseline / Paddle 映射 |
| `control_plane/fp8/reports/fp8_usage.md` | A7 | 使用说明 |

## 9. 测试和验收

### 9.1 测试层级

1. 协议层测试
2. backend adapter 测试
3. baseline 对齐测试
4. E2E 数值和梯度测试
5. benchmark 与稳定性测试

### 9.2 必测项

- forward vs `tests/reference_layers/standalone_moe_layer`
- backward vs `tests/reference_layers/standalone_moe_layer`
- forward / backward vs SonicMoE BF16
- compiled / non-compiled
- bias on / off
- QuACK on / off
- compatibility adapter 输入输出对齐

### 9.3 验收标准

- torch 侧增强版 MoE 层可独立工作
- Hopper 和 Blackwell 至少各有一条可用 FP8 path
- forward / backward / optimizer step 闭环成立
- BF16 路径无回归
- 相对 `tests/reference_layers/standalone_moe_layer` 的主要 case 误差控制在 2% 到 3% 范围
- 所有超出门限的差异都已归因
- compatibility 方案不侵入 SonicMoE 内部实现

## 10. 风险

### 10.1 技术风险

- Paddle 的 blockwise / dense 假设不适用于 SonicMoE grouped path
- Hopper 和 Blackwell scale encoding 不一致
- cuBLASLt baseline 很容易被误当成最终路线
- fused path 可能引入无法解释的精度漂移

### 10.2 组织风险

- 多 agent 同时改 `functional/__init__.py` 和 `backward.py`
- baseline 对齐没人持续维护
- 性能优化过早介入

对应策略：

- ownership 固定
- A6 维护 diff trace
- 性能优化放到 functional 闭环之后

## 11. 第一批任务

1. A1 冻结 dtype / backend / scale 协议。
2. A6 从 `tests/reference_layers/standalone_moe_layer` 抽出最小 reference path 和对齐张量定义。
3. A6 补齐 Paddle kernel 到 SonicMoE 的语义映射表。
4. A4 设计 torch 侧 FP8 reference path 入口。
5. A2 审核 Hopper grouped_gemm 现有 float8 边界。
6. A3 审核 QuACK 路线的 scale 和 backward 能力。
7. A5 建 baseline 对齐测试骨架。
8. A7 建 benchmark 骨架。

这 8 项完成后，再进入大规模编码，返工会少很多。