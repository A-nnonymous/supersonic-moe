# FP8 Engineering Log

本文件不再按“想到哪写到哪”的流水账组织，而是按**主线里程碑 / 实验线 / 经验教训 / 当前缺口**组织。

---

## 0. 读法

如果你只想知道“现在到底到了哪一步”，请只看：

- `## 1. 主线里程碑`
- `## 3. 当前最可信的数据`
- `## 5. 经验与教训`
- `## 6. 当前缺口`

### 0.1 论文 8 个算子 ↔ 工程函数 ↔ 当前 dtype / 支持状态

后续所有进展汇报，默认都按这张表对齐；不再只说“forward/backward 某段”，而要明确是论文里的哪一个算子、工程里对应哪一个函数、baseline 是什么 dtype、现在额外支持什么 dtype、以及 FP8 是在算子内还是算子边界上。

| 论文算子 | 工程函数 / 路径 | 论文变量 | baseline dtype | 当前支持 dtype | 当前支持方式 / 备注 | 当前完成度 | 进度预期 | 工程量预期 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up-proj (`varlen-M grouped GEMM + act`) | `sonicmoe/functional/forward.py::_up_projection_forward`；QuACK 路径主要是 `sonicmoe/quack_utils/gemm_interface.py::gemm_gated` | 输入 `X_e`，输出 pre-activation `H_e` 与 activation 后 `A_e` | 输入激活 `torch.bfloat16`；权重 `torch.bfloat16`；输出 `torch.bfloat16` | 输出后激活 buffer 额外支持 `torch.float8_e4m3fn`；scale 额外支持 `torch.float8_e8m0fnu` | **部分算子内融合**：QuACK `gemm_gated` 可直接写低精度 post-activation buffer；stable 主线仍主要是先算出 `A_e`，再做边界 FP8 quant/dequant | 部分完成，已是当前最成熟 FP8 入口 | 继续做局部 epilogue / saved-state / buffer 生命周期优化；不把 full-chain FP8 放到这条小步主线里 | 中 |
| down-proj (`varlen-M grouped GEMM`) | `sonicmoe/functional/forward.py::_down_projection_forward`；QuACK 路径主要是 `quack.gemm_interface::gemm` | 输入 `A_e`，输出 `Y_e` | 输入激活 `torch.bfloat16`；权重 `torch.bfloat16`；输出 `torch.bfloat16` | 实验线支持 `A_e` / `W2_e` 的 blockscaled FP8（`torch.float8_e4m3fn` + `torch.float8_e8m0fnu` scale） | **未主线化为原生 FP8 mainloop**；当前 stable 仍是恢复成 `bf16 A_e` 后再做 down-proj；experimental blockscaled path 仍依赖 grouped/static-capacity | stable 主线未完成 | 近期待继续做 glue / layout / router 过渡的小优化；原生 flat-varlen FP8 mainloop 默认记入大工程目录 | 大 |
| expert aggregation (forward) | `sonicmoe/functional/forward.py::_router_forward` | `O_t = Σ π_{t,e} S_{t,e} Y_{e,t}` | `Y_e` / `O` 为 `torch.bfloat16`；score 为 `torch.float32` | 暂无额外 FP8 主线支持 | 不是当前 FP8 主战场；保持和 baseline 一致 | 基本未动 | 短期只做 reshape / reduction / metadata glue 层小优化，不单独追求 FP8 化 | 小-中 |
| down-proj act grad | `sonicmoe/functional/backward.py::_down_projection_backward_act`；QuACK 路径会用 `sonicmoe/quack_utils/gemm_interface.py::gemm_dgated` | `dH_e` / `dA_e` / router score grad side products | `dout / H_e / dH_e / dA_e` 以 `torch.bfloat16` 为主；部分 reduce 为 `torch.float32` | 本地 probe 已支持输出 `torch.float8_e4m3fn` 的 router-weighted activation side product | **已局部打 through**：`gemm_dgated` 可产出 FP8 后激活 side product，但未接成 stable backward 主线 | 局部打通，主线未闭环 | 近期待继续做 reduction / scratch / dtype-contract 的局部优化；若要把 FP8 `S·A_e` 接到主线则会撞上更大工程 | 中 |
| down-proj weight grad (`varlen-K grouped GEMM`) | `sonicmoe/functional/backward.py::_down_projection_backward_weight` | `dW2_e` | 输入 `dout` / routed activation 为 `torch.bfloat16`，累加到 `torch.float32` | 暂无 stable FP8 支持 | 当前 blocker 是 mixed-dtype 合同 + Blackwell-native kernel 路线未完成 | 基本未做 | 默认并行排队，不阻塞主线；只接受为后续大工程做最小 probe，不在当前小步迭代里硬推 | 大 |
| up-proj act grad | `sonicmoe/functional/backward.py::_up_projection_backward_act` | `dX_e` / `dH_e` | `torch.bfloat16` 主线 | 暂无 stable FP8 支持 | 仍按论文 baseline 合同走 | 基本未动 | 近期待只做 metadata / saved-state / wrapper overhead 优化，不优先做算子级 FP8 改写 | 中 |
| up-proj weight grad (`varlen-K grouped GEMM`) | `sonicmoe/functional/backward.py::_up_projection_backward_weight` | `dW1_e` | 输入激活 / 梯度 `torch.bfloat16`，累加到 `torch.float32` | 暂无 stable FP8 支持 | 仍按论文 baseline 合同走 | 基本未动 | 近期待只做低风险 bookkeeping / saved-state 优化；真正 FP8 化优先级低于 down-proj 主问题 | 中 |
| backward expert aggregation | `sonicmoe/functional/backward.py::_token_broadcast_backward`；router score 反向为 `_softmax_topk_bwd` | `dX_t = Σ reverse_scatter(dX_e)` 与 router score grad | 激活梯度 `torch.bfloat16`；router score grad path 用 `torch.float32` score 合同 | 暂无额外 FP8 主线支持 | 不是当前 FP8 主战场 | 基本未动 | 只在出现明确 buffer / scatter 热点时做小修；不主动扩 scope | 小 |

### 0.2 论文变量 ↔ 工程变量（之后汇报不要只写简写）

后续汇报中，不应再只写 `z / y1 / y1s / STE` 这类简写，而应写成“论文变量名 + 工程变量名”。最低要求格式：

- `up-projection pre-activation output H_e（工程变量 z）`
- `activated expert intermediate A_e（工程变量 y1）`
- `router-weighted activated expert intermediate S_{t,e}·A_e（工程变量 y1s）`

| 工程变量名 | 论文变量 / 含义 | baseline dtype | 当前支持 dtype | 现在是怎么支持的 |
| --- | --- | --- | --- | --- |
| `x` | layer input `X` | `torch.bfloat16` | `torch.bfloat16` | stable 主线输入；未主线化为 FP8 输入 |
| `router_w` | router weight | `torch.bfloat16` | `torch.bfloat16` | stable 主线 |
| `router_logits` | routing scores `S`（top-k 之前） | `torch.bfloat16` 输入经 `F.linear` 产出；top-k score 计算时会升到 `torch.float32` | `torch.bfloat16` / `torch.float32` 计算合同 | 默认保持 `score` 路径数值稳定；代码里明确不默认降成 bf16 score |
| `topk_scores` | selected routing scores `S_{t,e}` | `torch.float32` | `torch.float32` | stable 主线；forward aggregation 与 backward router grad 都依赖这条合同 |
| `topk_indices` | activated expert ids（对应论文中的 `π` 非零位置） | `torch.int32` | `torch.int32` | metadata 构造与 gather/scatter 索引 |
| `x_gather_idx` | `Gather(X, π_{:,e})` 的 gather map | `torch.int32` | `torch.int32` | varlen/grouped GEMM metadata |
| `expert_frequency_offset` | 每个 expert 的 prefix-sum token count（`T_e` 的 packed 边界） | `torch.int32` | `torch.int32` | varlen-M / varlen-K grouped GEMM metadata |
| `s_scatter_idx` | 从 token-major 到 expert-major 的 scatter map | `torch.int32` | `torch.int32` | metadata |
| `s_reverse_scatter_idx` | 从 expert-major 回 token-major 的 reverse scatter map | `torch.int32` | `torch.int32` | aggregation / reverse aggregation metadata |
| `w1` | up-projection weight `W1_e` | `torch.bfloat16` | `torch.bfloat16` | stable 主线；未主线化为 FP8 weight |
| `z` | up-projection pre-activation output `H_e = X_e W1_e` | `torch.bfloat16` | `torch.bfloat16` | stable 主线；之后可作为 fused FP8 boundary 的输入 |
| `y1` | activated expert intermediate `A_e = act_func(H_e)` | `torch.bfloat16` | `torch.bfloat16`；局部可临时为 `torch.float8_e4m3fn` buffer | QuACK `gemm_gated` 可直接写低精度 post-activation buffer，但 stable 主线仍恢复为 `bf16 A_e` |
| `restored_out` | 边界 quant/dequant 后恢复的 `A_e` buffer | `torch.bfloat16` | `torch.bfloat16` | `apply_preact_activation_fp8_protocol_cutely_fused(..., restored_out=...)` |
| `w2` | down-projection weight `W2_e` | `torch.bfloat16` | stable 主线 `torch.bfloat16`；实验线 `torch.float8_e4m3fn` + `torch.float8_e8m0fnu` scale | experimental blockscaled weight path 只在 grouped/static-capacity 路线上成立 |
| `y2` | per-expert down-projection output `Y_e = A_e W2_e` | `torch.bfloat16` | `torch.bfloat16` | stable 主线 |
| `o` | final aggregated output `O` | `torch.bfloat16` | `torch.bfloat16` | stable 主线 |
| `dout` | output gradient `dO` | `torch.bfloat16` | `torch.bfloat16` | stable backward 输入 |
| `dz` | gradient of `H_e` / `z` | `torch.bfloat16` | `torch.bfloat16` | stable backward |
| `dx_expanded` | expert-major expanded input gradient before reverse aggregation | `torch.bfloat16` | `torch.bfloat16` | up-proj act grad 输出 |
| `dx_reduced` | token-major reduced input gradient after reverse aggregation | `torch.bfloat16` | `torch.bfloat16` | `_token_broadcast_backward` 输出 |
| `y1s` | router-weighted activated expert intermediate `S_{t,e} · A_e`，用于 `dW2_e` | baseline 主线 `torch.bfloat16` | probe 已支持 `torch.float8_e4m3fn` | 本地 `gemm_dgated` 已能产出 FP8 `y1s`；但后续 `down-proj weight grad` 还不支持稳定消费 |
| `dw1 / dw2` | `dW1_e / dW2_e` | `torch.float32` accumulation target | `torch.float32` | stable 主线 |
| `STE` | straight-through estimator（不是论文变量，而是边界训练技巧） | N/A | N/A | 当前 stable 主线默认不依赖它；只在特定边界适配逻辑里作为可选训练技巧存在 |

### 0.3 当前“FP8 到底做到哪”统一口径

如果后续要回答“当前进度主要是在论文 8 个算子外做 FP8 外挂，还是往算子内融合 FP8 quant/dequant”，统一口径如下：

- **大多数当前进展仍属于算子边界 / 算子之间的 FP8 contract 优化**
  - 代表路径：`up-proj -> fused fp8 boundary -> down-proj`
- **只有少数位置已经进入算子内融合**
  - 主要是 QuACK `gemm_gated` / `gemm_dgated` 这类带 activation epilogue 的路径
  - 它们已经能直接处理或产出低精度 `postact`
- **down-proj forward 与大部分 backward 核心算子，尚未成为原生 FP8 主线**

### 0.4 当前 dtype / support 统一口径

- stable 主线权重：
  - `router_w / w1 / w2`：`torch.bfloat16`
- stable 主线激活：
  - `x / z / y1 / y2 / dout / dz / dx`：主要是 `torch.bfloat16`
- router / top-k：
  - `topk_scores`：主线按 `torch.float32`
  - indices / offsets：`torch.int32`
- stable FP8 activation protocol：
  - activation data：`torch.float8_e4m3fn`
  - scales：`torch.float8_e8m0fnu`
  - stable granularity：`1x128`
- 实验线 blockscaled down-proj：
  - `w2` 可量化成 `e4m3 + e8m0(1x32)`
  - 但这条 grouped/static-capacity 路线仍不是 stable 主线

### 0.5 大工程目录（默认并行排队，不阻塞当前小步优化）

这部分不是“今天顺手补一补”能完成的工作。后续如果某项优化直接落到这些边界上，应默认把它记到这里，单独并行推进，而不是把主线迭代拖进大内核工程。

| 大工程名 | 范围 | 当前状态 | 为什么是大工程 | 进度预期 | 工程量预期 |
| --- | --- | --- | --- | --- | --- |
| Project 1 umbrella：native FP8 GEMM 能力集（不是单点问题） | umbrella，仅用于统筹 `native fp8 up-proj`、`native fp8 down-proj`、`fp8 weight storage`、`precision validation` 4 个子工程 | 已拆分定义，当前绕过 | 它同时覆盖输入激活、权重存储、输出/累加与验收，不应再被当成一个单点工程 | 后续按 1.1/1.2/1.3/1.4 子工程顺序推进 | 特大 |
| Blackwell-native mixed-dtype down-proj weight-grad | 让 `down-proj weight grad` 稳定消费 `bf16 dO + fp8 (S·A)` | 已立项，当前绕过 | 需要 mixed-dtype / scaled weight-grad 合同，还要补齐 Blackwell-native kernel 路线 | 维持并行排队；等更核心 forward 主线稳定后再集中攻坚 | 大 |
| rank-flexible blockscaled varlen down-proj | 让 flat varlen-M activation + blockscaled scale factor 不再依赖 grouped/static-capacity | 已立项，当前绕过 | 当前 blocker 在上游 blockscaled + varlen 的 rank/layout 设计，不是 SonicMoE glue 层小修能解决 | 先持续确认 blocker 边界；若无上游改动，不在当前仓库里强推 | 特大 |

### 0.6 大工程输入输出契约冻结（bf16 SonicMoE 作为金标准）

| 大工程名 | 冻结输入契约 | 冻结输出契约 | bf16 金标准来源 | 验收单测 |
| --- | --- | --- | --- | --- |
| 1.1 native FP8 up-proj | token-major `x`、routing metadata、`W1_e`；允许未来内部变成 FP8 input/postact，但 token/expert 语义不可变 | 最终 `O`、`expert_frequency` 与 bf16 主线一致 | `enable_quack_gemm(True)` + `moe(..., kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=None)` | `tests/fp8_large_project_contract_test.py::test_native_fp8_upproj_bf16_gold_contract` |
| 1.2 native FP8 down-proj | `A_e` / routing metadata / `W2_e`；允许未来内部 mainloop 改成 FP8 输入消费 | 最终 `O`、`expert_frequency` 与 bf16 主线一致 | 同上 | `tests/fp8_large_project_contract_test.py::test_native_fp8_downproj_bf16_gold_contract` |
| 1.3 FP8 weight storage | `W1_e / W2_e` 可改成 FP8 存储 + scale，但外部路由和 shape 合同不可变 | 最终 `O`、`expert_frequency` 与 bf16 主线一致 | 同上 | `tests/fp8_large_project_contract_test.py::test_fp8_weight_storage_bf16_gold_contract` |
| Blackwell-native mixed-dtype down-proj weight-grad | `dO`、router-weighted activated expert intermediate `S_{t,e}·A_e`、routing metadata；未来允许 `S_{t,e}·A_e` 变成 fp8 + scale 合同 | `dW2_e` 的 shape、dtype 语义、有限值、与 bf16 主线的一致性 | `enable_quack_gemm(True)` + bf16 `moe(...).backward(...)` 后的 `moe.c_proj.weight.grad` | `tests/fp8_large_project_contract_test.py::test_mixed_dtype_downproj_weight_grad_bf16_gold_contract` |
| rank-flexible blockscaled varlen down-proj | flat varlen `A_e`、`expert_frequency_offset`、`expert_indices` / scatter metadata；未来允许 blockscaled scale-factor layout 变化，但输入 token/expert 语义不可变 | token-major `O` 与 `expert_frequency`；未来内部可 blockscaled / rank-lift，但对外输出 contract 不变 | `moe_general_routing_inputs(..., bf16)` | `tests/fp8_large_project_contract_test.py::test_rank_flexible_varlen_downproj_bf16_gold_contract` |

### 0.7 大工程详细规划（冻结后按阶段推进）

#### 0.7.1 Project 1 umbrella：native FP8 GEMM 能力集（必须拆分，不是单点问题）

拆分结论：

- 它至少包含 4 个彼此独立、依赖关系也不同的单点问题：
  1. `1.1 native fp8 up-proj input/output contract`
  2. `1.2 fp8 weight storage & quantization contract`
  3. `1.3 native fp8 down-proj input/output/accumulator contract`
  4. `1.4 precision / numerics / performance validation`

需要触达的论文算子：

- `up-proj`
- `down-proj`
- （可选 side product）`down-proj act grad`

不属于本工程 umbrella 的算子：

- `down-proj weight grad`（那是 Project 2）
- `expert aggregation`
- `router / top-k`

推荐顺序：

1. **1.1 native fp8 up-proj**
   - 先稳定 `gemm_gated` 的 FP8 input/postact 合同
   - 工程位置：
     - `sonicmoe/quack_utils/gemm_interface.py::gemm_gated`
     - `sonicmoe/quack_utils/gemm_gated.py`
     - `sonicmoe/functional/__init__.py`
2. **1.2 fp8 weight storage**
   - 独立于 1.1，可并行推进
   - 先解决 `W1_e / W2_e` 的持久化与 scale contract
   - 工程位置：
     - `sonicmoe/moe.py`
     - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
     - `sonicmoe/quack_utils/gemm_interface.py`
3. **1.3 native fp8 down-proj**
   - 依赖 1.1 的 activation contract 稳定后再推进
   - 工程位置：
     - `quack.gemm_interface::gemm`
     - `sonicmoe/functional/__init__.py`
     - `sonicmoe/functional/forward.py::_down_projection_forward`
4. **1.4 验收与性能收敛**
   - correctness / bf16-gold / benchmark gate 全部通过后再谈默认接入

明确不做：

- 不再把 umbrella 本身当作“一个点改完就结束”的工程
- 不把“局部 epilogue 写一个 fp8 buffer”误报成工程 1 完成

#### 0.7.2 Blackwell-native mixed-dtype down-proj weight-grad

阶段拆分：

需要触达的论文算子：

- `down-proj act grad`
- `down-proj weight grad`

推荐顺序：

1. `2.1 helper / wrapper 打通`
2. `2.2 Blackwell mixed-dtype kernel`
3. `2.3 _down_projection_backward_weight` 消费 FP8 `S·A_e`
4. `2.4 主线 backward 接入 + acceptance`

明确不做：

- 不在 helper 层打补丁后就宣布问题解决；必须以 `dW2_e` 对齐 bf16 gold 为准

#### 0.7.3 rank-flexible blockscaled varlen down-proj

阶段拆分：

需要触达的论文算子：

- `down-proj`

推荐顺序：

1. `3.1 reverse-scatter / metadata generalization`
2. `3.2 rank-flexible scale packing`
3. `3.3 varlen-aware blockscaled mainloop`
4. `3.4 runtime toggle / acceptance / benchmark`

明确不做：

- 不把 grouped/static-capacity 的实验路径包装一下就当作 flat varlen 主线完成

### 0.8 baseline / opt 验收入口（开发完成可直接挂载）

bf16 baseline correctness gate：

```bash
make test-large-project-baseline
```

bf16 baseline performance gate：

```bash
make bench-large-project-baseline FP8_LARGE_PROJECT_BENCH_SHAPE=8192,4096,1024,128,8
```

未来 operator-opt correctness gate：

```bash
make test-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_NATIVE_FP8_UPPROJ=1"
make test-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_NATIVE_FP8_DOWNPROJ=1"
make test-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_FP8_WEIGHT_STORAGE=1"
make test-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_MIXED_DTYPE_DOWNPROJ_DW2=1"
make test-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_RANKFLEX_VARLEN_DOWNPROJ=1"
```

未来 operator-opt performance gate：

```bash
make bench-large-project-opt FP8_OPERATOR_OPTS="SONIC_MOE_OPT_NATIVE_FP8_UPPROJ=1" FP8_LARGE_PROJECT_BENCH_SHAPE=8192,4096,1024,128,8
```

规则：

1. baseline bf16 路径始终是金标准，不允许被 opt 默认覆盖
2. 每个新算子必须先挂 operator-opt env，再接 correctness gate
3. correctness 过关后，才允许进入 performance gate

---

## 1. 主线里程碑

### 1.1 Blackwell FP8 protocol 骨架已落地

关键文件：

- `sonicmoe/functional/fp8_protocol.py`
- `sonicmoe/functional/fp8_quant.py`
- `sonicmoe/functional/fp8_reference.py`

当前协议范围：

- activation dtype：`e4m3`
- scale encoding：`e8m0`
- stable granularity：`1x128`
- runtime target：Blackwell + QuACK

意义：

- FP8 工作已经从“概念讨论”进入“协议明确、接口固定、可持续演化”的阶段

### 1.2 preact fused FP8 boundary 已落地

关键文件：

- `sonicmoe/functional/fp8_cutely_fused.py`

关键路径：

- `z -> fused_weighted_swiglu_act_quant_best -> fused_act_dequant_best`

意义：

- 当前 stable 主线已经不是简单 reference quant/dequant 拼装
- 已有真实 fused boundary

### 1.3 stable 主线已复用 `y1`，去掉语义冗余路径

关键文件：

- `sonicmoe/functional/__init__.py`

已落地优化：

- fused preact dequant 支持 `restored_out=...`
- stable path 直接复用 QuACK `gemm_gated(...)` 产出的 `y1`
- preact boundary 包在 `torch.no_grad()` 下
- 默认去掉 `STE`

为什么这是对的：

- `_UpProjection.forward(...)` 返回的 `y1` 已 `mark_non_differentiable`
- `_DownProjection.backward(...)` 也不是沿前向 `y1` 回梯度
- 所以这条 `STE` 在主线里是语义冗余，而不是“训练必要组件”

### 1.4 benchmark 指标基础设施已补齐

关键文件：

- `benchmarks/moe-cute.py`

已落地能力：

- `--report_fp8_metrics`
- `--report_stage_memory`
- `--report_fp8_analysis`
- backward RMSE：
  - `dx_rmse`
  - `dw1_rmse`
  - `dw2_rmse`
  - `drouter_w_rmse`
- GPU peak 污染修复：
  - metrics snapshot 改为 CPU 保存

意义：

- 之后所有“fp8 比 bf16 快还是慢”的判断终于有了较可靠的数据面

### 1.5 预分配输出 buffer 合同已铺开

关键文件：

- `sonicmoe/functional/fp8_quant.py`
- `sonicmoe/functional/fp8_cutely_fused.py`

已落地接口：

- `round_scale_to_e8m0(..., out=...)`
- `quantize_activation_blockwise(..., out=..., scale_out=...)`
- `dequantize_activation_blockwise(..., out=...)`
- `apply_*_fp8_protocol_cutely_fused(..., scale_out=...)`

意义：

- 这是后续完全 cudagraph-compatible FP8 路线的接口基础

### 1.6 2026-03-25：QuACK inference fastpath 第 1 轮落地

关键文件：

- `sonicmoe/functional/__init__.py`
- `sonicmoe/quack_utils/gemm_interface.py`
- `tests/fp8_protocol_test.py`

本轮实际改动：

- 给 `moe_TC_softmax_topk_layer(...)` 增加了 QuACK inference fastpath：
  - `is_inference_mode_enabled=True` 且走 QuACK 时，不再复用训练态的 `_UpProjection.apply -> _DownProjection.apply` autograd 包装
  - 改成 forward-only 直接调用：
    - router/topk metadata
    - `gemm_gated(...)`
    - 可选 `apply_preact_activation_fp8_protocol_cutely_fused(...)`
    - `gemm(...)`
    - `_router_forward(...)`
- inference fastpath 现在在 forward-only 路径里显式 `torch.no_grad()`，并在 down-proj 前尽早释放 `z/y1`
- `gather_A` 场景下给 `gemm_gated` autotune 增加了更严格的 invalid-config 剪枝，避免 `cluster_n != 1` 的错误配置被拿来试
- `_DownProjection.forward()` 不再为 backward 保存实际上没有被用到的 `selected_experts`

本轮原本想做、但被代码现实挡住的点：

- 原计划是让 inference fastpath 在 **不需要 fp8 boundary** 时直接 `store_preact=False`，从而连 `z` 都不落地
- 但实测发现当前 QuACK `gather_A` up-proj 合同仍要求 `preact_out/D` 存在；否则会在 scheduler / kernel 路径上直接断言失败
- 这意味着：
  - **今天这轮不能把 inference 的 `z` 彻底删掉**
  - 但我们已经把这个 blocker 明确定位成 **现有 QuACK gather-A 合同限制**，不是 SonicMoE 上层逻辑问题

为什么这轮仍然值得保留：

- 当前高频 inference 路径以前虽然不 backward，但仍沿用了训练态自定义 autograd Function 包装
- 这会把不必要的 graph / ctx / saved-tensor 生命周期带进来
- 新 fastpath 至少做到了两件正确的事：
  - inference 不再走训练态 autograd 包装
  - `z` 不再跨到 down-proj / router 之后才自然释放，而是 forward 中段就能尽早释放

精度结果：

- bf16 inference fastpath vs. 旧路径：
  - `max_abs_o = 0.0`
  - `max_abs_logits = 0.0`
  - `expert_frequency` 完全一致
- fp8 inference fastpath vs. 旧路径：
  - `max_abs_o = 0.0`
  - `max_abs_logits = 0.0`
  - `expert_frequency` 完全一致

也就是说：

- **本轮没有引入任何可见数值回退**
- 至少在当前验证 shape 上，forward 结果是逐元素一致的

性能 / 显存结果（注意：共享机 8 卡持续高占用，以下 timing 噪声较大；更可信的是同窗口前后对比与显存趋势）：

- 测量环境：
  - `CUDA_VISIBLE_DEVICES=1`
  - shared GPU 高占用背景
  - shape：`4096,4096,1024,128,8`
  - mode：`is_inference_mode=True`

- 改前：
  - bf16：`mean 3.422 ms`, `min 2.748 ms`, `peak 3594.75 MiB`
  - fp8：`mean 3.442 ms`, `min 1.945 ms`, `peak 3595.75 MiB`

- 改后：
  - bf16：`mean 3.486 ms`, `min 3.347 ms`, `peak 3440.80 MiB`
  - fp8：`mean 3.346 ms`, `min 2.458 ms`, `peak 3595.75 MiB`

正确解释：

- **最稳定的收益是 bf16 inference peak memory 下降了约 `153.95 MiB`**
- fp8 inference mean time 在这组噪声环境里有小幅改善（`3.442 -> 3.346 ms`），但幅度还不够大，不能夸大
- bf16 inference mean time 没有拿到稳定优势，说明：
  - “去 autograd 包装 + 提前释放临时张量”是对的
  - 但如果不能把 `z` 真正从 up-proj inference 合同里拿掉，收益仍然有限

验证结果：

- `python -m py_compile sonicmoe/functional/__init__.py sonicmoe/quack_utils/gemm_interface.py tests/fp8_protocol_test.py`
- `CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py`
- 结果：`23 passed`

下一步规划：

1. 继续沿着 **真正去掉 inference `z`** 的方向推进，但目标应该转为：
   - 改 QuACK gather-A up-proj 合同
   - 或者给它补一个允许 `preact_out=None` 的合法 scheduler / kernel 路径
2. 如果 QuACK 侧今天还动不了，就把主精力重新拉回更值钱的方向：
   - `varlen fp8 postact + scales`
   - varlen-preserving fp8 down-proj mainloop
3. 保持当前结论清晰：
   - 本轮不是 endgame
   - 但它把 inference 路径里一层不必要的训练态包装拿掉了，并且把下一个真正 blocker 定位清楚了

### 1.7 2026-03-25：QuACK gather-A inference 允许 `D=None`，真正去掉非-FP8 边界场景下的 `z`

关键文件：

- `sonicmoe/quack_utils/gemm_gated.py`
- `sonicmoe/functional/__init__.py`
- `tests/fp8_protocol_test.py`

本轮实际改动：

- 在本地 `GemmGatedMixin` 里覆写了 scheduler 参数构造逻辑：
  - 对 `varlen_m + gather_A + mD is None` 的场景，不再沿用上游 `assert mD is not None or not self.gather_A`
  - 改为直接使用 `VarlenMTileSchedulerArguments(total_m=varlen_args.mAIdx.shape[0])`
- 基于这个本地调度修复，把 QuACK inference fastpath 恢复成：
  - `store_preact=False` when `fp8_protocol is None` or `upproj_epilogue_precision != fp8`
- 新增测试显式锁住：
  - 非 FP8 boundary 的 QuACK inference fastpath 必须 `store_preact=False`
  - fastpath 输出仍然 `requires_grad=False`
  - 全量 `tests/fp8_protocol_test.py` 继续通过

为什么这轮值得做：

- 第 1 轮已经证明：
  - inference fastpath 本身是对的
  - 但没有真正去掉 `z` 时，收益有限
- 这轮把真正的本地 blocker 拆掉了：
  - 不是算法不允许
  - 而是上游 scheduler 对 `gather_A + D=None` 做了保守限制
- 现在至少对 **不需要 fp8 boundary 的 inference**，`z` 已经可以不落地

精度结果：

- bf16 inference fastpath vs. 标准路径：
  - `max_abs_o = 0.0`
  - `max_abs_logits = 0.0`
  - `expert_frequency` 完全一致
- fp8 inference fastpath vs. 标准路径：
  - `max_abs_o = 0.0`
  - `max_abs_logits = 0.0`
  - `expert_frequency` 完全一致

结论：

- **本轮仍然没有引入任何数值回退**
- gather-A `D=None` 只是把不必要的 preact materialization 删掉，不改变最终算子语义

性能 / 显存结果（同样处于共享满卡环境，主要看趋势，不夸大 timing）：

- 对比基线：上一轮 fastpath 已落地、但 `store_preact` 仍被迫开启
- 测量环境：
  - `CUDA_VISIBLE_DEVICES=1`
  - shared GPU 高占用背景
  - shape：`4096,4096,1024,128,8`
  - mode：`is_inference_mode=True`

- 本轮改后：
  - bf16：`mean 3.335 ms`, `min 3.259 ms`, `peak 3440.80 MiB`
  - fp8：`mean 3.292 ms`, `min 2.438 ms`, `peak 3595.75 MiB`

正确解释：

- bf16 inference mean 有小幅改善（相对上一轮 `3.486 -> 3.335 ms`），说明去掉 `z` 对高频路径确实有帮助
- 但 bf16 inference peak 没有继续下降，说明：
  - 当前测量口径下的峰值已经不完全由 `z` 决定
  - 或者共享机噪声把这一层收益淹掉了
- fp8 inference peak 没变也符合预期：
  - 当 fp8 boundary 仍存在时，`z` 仍然是必须输入

验证结果：

- `python -m py_compile sonicmoe/functional/__init__.py sonicmoe/quack_utils/gemm_gated.py sonicmoe/quack_utils/gemm_interface.py tests/fp8_protocol_test.py`
- `CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py`
- 结果：`23 passed`

本轮顺手完成的高 ROI 探针：

- 尝试直接把现有 `blockscaled_fp8_gemm` 从 grouped/static-capacity 改成 **flat varlen-M**
- 结果没有直接打通，但 blocker 非常清楚：
  - `GemmDefaultSm100(sf_vec_size=32)` 当前 blockscaled A 路径要求 scale-factor layout 按 rank-3 形态构造
  - flat varlen-M 的 rank-2 activation 在 `tile_atom_to_shape_SF(...)` 处直接因 rank 假设失败
  - 这说明当前真正的下一步不是继续优化 grouped，而是优先评估 **rank-lifted A/SF layout 包装** 是否能把 flat varlen-M 接进来

下一步规划：

1. 继续追 `flat varlen-M blockscaled fp8 down-proj`
   - 先尝试 rank-lift / layout-promotion
   - 目标是复用现有 `GemmDefaultSm100(sf_vec_size=32)`，而不是回到 grouped/static-capacity
2. 如果 rank-lift 证伪，再决定是否：
   - 在本仓库本地派生一个更贴近 SonicMoE 合同的 blockscaled wrapper
   - 或把这部分视为 `quack/cutlass` 外部依赖层 blocker
3. 保持优先级判断不变：
   - inference 小修不是 endgame
   - **真正的 ROI 仍在 direct / varlen-preserving fp8 down-proj mainloop**

### 1.8 2026-03-25：本地 `gemm_dgated` 已打通 FP8 `y1s/postact_out`，backward blocker 前移到更外层

关键文件：

- `sonicmoe/quack_utils/gemm_dgated.py`
- `tests/fp8_protocol_test.py`

本轮实际改动：

- 给本地 `gemm_dgated` 包装层补上了 runtime FP8 dtype 处理：
  - 补齐 `torch.float8_e4m3fn -> cutlass.Float8E4M3FN`
  - 对 runtime FP8 tensor 改成按底层 `uint8` storage 走 `from_dlpack(...)`，再显式回填 `cute` element type
  - 不再依赖上游 `quack.gemm_wrapper_utils` 里尚未覆盖 float8 的 dtype 提取路径
- 新增 Blackwell-only 回归测试，锁住：
  - `gemm_dgated(..., postact_dtype=torch.float8_e4m3fn)` 可以真实运行
  - 输出合同为：
    - `dx.dtype == torch.bfloat16`
    - `y1s.dtype == torch.float8_e4m3fn`
    - `ds.dtype == torch.float32`

为什么这轮值得做：

- 之前只知道“想把 backward 的 `y1s` 压成 FP8 时会失败”，但失败边界并不清楚
- 这轮把问题拆清楚了：
  - **`gemm_dgated` wrapper / kernel 本身不是 blocker**
  - 现在前半段已经能在 Blackwell 上真实产出 FP8 `y1s`
  - 真正的后续阻塞点已经前移到：
    - `convert_torch_tensor_to_cute_tensor(...)` 对 float8 的 DLPack 支持
    - 以及现有 down-proj weight-grad kernel 的更外层架构边界

精度结果：

- 新增回归测试里，`y1s.float()` 全 finite，无 NaN
- 本轮没有修改主线训练 / 推理语义，只是把 wrapper 能力边界打开并用测试锁住

性能 / 显存结果：

- 本轮属于 backward 边界打通，不是主线端到端优化落地
- **没有宣称新的端到端加速或显存下降**
- 但它的重要意义是：
  - backward transient 压缩路线前半段已证实可行
  - 后续优化可以更集中，不再误判 `gemm_dgated` 自身为根因

验证结果：

- `python -m py_compile sonicmoe/quack_utils/gemm_dgated.py tests/fp8_protocol_test.py`
- `CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/fp8_protocol_test.py -k dgated_can_emit_fp8_postact_buffer`
- 最小实跑 probe：
  - `gemm_dgated(..., postact_dtype=torch.float8_e4m3fn)` => 成功，输出 `y1s.dtype == torch.float8_e4m3fn`

下一步规划：

1. 评估是否值得把 `convert_torch_tensor_to_cute_tensor(...)` 扩成 runtime FP8-friendly，让 down-proj weight-grad 至少能真正接到 FP8 `y1s`
2. 同步确认现有 `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd` 是否本质上仍是 Hopper-only；如果是，不在这条内核上做过量 patch
3. 保持 ROI 判断不变：
   - 这轮是把 backward FP8 `y1s` 从“猜测可行”推进到“前半段已证实可行”
   - 但真正决定大 shape 30%+ 收益的，仍更可能是 varlen-preserving forward/down-proj 主线收敛

### 1.9 2026-03-25：通用 runtime-FP8 `cute` conversion 已打通，down-proj weight-grad blocker 收敛为“大工程级”边界

关键文件：

- `sonicmoe/utils.py`
- `tests/fp8_protocol_test.py`

本轮实际改动：

- 扩展 `convert_torch_tensor_to_cute_tensor(...)`，让它支持 runtime `torch.float8_e4m3fn`
  - FP8 tensor 先按底层 `uint8` storage 走 `from_dlpack(...)`
  - 再显式回填 `cute` element type 为 `Float8E4M3FN`
  - 保持原有 `stream` 包装和 compact-shape 标注逻辑不变
- 新增回归测试，锁住通用 conversion 现在能接受 runtime FP8 tensor

为什么这轮值得做：

- 上一轮已经证明 `gemm_dgated` 能真实产出 FP8 `y1s`
- 这轮把更外层的 tensor-conversion blocker 也拆掉了
- 这样再去探 down-proj weight-grad，报错就更“干净”，不会再把 helper 层和 kernel 层混在一起

精度结果：

- 本轮没有改动主线数值路径，只扩展了 tensor conversion 能力
- 新增 conversion 回归仅验证合同正确，不引入新的数值近似

性能 / 显存结果：

- 本轮仍属于协议 / kernel 边界收敛
- **没有新的端到端性能或显存收益可宣称**

本轮确认后的真实 blocker：

- 当 `_down_projection_backward_weight(...)` 直接接收 FP8 `y1s` 时：
  - float8 的 tensor conversion 已不再是问题
  - 真实报错变成：`Type mismatch: BFloat16 != Float8E4M3FN`
- 同时，独立 bf16 probe 仍暴露：
  - 现有 `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd` 带有 `sm_90a` 假设
  - 在当前 Blackwell 环境下会报 `expects arch to be Arch.sm_90a, but got Arch.sm_100a`

结论：

- 这说明当前 down-proj weight-grad 的剩余问题已经不是“小补丁级”：
  1. 需要 mixed-dtype / scaled weight-grad 合同，能消费 `bf16 dout + fp8 y1s`
  2. 同时还需要更 Blackwell-friendly 的 weight-grad kernel 路线
- 这已经符合“如果必须做、绕不过去，可以单列项目”的标准
- 我会把它视为一个 **大型工程候选**：
  - `Blackwell-native mixed-dtype down-proj weight-grad`

验证结果：

- `python -m py_compile sonicmoe/utils.py tests/fp8_protocol_test.py`
- conversion probe：
  - `convert_torch_tensor_to_cute_tensor(fp8_tensor, ...)` => 成功，element type 为 `Float8E4M3FN`
- weight-grad probe：
  - bf16 `y1s` => 暴露 `sm_90a` 架构边界
  - fp8 `y1s` => 暴露 `BFloat16 != Float8E4M3FN` mixed-dtype 边界

下一步规划：

1. 将 `Blackwell-native mixed-dtype down-proj weight-grad` 作为日志中的大型工程候选保留
2. 日常自主迭代继续优先做 ROI 更高、可小步验证的项：
   - varlen-preserving fp8 down-proj 主线
   - 以及能在不重写 weight-grad 大内核的前提下减少 backward transient 的局部优化
3. 如果后续必须触及这块，就按单独项目方式推进，而不是在当前主线上零碎打补丁

### 1.10 2026-03-25：`flat varlen-M blockscaled` 本地猴补丁 prototype 再次证实是上游 rank/layout 边界，不适合在主线里碎修

本轮动作：

- 没有直接改仓库主代码
- 用一次性 Python prototype 临时猴补丁了两处上游边界，想验证是否存在“低成本本地 wrapper”空间：
  - `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF`
  - `quack.varlen_utils.VarlenManager.offset_batch_A`
- 目标是让：
  - flat 2D varlen-M activation
  - flat 2D packed scales
  - 直接走 `GemmDefaultSm100(sf_vec_size=32)` + `varlen_m`
  - 不再先 pack 成 grouped/static-capacity

prototype 结果：

- 没有打通
- 而且报错非常集中，继续指向同一类根因：
  - `tile_atom_to_shape_SF(...)` 仍要求 target shape rank 与 order `(2, 1, 3)` 对齐
  - 即使临时 rank-lift，后续 `gSFA_mkl = cute.local_tile(...)` 仍会因 coordinate/profile 不匹配失败
  - `offset_batch_A(...)` 这类按 rank-2 `(m, k)` 写死的 varlen 偏移逻辑也仍然是局部障碍

为什么这轮重要：

- 这不是再一次“猜测可能不行”
- 而是已经用更接近最终目标的猴补丁 prototype 证明：
  - **问题不在 SonicMoE 上层 glue**
  - **问题也不只是一个 helper 函数签名**
  - 真正卡点在上游 blockscaled + varlen 的 rank/layout 设计本身

精度结果：

- 本轮没有落地主线算子，因此无新的精度变化

性能 / 显存结果：

- 本轮仍属于 feasibility probe
- **没有新的端到端性能或显存收益**

结论：

- 继续在当前主线里零碎 patch 这条 `flat varlen-M blockscaled` 路线，ROI 已明显下降
- 这条线如果必须推进，也已经符合单列项目的标准
- 我把它记录成第二个 **大型工程候选**：
  - `rank-flexible blockscaled varlen down-proj`

验证记录：

- 一次性 prototype 在当前 Blackwell 环境下编译阶段失败，典型报错包括：
  - `expects target shape and order operands have same rank`
  - `unable to compute crd2idx`
  - `failed to construct a valid coordinate`

下一步规划：

1. 不在主线里继续硬凿这条上游 rank/layout 边界
2. 继续优先做不依赖大型内核工程的高 ROI 项
3. 如后续必须做：
   - 将 `rank-flexible blockscaled varlen down-proj` 作为单独项目推进
   - 目标应是上游级别的 blockscaled+varlen rank contract 设计，而不是 SonicMoE 侧零碎猴补丁

### 1.11 2026-03-25：QuACK inference fastpath 进一步去掉 router/top-k 的训练态 wrapper；workspace cache 试验未保留

关键文件：

- `sonicmoe/functional/__init__.py`
- `tests/fp8_protocol_test.py`

本轮实际改动：

- 在 QuACK inference fastpath 中，不再通过 `TC_Softmax_Topk_Router_Function.apply(...)` 走训练态 autograd 包装来拿 `topk_scores/topk_indices`
- 改成 inference-only 直接预分配：
  - `topk_scores`
  - `topk_indices`
  - 然后直接调用 `_softmax_topk_fwd(...)`
- 新增回归测试锁住：
  - inference fastpath 必须直接走 `_softmax_topk_fwd(...)`
  - 不再调用 `TC_Softmax_Topk_Router_Function.apply(...)`

本轮同时做过、但**没有保留**的试验：

- 我尝试过给 inference fastpath 增加内部 workspace cache，想复用：
  - router metadata buffer
  - upproj preact / postact 临时 buffer
  - downproj 输出临时 buffer
- 但在当前 shared-GPU steady-state probe 下，没有拿到稳定正收益，因此**撤回了这部分缓存逻辑**，不让主线带着一个无证据收益的复杂度点继续前进

为什么这轮值得保留：

- 这轮虽然不是大性能突破，但它让 inference fastpath 的语义更一致：
  - up-proj / down-proj 已经不走训练态 autograd wrapper
  - router/top-k 现在也不再走训练态 wrapper
- 这使得当前 QuACK inference fastpath 更接近“真正的 forward-only 路径”，后续再做更细的 inference 优化时，边界会更清晰

精度结果：

- `tests/fp8_protocol_test.py` 中：
  - inference no-boundary fastpath 与标准路径仍数值一致
  - inference fp8-boundary fastpath 与标准路径仍数值一致
- 本轮没有引入新的数值回退

性能 / 显存结果（shared GPU，高噪声；只做保守解读）：

- 用临时“旧版 apply-topk inference path”脚本对比当前 landed direct-topk cleanup：
  - old apply-topk：`mean 3.347 ms`, `min 3.314 ms`, `peak 3434.77 MiB`
  - new direct-topk：`mean 3.370 ms`, `min 3.009 ms`, `peak 3434.77 MiB`
- 结论：
  - **没有观察到稳定、可宣称的端到端性能收益**
  - peak memory 也没有变化
  - 因此这轮应被视为 **语义/合同清理**，而不是性能里程碑

workspace cache 试验结果（未保留，仅记录）：

- cache off：`mean 3.314 ms`, `min 2.721 ms`, `peak 3498.75 MiB`
- cache on：`mean 3.383 ms`, `min 3.306 ms`, `peak 3498.75 MiB`
- 结论：
  - 当前实现下 cache 路线没有拿到稳定收益
  - 不值得把它留在主线里增加复杂度

验证结果：

- `python -m py_compile sonicmoe/functional/__init__.py tests/fp8_protocol_test.py`
- `CUDA_VISIBLE_DEVICES=1 USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py`
- 结果：`25 passed`

下一步规划：

1. 把本轮结论定性为“清理完成”，不要误判成性能突破
2. 继续寻找真正有 ROI 的局部优化，优先级仍然应避开两类大型工程候选：
   - `Blackwell-native mixed-dtype down-proj weight-grad`
   - `rank-flexible blockscaled varlen down-proj`
3. 后续若做 inference 优化，优先考虑：
   - 能减少真实大 buffer 生命周期
   - 或能减少 kernel/contract 层的冗余 materialization
   - 而不是单纯为了减少 `torch.empty(...)` 次数而引入缓存复杂度

### 1.12 2026-03-25：16 机队列空闲资源扫描 / 远端 launch 流程已工程化

关键文件：

- `tools/cluster_idle_launch.py`
- `Makefile`
- `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md`

本轮实际改动：

- 新增 `tools/cluster_idle_launch.py`
  - `scan`：从当前队列环境变量 `PADDLE_TRAINERS / TRAINER_INSTANCES` 解析 16 个 node
  - 通过 `mpirun + nvidia-smi` 扫描每个 node 的 8 张 GPU 空闲情况
  - 按 `idle_gpus` 排序输出
  - 同时保留：
    - `host`：人类可读主机名
    - `launch_host`：实际用于 `mpirun --host` 的 IP
- 新增 Makefile 入口：
  - `make cluster-scan`
  - `make cluster-launch CLUSTER_COMMAND="..."`
- 已把完整使用说明追加到 `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md`

为什么这轮值得做：

- 当前本机 8 卡经常全部高占用，继续只盯本机做 benchmark 会显著放慢迭代
- 这个队列本身就是 16 机、gpfs 共享存储可复用代码和结果
- 因此“先扫空闲 node，再在空闲 node 上 launch 实验”应该成为默认工作流，而不是临时手工操作

当前验证结果：

- 本机 local GPU probe 仍显示 8/8 busy
- 16 机全量 scan 时，当前时间窗发现至少 2 台全空闲：
  - `tjzj-inf-sci-k8s-bzz2-0275`
  - `tjzj-inf-sci-k8s-bzz2-0342`
- 脚本验证通过：
  - `python tools/cluster_idle_launch.py scan`
  - `python tools/cluster_idle_launch.py launch --gpu-count 1 --command 'echo hello-from-cluster-launch' --dry-run`
  - `make cluster-scan`

注意事项：

- launch 当前按 `launch_host(IP)` 走 `mpirun --host`，避免部分 hostname 解析失败
- 如果当前 16 机里没有满足条件的空闲 GPU，脚本会直接失败退出，不会去争抢忙卡
- 这轮是实验调度工具化，不是内核性能优化；它的价值在于让后续 benchmark / ablation 能更稳定地落到空闲机器上执行

下一步规划：

1. 后续所有 benchmark / regression / ablation，优先走这套 `scan -> launch` 流程
2. 如果需要，可以继续扩展：
   - 指定 host 白名单
   - 指定结果日志目录
   - 自动把 stdout/stderr 重定向到 `reports/` 或 `output/`
3. 保持当前主线优化与“大工程目录”并行推进，不再被本机 8 卡高占用卡住

### 1.13 2026-03-25：QuACK 训练路径 `_UpProjection` 不再保存无用 `s_scatter_idx`，并把 0 段表格升级为“现状 + 进度预期 + 工程量预期”

关键文件：

- `sonicmoe/functional/__init__.py`
- `reports/fp8_upgrade/ENGINEERING_LOG.md`

这轮改动对应论文算子：

- `up-proj` 反向相关路径（主要是 `up-proj act grad / up-proj weight grad` 的工程 glue 层）
- 类型：**算子内工程化优化**，不是新的 FP8 数学路径

本轮实际改动：

- `_UpProjection.forward()` 现在会把 forward 时实际采用的 backend（`use_quack_gemm`）写入 `ctx`
- `_UpProjection.backward()` 改为严格沿用 forward 时记录下来的 backend，而不是再次查询运行时全局开关
- 当 forward 走 QuACK 路径时，不再把 `_up_projection_backward_act()` 才需要的 `s_scatter_idx` 保存进 autograd `saved_tensors`
  - 因为 QuACK backward 路径根本不会消费这份 index map
  - 之前这份张量只是被无条件保存，但在 QuACK backward 分支里是死数据
- 工程日志第 0 段的算子表 / 大工程表新增：
  - `当前完成度`
  - `进度预期`
  - `工程量预期`
  之后可以直接从第 0 段看出哪些工作没做、近期是否应该推进、属于小步优化还是大工程

为什么这轮值得做：

- 这是一个低风险、确定正确的训练侧内存 / 工程稳定性修正：
  - **内存侧**：少保存一份 `TK` 长度的 `torch.int32` metadata
  - **稳定性侧**：backward 不再依赖“执行 backward 时全局开关是否还和 forward 一致”这种隐式假设
- 对于大 shape，单层理论 saved-state 减少量约为：
  - `4 * T * K` bytes
  - 例如 `T=32768, K=8` 时约节省 `1.0 MiB / layer`

精度 / 性能 / 显存影响：

- 精度变化：**无**
  - 没改算子数学，只改 autograd 保存面与 backward 分支选择方式
- 理论显存变化：**训练态小幅下降**
  - 下降来源是 QuACK up-proj backward 不再保留无用 `s_scatter_idx`
- 理论性能变化：**预计中性到轻微正向**
  - 这轮主要收益不是 kernel 算力，而是减小 autograd 保存面和降低上下文错配风险

验证：

- `python -m py_compile sonicmoe/functional/__init__.py`
- `python tools/cluster_idle_launch.py launch --gpu-count 1 --command "source .../envs/xfer/bin/activate && python -m pytest -q tests/fp8_protocol_test.py"`
  - 空闲 node 远端回归已通过（`exit code 0`）

下一步规划：

1. 若远端回归通过，把这轮优化标记为稳定 landed
2. 继续沿着 backward-side 的局部 ROI 做：
   - `down-proj act grad` 的 `ds_partial` reduction / scratch 优化
   - 或其他不跨入“大工程目录”的 metadata / buffer 生命周期优化
3. 继续避免把主线拖入：
   - full-chain FP8 GEMM
   - Blackwell-native mixed-dtype `down-proj weight grad`
   - rank-flexible blockscaled varlen down-proj

### 1.14 2026-03-25：收掉剩余一批非大工程高 ROI 小改动，并冻结 3 个大工程的契约 / 验收入口

关键文件：

- `sonicmoe/functional/backward.py`
- `sonicmoe/functional/triton_kernels/__init__.py`
- `tests/fp8_large_project_contract_test.py`
- `reports/fp8_upgrade/ENGINEERING_LOG.md`

这轮对应的论文算子：

- `down-proj act grad`
- `expert aggregation (forward)` 的 metadata 统计 glue 层
- 以及 3 个大工程的**输入输出契约冻结**（不是实现完成）

本轮实际改动：

- `down-proj act grad`
  - 把 `ds_partial.sum(...)+copy_` 改成直接 `torch.sum(..., out=ds)`
  - 把 `new_ds_partial.sum(...)+copy_` 改成直接 `torch.sum(..., out=ds)`
  - 对 `N==1` 的分支，去掉了无意义的 `.to(dtype=ds.dtype)` 临时张量
- `router metadata`
  - 把 `expert_frequency.copy_(col_partial_sum_trans.sum(...))` 改成直接 `torch.sum(..., out=expert_frequency)`
- 大工程侧：
  - 新增 `tests/fp8_large_project_contract_test.py`
  - 把以下 3 个大工程的 bf16-gold acceptance 入口固定下来：
    - `full-chain FP8 GEMM`
    - `Blackwell-native mixed-dtype down-proj weight-grad`
    - `rank-flexible blockscaled varlen down-proj`
  - 在第 0 段新增：
    - `0.6 大工程输入输出契约冻结`
    - `0.7 大工程详细规划`

为什么这轮值得做：

- 这些改动都在**高频训练路径**里：
  - `down-proj act grad` 的 `ds` reduction
  - router metadata 的 expert histogram 汇总
- 收益不来自大 kernel 改写，而是来自：
  - 减少中间临时张量
  - 缩短 scratch 生命周期
  - 固定未来大工程不会漂移的 contract / acceptance 入口

精度 / 性能 / 显存变化：

- 精度变化：**无**
  - 归约数学未改，仍以 bf16 SonicMoE 路径为金标准
- 理论性能变化：**小幅正向**
  - 主要来自少一次 `sum -> 临时向量 -> copy_` 的中间写回
- 理论显存变化：**小幅下降**
  - 主要来自少一份 reduction 临时输出和更短的 scratch 生命周期

验证：

- `python -m py_compile sonicmoe/functional/__init__.py sonicmoe/functional/backward.py sonicmoe/functional/triton_kernels/__init__.py tests/fp8_protocol_test.py tests/fp8_large_project_contract_test.py`
- `python tools/cluster_idle_launch.py launch --gpu-count 1 --command "source .../envs/xfer/bin/activate && python -m pytest -q tests/fp8_protocol_test.py tests/fp8_large_project_contract_test.py"`
  - 空闲 node 远端回归已通过（`exit code 0`）

结论：

- 截至本轮，**大工程之外、明确高 ROI 且低风险的局部 buffer / saved-state / reduction 优化，已经基本收尾**
- 后续如果继续出现 ROI 明显的小改动，应满足至少一条：
  - 不跨入 `0.5` 大工程目录
  - 不破坏 bf16 baseline 合同
  - 能明确减少 buffer / scratch / wrapper 开销
- 否则默认转入大工程并行推进，不再以“小改动”名义混入主线

---

## 2. 实验线与结论

### 2.1 blockscaled down-proj：做到了哪里

关键文件：

- `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`

已做过的优化：

- 融合 `pack+quant`
- 不再物化完整 `grouped_a`
- 不再把 grouped output 再 unpack 成 flat `y2`
- grouped reverse scatter index 直接复用 `selected_experts`

结论：

- 前半段已经不是主矛盾
- 当前真正的墙是：
  - `grouped_out`
  - static capacity
  - grouped layout 到 router 聚合的过渡层

### 2.2 dummy postact buffer：能跑，但不是主线

开关：

- `SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER=1`

想法：

- 让 up-proj 先给出 fp8 dummy postact
- 再只靠 `z` 重建 `bf16 y1`

结论：

- 已验证可跑
- 但真实性能更差
- 默认关闭

### 2.3 backward runtime-fp8 `y1s`：前半段已打通，真正 blocker 已收敛到更外层

尝试位置：

- `_DownProjection.backward()`

当前更准确的结论：

- 本地 `gemm_dgated(..., postact_dtype=torch.float8_e4m3fn)` 已可真实运行
- 也就是说：
  - backward 的 `y1s` 压成 FP8 并不是在 `gemm_dgated` 这里就被证伪
  - 前半段现在已经是可行路线

当前真实 blocker：

- 现有 `_down_projection_backward_weight(...)` 若直接喂 FP8 `y1s`，首先会卡在：
  - `convert_torch_tensor_to_cute_tensor(...)` 对 float8 直接走 `from_dlpack(...)`
  - 当前 helper 路径会报：`float8 types are not supported by dlpack`
- 独立最小 probe 还显示：
  - `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd` 仍带有 `sm_90a` 假设
  - 说明 weight-grad 这条线和 Blackwell / QuACK 主线的边界还没有完全对齐

结论：

- backward fp8 `y1s` 不再是“已证伪”
- 但它也绝不是只补一个 dtype 参数就能端到端打通
- 现在更准确的说法应该是：
  - **前半段可行**
  - **后半段还需要 FP8-friendly tensor conversion + 更 Blackwell-native 的 weight-grad 合同**

### 2.4 static fp8 weight benchmark 合同：已做对

关键文件：

- `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- `sonicmoe/moe.py`
- `benchmarks/moe-cute.py`

已落地能力：

- protocol-aware `w2` cache key
- `prefetch_blockscaled_w2_fp8(...)`
- `clear_blockscaled_fp8_weight_cache()`
- `MoE.prefetch_fp8_weights(...)`
- `MoE.clear_fp8_weight_cache()`

结论：

- benchmark 已经做到：
  - bf16 权重由 seed 决定
  - fp8 weight 在计时前静态量化
  - 首轮在线量化不再污染 timing
- 但 blockscaled 路线仍明显慢于 bf16

这意味着：

- 主矛盾不是在线 weight quant
- 主矛盾仍是 grouped/static-capacity 过渡层

---

## 3. 当前最可信的数据

### 3.1 stable 主线，shape `8192,4096,1024,128,8`（2026-03-25 同机空闲卡复测）

复测环境：

- host `10.51.203.82`（空闲卡）
- `USE_QUACK_GEMM=1`
- bf16 与 fp8 在**同一台空闲机器**上重测，避免共享卡噪声

当前最可信结果：

- bf16：
  - inf / train fwd / e2e / bwd `1.930 / 1.944 / 5.676 / 3.746 ms`
- stable fp8：
  - output RMSE `0.01074111`
  - loss RMSE `0.00000025`
  - metrics probe：`bf16_peak_mib=7690.38`, `fp8_peak_mib=7572.50`
  - stagewise final peak：`7572.50 -> 7702.50 MiB`
  - inf / train fwd / e2e / bwd `1.947 / 2.144 / 5.871 / 3.924 ms`

相对 bf16 baseline：

- inference：`-0.88%`
- train fwd：`-10.29%`
- e2e：`-3.43%`
- backward：`-4.75%`

当前正确解释：

- **这版 stable fp8 在干净复测下没有性能收益**
- 理论上它仍可能靠更小的 activation payload 获益，但前提是：
  - `saved_traffic + overlap > quant/dequant + scales + extra_buffer + sync`
- 当前这条主线属于：
  - `up-proj -> fp8 boundary -> bf16 restored A_e -> down-proj`
  - 即**算子边界 fp8**，不是原生 full-chain fp8 算子
- 因此当前没有赢的根因是：
  1. 核心算子主体并没有变成原生 fp8 mainloop
  2. 额外引入了 quant / dequant / scale 处理
  3. 还会物化压缩态与恢复后的 `bf16 A_e`
  4. backward 侧仍有明显 transient overhead（stagewise `+130 MiB`）

收益来源若未来出现，应该主要来自：

- 更小的 `A_e` 边界 payload 带来的 HBM/L2 traffic 下降
- 更小 working-set 带来的 cache / residency 改善
- 若后续打通原生 fp8 算子，还可能出现 kernel 内部融合收益

但**当前这版收益没有出现**，因此不能再把“加了 quant/dequant 仍可能赢”当作已被当前主线实证支持的结论。

### 3.1.1 stable 主线，shape `32768,4096,1024,128,8`（并发复测状态）

- 已按并行规则把 bf16 / fp8 分别发到两台空闲 host：
  - bf16：`10.51.203.76`
  - fp8：`10.51.203.82`
- 两边都被 `SIGKILL (exit code 137)` 打断，未拿到可用 timing

当前解释：

- 这不是“fp8 比 bf16 快/慢”的证据
- 只说明当前这组超大 shape 在本轮空闲节点条件下没有稳定完成
- 后续如果还要复测 `32768`，应单独走：
  - 更轻量的 timing 口径
  - 或先确认该 host 的可用显存 / 作业限制

### 3.2 stable 主线，shape `4096,4096,1024,128,8`

- bf16：
  - peak `7049.88 MiB`
  - inf / train fwd / e2e / bwd `1.141 / 1.210 / 3.437 / 2.296 ms`
- stable fp8：
  - peak `6931.00 MiB`
  - output RMSE `0.01073675`
  - loss RMSE `0.00000021`
  - inf / train fwd / e2e / bwd `1.125 / 1.697 / 3.733 / 2.608 ms`

正确解释：

- 显存优势稳定存在
- 训练前向 / e2e 还没有彻底超过 bf16

### 3.3 blockscaled static weight benchmark，shape `1024,512,512,32,4`

- bf16：
  - peak `380.25 MiB`
  - inf / train fwd / e2e / bwd `0.184 / 1.187 / 3.184 / 3.000 ms`
- static fp8 `w2`：
  - peak `144.78 MiB`
  - output RMSE `0.00131902`
  - loss RMSE `0.00000013`
  - inf / train fwd / e2e / bwd `0.275 / 2.683 / 5.066 / 4.790 ms`

正确解释：

- 省显存很大
- 但变慢也非常明显
- 这不是因为在线量化，而是因为路径本身不对

---

## 4. 理论账本

### 4.1 `4096,4096,1024,128,8`

- 当前稳定主线：
  - `stable_fp8_saved_payload_mib=31.00`
- 若打通 varlen-friendly direct FP8 mainloop：
  - `direct_fp8_boundary_saved_mib=97.75`
- 若进一步把 `w1/w2` 做成 FP8 存储：
  - `aggressive_weight_saved_mib=1524.00`
- 合并理论上限：
  - `aggressive_total_saved_mib=1555.75`

### 4.2 结论

- 现在最值得追的大头非常明确：
  1. 去掉 `bf16` 回退边界
  2. 做真正的 weight fp8 存储与消费
- 小于这个量级的收益，不值得为之破坏 SonicMoE 的核心内存合同

---

## 5. 经验与教训

### 5.1 一定要继续坚持的做法

1. **先问“是否符合 SonicMoE 的内存合同”**
   - 再问“是不是更激进的 fp8”

2. **所有性能结论都必须带 bf16 baseline**

3. **所有大 shape 结论都尽量同时给出：**
   - metrics cold run
   - stagewise raw probe
   - 理论账本

4. **先修 benchmark 合同，再讨论性能输赢**

5. **把“为什么不是主线”写清楚**
   - 比“这条路可以跑”更重要

### 5.2 明确的反模式

1. 把 grouped/static-capacity 当作默认演进方向
2. 把 toy case 提升误判成真实主线进展
3. 忽略 SonicMoE 的复用与调度，只盯 dtype
4. 把一份 raw peak 当成最终唯一真相
5. 因为某条实验数值正确，就误判它接近 endgame

---

## 6. 当前缺口

### 6.1 真正缺的算子 / 合同

1. `varlen FP8 postact + scales`
2. `gather-A preserving down-proj fp8 mainloop`
3. `backward mixed-dtype / scaled GEMM`
4. `persistent static FP8 weight storage`
5. `fully cudagraph-compatible FP8 path`

### 6.2 当前最需要避免的错误

- 不要继续把工程精力重投到 grouped/static-capacity 主线化
- 除非目标明确是做“对照实验”，而不是做真正交付路径

### 6.3 Sprint: Native FP8 Tensor Core — 三路并行（进行中）

**目标**: 全流程涉及到 GEMM 的，内部都调用 FP8 tensor core，使用 FP32 main-loop 累加。

**关键技术决策**:
- Blackwell tcgen05.mma 原生支持 1×32 UE8M0 scale — 不需要在 matmul 内部 descale
- gemm_gated / gemm_dgated (sonicmoe fork) 已支持 fp8 A/B 输入
- 标准 quack.gemm 不支持 fp8 — `torch2cute_dtype_map` 缺少 fp8 映射
- gemm_dgated 约束: Out/PreAct 必须 bf16 (element_size==2 断言)，A/B 可以 fp8

**三路并行 Agent**:

| Agent | 范围 | GPU | 状态 |
|-------|------|-----|------|
| P1 fp8-varlen-kernel | blockscaled_fp8_gemm_varlen | GPU 1 | 进行中 |
| P2 fp8-forward-pipeline | up-proj fp8 tensor cores + FP8 boundary | GPU 2 | 进行中 |
| P3 fp8-backward-benchmark | backward dgated fp8 + benchmark fp8 inputs | GPU 3 | 进行中 |

---

## 7. 当前工作树里最重要的非文档改动

- `benchmarks/moe-cute.py`
  - backward RMSE
  - CPU snapshot fix
  - static weight prefetch benchmark plumbing
- `sonicmoe/moe.py`
  - fp8 weight prefetch / clear API
- `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - protocol-aware weight cache
  - prefetch / clear
- `sonicmoe/functional/__init__.py`
  - 更细的 backward stage-memory probe

备注：

- 我尝试过把 benchmark 进一步改成“只暴露一个 `--precision` 参数”
- 但那一版没有完成
- 我已把 benchmark 恢复到**可编译状态**
- 下一个 agent 不要从那份坏 patch 继续改

---

## 8. 当前推荐命令

### 快速回归

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

### 稳定主线 metrics / theory / stage memory

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py \
  --thiek 4096,4096,1024,128,8 \
  --dtype BFloat16 \
  --activation swiglu \
  --skip_test \
  --fp8_protocol blackwell \
  --report_fp8_metrics \
  --report_fp8_analysis \
  --report_stage_memory
```

### static fp8 weight benchmark（仅做实验，不要误判为主线）

```bash
USE_QUACK_GEMM=1 \
SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ=1 \
SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY=256 \
SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION=fp8 \
python benchmarks/moe-cute.py \
  --thiek 1024,512,512,32,4 \
  --dtype BFloat16 \
  --activation swiglu \
  --skip_test \
  --fp8_protocol blackwell \
  --report_fp8_metrics \
  --prefetch_fp8_weights
```
