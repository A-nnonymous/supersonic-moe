# Session 60 Engineering Log — Gate↔MLP Gradient Chain & Precision

## Key Lessons

### 67. torch-proxy backward 梯度张量与 native Paddle autograd 节点不兼容

**现象**: 当 gate 不使用 `no_grad()` 时，backward segfault 于 `TopkGradNode`/`CastGradNode`。

**根因**: SonicMoE 的 `_UpProjection`/`_DownProjection` 是 `torch.autograd.Function`，通过 Paddle torch-proxy 执行。backward 返回的梯度张量（ds、dx）的内部元数据（`paddle::Tensor::type()`）对 native Paddle autograd 节点不可见。

**触发条件**: ds 回传路径上存在**在 MLP forward/backward 之间新创建的** native Paddle autograd 节点：
- `_convert_routing_map_and_probs` 中的 `paddle.topk()` → TopkGradNode
- `_prepare_sonic_inputs` 中的 `.cast("float32")` → CastGradNode
- `paddle.amp.decorate` 注入的 Cast 节点 → CastGradNode

**修复**: 消除 ds 回传路径上的所有中间 Paddle autograd 节点：
1. 直接使用 gate 产出的 `topk_weights` [T,K] 和 `topk_indices` [T,K]
2. 路由元数据用 `general_routing_router_metadata`（纯整数运算，无 autograd）
3. 直接调用 `_UpProjection.apply()` + `_DownProjection.apply()`，`is_varlen_K=False`
4. 不使用 `paddle.amp.decorate`

**原理**: gate 自身 forward 时创建的 TopkGradNode 没问题（张量是正常的 native Paddle 张量）。但在 gate→MLP 之间插入的 TopkGradNode/CastGradNode 会在 backward 时接收 torch-proxy 产出的梯度张量 → segfault。

### 68. 单卡组网必须与 PaddleFleet 保持一致

PaddleFleet 的 `_forward_single_card_grouped_gemm_moe` (sonic-moe 分支):
- `is_varlen_K=False`, `K=K` (固定 topk)
- `topk_scores` [T, K] 直接传入 `_DownProjection.apply()`
- 路由元数据由 `fused_expert_parallel_TC_topk_router_metadata` (CUDA op) 构造
- Gate 无 `no_grad()`，ds 正常回传

之前测试使用的 `moe_general_routing_inputs` 走的是 `is_varlen_K=True` 路径，需要将 topk_scores 展平为 [TK] 并通过 `_prepare_sonic_inputs` 做 padding + boolean indexing，这些操作创建了有害的 autograd 节点。

### 69. `paddle.amp.decorate` 与 FP8 不兼容

AMP O2 在层级注入 Cast 节点。FP8 backward 中的 `dz.untyped_storage().resize_(0)` 释放张量存储后，AMP 的 CastGradNode 访问悬空指针。生产 PaddleFleet 也不对 MoE 层使用 AMP decorate。

### 70. ds 在 torch-proxy `.apply()` 下的叶节点可观测性

`_DownProjection.backward` 内部正确计算了 ds（经组网验证：gate 的 TopkGradNode 收到了 ds 回传）。但当 `topk_scores` 是独立的 `paddle.to_tensor` 叶节点时，`.grad` 属性不被填充。这是 Paddle torch-proxy 的梯度累积机制限制，不影响实际训练（gate 连接时 ds 正常回传）。

### 71. Path A vs Path B 精度对比

| 指标 | RRMSE (%) | Cosine | 说明 |
|---|---|---|---|
| output | 0.0004 | 1.000000 | 几乎 bit-exact |
| dx | 0.0027 | 1.000000 | 几乎 bit-exact |
| dw1 | 0.1656 | 0.999896 | 来自 is_varlen_K 路径差异 |
| dw2 | 0.1656 | 0.999928 | 来自 is_varlen_K 路径差异 |

### 72. `_topk_scores_needs_grad` 必须默认 True（Paddle torch-proxy bug）

**根因**：`_DownProjection.forward` 用 `not topk_scores.stop_gradient` 判断是否计算 ds。但 Paddle torch-proxy 的 `.apply()` 在传入 forward 前将所有 tensor input 的 `stop_gradient` 设为 True（模仿 PyTorch `Function.apply()` detach 行为），且不提供 `ctx.needs_input_grad` 接口。

**后果**：`ctx._topk_scores_needs_grad = False`，ds 不被计算，gate weight 梯度为零，router 无法训练。

**修复**：`ctx._topk_scores_needs_grad = True`（始终计算 ds）。在 MoE 训练中 topk_scores 始终需要梯度。若 caller 不需要 ds，autograd engine 自动丢弃。

**验证**：修复后 gate_w.grad.norm = 0.0558（之前为 0.0），x.grad.norm 从 0.0188 增至 0.0225（+19%，ds 通过 gate 贡献了额外梯度到 hidden_states）。
