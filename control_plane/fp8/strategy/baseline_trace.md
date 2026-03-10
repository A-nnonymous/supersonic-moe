# FP8 Baseline Trace

## 1. 文档目的

这份文档只做一件事：把 Paddle / `tests/reference_layers/standalone_moe_layer` / SonicMoE 三者之间的语义映射写清楚，避免后续 agent 在不同基线上推进。

基线优先级：

1. SonicMoE 决定上层算法和公开 API
2. `tests/reference_layers/standalone_moe_layer` 决定数值和训练可用性基线

当前仓库内 reference baseline 根路径：`tests/reference_layers/standalone_moe_layer`

关键兼容 shim：`tests/reference_layers/standalone_moe_layer/moe_standalone/compat.py`
3. Paddle kernel 决定局部 FP8 语义参考

## 2. 已审查的 Paddle 文件

| Paddle 文件 | 已确认能力 | 对 SonicMoE 的价值 | 结论 |
|------------|------------|--------------------|------|
| `fused_weighted_swiglu_act_quant_kernel.cu` | BF16 -> SwiGLU -> optional prob -> FP8 quant | 明确前向 gate + quant 语义 | 直接参考语义 |
| `fused_swiglu_weighted_bwd_kernel.cu` | gate backward + probs_grad + o2_s | 明确 backward 输出关系 | 直接参考语义 |
| `fused_act_dequant_kernel.cu` | FP8 -> BF16 dequant，支持 `float32` 和 `ue8m0/e8m0` scale | 明确 Hopper / Blackwell scale decoding | 直接参考语义 |
| `fp8_quant_blockwise_kernel.cu` | `128x128` blockwise quant，pow2 scale，ue8m0 scale | 明确 quant 块粒度与 scale 编码 | 直接参考量化口径 |
| `fp8_gemm_blockwise_kernel.cu` | 基于 cuBLASLt 的 subchannel / blockwise FP8 GEMM | 提供 baseline GEMM 路线 | 只做 baseline，不做最终 fused 模板 |
| `moe_permute_kernel.cu` | DeepEP 风格 dispatch / permute | 可参考 dispatch 思路 | 不直接迁移 |
| `moe_unpermute_kernel.cu` | token combine / unzip -> zip | 可参考 combine 语义 | 不直接迁移 |
| `fused_transpose_split_quant_kernel.cu` | 非 grouped 专家重整 + quant | 可参考数据重整逻辑 | 大概率不直接复用 |
| `fused_stack_transpose_quant_kernel.cu` | stack + transpose + quant | 可参考 dense-blockwise quant 数据排布 | 大概率不直接复用 |

## 3. 关键技术储备结论

### 3.1 已经有的能力

- 已有成熟的 gate + quant 融合前向语义
- 已有匹配的 gate backward 语义
- 已有 Hopper / Blackwell 两类 scale 编码口径
- 已有 blockwise quant/dequant 能力
- 已有基于 cuBLASLt 的 FP8 GEMM baseline
- 已有一套普通 FP8-MoE 训练 / 推理 baseline 组网

### 3.2 还没有的能力

- 面向 SonicMoE grouped path 的最终 fused FP8 GEMM 主路线
- 与 SonicMoE router / metadata 完全一致的 FP8 dispatch 主路线
- 统一的 torch-side scale 协议和 compatibility adapter

## 4. SonicMoE 中的落位方式

| 能力 | SonicMoE 中的落位 | 处理策略 |
|------|-------------------|----------|
| gate + quant 前向语义 | `functional/forward.py` / backend adapter | 保持语义，按 torch API 重写 |
| gate backward 语义 | `functional/backward.py` / autograd | 保持输出关系，允许分步实现 |
| dequant 语义 | backend adapter / testing reference | 统一 scale decoding 抽象 |
| blockwise quant 语义 | scale 协议 / reference path | 先做 reference path，再决定 fused 程度 |
| cuBLASLt FP8 GEMM | baseline 或 fallback | 不当作最终优化目标 |
| permute / unpermute 思路 | router / metadata 优化参考 | 只借鉴，不迁移结构 |
| transpose / split / stack quant | 非 grouped 数据重整参考 | 优先不引入 SonicMoE 主路径 |

## 5. baseline 对齐规则

后续所有实现都要按下面的顺序对齐：

1. 先对齐 `tests/reference_layers/standalone_moe_layer` 的 reference path
2. 再对齐 SonicMoE BF16 path
3. 最后比较 fused path 与 reference path 的偏差

如果 fused path 与 baseline 有差异，必须记录：

- 差异出现在 forward 还是 backward
- 差异出现在 quant、gate、gemm、dispatch 还是 combine
- 是否与 scale 编码退化有关
- 是否会阻塞上线

## 6. compatibility 最小适配面

Paddle compatibility 只允许暴露下面这些概念：

- 输入张量
- router/topk 结果或等价 metadata
- expert 权重
- scale 张量或 scale 编码
- 输出张量

compatibility 不应该承担：

- 决定 SonicMoE 内部 kernel 分层
- 决定公开 API
- 把 Paddle 的 dispatch 结构强行映射到 SonicMoE

## 7. 直接指导工程的结论

1. 先做 torch reference path，不要一上来就做完全 fused。
2. 把 Paddle 的量化、反量化、gate 语义抽成协议，而不是直接搬 kernel 结构。
3. Hopper / Blackwell 的差异重点在 scale encoding 和 backend adapter，不在上层 API。
4. `tests/reference_layers/standalone_moe_layer` 是主要金标准；Paddle kernel 是局部语义金标准。
5. cuBLASLt FP8 GEMM 可以支撑 baseline，但不能替代 SonicMoE 需要的 grouped fused 路线。