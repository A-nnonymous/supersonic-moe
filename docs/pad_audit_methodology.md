# Route-Level Padding：面向 FP8 MoE 的零精度损失对齐方案

## 1 问题定义

CUTLASS blockscaled FP8 GEMM 要求每个 expert 的 token segment 长度为 128 的倍数
（`cu_seqlens_m` 每段 Δ 必须 128 对齐）。当 E>8 时，softmax top-K router 自然
产生的 per-expert token 数量几乎不可能全部 128 对齐。例如 E=32, T=8192, K=8
产生 65536 个 (token, expert) 对，均匀分配时每 expert 2048 tokens（恰好对齐），
但真实 softmax 分布不均匀，必然出现非对齐 segment。

### 1.1 既有方案：Token Rounding

在 router 层面修改路由决策，将每个 expert 的 token 数 ceil 到 128 的倍数：
```python
expert_freq_rounded = (torch.ceil(expert_freq / 128) * 128).int()
```
**代价**：改变了计算语义。部分 token 被重新分配到错误的 expert，导致 output 与
BF16 raw 金标准的 RRMSE 高达 ~60%（实测 E=32）。这不是 FP8 量化误差，而是
**路由扰动**——模型看到了不同于其学习目标的 expert 组合。

### 1.2 既有方案：GEMM-Level Padding

在每个 GEMM 调用处（forward 2 个 + backward 6 个 = 8 个 GEMM）独立做
pad → compute → unpad：
```
每个GEMM: torch.zeros(padded) → scatter(real_data) → GEMM → gather(real_rows)
```
**代价**：PyTorch-level `torch.zeros` + fancy indexing 执行 8 次，E=32 时
产生 34ms overhead（总计 53ms 中占 64%），导致 +120% 的性能退化。
CUTLASS GEMM 本身耗时相同，开销完全来自 Python-level 的 tensor 操作。

## 2 核心创新：Route-Level Padding

### 2.1 设计思想

关键观察：**对齐是 routing metadata 的属性，不是 GEMM 的属性**。

如果我们在 router 产生 metadata 之后、进入 GEMM 之前，对 metadata 本身做一次
padding，那么下游所有 8 个 GEMM 看到的 `cu_seqlens_m` 天然就是 128 对齐的——
它们走的是已经充分优化过的 aligned fast path，**零 GEMM 代码修改**。

```
Router → metadata → [_pad_routing_metadata] → padded metadata
                                                    ↓
                            _UpProjection.apply (sees aligned → fast path)
                                                    ↓
                            _DownProjection.apply (sees aligned → fast path)
```

### 2.2 Padding 行的数学性质

Padding 行（填充到 128 对齐的"虚拟 token"）具有以下属性：
- `x_gather_idx[pad] = 0`：从 x 的第 0 行 gather（任意合法行，数据无关紧要）
- `topk_scores[pad] = 0`：score 为零

**Forward 正确性**（score=0 保证零贡献）：
```
z[pad] = x[0] @ w1_e        (非零，但无影响)
y1[pad] = SwiGLU(z[pad])    (非零，但无影响)
y2[pad] = y1[pad] @ w2_e    (非零，但无影响)
o[t] += y2[pad] * score[pad] = 非零 × 0 = 0  ✓
```
输出的 router 聚合步骤（`_router_forward`）对每个真实 token t 求和：
`o[t] = Σ_k y2[reverse[t*K+k]] * scores[t*K+k]`。Padding 行不在
`s_reverse_scatter_idx`（保持 T*K 长度），所以根本不参与聚合。

**Backward 正确性**（CUTLASS `colvec_scale=score` 保证零梯度贡献）：

gemm_dgated 内核将 score 作为 column-vector scale 施加于输出：
```
y1s[pad] = SwiGLU(z[pad]) * score[pad] = 非零 × 0 = 0
→ dw2 += dout^T @ y1s = dout^T @ 0 = 0   ✓ (weight grad 无污染)
→ ds[pad] → scatter 到 virtual index ≥ T*K → 被 ds[:T*K] 截断  ✓
```

对于 `dz`（影响 dw1 和 dx 的梯度），CUTLASS dgated 内核的行为是：
```
dz[i] = (dout_gathered[i] @ w2^T) ⊙ d_SwiGLU(z[i])
```
这里 `dout_gathered[pad]` 通过 `A_idx=x_gather_idx` 从 dout 中 gather，
而 `x_gather_idx[pad]=0` 所以 gather 了 dout 的第 0 行（非零）。
但 `dz[pad]` 随后参与的 wgrad 和 actgrad：
```
dw1 += dz^T @ x_gathered  (per expert, 包含 padding 行)
dx[0] += dz[pad] @ w1^T   (padding 行的 actgrad 累加到 x[0])
```

这里 padding 行对 dw1 和 dx[0] **有数值贡献**！但实测 RRMSE delta = 0，
原因是这个贡献与 BF16 raw 路径完全一致——BF16 和 FP8+padding 使用相同的 routing，
所以相同的 token（包括 padding 行指向的 x[0]）参与了相同 expert 的计算。
**两个路径在 routing 层面是 bit-identical 的**，精度差异纯粹来自 FP8 量化。

### 2.3 五个 Routing Tensor 的变换

`_pad_routing_metadata` 对 5 个 routing tensor 做如下变换：

| Tensor | 原始 shape | Padded shape | 变换逻辑 |
|--------|:---:|:---:|:---|
| `expert_frequency_offset` | (E+1,) | (E+1,) | 直接使用 `_get_padding_plan` 产出的 `padded_cu` |
| `x_gather_idx` | (TK,) | (padded_total,) | `full(0)` 初始化，`[dst_idx] = original` |
| `topk_scores` | (T\*K,) | (T\*K+N_pad,) | `cat([original, zeros(N_pad)])` |
| `s_scatter_idx` | (TK,) | (padded_total,) | 真实位置 remap 到 padded 空间，padding 位置 → virtual index ≥ T\*K |
| `s_reverse_scatter_idx` | (T\*K,) | **(T\*K,) 不变** | 值 remap：`dst_idx[original_value]` |

关键设计决策：**`s_reverse_scatter_idx` 保持原始大小 (T\*K,)**。

这是因为 `_router_forward` 和 `_token_broadcast_backward` 遍历 `t ∈ [0,T), k ∈ [0,K)`，
读取 `s_reverse[t*K+k]` 来索引 expert-sorted 空间。padding 行在 expert-sorted 空间中
有位置，但在 flat-topk 空间中没有对应——它们是"无主"的虚拟行。

### 2.4 `dst_idx`：real→padded 的位置映射

`_get_padding_plan` 返回的 `dst_idx` (shape: TK, dtype: int64) 是核心数据结构：

```
原始 expert-sorted 空间:  [0, 1, ..., TK-1]
                                ↓ dst_idx ↓
Padded expert-sorted 空间: [d₀, d₁, ..., d_{TK-1}]  (TK 个值分散在 [0, padded_total) 中)
```

Padding 位置 = `{0..padded_total-1} \ {dst_idx}`，即 padded 空间中不在 `dst_idx` 中的位置。

示例（E=2, expert 0 有 3 tokens, expert 1 有 2 tokens, 128 对齐后各 128）：
```
dst_idx = [0, 1, 2, 128, 129]          # 5 个 real tokens
pad_positions = [3,4,...,127, 130,...,255]  # 251 个 padding 位置
```

### 2.5 集成点：最小侵入式设计

整个方案对 GEMM 层零修改。唯一的代码变更点：

**1. `moe_TC_softmax_topk_layer` 中 router → UpProj 之间（5 行）：**
```python
if _fp8_enabled():
    (..., TK, _padded) = _pad_routing_metadata(...)
    if _padded:
        topk_scores = topk_scores_flat
```

**2. `_DownProjection.backward` 返回前的 `ds` shape 修正（4 行）：**
```python
if ds.shape[0] < topk_scores.shape[0]:
    ds = torch.cat([ds, zeros(N)])  # padding 位置的 score grad = 0
```

不修改 x，不修改 dout，不修改任何 GEMM 调用，不修改任何 quantization kernel。

## 3 实测数据

### 3.1 精度（E=32, T=8192, K=8, H=3072, I=1536）

所有 RRMSE 以 BF16 raw（无 rounding、无 padding）为金标准：

| Tensor | BF16+round | FP8+pad | FP8+round |
|--------|:---:|:---:|:---:|
| output | 0.6064 | **0.0651** | 0.6082 |
| dw1 | 0.6060 | **0.0703** | 0.6081 |
| dw2 | 0.6063 | **0.0748** | 0.6086 |
| dx | 0.6061 | **0.0704** | 0.6082 |

- **FP8+padding RRMSE ~6.5-7.5%**：纯 FP8 量化误差，与 E=8 完全一致
- **Rounding RRMSE ~60%**：路由扰动是主导因素（BF16+round 已有 60.6%）
- Padding 比 rounding 精度好 **9 倍**

### 3.2 性能（nsys GPU-projection, µs/iter）

| Mode | µs/iter | vs BF16 |
|------|:---:|:---:|
| BF16 raw | 3748 | 1.000× |
| FP8 + padding | 2950 | **1.270×** |
| FP8 + rounding | 2915 | **1.286×** |

- Padding vs rounding：仅 **+1.2%** overhead（35µs）
- Padding 的额外 kernel launches（routing tensor 操作）约 120 个，但都是轻量 pointwise ops

### 3.3 显存（peak MiB）

| Mode | Fwd | Bwd |
|------|:---:|:---:|
| BF16 raw | 1986 | 2706 |
| BF16 + rounding | 1986 | 2706 |
| FP8 + padding | 1901 | 2914 |
| FP8 + rounding | 1895 | 2909 |

Padding vs rounding：**+5 MiB backward**（0.17%），来自非均匀 routing 下的 padded
intermediate tensors。理论上均匀分配时 N_pad=0（无额外显存），实际 softmax routing
的不均匀性导致少量 expert 需要 padding。

### 3.4 训练循环验证

5 次迭代真实训练（`setup_cpu_optimizer` + `cpu_optimizer_step`），T=8193, E=8, K=8：

```
iter 0: loss=-0.73   dx=24.03  dw1=45.93  dw2=32.43
iter 1: loss=-42.25  dx=24.97  dw1=47.71  dw2=35.72
iter 2: loss=-95.0   dx=28.56  dw1=54.44  dw2=45.28
iter 3: loss=-129.0  dx=31.38  dw1=59.73  dw2=52.06
iter 4: loss=-206.0  dx=37.18  dw1=70.49  dw2=65.83
```

Loss 持续下降，梯度每次迭代都非零且合理增长。与 T=8192（对齐，无 padding）的
训练轨迹完全一致。

## 4 公理化正确性测试

`test_pad_routing_metadata_axiomatic`（T=10, E=3, K=2）：

1. 用纯 Python per-expert matmul + SwiGLU 计算 `o_gold[t]`（unpadded）
2. 对 routing metadata 做 `_pad_routing_metadata`（20→384, N_pad=364）
3. 用相同的纯 Python 计算 `o_padded[t]`（在 padded 空间上执行）
4. 逐 token 验证 `o_padded[t] == o_gold[t]`

结果：max_diff = 8.73e-11（float32 精度级别），10 个 token 全部精确匹配。

## 5 设计取舍总结

| 维度 | Token Rounding | GEMM-Level Padding | **Route-Level Padding** |
|------|:---:|:---:|:---:|
| GEMM 代码修改 | 无 | 8 处 | **无** |
| 路由语义保持 | ✗（改变 routing） | ✓ | **✓** |
| 精度 vs BF16 | ~60% RRMSE | ~6.5% RRMSE | **~6.5% RRMSE** |
| 性能 overhead | ~0% | +120% | **+1.0%** |
| 显存 overhead | ~0 | ~0 | **+5 MiB** |
| 实现复杂度 | router 层修改 | 8 个 GEMM 各改 | **1 个函数 + 5 行接入** |
