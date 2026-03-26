# FP8 Next-Agent Handoff

本文件的目标只有一个：**让下一个 agent 在最短时间内接住主线，不重复踩坑。**

---

## 0. 先说一句人话

当前最接近正确方向的，不是 blockscaled 实验线，而是：

- 保住 SonicMoE 的 `varlen/gather-A` 合同
- 让 up-proj epilogue 直接产出 `varlen FP8 postact + scales`
- 让 down-proj mainloop 在**不引入 grouped/static capacity** 的前提下直接消费这些 FP8 激活

换句话说：

- **不要把 grouped/static-capacity 当成最终方案**
- **不要把“已经能跑”误当成“已经接近最优”**

---

## 1. 用户偏好与工作方式

这些不是建议，是本 session 已经被反复强调的硬约束：

### 1.1 目标

- 目标是**极致性能 + 极致显存收益**
- 不接受“跑通就行”
- 也不接受“只做一个能工作的实验原型就交差”

### 1.2 设计哲学

用户希望我们真正学习 SonicMoE 的精髓，而不是机械把 tensor 换成 FP8：

- 少中间结果
- 少长驻留 buffer
- 少额外调度
- 守住 `varlen/gather-A`
- 尽量避免 grouped/static-capacity 这类破坏 SonicMoE 内存合同的设计

### 1.3 benchmark / debug 方法

用户反复要求：

- 用真实大 shape，不要只看 toy case
- 同时做：
  - bf16 baseline 对排
  - 理论账本
  - stagewise memory
  - backward precision / RMSE
- 如果 fp8 变慢，必须分析：
  - 是算子本身慢
  - 还是边界搬运慢
  - 还是中间结果驻留太大
  - 还是组网/复用方式偏离 SonicMoE

### 1.4 benchmark 合同

用户明确要求：

- 用 seed 控制 bf16 权重初始化
- 如果要测 fp8 weight，必须在**运行前静态量化**
- 不能把首轮在线量化混进 timing
- 最终用户视角下，bf16 / fp8 的使用逻辑应该尽量一致，最好只差一个参数

### 1.5 工程记录要求

- 每次工程日志都要带相对 bf16 baseline 的性能/精度关系
- 所有结论都要尽量避免误导
- report 是给人和 agent 看的，不是流水账

---

## 2. 当前代码树的真实状态

### 2.1 已落地且应视为稳定主线的部分

#### A. preact fused FP8 boundary

- 文件：
  - `sonicmoe/functional/fp8_cutely_fused.py`
- 已落地能力：
  - preact fused quant/dequant
  - `restored_out=...`
- 关键意义：
  - dequant 可以直接回写到调用方提供的 `y1` buffer

#### B. stable QuACK mainline 复用 `y1`

- 文件：
  - `sonicmoe/functional/__init__.py`
- 已落地能力：
  - stable path 直接复用 `gemm_gated(...)` 产出的 `y1`
  - preact FP8 boundary 包在 `torch.no_grad()` 下
  - 默认去掉语义冗余的 `STE`

#### C. benchmark 指标面

- 文件：
  - `benchmarks/moe-cute.py`
- 已落地能力：
  - `--report_fp8_metrics`
  - `--report_stage_memory`
  - `--report_fp8_analysis`
  - backward RMSE
  - CPU snapshot 防止污染后续 GPU peak

#### D. 预分配 / cudagraph-safe 基础设施

- 文件：
  - `sonicmoe/functional/fp8_quant.py`
  - `sonicmoe/functional/fp8_cutely_fused.py`
- 已落地能力：
  - `round_scale_to_e8m0(..., out=...)`
  - `quantize_activation_blockwise(..., out=..., scale_out=...)`
  - `dequantize_activation_blockwise(..., out=...)`
  - `apply_*_fp8_protocol_cutely_fused(..., scale_out=...)`

### 2.2 已存在但仍是实验线的部分

#### A. blockscaled down-proj

- 文件：
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- 已做过的优化：
  - 吃掉了 `grouped_a` 的完整物化
  - pack+quant 融合
  - 不再把 grouped output 再 unpack 成 flat `y2`
  - grouped reverse scatter index 直接复用 `selected_experts`
- 但仍不是主线，因为：
  - 需要 `static capacity`
  - 需要 grouped layout
  - `grouped_out` 本身过大
  - router 聚合过渡层成本太高

#### B. dummy postact buffer

- 开关：
  - `SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER=1`
- 状态：
  - 默认关闭
  - 仅实验价值

#### C. static fp8 weight benchmark plumbing

- 文件：
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - `sonicmoe/moe.py`
  - `benchmarks/moe-cute.py`
- 已落地能力：
  - `prefetch_blockscaled_w2_fp8(...)`
  - `clear_blockscaled_fp8_weight_cache()`
  - `MoE.prefetch_fp8_weights(...)`
  - `MoE.clear_fp8_weight_cache()`
- 结论：
  - benchmark 合同是对的
  - 但它只证明了“在线 weight quant 不是主矛盾”
  - 没有证明 blockscaled 是正确主线

### 2.3 已证伪 / 不要重复的方向

#### A. grouped `fp8-direct-downproj`

- 结论：
  - 已用真实 shape 证伪
  - 仅仅去掉 bf16 回退，不足以抵消 grouped/static-layout 的开销

#### B. `_DownProjection.backward()` 上的 runtime-fp8 `y1s` 最短路径

- 结论：
  - 已证伪
- blocker：
  - plain `gemm` 要求 `A/B` 同 dtype
  - `dout.T` 是 bf16
  - `gather_A` 合法 config 更苛刻

#### C. 把 stagewise 与 cold metrics peak 混写

- 结论：
  - 这是错误的
- 后续任何 agent 都不要再这样写结论

---

## 3. 最可信的数据结论（2026-03-27 全链条 FP8）

### 3.1 Shape 4096,4096,1024,128,8（中等 shape）

| 指标 | BF16 | FP8-perf | Delta |
|------|------|----------|-------|
| Fwd inference (ms) | 2.501 | 1.541 | **-38.4%** |
| Fwd training (ms) | 2.566 | 1.686 | **-34.3%** |
| E2E fwd+bwd (ms) | 6.352 | 5.416 | **-14.7%** |
| Backward (ms) | 3.851 | 3.875 | +0.6% |
| Peak mem (MiB) | 7,050 | 10,122 | +43.6% |

### 3.2 Shape 8192,4096,1024,128,8（大 shape）

| 指标 | BF16 | FP8-perf | Delta |
|------|------|----------|-------|
| Fwd inference (ms) | 4.152 | 2.404 | **-42.1%** |
| Fwd training (ms) | 4.499 | 2.178 | **-51.6%** |
| E2E fwd+bwd (ms) | 12.780 | 7.666 | **-40.0%** |
| Backward (ms) | 8.627 | 5.263 | **-39.0%** |
| Peak mem (MiB) | 7,690 | 10,762 | +39.9% |

### 3.3 内存分析

+3 GiB 来自 FP8 权重缓存（4 个 cache 条目）。这是 **结构性开销**，无法在当前架构下消除：
- BF16 参数必须保留（optimizer Adam m/v 需要 fp32 master weight → bf16 param）
- FP8 缓存是额外的 permuted 副本，用于 CUTLASS kernel 的特定 layout 需求
- 推理场景或 FP8 optimizer 可以消除此开销（未来方向）

### 3.4 精度

output max abs diff: 0.003, grad max abs diff: 0.013-0.038。符合 FP8 E4M3 理论精度。
---

## 4. SonicMoE 设计精髓 vs 当前 FP8 偏差

这是最重要的一节。

### 4.1 SonicMoE 的精髓

SonicMoE 真正宝贵的不是“有一个 MoE kernel”，而是：

- `varlen/gather-A`
- 避免为每个 expert 构造巨大静态中间 buffer
- 尽量把工作保持在原有调度/路由形态里

### 4.2 当前 stable 主线为什么比 blockscaled 更像正确方向

因为 stable 主线虽然还会：

- `z -> quant -> dequant -> bf16 y1`

但它至少仍然保住了：

- varlen routing
- gather-A 风格
- 没有回到 grouped/static-capacity 这类大中间结果设计

### 4.3 当前 blockscaled 为什么偏离 SonicMoE 精髓

因为它重新引入了：

- grouped layout
- static expert capacity
- `grouped_out`

它虽然在“fp8 weight / fp8 mainloop”这个局部更激进，但在整体内存合同上更不 SonicMoE。

---

## 5. 剩余缺口

### 5.1 ~~缺口一：varlen FP8 postact + scales~~ → **已关闭**
gemm_gated 直接产出 fp8 postact，下游 quack.gemm 直接消费。

### 5.2 ~~缺口二：gather-A preserving fp8 down-proj~~ → **已关闭**
通过 monkey-patch quack.gemm 支持 fp8，保持 varlen/gather-A 合同。

### 5.3 ~~缺口三：backward mixed-dtype / scaled GEMM~~ → **已关闭**
gemm_dgated 产出 fp8 y1s，直接消费于 weight grad；dz/dx 全链条 fp8。

### 5.4 缺口四：E8M0 blockscaling（未关闭）
当前量化为简单 `.to(fp8)` cast（per-tensor），无 scale factor。
Blackwell 原生支持 1x32 UE8M0 blockscaling，需要 gemm_gated/gemm_dgated 传入 sf_vec_size + CUTLASS upstream rank 修复。

### 5.5 缺口五：FP8 optimizer / weight storage（未关闭）
当前 BF16 参数 + FP8 缓存 = 双倍权重存储。
需要 FP8 optimizer 或 FP8 parameter storage 才能消除 3 GiB 开销。
---

## 6. 经验与教训

### 6.1 有效经验

1. **优先对排真实大 shape**
   - toy case 容易误导

2. **先看理论账本，再做工程选择**
   - 没有理论上限约束，很容易在小收益点上反复打磨

3. **用 stagewise memory 找主矛盾**
   - 它能迅速区分是 kernel 问题还是边界问题

4. **显式记录“为什么某条路不是主线”**
   - 这比记录“某条路能跑通”更重要

5. **先把 benchmark 合同做正确**
   - 不然所有结论都会有污染

### 6.2 反模式

1. 把 grouped/static-capacity 当成“更接近 fp8 endgame”
2. 因为某条路数值正确，就误判它接近最优
3. 把 cold metrics 与 stagewise raw probe 混成单一结论
4. 因为局部 microbench 提速，就误判主线主矛盾已解决
5. 忽略 SonicMoE 的调度与复用设计，只从 dtype 角度看问题

---

## 7. 当前工作树的关键信息

### 7.1 当前有价值的本地改动

- `benchmarks/moe-cute.py`
  - backward RMSE 指标
  - CPU snapshot 修复
  - static fp8 weight prefetch benchmark plumbing
- `sonicmoe/moe.py`
  - fp8 weight prefetch / clear API
- `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - protocol-aware weight cache
  - prefetch / clear
- `sonicmoe/functional/__init__.py`
  - 更细的 backward stage-memory probe

### 7.2 当前不要继续推进的半成品

- 刚刚尝试过的“benchmark 对外只暴露一个 `--precision` 参数”改造**没有落地**
- 我已把 `benchmarks/moe-cute.py` 恢复到可编译状态
- 这件事应在后续单独、完整地做，不要从坏 patch 继续接着改

---

## 8. 推荐的首个动作

下一个 agent 接手后，建议顺序如下：

1. 先读：
   - `reports/README.md`
   - 本文件
   - `reports/fp8_upgrade/ENGINEERING_LOG.md`

2. 再看代码：
   - `sonicmoe/functional/__init__.py`
   - `sonicmoe/functional/fp8_cutely_fused.py`
   - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
   - `benchmarks/moe-cute.py`

3. 然后做一次快速回归：

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

4. 接着先跑稳定主线，而不是 blockscaled：

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

---

## 9. 下一阶段真正值得做的事

### P0

- 做 `varlen-friendly FP8 epilogue/mainloop`
- 保住 SonicMoE 内存合同

### P1

- 继续拆 backward transient overhead
- 让 large-shape peak 和 e2e 同时稳定优于 bf16

### P2

- 把 static fp8 weight 从 benchmark 合同推进到真实训练合同

### 明确不建议优先做

- 继续把 grouped/static-capacity blockscaled 往默认主线硬推

