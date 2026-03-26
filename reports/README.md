# SonicMoE FP8 Reports

本目录只保留**当前状态、已验证事实、下一步主线**。不要把这里当成草稿堆。

## 1. 先看什么

- 人类读者先看本文件。
- 新 agent 接手先看 `reports/fp8_upgrade/HANDOFF.md`。
- 需要具体过程、数据和经验教训时，再看 `reports/fp8_upgrade/ENGINEERING_LOG.md`。

## 2. 当前结论

- 权威环境：
  - `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- 黑盒前提：
  - Blackwell
  - QuACK GEMM 路径必须打开：`USE_QUACK_GEMM=1`
- 当前**稳定主线**不是 blockscaled，而是：
  - QuACK up-proj
  - preact fused FP8 boundary
  - down-proj 仍走 SonicMoE 原生 `varlen/gather-A` 友好的 bf16 主循环
- 当前**真实可交付结论**：
  - FP8 主线已经在真实大 shape 上拿到 inference 优势
  - `8192,4096,1024,128,8` 上 e2e 也曾在同时间窗微幅领先 bf16
  - 但训练主矛盾仍是：
    - `quant -> dequant -> bf16` 边界回退
    - backward transient overhead
- 当前**不应误判**的点：
  - `report_fp8_metrics` 和 `--report_stage_memory` 是两套不同测量口径
  - large-shape 下不能把 cold metrics peak 和 stagewise final peak 混写成一个结论

## 3. 当前稳定主线已经做成了什么

- `sonicmoe/functional/fp8_cutely_fused.py`
  - preact fused quant/dequant 已落地
  - 支持 `restored_out=...`
- `sonicmoe/functional/__init__.py`
  - QuACK stable path 默认复用 `gemm_gated(...)` 已产出的 `y1`
  - preact FP8 boundary 在 `torch.no_grad()` 下执行
  - 已去掉语义冗余的 STE 混合
- `benchmarks/moe-cute.py`
  - 已支持：
    - `--report_fp8_metrics`
    - `--report_stage_memory`
    - `--report_fp8_analysis`
  - 已支持 backward RMSE 指标
  - 已避免 metrics snapshot 污染后续 GPU peak

## 4. 当前实验线是什么，为什么不是主线

### 4.1 blockscaled down-proj

- 文件：
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- 现状：
  - 已有 pack+quant 融合
  - 已不再把 grouped 输出重新 unpack 成 flat `y2`
  - 已支持 `w2` 静态 fp8 cache / prefetch
- 但它**仍不是主线**，根因不是“没预量化权重”，而是：
  - `grouped_out`
  - `static capacity`
  - grouped layout 到 router 聚合的过渡层
- 也就是说：
  - 它违背了 SonicMoE 最精髓的 `varlen/gather-A` 省显存合同

### 4.2 dummy postact buffer

- 开关：
  - `SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER=1`
- 作用：
  - up-proj 请求 fp8 dummy postact，再由 boundary 用 `z` 重建 bf16 `y1`
- 结论：
  - 默认关闭
  - 只保留实验价值

### 4.3 backward runtime-fp8 `y1s`

- 尝试过 `_DownProjection.backward()` 上的最短路径
- 已证伪
- blocker：
  - plain `gemm` 要求 `A/B` 同 dtype
  - `gather_A` 合法 config 更苛刻
- 结论：
  - backward fp8 需要新的 mixed-dtype / scaled GEMM 合同
  - 不是补一个 `postact_dtype` 就能成功

## 5. 当前最可信的数据点

### 5.1 稳定主线，大 shape

- shape `8192,4096,1024,128,8`
- 同时间窗一份代表性结果：
  - bf16：
    - peak `7690.63 MiB`
    - inf / train fwd / e2e / bwd `4.081 / 3.942 / 5.728 / 1.647 ms`
  - stable fp8：
    - peak `7700.75 MiB`
    - output RMSE `0.01074111`
    - loss RMSE `0.00000025`
    - inf / train fwd / e2e / bwd `1.842 / 2.111 / 5.586 / 3.743 ms`
- 但同 shape 的 stagewise raw probe 还显示：
  - final peak `7690.63 -> 7572.75 MiB`
  - backward 三段有明显 transient overhead

### 5.2 稳定主线，中等 shape

- shape `4096,4096,1024,128,8`
- 代表性结果：
  - bf16：
    - peak `7049.88 MiB`
    - inf / train fwd / e2e / bwd `1.141 / 1.210 / 3.437 / 2.296 ms`
  - stable fp8：
    - peak `6931.00 MiB`
    - output RMSE `0.01073675`
    - loss RMSE `0.00000021`
    - inf / train fwd / e2e / bwd `1.125 / 1.697 / 3.733 / 2.608 ms`

### 5.3 static fp8 weight benchmark 合同

- 已做成：
  - 用 seed 控制 bf16 权重初始化
  - 在 benchmark 前预量化 `w2`
  - 不再把首轮在线量化成本混入 timing
- 但当前它**仍然慢于 bf16**
- 这进一步证明：
  - 当前 blockscaled 慢点不在在线 weight quant
  - 而在 grouped/static-capacity 过渡层本身

## 6. 当前理论账本

- `4096,4096,1024,128,8`
- 当前稳定主线：
  - `stable_fp8_saved_payload_mib=31.00`
- 若打通真正的 varlen-friendly direct FP8 mainloop：
  - `direct_fp8_boundary_saved_mib=97.75`
- 若再把 `w1/w2` 做成 FP8 存储：
  - `aggressive_weight_saved_mib=1524.00`
- 合并上限：
  - `aggressive_total_saved_mib=1555.75`

这组数字就是当前取舍标准：

- 小于这个量级的收益，不值得为之破坏 SonicMoE 的核心内存合同

## 7. 当前缺少的关键算子 / 合同

当前真正缺的不是更多 flag，而是下面这些能力：

1. `varlen FP8 postact + scales`
   - up-proj epilogue 直接产出可被下游消费的 varlen fp8 激活

2. `gather-A preserving down-proj fp8 mainloop`
   - 不引入 grouped/static layout
   - 不引入显式 capacity

3. `backward mixed-dtype / scaled GEMM contract`
   - 让 backward act / weight 真正迁移到 fp8 相关路径

4. `persistent static FP8 weight storage`
   - 不只是 global lazy cache
   - 最终要演进到 module-level / training-realistic 合同

5. `cudagraph-compatible fp8 path`
   - 预分配输出
   - 无动态分配
   - 无 runtime 回退

## 8. 用户偏好（本 session 的硬约束）

- 不接受“跑通就行”
- 目标是**极致性能 + 极致显存收益**
- 必须学习 SonicMoE 的设计精髓：
  - 少中间结果
  - 少长驻留
  - 守住 `varlen/gather-A`
- 性能判断必须：
  - 对排 bf16 baseline
  - 对排真实大 shape
  - 同时看理论账本与实际 stage memory
- 如果 fp8 变慢，要先回答：
  - 是算子慢
  - 还是边界/中间结果慢
  - 还是组网/复用方式偏离 SonicMoE
- benchmark 应尽量做到：
  - bf16 与 fp8 使用逻辑一致
  - 最好只差一个参数
  - 不应要求用户理解一堆内部 flag

## 9. 当前推荐入口

### 快速回归

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py
```

### 稳定主线 metrics

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

## 10. 接下来应该做什么

- 第一优先级：
  - 继续把稳定主线往 **varlen-friendly FP8 epilogue/mainloop** 推
- 第二优先级：
  - 继续定位并缩小 backward transient overhead
- 第三优先级：
  - 让 static fp8 weight 从 benchmark 合同升级为真实训练合同
- 不建议优先做：
  - 继续深挖 grouped/static-capacity blockscaled 主线化

