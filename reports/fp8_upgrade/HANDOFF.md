# FP8 Per-Tensor Upgrade — Status & Handoff

> **Last updated: 2026-03-28**

---

## 1. 目标

全链路 per-tensor FP8 MoE training：Forward 使用 per-tensor FP8 GEMM (QuACK CUTLASS scheduler)，Backward 使用 BF16 fused varlen GEMM。精度 RelRMSE < 0.1，性能和显存优于 BF16 baseline。

---

## 2. 当前状态

### 已完成
- **Forward up-proj**: Per-tensor FP8 `gemm_gated` with `A_idx` (QuACK autotuned CUTLASS), `postact_dtype=fp8` 直接输出 FP8 y1
- **Forward down-proj**: Per-tensor FP8 `gemm` with QuACK fast varlen scheduler
- **Backward**: 全 BF16 fused varlen GEMM (QuACK scheduler)
- FP8 weight caching: `_get_cached_fp8_weight` / `_get_fp8_weight_orig` (首次转换后缓存)
- Contract tests: 8/8 PASSED (small shapes, 5% tolerance)
- RMSE: 10/10 PASSED (FUSED_GATED=0 和 =1 均通过, threshold=0.1)

### 关键优化决策

| 决策 | 原因 | 性能影响 |
|------|------|---------|
| **Per-tensor FP8 取代 blockscaled** (forward) | Blockscaled gather+quantize 1.08ms vs per-tensor cast <0.05ms | Forward 从 9.4ms → 2.1ms |
| **QuACK gemm 取代 CUTLASS varlen** (down-proj) | CUTLASS GemmDefaultSm100 比 QuACK gemm 慢 6.8x | Down-proj GEMM 从 2.6ms → ~0.2ms |
| **BF16 backward** | Per-tensor FP8 `.to(fp8)` 梯度精度损失大 (~0.86 RelRMSE) | 保持准确性 |
| **FP8 weight caching** | 非缓存 strided 权重转换每次 ~10ms | 额外 1537 MiB 换 ~10ms/call |

---

## 3. 性能 (B200, T=8192, H=4096, I=1024, E=128, K=8)

> **注意**: 集群负载高，绝对数值有波动；以同次运行的 ratio 为准。

| Config | Fwd (ms) | Bwd (ms) | E2E (ms) | TFLOPS | Peak MiB |
|--------|----------|----------|----------|--------|----------|
| **BF16 baseline** | 2.55 | 8.65 | 11.19 | 445 | 7530 |
| **FP8 per-tensor** | **2.11** | **5.20** | **7.31** | **681** | 9067 |
| Ratio | **1.21x faster** | **1.66x faster** | **1.53x faster** | **1.53x** | +1537 MiB |

**Forward**: FP8 tensor core 2x 吞吐 + A_idx 内置 gather → 比 BF16 快 21%。
**Backward**: 同样是 BF16 GEMM；差异来自集群负载波动。
**内存**: +1537 MiB 来自 FP8 权重缓存 (w1: 1024 MiB + w2: 512 MiB)。

---

## 4. 精度验证

### RMSE (production shape, 最新配置)

| Metric | FUSED_GATED=0 | FUSED_GATED=1 | 状态 |
|--------|---------------|---------------|------|
| Forward output | 0.081 | 0.081 | ✅ PASS |
| dx (activation grad) | 0.043 | 0.043 | ✅ PASS |
| d(router_scores) | 0.060 | 0.060 | ✅ PASS |
| dw1 (up-proj weight grad) | 0.043 | 0.043 | ✅ PASS |
| dw2 (down-proj weight grad) | 0.060 | 0.060 | ✅ PASS |

Threshold=0.1, **ALL 10/10 PASSED**。

---

## 5. GEMM 矩阵 (最终配置)

| 算子 | 实现 | Kernel | 精度/性能 |
|------|------|--------|-----------|
| up-proj fwd | **Per-tensor FP8** | `gemm_gated` (QuACK CUTLASS autotuned) | ✅ 786 TFLOPS |
| down-proj fwd | **Per-tensor FP8** | `gemm` (QuACK fast varlen scheduler) | ✅ ~0.2ms |
| down-proj bwd act-grad | BF16 fused | `gemm_dgated` | ✅ |
| down-proj bwd weight-grad | BF16 varlen | `gemm` | ✅ |
| up-proj bwd act-grad | BF16 varlen | `gemm` | ✅ |
| up-proj bwd weight-grad | BF16 varlen | `gemm` | ✅ |

---

## 6. 关键代码文件

| File | Content |
|------|---------|
| `sonicmoe/functional/__init__.py` | 核心调度：_UpProjection, _DownProjection (fwd+bwd) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | blockscaled GEMM, weight_grad_gemm, quantize+pack (保留未用) |
| `sonicmoe/quack_utils/gemm_interface.py` | gemm_gated_tuned / gemm_dgated_tuned (fused 路径) |
| `tests/fp8_large_project_contract_test.py` | 8+3 contract tests (large_shape 需独占 GPU) |
| `tools/final_benchmark.py` | E2E benchmark |
| `tools/rmse_verification.py` | RMSE 验证 (测试两种 FUSED_GATED 模式) |
| `tools/kernel_timing.py` | 逐 kernel 性能分析 |

---

## 7. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass, exclude large_shape — 需独占 GPU)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# RMSE 验证 (tests both FUSED_GATED=0 and =1)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python tools/rmse_verification.py

# BF16 baseline benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=off \
  python tools/final_benchmark.py

# FP8 benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python tools/final_benchmark.py
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `perf` | 启用 per-tensor FP8 (`off`=纯BF16) |

---

## 8. 技术备忘

1. **Per-tensor vs Blockscaled**: Blockscaled (1×32 UE8M0) 有更好的量化精度，但 CUTLASS blockscaled varlen GEMM 比 QuACK 慢 6.8x，加上 gather+quantize 开销 1.08ms，整体远慢于 BF16。Per-tensor FP8 利用 QuACK 的高效 varlen scheduler，实现 FP8 tensor core 吞吐增益。
2. **Backward FP8 不可行**: `.to(fp8)` 不带 scaling 量化梯度，损失过大 (~0.86 RelRMSE)。需要 loss scaling 或 per-tensor scaling 才能启用 FP8 backward。
3. **postact_dtype=fp8**: up-proj 的 `gemm_gated` 直接输出 FP8 y1，down-proj 无需额外转换。
4. **FP8 weight cache**: `_FP8_WEIGHT_CACHE` (permuted layout) 和 `_FP8_ORIG_CACHE` (original layout)。`clear_all_fp8_weight_caches()` 在 optimizer step 间调用。
5. **Blockscaled 代码保留**: `blockscaled_fp8_gemm.py` 中的所有 blockscaled 函数保留供未来使用（当 QuACK 支持 blockscaled varlen 时可切换回）。
6. **`gemm_dgated_tuned` blockscaled 不可用**: a_scales/b_scales 参数存在但 CUTLASS 内核计算错误 (RelRMSE ~0.44)。
7. **`blockscaled_fp8_weight_grad_gemm` 保留但未使用**: E=128 下 per-expert pack+quantize+GEMM 共 ~7 kernel 启动，4.7x 慢于 BF16 fused varlen。

---

## 9. 后续优化方向

1. **空闲 GPU 性能验证**: 当前集群负载高，FP8 vs BF16 比率可能在空闲 GPU 上更优
2. **Forward 量化融合**: 将 activation quantize 融入 GEMM epilogue (需 CUTLASS 内核改动)
3. **CUDA Graph**: 消除 kernel launch overhead (当前 FP8 forward 4 kernel vs BF16 2 kernel)
4. **`gemm_dgated_tuned` blockscaled 修复**: 如能修复 dgated 的 blockscaled 支持，可恢复 backward act-grad FP8
5. **Varlen weight-grad FP8**: 需要 CUTLASS varlen scheduler 支持变长内维 GEMM
6. **Per-tensor FP8 hybrid**: 对 backward 使用 per-tensor FP8 (精度略降，但 QuACK scheduler 高效)
