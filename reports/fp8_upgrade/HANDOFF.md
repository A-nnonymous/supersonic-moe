# FP8 Blockscaled Upgrade — Status & Handoff

> **Last updated: 2025-07-25**

---

## 1. 架构决策: Hybrid FP8 Forward + BF16 Backward

经过完整的 profiling 和多轮迭代，采用 **Hybrid 策略**：

- **Forward**: FP8 blockscaled up-proj GEMM (利用 FP8 2x 计算吞吐) + BF16 down-proj GEMM
- **Backward**: 全 BF16 (fused `gemm_dgated` + varlen `gemm` weight-grad)

**为什么不用全链路 FP8 backward？**
MoE workload 在当前 shape (E=128, K=8, I=1024) 是 **IO-bound 而非 compute-bound**。BF16 已达 1000+ TFLOPS (55% B200 peak)。FP8 backward 的 quantize 开销 (pack + scale + Triton kernel launches) 远超 2x 计算收益。实测全链路 FP8 backward 为 21.9ms，切到 BF16 后为 3.7ms — **5.9x 加速**。

---

## 2. 性能数据 (B200, T=8192, H=4096, I=1024, E=128, K=8)

| Config | Fwd (ms) | Bwd (ms) | E2E (ms) | TFLOPS | Peak MiB |
|--------|----------|----------|----------|--------|----------|
| BF16 baseline | 1.65 | 3.26 | 4.95 | 1005 | 7530 |
| **FP8 Hybrid (推荐)** | **2.96** | **3.70** | **6.65** | **748** | **8587** |
| FP8 原始 (全链路 blockscaled) | 2.73 | 21.89 | 24.62 | 202 | 12652 |

**改进**: E2E **3.7x 加速** (24.62ms → 6.65ms)，显存 **32% 减少** (12652 → 8587 MiB)。

**Forward 显存 peak 7109 MiB < BF16 的 7530 MiB** — fused gather+quantize 消除了 BF16 gather intermediate。

---

## 3. 精度验证

### RMSE (production shape, FUSED_GATED=0)

| Metric | RelRMSE | 状态 |
|--------|---------|------|
| Forward output | 0.053 | ✅ PASS |
| dx (activation grad) | 0.038 | ✅ PASS |
| d(router_scores) | 0.053 | ✅ PASS |
| dw1 (up-proj weight grad) | 0.038 | ✅ PASS |
| dw2 (down-proj weight grad) | 0.053 | ✅ PASS |

**ALL 5/5 PASSED** (threshold=0.1). 之前全链路 FP8 的 dx/dw1 RelRMSE 为 333,000+ (catastrophic)。

### Contract Tests: **8/8 PASSED** (5% tolerance, small shapes)

---

## 4. 已知问题

### FUSED_GATED=1 有 ~3x 系统性误差 (CUTLASS kernel bug)

`gemm_gated_tuned` fused 内核在 production shapes 下 forward output 有 ~3x scaling 错误 (RelRMSE=2.10)。
- Small shapes 下正常 (contract tests pass)
- 仅影响 FUSED_GATED=1 路径；FUSED_GATED=0 (默认推荐) 不受影响
- 根因在 CUTLASS fused GEMM+SwiGLU kernel，非 Python dispatch 层面
- **Workaround**: 使用 `SONIC_MOE_FP8_FUSED_GATED=0`

### Forward 性能 gap (2.96ms vs BF16 1.65ms)

FP8 forward 仍比 BF16 慢 79%。原因：
1. `gather_quantize_and_pack_activation(x)` — Triton quantize kernel (~0.3ms)
2. `blockscaled_fp8_gemm_varlen(x, w1)` — unfused GEMM，比 BF16 fused `gemm_gated` 慢
3. BF16 `gemm_gated` = 1 kernel (GEMM+SwiGLU fused)；FP8 = 3 kernels (quantize + GEMM + SwiGLU)

修复 FUSED_GATED=1 bug 可缩小 gap (fused path: 2.62ms)。

---

## 5. 代码架构

### 当前 GEMM 使用矩阵

| 算子 | Forward | Backward |
|------|---------|----------|
| up-proj act | FP8 blockscaled `gemm_varlen` | BF16 `gemm()` varlen |
| up-proj weight | — | BF16 `gemm()` varlen |
| down-proj act | BF16 `gemm()` varlen | BF16 `gemm_dgated()` fused |
| down-proj weight | — | BF16 `gemm()` varlen |

### 关键机制

- **`_PREQUANTIZED_SCALES`**: Dict-based FP8 tensor 复用 (已简化 — backward 不再使用)
- **`_evict_per_tensor_caches_once()`**: 首次 blockscaled 调用时清理旧 per-tensor FP8 weight cache
- **Forward peak 优化**: `gather_quantize_and_pack_activation` fused kernel 消除 BF16 gather intermediate

### 本次变更摘要

1. **Down-proj backward**: blockscaled FP8 → BF16 `gemm_dgated()` fused
2. **Up-proj backward**: blockscaled FP8 → BF16 `gemm()` varlen
3. **Down-proj forward**: blockscaled FP8 → BF16 `gemm()` varlen
4. **Up-proj forward (separate)**: fused gather+quantize + standalone SwiGLU
5. **新增**: `dequantize_blockscaled_fp8` Triton kernel
6. **修复**: RMSE 工具 retain_graph bug

---

## 6. 下一步优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 修复 FUSED_GATED=1 ~3x bug | CUTLASS `gemm_gated_tuned` scaling 错误 |
| P1 | FP8 weight-only 存储 | 只存 FP8 weights，减少 ~1GB 内存 |
| P2 | Shape-adaptive FP8 | 大 shape 用 FP8，小 shape fallback BF16 |
| P3 | 全链路 FP8 backward (大 shape) | 当 MoE 变 compute-bound (I=4096+) 时重新启用 |

---

## 7. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass, exclude large_shape)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# FP8 Hybrid benchmark (推荐配置)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_FUSED_GATED=0 \
  python tools/final_benchmark.py

# RMSE verification
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/rmse_verification.py
```

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `perf` | FP8 hybrid (`off`=纯BF16) |
| `SONIC_MOE_FP8_FUSED_GATED` | `0` | ⚠️ `1` 有 ~3x bug |

---

## 8. 教训

1. **MoE IO-bound**: 当前 shapes 下 BF16 达 55% peak。FP8 quantize 开销 > 2x 计算收益。Hybrid 是正确权衡。
2. **blockscaled `.to(bf16)` 有损**: 丢失 per-block e8m0 scale。需显式 dequantize。
3. **128-row alignment**: 详见 [`BLOCKSCALED_ALIGNMENT.md`](BLOCKSCALED_ALIGNMENT.md)。
4. **QuACK varlen alignment**: `colvec_scale` 必须 `.float()` — BF16 指针 offset 可能不满足 async copy 32-bit alignment。
