# FP8 Blockscaled Upgrade — Status & Handoff

> **Last updated: 2026-03-27**

---

## 1. 目标

全链路 blockscaled FP8 (1x32 UE8M0) MoE training：Forward + Backward 全部使用 blockscaled FP8 GEMM，实现 E2E 性能超越 BF16 baseline，显存更低。

---

## 2. 当前状态

### 已完成
- Forward up-proj: blockscaled FP8 gather+quantize + GEMM + SwiGLU (FUSED_GATED=0 正常)
- Forward down-proj: BF16 GEMM (待切换 FP8)
- Backward: 全 BF16 (待切换 FP8)
- Triton kernels: fused SwiGLU+quantize (forward/backward), gather+quantize+ISA-pack, dequantize
- `blockscaled_fp8_weight_grad_gemm` 已实现并导出
- Contract tests: 8/8 PASSED (small shapes, 5% tolerance)

### 阻塞项
| 问题 | 影响 | 状态 |
|------|------|------|
| FUSED_GATED=1 在 production shapes 有 ~3x scaling 误差 | Forward 必须用 3-kernel separate path (慢) | 诊断中 |
| Backward 全 BF16 | 无法利用 FP8 2x 吞吐 | 待实现 |
| Down-proj forward 为 BF16 | 丢失 FP8 forward 收益 | 待实现 |

### 性能现状 (B200, T=8192, H=4096, I=1024, E=128, K=8)

| Config | Fwd (ms) | Bwd (ms) | E2E (ms) | TFLOPS |
|--------|----------|----------|----------|--------|
| **BF16 baseline** | **1.65** | **3.26** | **4.95** | **1005** |
| FP8 Hybrid (partial, FUSED_GATED=0) | 2.96 | 3.70 | 6.65 | 748 |

**当前 FP8 Hybrid 比 BF16 慢 34%** — 因为只有 up-proj forward 用了 FP8，且 FUSED_GATED bug 导致用了低效 separate path。

---

## 3. 精度验证

### RMSE (production shape, Hybrid 配置)

| Metric | RelRMSE | 状态 |
|--------|---------|------|
| Forward output | 0.053 | PASS |
| dx (activation grad) | 0.038 | PASS |
| d(router_scores) | 0.053 | PASS |
| dw1 (up-proj weight grad) | 0.038 | PASS |
| dw2 (down-proj weight grad) | 0.053 | PASS |

Threshold=0.1, ALL 5/5 PASSED。

---

## 4. GEMM 使用矩阵 (目标 vs 当前)

| 算子 | 当前 | 目标 |
|------|------|------|
| up-proj fwd act | FP8 blockscaled `gemm_varlen` | FP8 fused `gemm_gated` (修 bug) |
| down-proj fwd act | BF16 `gemm()` | FP8 blockscaled |
| down-proj bwd act | BF16 `gemm_dgated()` | FP8 blockscaled `gemm_dgated` 或 separate |
| down-proj bwd weight | BF16 `gemm()` | FP8 `blockscaled_fp8_weight_grad_gemm` |
| up-proj bwd act | BF16 `gemm()` | FP8 blockscaled `gemm_varlen` |
| up-proj bwd weight | BF16 `gemm()` | FP8 `blockscaled_fp8_weight_grad_gemm` |

---

## 5. 关键代码文件

| File | Content |
|------|---------|
| `sonicmoe/functional/__init__.py` | 核心调度：_UpProjection, _DownProjection (fwd+bwd) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | blockscaled GEMM, weight_grad_gemm, quantize+pack |
| `sonicmoe/quack_utils/swiglu_triton.py` | fused SwiGLU+quantize Triton kernels |
| `sonicmoe/quack_utils/gemm_interface.py` | gemm_gated_tuned (FUSED_GATED 路径) |
| `tests/fp8_large_project_contract_test.py` | 8 contract tests |
| `tools/final_benchmark.py` | E2E benchmark |

---

## 6. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass, exclude large_shape for pre-existing NaN)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# BF16 baseline benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/final_benchmark.py

# FP8 benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python tools/final_benchmark.py
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `perf` | 启用 FP8 (`off`=纯BF16) |
| `SONIC_MOE_FP8_FUSED_GATED` | `1`(默认) | fused GEMM+SwiGLU+blockscaled (当前有 bug) |

---

## 7. 技术备忘

1. **128-row alignment**: blockscaled FP8 TMA 加载要求 M-dim 为 128 倍数。Token rounding routing 保证对齐。
2. **blockscaled `.to(bf16)` 有损**: 丢失 per-block e8m0 scale。需 `dequantize_blockscaled_fp8` 显式反量化。
3. **QuACK varlen alignment**: `colvec_scale` 必须 `.float()` — BF16 指针 offset 不满足 async copy 32-bit alignment。
4. **`_PREQUANTIZED_SCALES` dict**: 在 autograd Function 边界间传递 pre-packed blockscaled (fp8_data, scales)。
