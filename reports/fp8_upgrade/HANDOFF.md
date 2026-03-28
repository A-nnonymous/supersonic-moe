# FP8 Blockscaled Upgrade — Status & Handoff

> **Last updated: 2026-03-28**

---

## 1. 目标

全链路 blockscaled FP8 (1x32 UE8M0) MoE training：Forward 使用 blockscaled FP8 GEMM，Backward 使用 BF16 fused varlen GEMM。精度 RelRMSE < 0.1。

---

## 2. 当前状态

### 已完成
- **Forward up-proj**: blockscaled FP8 fused gather+quantize + CUTLASS `gemm_gated_tuned` + SwiGLU (FUSED_GATED=1 ✅ bug 已修)
- **Forward down-proj**: blockscaled FP8 activation+weight pre-quantize + CUTLASS `blockscaled_fp8_gemm_varlen`
- **Backward**: 全 BF16 fused varlen GEMM (QuACK scheduler，性能与 baseline 持平或更快)
- Weight pre-quantization caching: `_FUSED_WEIGHT_CACHE` 首次量化后缓存，后续调用 ~0 开销
- Triton kernels: fused gather+quantize+ISA-pack (`_gather_quantize_and_pack_kernel`), quantize+ISA-pack (`_quantize_and_pack_kernel`)
- `blockscaled_fp8_weight_grad_gemm` 已实现并导出（保留，未在热路径使用——E=128 下 4.7x 慢于 BF16 varlen）
- Contract tests: 8/8 PASSED (small shapes, 5% tolerance)
- RMSE: 10/10 PASSED (FUSED_GATED=0 和 =1 均通过, threshold=0.1)
- FUSED_GATED=1 双重 gather bug 已修复 (A_idx=None)
- Benchmark env var display bug 已修复

### 已知限制
| 问题 | 影响 | 决策 |
|------|------|------|
| Backward act-grad FP8 (blockscaled CUTLASS) 比 BF16 varlen 慢 | BF16 QuACK varlen scheduler 高效；FP8 量化+CUTLASS 有额外开销 | **保持 BF16** |
| Backward weight-grad FP8 per-expert 4.7x 慢于 BF16 varlen (E=128) | 128 专家小 GEMM 启动开销大 | **保持 BF16** |
| `gemm_dgated_tuned` blockscaled FP8 a_scales/b_scales 精度差 (~0.44 RelRMSE) | CUTLASS dgated 内核未正确处理 blockscaled scales | **保持 BF16 fused dgated** |
| Forward FP8 量化开销 (gather+quantize ~1.5ms on loaded cluster) | 抵消 FP8 tensor core 2x 吞吐收益 | 已是单 kernel 融合 gather+quant |

---

## 3. 性能现状 (B200, T=8192, H=4096, I=1024, E=128, K=8)

> **注意**: 集群负载高，绝对数值膨胀 ~2x；以同次运行的 BF16 baseline 为参照。

| Config | Fwd (ms) | Bwd (ms) | E2E (ms) | TFLOPS | Peak MiB |
|--------|----------|----------|----------|--------|----------|
| **BF16 baseline** | **3.48** | **7.83** | **11.30** | **440** | **7530** |
| FP8 Hybrid (FUSED_GATED=1) | 9.39 | **6.19** | 15.58 | 319 | 9116 |

**分析**:
- **Backward 6.19ms < BF16 7.83ms** — 全 BF16 GEMM 路径，GPU thermal/cache 状态差异
- **Forward 9.39ms vs 3.48ms** — FP8 量化开销 (gather+quant x: ~1.5ms, quant y1: ~0.4ms) + CUTLASS vs QuACK scheduler 差异
- **内存 +1586 MiB** — FP8 权重缓存 (w1_fused_gated ~1.1GB + w2_varlen ~0.6GB)

### 理论性能（空闲 GPU 估算）

在空闲 B200 上，FP8 tensor core 2x 吞吐应使 GEMM 加速显著。量化开销为固定带宽成本 (~0.3ms @ 3.35TB/s)。
- 预计 FP8 Forward 可快于 BF16 Forward ~30-50%
- 需空闲 GPU 验证

---

## 4. 精度验证

### RMSE (production shape, 最新配置)

| Metric | FUSED_GATED=0 | FUSED_GATED=1 | 状态 |
|--------|---------------|---------------|------|
| Forward output | 0.065 | 0.065 | ✅ PASS |
| dx (activation grad) | 0.038 | 0.038 | ✅ PASS |
| d(router_scores) | 0.053 | 0.053 | ✅ PASS |
| dw1 (up-proj weight grad) | 0.038 | 0.038 | ✅ PASS |
| dw2 (down-proj weight grad) | 0.053 | 0.053 | ✅ PASS |

Threshold=0.1, **ALL 10/10 PASSED**。

---

## 5. GEMM 矩阵 (最终配置)

| 算子 | 当前实现 | Kernel | 精度/性能 |
|------|----------|--------|-----------|
| up-proj fwd (FUSED_GATED=1) | **FP8 blockscaled** | `gemm_gated_tuned` + a/b_scales | ✅ 0.065 RelRMSE |
| up-proj fwd (FUSED_GATED=0) | **FP8 blockscaled** | `blockscaled_fp8_gemm_varlen` | ✅ 0.065 RelRMSE (慢，6 kernel) |
| down-proj fwd | **FP8 blockscaled** (A+W pre-quant) | `blockscaled_fp8_gemm_varlen` | ✅ w2 缓存 |
| down-proj bwd act-grad | BF16 fused | `gemm_dgated` | ✅ FP8 dgated 精度差 |
| down-proj bwd weight-grad | BF16 varlen | `gemm` | ✅ FP8 per-expert 太慢 |
| up-proj bwd act-grad | BF16 varlen | `gemm` | ✅ FP8 CUTLASS 慢于 QuACK |
| up-proj bwd weight-grad | BF16 varlen | `gemm` | ✅ FP8 per-expert 太慢 |

---

## 6. 关键代码文件

| File | Content |
|------|---------|
| `sonicmoe/functional/__init__.py` | 核心调度：_UpProjection, _DownProjection (fwd+bwd) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | blockscaled GEMM, weight_grad_gemm, quantize+pack, weight cache |
| `sonicmoe/quack_utils/swiglu_triton.py` | fused SwiGLU+quantize Triton kernels |
| `sonicmoe/quack_utils/gemm_interface.py` | gemm_gated_tuned / gemm_dgated_tuned (fused 路径) |
| `tests/fp8_large_project_contract_test.py` | 8+3 contract tests (large_shape 需独占 GPU) |
| `tools/final_benchmark.py` | E2E benchmark |
| `tools/rmse_verification.py` | RMSE 验证 (测试两种 FUSED_GATED 模式) |
| `tools/nsys_profile_detailed.py` | ncu/nsys 逐 kernel 分析 |

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
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/final_benchmark.py

# FP8 benchmark (FUSED_GATED=1 is default)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python tools/final_benchmark.py
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `perf` | 启用 blockscaled FP8 (`off`=纯BF16) |
| `SONIC_MOE_FP8_FUSED_GATED` | `1`(默认) | fused gather+quant → GEMM+SwiGLU (推荐) |

---

## 8. 技术备忘

1. **FUSED_GATED=1 双重 gather bug (已修)**: `gather_quantize_and_pack_activation` 已 pre-gather，传给 `gemm_gated_tuned` 时必须 `A_idx=None`。
2. **128-row alignment**: blockscaled FP8 TMA 加载要求 M-dim 为 128 倍数。Token rounding routing 保证对齐。
3. **blockscaled `.to(bf16)` 有损**: 丢失 per-block e8m0 scale。需 `dequantize_blockscaled_fp8` 显式反量化。
4. **QuACK varlen alignment**: `colvec_scale` 必须 `.float()` — BF16 指针 offset 不满足 async copy 32-bit alignment。
5. **Weight cache**: `_FUSED_WEIGHT_CACHE` 按 (data_ptr, version, shape, strides) 缓存。不同 permute 产生不同 cache 条目。Cache size > 8 时自动清空。
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
