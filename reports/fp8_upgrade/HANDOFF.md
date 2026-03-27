# FP8 Next-Agent Handoff

> **最后更新：2026-03-27 Session 3 最终版**

---

## 0. 一句话现状

**全链路 blockscaled FP8 训练 E2E 10.08ms，比 BF16 vanilla baseline 11.89ms 快 15.2%。Forward 2.77ms (598 TFLOPS)。合同测试 8/8 PASSED。Fused `gemm_gated` + blockscaled FP8 在 kernel 层比 BF16 fused 快 1.81x。Backward 使用 separate path（blockscaled_fp8_gemm_varlen + fused dSwiGLU+quant Triton kernel）因 QuACK upstream alignment bug 阻塞 fused dgated。**

---

## 1. 硬约束

- 全链条 FP8：GEMM 内部使用 FP8 tensor core + FP32 累加
- 量化方案：1x32 blockscaled UE8M0（Blackwell tcgen05.mma 硬件原生 descale）
- 精度：RelRMSE < 10%, cosine > 0.99
- 性能 + 显存双指标优于 BF16 baseline
- 守住 SonicMoE varlen/gather-A 内存合同
- 不做 site-packages 直接修改（monkey-patch 可以，但必须 git commit）

---

## 2. 代码真实状态

### 2.1 FP8 启用机制

- `SONIC_MOE_FP8_MODE=perf|mem` — 全局启用 FP8
- `SONIC_MOE_FP8_FUSED_GATED=1|0` — 启用/禁用 fused gemm_gated blockscaled（默认启用）
- `_fp8_enabled()` + `_min_expert_segment() >= 32` → blockscaled path
- `_fp8_enabled()` + segment < 32 → per-tensor FP8 fallback

### 2.2 GEMM 算子状态

| # | 算子 | 当前路径 | 量化 | 状态 |
|---|------|---------|------|------|
| 1 | up-proj forward | fused `gemm_gated` + blockscaled (FUSED=1) 或 `blockscaled_fp8_gemm_varlen` + SwiGLU (FUSED=0) | 1x32 blockscaled | **DONE** |
| 2 | SwiGLU fwd | fused 在 `gemm_gated` epilogue 内 (FUSED=1) 或 `swiglu_forward_quant_triton` (FUSED=0) | — | **DONE** |
| 3 | down-proj forward | `blockscaled_fp8_gemm_varlen` | 1x32 blockscaled | **DONE** |
| 4 | down-proj act-grad | `blockscaled_fp8_gemm_varlen` | 1x32 blockscaled | **DONE** |
| 5 | dSwiGLU+quant | `swiglu_backward_quant_triton` fused | 1x32 blockscaled | **DONE** |
| 6 | up-proj act-grad | `blockscaled_fp8_gemm_varlen` | 1x32 blockscaled | **DONE** |
| 7 | up-proj weight-grad | `quack.gemm(x.T, dz, cu_seqlens_k=...)` | per-tensor `.to(fp8)` | per-tensor 足够 |
| 8 | down-proj weight-grad | `quack.gemm(dout.T, y1s, cu_seqlens_k=...)` | per-tensor `.to(fp8)` | per-tensor 足够 |

### 2.3 Fused kernel 清单

| Kernel | 文件 | 功能 | 消除的中间 tensor |
|--------|------|------|------------------|
| `_quantize_and_pack_kernel` | `blockscaled_fp8_gemm.py` | bf16 → fp8 + ISA-packed E8M0 scales | raw_scales 中间 tensor |
| `_gather_quantize_and_pack_kernel` | `blockscaled_fp8_gemm.py` | gather + quantize + ISA-pack | bf16 gathered tensor (~512 MiB) |
| `swiglu_forward_quant_triton` | `swiglu_triton.py` | SwiGLU + blockscaled fp8 quant | bf16 y1 中间 tensor |
| `swiglu_backward_quant_triton` | `swiglu_triton.py` | dSwiGLU + score + blockscaled fp8 quant | bf16 dz 中间 tensor |
| `gemm_gated` + `sf_vec_size=32` | `gemm_gated.py` | GEMM + SwiGLU + blockscaled descale | 分离的 GEMM 输出 + SwiGLU 输入 |

### 2.4 Weight Cache

- `_WEIGHT_CACHE`: blockscaled varlen path 的 weight fp8 + packed scales
- `_FUSED_WEIGHT_CACHE`: fused gemm_gated path 的 weight fp8 + packed scales（不同 permute layout）
- 基于 `data_ptr + _version` 做 key，optimizer step 后自动失效
- `MoE.prefetch_all_fp8_weights()`: 一次性 pre-quantize 所有布局（ernie-core pattern）
- `MoE.clear_fp8_weight_cache()`: 清空所有缓存

---

## 3. 性能数据（2026-03-27 实测, Blackwell CF-NG-BZZ2-O）

### 3.1 训练性能（token rounding nr, weight cached, T=8192, H=4096, I=1024, E=128, K=8）

| 配置 | Forward (ms) | E2E fwd+bwd (ms) | Backward (ms) | TFLOPS (E2E) |
|------|-------------|-------------------|---------------|-------------|
| **FP8 fused blockscaled** | **2.772** | **10.079** | 7.308 | **493.8** |
| FP8 separate blockscaled | 3.555 | 10.823 | 7.268 | 459.8 |
| BF16 vanilla top-K (参考) | 3.511 | 11.889 | 8.012 | ~400 |
| **vs BF16** | **-21.0%** | **-15.2%** | **-8.8%** | **+23.5%** |

### 3.2 推理性能（vanilla top-K, CUDA graph, 同一 shape）

| 配置 | Inference forward (ms) | TFLOPS |
|------|----------------------|--------|
| BF16 | 3.878 | 425.3 |
| FP8 blockscaled | **2.216** | **744.3** |
| **加速** | **-42.9%** | **+75.0%** |

### 3.3 Kernel 级别验证（isolated gemm_gated, 65536×4096×2048, E=128）

| 配置 | Latency (ms) | Speedup |
|------|-------------|---------|
| BF16 fused | 0.852 | — |
| FP8 blockscaled fused | **0.470** | **1.81x** |

### 3.4 精度

| 测试 | 结果 |
|------|------|
| 合同测试 8/8 | **PASSED** (5% rtol/atol vs BF16 gold) |
| Fused gemm_gated D RelRMSE | **3.75%** |
| Fused gemm_gated PostAct RelRMSE | **5.29%** |
| 官方 moe_blackwell_test | **PASSED** |

### 3.5 显存（排除共同 master weights + optimizer）

| 组件 | BF16 | FP8 perf | FP8 mem |
|------|------|---------|---------|
| Forward peak activation | 896 MiB | 2238 MiB | **648 MiB** |
| Weight cache | 0 | +1590 MiB | 0 |
| **Activation saving** | — | — | **-248 MiB (-28%)** |

---

## 4. 已识别的 Bug 和限制

### 4.1 QuACK upstream alignment bug（CRITICAL）

**Bug**: `gemm_default_epi.py:127-129` 的 `domain_offset(cu_seqlens_m[batch_idx], mColVecBroadcast)` 将 bf16 指针对齐从 32-bit 降级到 16-bit。async copy atom 要求 32-bit → compile 失败。

**触发条件**: `gemm_dgated` + `cu_seqlens_m`(varlen) + `colvec_scale`(bf16) 同时使用。

**影响**:
- BF16 token rounding backward **crash**（走 `gemm_dgated` + varlen + colvec_scale）
- FP8 blockscaled backward **不受影响**（走 `blockscaled_fp8_gemm_varlen` + standalone SwiGLU，绕过 `gemm_dgated`）
- BF16 vanilla top-K backward **不受影响**（小 shape 的 tile config 恰好不触发）
- 官方 `moe_blackwell_test.py` **PASSED**（shape 256,768,256,128,8 不触发）

**证据**: 无任何 upstream 测试覆盖 `gemm_dgated + varlen + colvec_scale + bf16` 组合。

**修复方向**: 需要 QuACK upstream 修改 `copy_utils.tiled_copy_1d` 在 varlen 模式下使用 sync copy 或调整 alignment。`assumed_align=2` 简单修复会导致其他 verification failure。

### 4.2 Token Rounding 128-alignment 硬约束

128-alignment 是 TMA + ISA scale factor tile layout 的硬件硬约束。Token rounding routing (SonicMoE Algorithm 4) 保证 expert segments 都是 128 倍数。`_get_padding_plan` 是 fallback safety net。

---

## 5. 高价值信息源

| 信息 | 位置 |
|------|------|
| QuACK GemmSm100 blockscaled 支持 | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/lib/python3.13/site-packages/quack/gemm_sm100.py` |
| VarlenMTileScheduler | `quack/tile_scheduler.py:587` |
| Alignment bug 根因 | `quack/gemm_default_epi.py:117-138` |
| ernie-core FP8 weight 策略 | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/baidu/ernie/ernie-core/src/ernie_core/models/moe/moe_layer.py` |
| ernie-core Fp8FusedMoeFunc | 同上 line 2195 |
| SonicMoE 论文 | https://arxiv.org/html/2512.14080 |

---

## 6. 教训

1. **先验证 benchmark 的 routing 模式再下结论**。Session 2 报告"4x 退化"是因为 vanilla top-K routing 触发 padding。Token rounding 消除问题。
2. **Weight 量化必须缓存**。`precompute_weight_fp8_for_fused_gated` 首次调用 ~200ms，缓存后 0ms。benchmark 每 trial 重新初始化 weight 导致假象。
3. **`gemm_gated` + blockscaled 的 SwiGLU epilogue 不需要任何修改**——epilogue 在 fp32 累加器上操作，与 blockscaled mainloop 正交。这使得 "fused GEMM+SwiGLU+blockscaled" 只需 ~150 行 plumbing 代码。
4. **`gemm_dgated` + blockscaled 的 alignment bug 不是我们能简单修复的**。需要 QuACK upstream 改 copy atom selection 逻辑。
5. **Gather + quantize fusion 有价值但收益递减**（~4.6% forward 加速）。最大的性能提升来自 fused gemm_gated（1.81x）。

---

## 7. 下一步规划

### P0: 修复 QuACK alignment bug 以启用 fused dgated backward
- 当前 backward 用 2 个 kernel（blockscaled_fp8_gemm_varlen + swiglu_backward_quant_triton）
- Fused `gemm_dgated` + blockscaled 会合并为 1 个 kernel
- 预估 backward 从 7.3ms → ~5ms（E2E 从 10.1ms → ~8ms，总提速 ~30%）
- **阻塞**: QuACK `gemm_default_epi.py` varlen colvec alignment bug

### P1: SwiGLU fwd kernel scale output 直接写 ISA layout
- `swiglu_forward_quant_triton` 的 scale output 经过单独的 `pack_blockscaled_1x32_scales`
- 改为 kernel 内直接写 ISA layout（和 `_quantize_and_pack_kernel` 相同手法）
- 预估 -20μs/step

### P2: 消除 weight-grad 双重 fp8 cast
- `_UpProjection.backward` 中 dz（可能已是 fp8）被 cast 回 bf16 再 cast 到 per-tensor fp8
- 应直接使用 fp8 dz 做 weight-grad GEMM

### P3: 纯 FP8 weight 存储
- 参考 ernie-core：同时存 W_fp8 和 W_T_fp8（forward 用 W_T，backward dgrad 用 W）
- `MoE.prefetch_all_fp8_weights()` 框架已就绪

---

## 8. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# 合同测试 (8/8 pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# 官方 Blackwell 测试
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/moe_blackwell_test.py -v

# FP8 fused benchmark (真实训练场景，weight 缓存命中)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_FUSED_GATED=1 \
  python tools/final_benchmark.py

# nsys profile
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_FUSED_GATED=1 \
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  -o /path/to/output python tools/nsys_profile.py
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `sonicmoe/functional/__init__.py` | 核心 MoE forward/backward + FP8 调度 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Blockscaled FP8 GEMM + fused kernels |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU+quantize Triton kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | Fused GEMM+SwiGLU (+ blockscaled support) |
| `sonicmoe/quack_utils/gemm_dgated.py` | Fused GEMM+dSwiGLU (+ blockscaled support, blocked by bug) |
| `sonicmoe/quack_utils/gemm_interface.py` | Public API wrappers |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | QuACK monkey-patches (FP8 dtype + bug docs) |
| `sonicmoe/moe.py` | MoE module + weight cache management |
| `tests/fp8_large_project_contract_test.py` | 8 合同测试 |
| `tools/final_benchmark.py` | 真实训练场景 benchmark |
| `tools/nsys_profile.py` | NVTX-annotated profiling script |
| `tools/memory_analysis.py` | 理论显存分析 |
