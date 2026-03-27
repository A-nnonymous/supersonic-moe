# FP8 Next-Agent Handoff

> **最后更新：2026-03-27 Session 3 — 诚实最终版**

---

## 0. 一句话现状

**FP8 blockscaled 在当前 shape (8192,4096,1024,128,8) 下 E2E 训练比 BF16 慢 ~78%（13.8ms vs 7.8ms）。Forward 接近（2.8ms vs 2.5ms），backward 是主要差距（11.0ms vs 5.3ms）。根因：FP8 backward 每次 GEMM 前需要 activation quantize + ISA scale packing 开销，而 BF16 直接用 fused `gemm_dgated` 零额外开销。推理 forward 快 43%（2.2ms vs 3.9ms，CUDA graph）。精度合同测试 8/8 PASSED。**

**核心 insight：blockscaled FP8 的 quantize overhead（bf16→fp8+scales→ISA pack）在 MoE 的多次 GEMM 调用中累积，严重抵消了 FP8 tensor core 的 2x throughput 优势。除非能做到 zero-overhead quantization（如 ernie-core 的 FP8 dispatch 在通信前已量化），否则 FP8 在 MoE 训练中的收益有限。**

---

## 1. 硬约束

- 量化方案：1x32 blockscaled UE8M0（Blackwell tcgen05.mma 硬件原生 descale）
- 精度：RelRMSE < 10%, cosine > 0.99
- 不做 site-packages 直接修改（monkey-patch via git commit 可以）

---

## 2. 公平对比数据（同 shape, token rounding, weight cached）

**Shape: T=8192, H=4096, I=1024, E=128, K=8, TK=65920**

| 配置 | Forward (ms) | Backward (ms) | E2E (ms) | TFLOPS | Peak (MiB) |
|------|-------------|---------------|---------|--------|-----------|
| **BF16 (fused dgated)** | **2.471** | **5.293** | **7.764** | **641** | **7530** |
| FP8 fused fwd + fused bwd | 2.759 | 11.070 | 13.829 | 360 | 10241 |
| FP8 fused fwd + separate bwd | 2.772 | 7.308 | 10.079 | 494 | 10935 |

**推理（CUDA graph, vanilla top-K）：**

| 配置 | Inference (ms) | TFLOPS |
|------|---------------|--------|
| BF16 | 3.878 | 425 |
| FP8 blockscaled | **2.216** | **744** |
| **加速** | **-42.9%** | **+75%** |

### 性能分析

**FP8 Forward (2.77ms) vs BF16 Forward (2.47ms) — 差 12%：**
- BF16: fused `gemm_gated` 单 kernel（零 quantize 开销）
- FP8: `gather_quantize_and_pack` (~0.05ms) + fused `gemm_gated` blockscaled (~0.47ms 仅 GEMM) + weight cache lookup
- GEMM kernel 本身 FP8 快 1.81x，但 quantize overhead 吃掉了大部分收益

**FP8 Backward — 主要差距来源：**
- BF16 fused dgated: 单 CUTLASS kernel 完成 GEMM + dSwiGLU + score weighting
- FP8 separate bwd: `gather_quantize_and_pack` + `blockscaled_fp8_gemm_varlen` + `swiglu_backward_quant_triton` + per-tensor weight grad — 至少 4 个 kernel
- FP8 fused dgated bwd: 仍需要 activation quantize + weight quantize 前置，且 dgated epilogue 性能在 blockscaled 下不理想

### 精度

| 测试 | 结果 |
|------|------|
| 合同测试 8/8 | **PASSED** (5% rtol/atol) |
| Official moe_blackwell_test | **PASSED** |
| Fused gemm_gated D RelRMSE | 3.75% |
| Fused gemm_gated PostAct RelRMSE | 5.29% |

### 显存（排除共同 master weights）

| 组件 | BF16 | FP8 perf | FP8 mem |
|------|------|---------|---------|
| Forward peak activation | 896 MiB | 2238 MiB | 648 MiB |
| Weight cache | 0 | +1590 MiB | 0 |

---

## 3. 代码状态

### GEMM 算子

| 算子 | BF16 路径 | FP8 路径 | 状态 |
|------|---------|---------|------|
| up-proj forward | fused `gemm_gated` | fused `gemm_gated` + blockscaled | DONE |
| down-proj forward | `quack.gemm` varlen | `blockscaled_fp8_gemm_varlen` | DONE |
| down-proj act-grad | fused `gemm_dgated` | separate `blockscaled_fp8_gemm_varlen` + SwiGLU bwd | DONE |
| up-proj act-grad | `quack.gemm` varlen | `blockscaled_fp8_gemm_varlen` | DONE |
| weight-grad (×2) | `quack.gemm` varlen-K | per-tensor `.to(fp8)` + `quack.gemm` | DONE |

### 已修复的 Bug

- **QuACK varlen alignment bug**: `gemm_default_epi.py:127-129` `domain_offset` 降低 bf16 colvec 指针对齐。**修复**: `colvec_scale=s.float()` — fp32 指针动态偏移后仍 32-bit aligned。

### Fused Kernel 清单

| Kernel | 功能 | 节省 |
|--------|------|------|
| `_quantize_and_pack_kernel` | bf16→fp8+ISA scales (1 kernel) | raw_scales tensor |
| `_gather_quantize_and_pack_kernel` | gather+quant+pack (1 kernel) | ~512 MiB bf16 intermediate |
| `swiglu_forward_quant_triton` | SwiGLU+blockscaled quant | bf16 y1 intermediate |
| `swiglu_backward_quant_triton` | dSwiGLU+score+blockscaled quant | bf16 dz intermediate |

---

## 4. 高价值信息源

| 信息 | 位置 |
|------|------|
| QuACK blockscaled + varlen 支持 | `quack/gemm_sm100.py:423` GemmSm100.__call__ accepts mSFA/mSFB |
| VarlenMTileScheduler | `quack/tile_scheduler.py:587` |
| alignment bug 根因 | `quack/gemm_default_epi.py:117-138` varlen domain_offset |
| ernie-core FP8 weight 策略 | ernie-core `src/ernie_core/models/moe/moe_layer.py:2195` Fp8FusedMoeFunc |
| SonicMoE 论文 | https://arxiv.org/html/2512.14080 |
| 128-alignment 硬约束分析 | `reports/fp8_upgrade/BLOCKSCALED_ALIGNMENT.md` |

---

## 5. 教训

1. **BF16 fused `gemm_dgated` 极强**。单 CUTLASS kernel 完成 GEMM+dSwiGLU+score weighting，零额外开销。FP8 要赢需要把 quantize overhead 做到 ~0。
2. **Activation quantize 是 FP8 MoE 的核心瓶颈**。每次 GEMM 前的 bf16→fp8+ISA-pack 开销累积（4-6 次/step），严重抵消 FP8 tensor core 优势。
3. **ernie-core 的方案**：FP8 dispatch（通信前已量化）+ deep_gemm（C++ 级别 GEMM，无 Python overhead）+ 每个 GEMM 只做 1 次量化。SonicMoE 的 CUTLASS DSL 路径有更多 Python/Triton kernel launch overhead。
4. **`colvec_scale=s.float()` 修复 alignment bug**：fp32 元素 4 bytes，动态偏移后永远 32-bit aligned。简单有效。
5. **FP8 推理有明确优势**（-43%）因为只有 forward，quantize 只做 1-2 次。

---

## 6. 下一步方向

### P0: 消除 activation quantize overhead
- 当前每次 blockscaled GEMM 前都要 `quantize_and_pack_activation`（bf16→fp8+ISA pack，~30-50μs/call × 4-6 calls = 200μs+）
- 方案 A: 全 pipeline fp8 保持，避免 bf16↔fp8 往返（需要 GEMM 输出直接为 fp8）
- 方案 B: 参考 ernie-core 的 deep_gemm，用 C++ kernel 替代 Python Triton quantize
- 方案 C: 将 quantize 融合到 CUTLASS GEMM prologue（mainloop 内做 on-the-fly quantize）

### P1: 利用 fused dgated + blockscaled（已有基础设施）
- `gemm_dgated` + `a_scales/b_scales` 已能编译运行（13.8ms），但性能不如 separate path
- 需要 profile 分析 fused dgated blockscaled 的 overhead 来源（autotune config？epilogue overhead？）

### P2: Weight-grad blockscaled
- 当前 per-tensor `.to(fp8)` cast，精度可能不够稳定
- 需要 per-operator RMSE 分析确认是否需要改进

---

## 7. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# 合同测试 (8/8 pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# BF16 E2E benchmark (token rounding, fused dgated)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=off \
  python tools/final_benchmark.py

# FP8 E2E benchmark (token rounding, fused gated fwd + separate bwd)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_FUSED_GATED=1 \
  python tools/final_benchmark.py

# nsys profiles
output/fp8_fused_profile.nsys-rep
output/bf16_fwd_profile.nsys-rep
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `sonicmoe/functional/__init__.py` | 核心 forward/backward + FP8 调度 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Blockscaled GEMM + fused kernels |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU+quantize kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | Fused GEMM+SwiGLU (+ blockscaled) |
| `sonicmoe/quack_utils/gemm_dgated.py` | Fused GEMM+dSwiGLU (+ blockscaled) |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | QuACK patches (FP8 dtype + bug docs) |
| `tools/final_benchmark.py` | Fair comparison benchmark |
| `tools/nsys_profile.py` | NVTX profiling |
