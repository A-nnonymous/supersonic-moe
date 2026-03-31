# Blockscaled FP8 MoE — Handoff

> **Last updated: 2026-03-31, Session 21**
> **Status: FP8 is 18% SLOWER than official BF16 baseline. Previous "1.59x speedup" was measured against inflated fork BF16. Three-way nsys comparison reveals true gap.**

---

## 1. 当前状态 — ⚠️ 实事求是

**全链路 blockscaled FP8 (1×32 UE8M0) MoE forward + backward 功能完成。FP8 wgrad 已集成（默认关闭，需要用 SONIC_MOE_FP8_WGRAD=1 开启）。**

### ⚠️ 关键发现：此前上报的 "1.59x speedup" 是与自身 fork 的 BF16 做对比，而非与官方 BF16 baseline

**三方 nsys 对比（NVTX GPU Projection，sync barriers，production shape T=4096 H=4096 I=1024 E=128 K=8）：**

| Mode | Forward (µs) | Backward (µs) | Total (µs) | vs Official BF16 |
|------|-------------|---------------|------------|-------------------|
| **Official BF16** (quack 0.2.5) | 777 | 1698 | **2475** | 1.00x (baseline) |
| Fork BF16 (quack 0.3.7) | 800 | 3781 | 4581 | 0.54x ❌ |
| Fork FP8 (quack 0.3.7) | 935 | 1995 | 2930 | **0.84x** ❌ |

- **FP8 比官方 BF16 慢 455µs (18%)**
- Fork BF16 慢 1.85x 是因为 backward 有 2101µs contiguous copy overhead（详见 §3）
- 此前 "1.59x" 数字 = 4581/2930，是拿慢了的 fork BF16 做分母，不具参考价值

**nsys profile 文件**:
- `reports/sonic_official_bf16.sqlite` — 官方 BF16 (quack 0.2.5, AUTHORITATIVE baseline)
- `reports/sonic_fork_bf16_v4.sqlite` — Fork BF16 (quack 0.3.7)
- `reports/sonic_fork_fp8_v4.sqlite` — Fork FP8 (quack 0.3.7)

**分析工具**: `tools/nsys_full_breakdown.py` 解析 sqlite → NVTX GPU projection + kernel breakdown

---

## 2. 精度（达标，8/8 contract tests PASS）

| Metric | Small (T=256) | Production (T=4096) | 阈值 | 状态 |
|--------|--------------|---------------------|------|------|
| Forward RelRMSE | 6.56% | 6.61% | <10% | ✅ |
| dx grad RelRMSE | 6.54% | — | <10% | ✅ |
| dw2 RelRMSE | 5.35% | — | <10% | ✅ |
| Correlation | 0.998 | 0.998 | >0.99 | ✅ |
| FP8 wgrad (varlen_k) | 3.74% | 3.75% | <10% | ✅ |

验证命令:
```bash
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

---

## 3. 三方 Kernel Breakdown（nsys NVTX GPU Projection）

### 3.1 Official BF16 Forward (777µs) — THE REAL BASELINE

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmGatedSm100` (up-proj+SwiGLU) | 418 | 53.8% | **Fused** GEMM+SwiGLU (quack 0.2.5 独有) |
| `GemmDefaultSm100` (down-proj) | 231 | 29.7% | BF16 GEMM |
| `token_gather_sum_kernel` | 47 | 6.0% | scatter-reduce |
| misc | 81 | 10.4% | — |

### 3.2 Official BF16 Backward (1698µs) — 无 contiguous copy

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmDefaultSm100` ×3 (dact + 2× wgrad) | 1297 | 76.4% | BF16 GEMM |
| `GemmDGatedSm100` (fused dact+SwiGLU_bwd) | 258 | 15.2% | fused backward |
| misc | 144 | 8.5% | — |

### 3.3 Fork BF16 Backward (3781µs) — 被 contiguous copy 拖慢

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| **`elementwise_kernel` ×2** | **2101** | **55.6%** | **contiguous copy, 根因见 §3.6** |
| `GemmDefaultSm100` ×3 | 1264 | 33.4% | BF16 GEMM |
| `GemmDGatedSm100` | 291 | 7.7% | fused backward |
| misc | 125 | 3.3% | — |

### 3.4 Fork FP8 Forward (935µs)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmDefaultSm100` ×2 (decomposed up+down) | 480 | 51.3% | FP8 GEMM |
| `_swiglu_fwd_quant_pack_zsave_kernel` | 229 | 24.5% | **Triton: 优化重点 #1** |
| `_gather_quantize_and_pack_kernel` | 98 | 10.4% | gather + FP8 quant |
| `token_gather_sum` | 48 | 5.1% | scatter-reduce |
| misc | 81 | 8.7% | — |

### 3.5 Fork FP8 Backward (1995µs)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmDefaultSm100` ×4 (2× FP8 act + 2× BF16 wgrad) | 1336 | 67.0% | GEMM 集合 |
| `_swiglu_bwd_quant_pack_kernel` | 384 | 19.2% | **Triton: 优化重点 #2** |
| `_gather_quantize_and_pack_kernel` | 97 | 4.8% | dout quant |
| `token_gather_sum` | 46 | 2.3% | — |
| `_dequant_blockscaled_fp8_kernel` | 39 | 1.9% | z_fp8 → bf16 |
| misc | 94 | 4.7% | — |

### 3.6 Fork BF16 Contiguous Copy 根因

Fork backward 有 2101µs elementwise_kernel 而 Official 没有。原因：
- Fork 的 gemm_dgated.py 使用 `TileStore("mPostAct")` (quack 0.3.7 接口)
- Official 使用 dataclass-based epilogue (quack 0.2.5 接口)
- Fork 的 TileStore 可能产生非连续 y1s 输出
- Fork 在 `_DownProjection.backward` line 921 有 `dout = dout.contiguous()`，产生大额 copy
- **这不是 bug，是 quack 0.3.7 的 layout 行为差异**

---

## 4. FP8 vs Official BF16: 455µs Gap 分析

| 来源 | FP8 额外开销 (µs) | FP8 节省 (µs) | 净影响 |
|------|-------------------|---------------|--------|
| **前向** | | | **+158µs** |
| 分解 GEMM 代替 fused GemmGated | +62 | — | +62 |
| `_swiglu_fwd_quant_pack_zsave_kernel` (SwiGLU + quant) | +229 | -418 (无 fused GemmGated) | — |
| `_gather_quantize_and_pack_kernel` | +98 | — | +98 |
| **反向** | | | **+297µs** |
| `_swiglu_bwd_quant_pack_kernel` | +384 | -258 (无 GemmDGated) | +126 |
| `_gather_quantize_and_pack_kernel` | +97 | — | +97 |
| `_dequant_blockscaled_fp8_kernel` | +39 | — | +39 |
| FP8 GEMM 加速 (act-grad) | — | -218 | -218 |
| wgrad 仍 BF16（无 FP8 加速） | 0 | 0 | 0 |
| **合计** | | | **~+455µs** |

**根本矛盾**: FP8 的 Triton quant/SwiGLU 开销 (~847µs) 远大于 FP8 GEMM 节省 (~218µs + fused 无法使用)

---

## 5. FP8 wgrad varlen_k（已实现，已集成，默认关闭）

### 5.1 实现概述

Column-wise quantize + TMA non-contiguous FP8 + CUTLASS varlen_k GEMM.
FP8 wgrad 本身 per-op 更快（276µs vs BF16 575µs），但由于 layout permutation copy 额外 ~637µs，E2E 反而变慢。

### 5.2 代码位置

```
sonicmoe/quack_utils/blockscaled_fp8_gemm.py:
  - _colwise_quantize_and_pack_kernel  (Triton kernel)
  - colwise_quantize_and_pack()        (Python wrapper)
  - _run_cutlass_blockscaled_gemm_varlen_k()  (CUTLASS launch + cache)
  - blockscaled_fp8_wgrad_varlen_k()   (All-in-one: quant + GEMM)

sonicmoe/functional/__init__.py:
  - _use_fp8_wgrad() feature flag (line ~134, 默认关闭)
  - _DownProjection.backward (line ~1011): FP8 wgrad w2
  - _UpProjection.backward (line ~683): FP8 wgrad w1

tools/test_fp8_wgrad_varlen_k.py: (Validation test, 4/4 PASS)
```

### 5.3 为什么默认关闭

FP8 wgrad GEMM 276µs vs BF16 575µs (2.08x faster per-op)。
但 colwise quant (235µs) + layout permutation copy (~637µs) 超过了 GEMM 节省。
**需要融合 quant 到 GEMM epilogue / 消除 permutation copy 才能获益。**

### 5.4 关键发现（高价值 info）

| 信息 | 来源 | 价值 |
|------|------|------|
| quack `gemm()` 内部执行 `B = B.mT` | `quack/gemm_interface.py:170` | B 永远非连续; TMA 处理 |
| FP8 无 k-major 约束 | `quack/gemm_sm100.py:2089-2165` | 仅 FP4 需要 k-major |
| `mark_layout_dynamic(leading_dim=X)` 验证 stride[X]==1 | `cutlass/cute/runtime.py:176-196` | transposed FP8 自然满足 |
| `permute_tensors(varlen_k=True)` 只 permute D 和 C | `quack/gemm_wrapper_utils.py:168-186` | A 保持 2D |
| FP8 对齐: contiguous dim 必须 ×16 | `quack/gemm_sm100.py:2307-2308` | H=4096, I=1024 都 OK |

---

## 6. Fused Blockscaled 前向 — CUTLASS DSL 限制

`GemmGatedSm100` + `sf_vec_size=32` → `CUDA_ERROR_ILLEGAL_INSTRUCTION`

根因: `cute.recast_layout(2, 1, ...)` 不感知 blockscaled MMA 的 TMEM 物理列布局。
需 CUTLASS C++ 或 quack ≥ 0.4.0 修复。这是前向 62µs gap 的根本来源——无法使用 fused GemmGated。

---

## 7. quack 版本兼容性矩阵（关键约束）

| Feature | PyPI 0.2.5 | PyPI 0.3.7 | Original xfer (custom build) |
|---------|-----------|-----------|----------------------------|
| `ArgumentsBase` | ✓ | ✗ | ✓ |
| `mlir_namedtuple` | ✗ | ✓ | ✓ |
| `GemmGatedSm100.num_epi_tensormaps` | ✓ | ✗ | ✓ |

- **Fork 需要 `mlir_namedtuple` → 必须 quack 0.3.7**
- **Official 需要 `ArgumentsBase` + `num_epi_tensormaps` → 必须 quack 0.2.5**
- **两者 API 不兼容，无法同环境运行**
- Official BF16 profile 是在 quack 临时被 0.2.5 覆盖时抓取的（之后恢复了 0.3.7）

---

## 8. Bug 修复历史

| Bug | 修复 |
|-----|------|
| 非对齐路由 7x 减速 | Gate FP8 on `_ALIGNMENT_ASSUMED`, fallback BF16 |
| Weight cache thrashing (limit 2 < 4) | Increase to 8 |
| `global _ALIGNMENT_ASSUMED` in `@staticmethod` | Explicit `global` |
| Benchmark gradient accumulation (+1.4ms) | `p.grad = None` |
| Per-expert FP8 wgrad 3.6x slower | Replaced with varlen_k |
| Official baseline `.contiguous()` crash | Monkey-patched in profiler |

---

## 9. 教训（高价值）

1. **benchmark 基线必须用官方原版** — 不能拿 fork BF16 做分母，quack 版本差异导致 1.85x 膨胀
2. **kernel-only ≠ E2E** — 永远以 nsys NVTX GPU projection (with sync barriers) 为准
3. **benchmark 必须 zero_grad** — `p.grad = None`, 否则 +1.4ms
4. **per-expert FP8 wgrad 是死路** — tpe=256 太小, overhead > 带宽节省
5. **varlen_k + TMA 才是正确 wgrad 路径** — 但 layout copy 需要消除
6. **nsys 必须加 sync barriers** — `torch.cuda.synchronize()` before/after each NVTX range
7. **quack 0.2.5 与 0.3.7 API 不兼容** — 不要在同一 env 切换安装
8. **ISA scale packing**: SF_TILE_M=128, SF_TILE_K=128, SF_VEC_SIZE=32, SF_TILE_STORAGE=512
9. **Triton 的 quant/SwiGLU 开销 (~850µs) 是 FP8 路径的主要瓶颈，不是 GEMM**

---

## 10. 环境

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# E2E benchmark
CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py

# FP8 wgrad prototype test (4/4 pass)
CUDA_VISIBLE_DEVICES=0 python tools/test_fp8_wgrad_varlen_k.py

# nsys profiling (需要在有 nsys 的远端节点运行)
# 远端节点: 10.51.200.142 或 10.51.196.84
python tools/profile_official_vs_fork.py --mode fork_fp8
python tools/nsys_full_breakdown.py reports/sonic_fork_fp8_v4.sqlite
```

| 包 | 版本 |
|---|------|
| quack-kernels | 0.3.7 |
| nvidia-cutlass-dsl | 4.4.2 |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |

---

## 11. 显存消耗

| 组件 | BF16 | FP8 | 节省 |
|------|------|-----|------|
| z tensor (per layer) | 128MB | 66MB | ~50% |
| Weight cache | 0 | ~2.4GB (global) | 额外 |
| Peak (single layer, E=128) | ~12GB | ~10GB | ~17% |

---

## 12. 下一步规划（优先级排序）

### 🔴 核心问题：FP8 比官方 BF16 慢 18%，需要消除 455µs gap

### P0: Triton SwiGLU 内核优化 (目标: -300µs)

**当前**: `_swiglu_fwd_quant_pack_zsave_kernel` 229µs, `_swiglu_bwd_quant_pack_kernel` 384µs

**分析** (来自 agent 研究):
- Forward: 229µs vs 44µs 理论下限 = 5.2x overhead. BLOCK_ROWS=8, 32 group iterations, 重复 E8M0 编码
- Backward: 384µs vs 49µs 理论下限 = 7.8x overhead. 64 次 sigmoid (transcendental), BLOCK_ROWS=8

**优化方向**:
1. 增大 BLOCK_ROWS (16/32) 提高 SM 占用率
2. 减少 group loop 迭代次数（fuse reduction）
3. Backward sigmoid 替换为多项式近似（最大 single-op 瓶颈）
4. 设置 `num_warps` / `num_stages` autotuning
5. 用 ncu 获取 roofline 数据，确定 compute-bound vs memory-bound

### P1: Forward 恢复 fused GEMM+SwiGLU (目标: -160µs)

官方 BF16 用 `GemmGatedSm100` (418µs) = fused GEMM+SwiGLU
Fork FP8 用 decomposed GEMM (263µs) + SwiGLU kernel (229µs) = 492µs

**路径**:
- 修复 CUTLASS DSL `recast_layout` bug 使 `GemmGatedSm100` 支持 blockscaled FP8
- 或用 CUTLASS C++ 手写 fused kernel
- 或升级到 quack ≥ 0.4.0 (如果上游修复了此 bug)

### P2: Gather+Quant 优化 (目标: -40µs)

`_gather_quantize_and_pack_kernel` 98µs × 2 = 196µs. 理论下限 54µs (1.83x).
已经比较高效，可尝试 BLOCK_ROWS={24,32} autotuning.

### P3: FP8 wgrad 的 layout copy 消除 (目标: -637µs for wgrad path)

当前 FP8 wgrad 因 permutation copy 净亏。需要:
- 修改 CUTLASS output layout 直接写入目标 shape
- 或将 quant 融入 GEMM epilogue

### P4: Multi-stream Overlap

act-grad 和 weight-grad 无数据依赖，可在不同 CUDA stream 并行。

---

## 13. Profiling 工具使用指南

| 工具 | 用途 | 用法 |
|------|------|------|
| `tools/bench_aligned_e2e.py` | CUDA event E2E benchmark | `CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py` |
| `tools/profile_official_vs_fork.py` | nsys 三方对比 profiling | `python tools/profile_official_vs_fork.py --mode {official,fork_bf16,fork_fp8}` |
| `tools/nsys_full_breakdown.py` | 解析 nsys sqlite → kernel breakdown | `python tools/nsys_full_breakdown.py reports/xxx.sqlite` |
| `tools/ncu_profile_kernels.py` | ncu 单 kernel profiling | `python tools/ncu_profile_kernels.py` |
| `tools/nsys_profile_comprehensive.py` | nsys + sync barriers | `python tools/nsys_profile_comprehensive.py` |
| `tools/test_fp8_wgrad_varlen_k.py` | FP8 wgrad 功能+精度测试 | `CUDA_VISIBLE_DEVICES=0 python tools/test_fp8_wgrad_varlen_k.py` |
| `tools/rmse_verification.py` | 精度详细验证 | `CUDA_VISIBLE_DEVICES=0 python tools/rmse_verification.py` |

---

## 14. 文件结构速查

```
sonicmoe/functional/__init__.py      # FP8/BF16 dispatch, autograd Functions, alignment gating
sonicmoe/quack_utils/blockscaled_fp8_gemm.py  # Core FP8 GEMM + wgrad varlen_k + quant kernels
sonicmoe/quack_utils/swiglu_triton.py # Triton SwiGLU fwd/bwd kernels (优化重点)
sonicmoe/quack_utils/gemm_dgated.py   # CUTLASS fused backward GEMM+SwiGLU
sonicmoe/quack_utils/gemm_gated.py    # CUTLASS fused forward GEMM+SwiGLU (BF16 only)
tests/fp8_large_project_contract_test.py # 8 contract tests
reports/fp8_upgrade/HANDOFF.md        # 本文档
reports/fp8_upgrade/BLOCKSCALED_ALIGNMENT.md # ISA scale packing 对齐说明
reports/*.sqlite                      # nsys profiles
```
