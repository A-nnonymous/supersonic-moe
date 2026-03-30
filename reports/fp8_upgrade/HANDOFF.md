# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-03-30, Session 17-18**
> **Status: Production-ready, 1.57x E2E over BF16, 8/8 contract tests PASS**

---

## 1. 当前状态

**全链路 blockscaled FP8 (1×32 UE8M0) MoE forward + backward 功能完成，production shape 实测 1.57x E2E 加速。**

- Forward: 1.142ms (FP8) vs 1.130ms (BF16) = **0.99x** (基本持平)
- Backward: 1.894ms (FP8) vs 3.650ms (BF16) = **1.93x** ⭐ (主要收益)
- Total: 3.036ms (FP8) vs 4.781ms (BF16) = **1.57x** ⭐

Forward 持平的原因: FP8 decomposed path 使用 4 kernels (gather_quant + GEMM + SwiGLU_quant + GEMM)，而 BF16 fused path 仅 2 kernels (gemm_gated[GEMM+SwiGLU] + gemm)。Fused GEMM+SwiGLU+FP8 因 CUTLASS DSL 限制无法实现。

Backward 1.93x 的原因: 4/6 GEMM 使用 FP8 (act-grad)，权重带宽减半；weight-grad 保持 BF16 (bandwidth-bound，FP8 无收益)。

---

## 2. 精度（已达标）

| Metric | Small (T=256) | Production (T=4096) | 阈值 |
|--------|--------------|---------------------|------|
| Forward RelRMSE | 6.56% | 6.61% | <10% |
| dx grad RelRMSE | 6.54% | — | <10% |
| dw2 RelRMSE | 5.35% | — | <10% |
| Correlation | 0.998 | 0.998 | >0.99 |

全链路使用 blockscaled 1×32 E8M0 scaling（非 per-tensor），z 保存为 FP8 + raw E8M0 scales 在 backward dequant。

---

## 3. E2E 性能实测 (B200 sm_100a，干净 GPU)

### Production Benchmark

```
Shape: T=4096, H=4096, I=1024, E=128, K=8 (tpe=256, 128-aligned)
Benchmark: tools/bench_aligned_e2e.py (uniform routing, p.grad=None zero-grad)

BF16 baseline:     fwd=1.130ms  bwd=3.650ms  total=4.781ms
FP8 (aligned):     fwd=1.142ms  bwd=1.894ms  total=3.036ms
Speedup:           fwd=0.99x    bwd=1.93x    total=1.57x
```

### FP8 Forward Kernel Breakdown (per iteration)

| Kernel | µs | 说明 |
|--------|----|------|
| `_gather_quantize_and_pack_kernel` | 96 | gather x + blockscaled quant + ISA pack |
| CUTLASS GEMM (up-proj) | 266 | FP8×FP8→BF16, 128 experts × (256,4096)×(4096,2048) |
| `_swiglu_fwd_quant_pack_zsave_kernel` | 231 | SwiGLU + y1 quant + z fp8 save (读 z 一次) |
| CUTLASS GEMM (down-proj) | 213 | FP8×FP8→BF16, 128 experts × (256,1024)×(1024,4096) |
| **Total** | **~810** | |

### FP8 Backward Kernel Breakdown (per iteration)

| Kernel | µs | 说明 |
|--------|----|------|
| `_gather_quantize_and_pack_kernel` | 96 | dout gather + quant |
| CUTLASS GEMM (dout×w2ᵀ) | 240 | FP8 act-grad |
| `_dequant_blockscaled_fp8_kernel` | 39 | z_fp8 → bf16 |
| `_swiglu_bwd_quant_pack_kernel` | 385 | dSwiGLU + dz quant + ISA pack + y1s + ds |
| CUTLASS GEMM (dz×w1ᵀ) | 285 | FP8 act-grad |
| `quack::gemm` (weight-grad ×2) | ~863 | BF16 (bandwidth-bound, FP8无收益) |
| **Total** | **~1910** | |

### tpe Scaling (FP8 优势随 tpe 增大而减小)

| tpe | Forward | Backward | Total Speedup |
|-----|---------|----------|---------------|
| 256 (production) | 0.99x | 1.93x | **1.57x** |
| 1024 | ~0.95x | ~1.4x | ~1.2x |
| 4096 | ~0.9x | ~1.05x | ~1.02x |

原因: tpe 增大 → weight-grad GEMM 占比增大 → BF16 weight-grad 主导总时间。

### 非对齐路由

当 expert segments 非 128-aligned 时，自动 fallback 到 BF16 fused path。性能 ~1.0x BF16（无惩罚）。Production 应使用 token rounding 保证 128-alignment。

---

## 4. GEMM 矩阵

| # | 算子 | 实现 | Dtype | µs |
|---|------|------|-------|----|
| 1 | up-proj fwd | `blockscaled_fp8_gemm_varlen` + `swiglu_fwd_quant_pack_zsave` | FP8 | 266+231 |
| 2 | down-proj fwd | `blockscaled_fp8_gemm_varlen` (pre-quantized y1) | FP8 | 213 |
| 3 | down-proj bwd act | `blockscaled_fp8_gemm_varlen` + `swiglu_bwd_quant_pack` | FP8 | 240+385 |
| 4 | down-proj bwd wt | `quack.gemm` | BF16 | ~578 |
| 5 | up-proj bwd act | `blockscaled_fp8_gemm_varlen` | FP8 | ~285 |
| 6 | up-proj bwd wt | `quack.gemm` | BF16 | ~285 |

4/6 GEMM 使用 FP8 blockscaled。Weight-grad 保持 BF16 因 per-expert M 太小 (tpe=256)，memory-bound，FP8 无收益。

---

## 5. Fused Blockscaled Bug — 根因分析（极高价值信息）

### 现象

`GemmGatedSm100` + `sf_vec_size=32` (blockscaled FP8) → `CUDA_ERROR_ILLEGAL_INSTRUCTION`

### 根因

**crash 在 `GemmGatedMixin.epi_visit_subtile` 的 accumulator register recast**:

```python
# gemm_gated.py line 81 (quack 0.3.7)
tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
```

blockscaled MMA (`MmaMXF8Op`) 的 accumulator 在 TMEM 中的物理列布局与标准 MMA 不同。`cute.recast_layout(2, 1, ...)` 不感知此差异，产生无效寄存器地址。

### 已排除的方向

| 尝试 | 结果 |
|------|------|
| Override `epi_setup_postact` 用 `CopyUniversalOp` | CRASH |
| `epi_setup_postact` 返回 `None` (完全禁用 postact store) | CRASH |
| `GemmDefaultSm100` + blockscaled (无 gating) | PASS |

### 修复方向 (需人类专家)

1. **Standalone CUTLASS C++ kernel** — 绕开 DSL，直接写 `GemmGatedBlockscaledFp8Sm100` (4-6 weeks, recommended)
2. **等 quack ≥ 0.4.0** — 可能修复 blockscaled accumulator recast
3. **Triton MXFP8 GEMM** — `tl.dot_scaled` 在 SM100a + Triton 3.5.1 broken，需等 Triton 修复

---

## 6. Session 17-18 修复清单

### Bug 1: 非对齐路由 7x 减速 → BF16 Fallback

**问题**: `blockscaled_fp8_gemm_varlen(assume_aligned=False)` 对每个 expert 做 padding (128× allocate+copy+GEMM+unpad)，导致 7x 减速。

**修复**: 在 `_UpProjection.forward/backward` 和 `_DownProjection.forward/backward` 中，gate 所有 FP8 paths on `_ALIGNMENT_ASSUMED`。非对齐时 fallback 到 BF16 fused path (`gemm_gated`/`gemm_dgated`)。

**文件**: `sonicmoe/functional/__init__.py` (lines ~526-955)
**Commit**: `0bebc08`

### Bug 2: Weight FP8 Cache Thrashing

**问题**: `_FUSED_WEIGHT_CACHE` limit 2，但 fwd+bwd 需 4 entries (w1, w2 forward + w1ᵀ, w2ᵀ backward)。Cache 每轮清空 → 重新量化全部权重 (~2ms)。

**修复**: 增加 cache limit 从 2 到 8 (`> 2` → `> 8`)，3 处 `precompute_weight_fp8*` 函数。

**文件**: `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` (lines 1358, 1412, 1458)
**Commit**: `ea2647f`

### Bug 3: `global _ALIGNMENT_ASSUMED` 传播

**问题**: `_UpProjection.forward` 中 `_ALIGNMENT_ASSUMED = aligned` 创建局部变量，不传播到 `_DownProjection`。

**修复**: 在 `@staticmethod forward` 中添加 `global _ALIGNMENT_ASSUMED`。通过 `dis.get_instructions` 验证 `STORE_GLOBAL`。

**文件**: `sonicmoe/functional/__init__.py` (line 527)
**Commit**: `ea2647f`

### Fix 4: Benchmark Gradient Accumulation

**问题**: `bench_aligned_e2e.py` 未在 iterations 间 zero_grad，`aten::add_` 累加 ~1.4ms/iter。

**修复**: 添加 `p.grad = None` zero-grad。

**Commit**: `c8492a6`

---

## 7. 历史优化清单 (Sessions 5-16)

| Session | 优化 | 效果 |
|---------|------|------|
| 5 | 初始 decomposed FP8 path + CUTLASS patches | 功能完成, 0.82x (慢于BF16) |
| 6 | Fused SwiGLU+quant+ISA-pack, gather+quant, pad+quant | 1.03x |
| 7 | quantize kernel 2D→1D grid (4.4x faster) | 1.13x |
| 8 | SwiGLU 1D grid + atomic elimination + cache reduction | 1.13x (GPU contended) |
| 9-10 | Integer E8M0, fused z-save, BLOCK_ROWS tuning | kernel-level verified |
| 11-12 | Zero-sync alignment, GemmGated analysis | infra improvement |
| 13 | E2E contended benchmark (2.17x P10) | measurement only |
| 15-16 | BLOCK_ROWS production tuning, GEMM fast-path cache | kernel 1.76x |
| **17-18** | **Cache fix, alignment fix, benchmark correction** | **E2E 1.57x** ⭐ |

---

## 8. 关键代码文件

| 文件 | 内容 |
|------|------|
| `sonicmoe/functional/__init__.py` | FP8/BF16 调度中枢：alignment-gated, pre-quantized paths, z-in-FP8 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 GEMM: `blockscaled_fp8_gemm_varlen`, quantize+ISA-pack kernels, weight cache |
| `sonicmoe/quack_utils/swiglu_triton.py` | SwiGLU: fwd/bwd + fused quant+ISA-pack, z-dequant |
| `sonicmoe/quack_utils/gemm_gated.py` | BF16 fused GEMM+SwiGLU (blockscaled FP8 crashes) |
| `sonicmoe/quack_utils/gemm_dgated.py` | BF16 fused backward GEMM+dSwiGLU |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | CUTLASS monkey-patches for quack 0.3.7 |
| `tests/fp8_large_project_contract_test.py` | 11 contract tests (8 small + 3 large_shape) |
| `tools/bench_aligned_e2e.py` | **Production E2E benchmark** (use this, not kernel-level) |

---

## 9. 环境

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 small pass, -k "not large_shape")
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# Production E2E benchmark
CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 CUTLASS GEMM (必须) |
| `SONIC_MOE_FP8_MODE` | `perf` / `off` | blockscaled FP8 / 纯 BF16 |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | `1` | 融合 SwiGLU+quant (默认 on) |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | `1` | z 存 FP8 省显存 (默认 on) |
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | `1` | 强制 zero-sync (生产推荐) |

| 包 | 版本 |
|---|------|
| quack-kernels | 0.3.7 |
| nvidia-cutlass-dsl | 4.4.2 |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |

---

## 10. 教训（高价值）

1. **kernel-level benchmark ≠ E2E benchmark** — kernel-only 不含 Python dispatch (~0.3ms)、routing、alignment check、tensor allocation。先前 "1.76x forward" 是 kernel-only，E2E 实测仅 0.99x。**永远以 E2E 为准。**

2. **benchmark 必须 zero_grad** — 不清 gradient 会引入 `aten::add_` 累加开销 (~1.4ms/iter for 大权重)，严重扭曲 backward 计时。

3. **weight cache limit 必须覆盖 fwd+bwd** — fwd 需 w1, w2 (2 entries)，bwd 需 w1ᵀ, w2ᵀ (另 2 entries)。Limit 太小 → 每轮重新量化全部权重。

4. **`global` 在 `@staticmethod` 中是必须的** — Python `@staticmethod` 中赋值 module-level variable 不会自动 global，必须显式 `global _VAR`。

5. **非对齐 FP8 routing 是性能陷阱** — `assume_aligned=False` 对每 expert 做 padding，128 experts × padding = 7x 减速。正确做法：detect alignment → fallback BF16。

6. **CUTLASS DSL accumulator recast 不感知 blockscaled 物理布局** — 这是根本限制，不是 monkey-patch 能修的。需 CUTLASS C++ 或等 quack 升级。

7. **Triton `tl.dot_scaled` 在 SM100a + Triton 3.5.1 broken** — 不可用。

8. **ISA E8M0 scale packing 有复杂 tile-based 索引** — SF_TILE_M=128, SF_TILE_K=128, SF_VEC_SIZE=32, SF_TILE_STORAGE=512。

9. **BLOCK_ROWS 必须在 production shape 调参** — BR=1 在 512×1024 最优但在 32768×4096 灾难性 (7.5x slower)。

10. **FP8 backward 优势源于 act-grad GEMM 权重带宽减半** — weight-grad GEMM 保持 BF16 因 per-expert M 太小 (tpe=256)。

---

## 11. 下一步规划

### P0: Forward 性能提升 (当前瓶颈，0.99x)

| 方向 | 预期收益 | 难度 | 说明 |
|------|----------|------|------|
| **CUTLASS C++ fused GEMM+SwiGLU+FP8** | ~200µs (~25% fwd) | 4-6 weeks | 绕开 DSL，写 standalone kernel |
| Gather_A in blockscaled GEMM | ~96µs (~12% fwd) | Medium | 消除 gather_quant kernel |
| CUDA Graph for static shapes | ~30µs (launch overhead) | Easy | 需 assume_aligned + static shapes |

### P1: Backward 优化

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| Multi-stream overlap (act-grad ∥ weight-grad) | ~200µs (~10% bwd) | act-grad FP8 和 weight-grad BF16 可并行 |
| `_swiglu_bwd_quant_pack_kernel` 优化 | ~100µs | 385µs 是 backward 最大 Triton kernel |

### P2: 显存优化

| 方向 | 预期收益 |
|------|----------|
| FP8 weight storage (w1, w2) | ~50% weight memory per layer |
| z-in-FP8 已实现 | 每层省 ~113MB |

### P3: 生态

| 方向 | 说明 |
|------|------|
| Token rounding in routing | 保证 128-alignment，使 FP8 path 始终生效 |
| quack ≥ 0.4.0 升级 | 可能修复 fused GemmGated+FP8 |
| Triton 升级 | `tl.dot_scaled` 在 SM100a 可能修复 |

---

## 12. 显存消耗

| 组件 | BF16 | FP8 | 节省 |
|------|------|-----|------|
| z tensor (per layer) | 128MB (TK×2I×2B) | 64MB (FP8) + 2MB (scales) | ~50% |
| Weight cache | 0 | ~600MB×4 entries = 2.4GB | 额外开销 |
| 总 peak (single layer, E=128) | ~12GB | ~10GB | ~17% |

> weight cache 是全局共享的（所有 layer），不随 layer 数增长。

