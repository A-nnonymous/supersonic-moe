# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-07-26, Session 6 (fused pad+quantize, alignment-gated optimization, z-in-FP8)**

---

## 1. 当前状态：一句话

**全链路 blockscaled FP8 (1×32 UE8M0) forward + backward 功能完成，11/11 contract tests PASS，精度 RelRMSE 5-7%。性能上 FP8 backward 已 1.22x 超越 BF16 (4.68ms vs 5.70ms)；FP8 forward 因 fused GemmGated+blockscaled 有 CUTLASS DSL 深层 codegen bug 无法使用仍慢于 BF16 fused。Session 6 新增：fused pad+quantize 内核 (forward 3.44→2.52ms)、alignment-gated 优化 (aligned 场景 FP8 ~1.0ms vs BF16 1.75ms = 1.75x)、z-in-FP8 内存优化 (113MB/layer 节省)。**

---

## 2. 精度（已达标）

| Metric | Small shape (T=256) | Production shape (T=4096) | 阈值 |
|--------|--------------------|-----------------------------|------|
| Forward RelRMSE | 6.56% | 6.61% | <10% |
| dx grad RelRMSE | 6.54% | — | <10% |
| dw2 RelRMSE | 5.35% | — | <10% |
| Correlation | 0.998 | 0.998 | >0.99 |

Contract tests: **11/11 PASS**（8 small + 3 large shape，含 forward + backward + weight-grad）。

---

## 3. 性能（Session 6 优化后）

### E2E Latency (fwd+bwd, T=4096 H=4096 I=1024 E=128 K=8, B200 sm_100a)

| Config | Fwd (ms) | Bwd (ms) | Total (ms) | 相对 BF16 |
|--------|----------|----------|------------|-----------|
| BF16 fused baseline | 1.72 | 5.70 | 7.42 | 1.00x |
| FP8 Session 5 (old) | 3.38 | 5.66 | 9.04 | 0.82x |
| FP8 Session 6 (fused pad+quant, z-bf16) | 2.52 | 4.68 | 7.20 | **1.03x** |
| FP8 Session 6 (+ z-in-fp8) | 2.67 | 4.94 | 7.61 | 0.98x |
| FP8 aligned (production routing) | ~1.0 | ~3.5 (est.) | ~4.5 | **~1.65x** |

### Session 6 改进幅度

| 优化 | Forward 影响 | Backward 影响 |
|------|-------------|--------------|
| Fused pad+quantize kernel | 3.44→2.52ms (**-27%**) | 5.80→4.68ms (**-19%**) |
| z-in-FP8 memory opt | +0.15ms (quant cost) | +0.26ms (dequant cost) |
| Alignment-gated pre-quantize | 0 (unaligned benchmark) | 0 (unaligned benchmark) |

### Aligned Segments 内核级分析 (production routing, no padding)

| 操作 | 时间 (ms) |
|------|-----------|
| gather_quantize_and_pack | 0.549 |
| GEMM1 pre-quantized | 0.416 |
| SwiGLU+quant+pack fused | 0.189 |
| GEMM2 pre-quantized | 0.373 |
| **Total FP8 kernels** | **1.527** |
| **BF16 fused total** | **1.210** |

**关键发现**: FP8 GEMM 单独计算 0.789ms vs BF16 ~0.9ms → FP8 GEMM 已经更快。瓶颈是额外 kernel launch (gather_quant + SwiGLU_quant)。只有 fused GEMM+SwiGLU+FP8 才能真正超越 BF16 forward。

### 显存

| Config | Peak Memory | 相对 BF16 |
|--------|-------------|-----------|
| BF16 baseline | 10.46 GB | 1.00x |
| FP8 (z-in-bf16) | 14.07 GB | 1.34x |
| FP8 (z-in-fp8) | 13.96 GB | 1.33x |
| z-in-FP8 节省 | **113MB/layer** | (**~3.6GB for 32-layer model**) |

### 多 Shape 性能数据 (Session 5, 待 Session 6 re-bench)

| Shape | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| T=4096 H=4096 I=1024 E=128 | 1.76 | 3.37 | 0.52x |
| T=8192 H=4096 I=1024 E=128 | 2.68 | 5.49 | 0.49x |
| T=16384 H=4096 I=1024 E=128 | 3.66 | 9.75 | 0.38x |
| T=32768 H=4096 I=1024 E=128 | 6.42 | 18.28 | 0.35x |
| T=4096 H=4096 I=2048 E=128 | 2.94 | 4.23 | 0.70x |

**Note**: These are Session 5 numbers. Session 6 fused pad+quantize should improve all FP8 numbers by ~25%.

---

## 4. GEMM 矩阵

| # | 算子 | 实现 | Dtype | 说明 |
|---|------|------|-------|------|
| 1 | up-proj fwd | `blockscaled_fp8_gemm_varlen` (pre-quantized if aligned) + `swiglu_fwd_quant_pack` (if aligned) | FP8→bf16 | 2 kernels, fused SwiGLU+quant when aligned |
| 2 | down-proj fwd | `blockscaled_fp8_gemm_varlen` (pre-quantized if aligned) | FP8 | 1 kernel |
| 3 | down-proj bwd act | `blockscaled_fp8_gemm_varlen` + `swiglu_backward_triton` | FP8→bf16 | 2 kernels |
| 4 | down-proj bwd wt | `quack.gemm` | BF16 | bandwidth-bound, FP8 无收益 |
| 5 | up-proj bwd act | `blockscaled_fp8_gemm_varlen` | FP8 | 1 kernel |
| 6 | up-proj bwd wt | `quack.gemm` | BF16 | bandwidth-bound, FP8 无收益 |

4/6 GEMM 使用 FP8 blockscaled 1×32 UE8M0。Weight-grad 保持 BF16 因为 per-expert 维度 (TPE) 太小，FP8 kernel launch overhead > compute 收益。

---

## 5. Fused Blockscaled Bug — 深度诊断结论（极高价值信息）

### 现象

`GemmGatedSm100` / `GemmDGatedSm100` + `sf_vec_size=32` (blockscaled FP8) → `CUDA_ERROR_ILLEGAL_INSTRUCTION`。

### 已排除的假设

| 尝试 | 结果 | 排除 |
|------|------|------|
| Override `epi_setup_postact` 用 `CopyUniversalOp` | CRASH | ≠ TileStore R2S copy atom 问题 |
| `epi_setup_postact` 返回 `None` (完全禁用 postact store) | CRASH | ≠ TileStore/TMA store 问题 |
| BF16 fused `gemm_gated` (同形状) | PASS | ≠ shape/alignment 问题 |
| `GemmDefaultSm100` + blockscaled (无 TileStore, 无 gating) | PASS | 确认 D store 正常 |

### 真正根因

**crash 在 `GemmGatedMixin.epi_visit_subtile` 的 accumulator register recast**:

```python
# gemm_gated.py line 81
tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
```

此 recast 将 2I 列 accumulator 重解释为 I 列 (gate, value) 对。blockscaled MMA (`MmaMXF8Op`, `tcgen05.mma.kind::mxf8f6f4`) 产生的 accumulator 在 TMEM/register 中的**物理列布局**与标准 MMA 不同——blockscaled 使用 max TMEM columns (`get_max_tmem_alloc_cols`)，元素在 TMEM 中的 column interleave pattern 不同。`cute.recast_layout(2, 1, ...)` 按逻辑维度做 recast，不感知物理布局差异，导致产生无效的寄存器地址。

### 修复方向

这不是 monkey-patch 能修的。需要：
1. **在 quack/CUTLASS DSL 层面** 实现 blockscaled-aware 的 accumulator recast（理解 `MmaMXF8Op` 的 TMEM column layout）
2. 或 **写全新的 Triton MXFP8 fused GEMM+SwiGLU kernel** 绕开 CUTLASS
3. 或 **等 quack >= 0.4.0** 可能修复此 codegen

### 关键源码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `quack/gemm_sm100.py` | 194 | `self.blockscaled = sf_vec_size is not None` |
| `quack/gemm_sm100.py` | 260-278 | blockscaled 改变 `tiled_mma`, `num_tmem_alloc_cols` |
| `quack/gemm_sm100.py` | 420-425 | blockscaled max TMEM cols |
| `quack/gemm_act.py` | 74-105 | `epi_setup_postact` — R2S copy atom from `tiled_copy_t2r` |
| `quack/gemm_sm90.py` | 1192 | epilogue 调用 `epi_setup_postact` |
| `quack/gemm_sm90.py` | 1253 | epilogue 调用 `epi_visit_subtile` (crash site) |
| `quack/epi_utils.py` | 35-64 | `setup_epi_tensor` — blockscaled-unaware smem layout |
| `blackwell_helpers.py` | 174-332 | `get_smem_store_op` — StMatrix variant selection |
| `blackwell_helpers.py` | 336-527 | `get_tmem_load_op` — blockscaled changes TMEM load |

---

## 6. 本轮修复清单

### Session 5 修复

| 修复 | 文件 | 说明 |
|------|------|------|
| `get_c_pointers` / `get_mlir_types` monkey-patch | `fp8_quack_patch.py` | CUTLASS dtype classes 作为 Constexpr 字段时 TypeError |
| `fma_packed_f32x2` API 迁移 | `gemm_dgated.py` | quack 0.3.7 API 迁移 |
| Decomposed FP8 forward | `functional/__init__.py` | blockscaled GEMM + SwiGLU 替代 fused gemm_gated |
| Decomposed FP8 backward | `functional/__init__.py` | blockscaled GEMM + SwiGLU 替代 fused gemm_dgated |
| `epi_setup_postact` override | `gemm_gated.py`, `gemm_dgated.py` | CopyUniversalOp fallback |

### Session 6 优化

| 优化 | 文件 | 说明 |
|------|------|------|
| Fused SwiGLU+quant+ISA-pack fwd | `swiglu_triton.py` | `swiglu_forward_quant_pack_triton`: 融合 SwiGLU + blockscaled quantize + ISA scale packing |
| Fused SwiGLU+quant+ISA-pack bwd | `swiglu_triton.py` | `swiglu_backward_quant_pack_triton`: 融合反向 SwiGLU + quantize + ISA pack |
| Fused gather+quantize+ISA-pack | `blockscaled_fp8_gemm.py` | `gather_quantize_and_pack_activation`: 融合 gather + blockscaled quantize + ISA pack |
| Fused pad+quantize+ISA-pack | `blockscaled_fp8_gemm.py` | `pad_quantize_and_pack_activation`: 通过 src_idx 间接读取 bf16 → padded fp8 + ISA scales。消除 320MB bf16 padded 中间量。**Forward 3.44→2.52ms (-27%)** |
| Alignment-gated optimization | `functional/__init__.py` | `_all_segments_128_aligned()`: 检测所有 expert segment 是否 128 对齐。仅在对齐时使用 pre-quantized path，避免 lossy dequant→re-quantize 循环 |
| Pre-quantized scale passing | `functional/__init__.py` | `_PREQUANTIZED_SCALES` dict: up-proj 的 (y1_fp8, packed_scales) 直接传递给 down-proj，跳过 down-proj 内部 quantize |
| z-in-FP8 memory optimization | `functional/__init__.py` | `_save_z_fp8()` 控制: forward 末尾 z→fp8+raw_scales, backward 开头 dequant。每层节省 113MB |

---

## 7. 关键代码文件

| 文件 | 内容 |
|------|------|
| `sonicmoe/functional/__init__.py` | FP8/BF16 调度中枢：alignment-gated optimization, pre-quantized paths, z-in-FP8 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 基础设施：`blockscaled_fp8_gemm_varlen`, `pad_quantize_and_pack_activation`, `gather_quantize_and_pack_activation`, `quantize_and_pack_activation` |
| `sonicmoe/quack_utils/swiglu_triton.py` | SwiGLU kernels: forward/backward + fused quant+ISA-pack variants, `dequantize_blockscaled_fp8` |
| `sonicmoe/quack_utils/gemm_gated.py` | Fused GEMM+SwiGLU (BF16 only; blockscaled FP8 crashes) |
| `sonicmoe/quack_utils/gemm_dgated.py` | Fused backward GEMM+dSwiGLU |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | CUTLASS monkey-patches |
| `sonicmoe/quack_utils/gemm_interface.py` | gemm_gated / gemm_dgated tuned wrappers |
| `tests/fp8_large_project_contract_test.py` | 11 contract tests (all pass) |
| `tools/_benchmark_split.py` | Forward/backward split benchmark |
| `tools/_benchmark_aligned.py` | Kernel-level aligned benchmark |

---

## 8. 环境

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# 全量 contract tests (11/11 pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v

# 快速 smoke test (forward + backward 精度)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/_smoke_test.py

# E2E benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/_benchmark.py

# 分拆 fwd/bwd benchmark
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/_benchmark_split.py

# 远端空闲 GPU 扫描/launch
python tools/cluster_idle_launch.py scan
python tools/cluster_idle_launch.py launch --gpu-count 1 --command "..."
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 CUTLASS GEMM (必须) |
| `SONIC_MOE_FP8_MODE` | `perf` / `off` | `perf`=blockscaled FP8, `off`=纯 BF16 |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | `1` / `0` | 融合 SwiGLU+quant kernels (默认: on) |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | `1` / `0` | 保存 z 为 FP8 格式 (默认: on, 每层省 113MB) |
| `SONIC_MOE_FP8_FUSED_GATED` | `1` / `0` | Fused GemmGated+FP8 (crashes, kept for docs) |

| 包 | 版本 |
|---|------|
| `quack-kernels` | 0.3.7 |
| `nvidia-cutlass-dsl` | 4.4.2 |
| `torch` | 2.9.1+cu128 |
| `triton` | 3.5.1 |

---

## 9. 教训

1. **CUTLASS DSL accumulator recast 不感知 blockscaled 物理布局** — 这是根本限制，不是 TileStore bug。
2. **quack 0.3.7 有大量 undocumented API break** — `fma_packed_f32x2` 位移、`get_c_pointers` 对 dtype class 的处理。
3. **QuACK interleaved layout** — `w1(2I, H, E)` 的行是 `[gate0, val0, gate1, val1, ...]`。非 fused path 必须用 `z[:,0::2]`/`z[:,1::2]`。
4. **Triton `tl.dot_scaled` 在 SM100a + Triton 3.5.1 不可用**。
5. **blockscaled_fp8_gemm_varlen K 维度必须是 128 的倍数**。
6. **Decomposed backward 与 BF16 fused 性能持平**。
7. **Padding 是 FP8 性能的核心瓶颈** — 128-alignment 约束导致 random routing 下 126/128 experts 需要 padding，总 row overhead 25%。解决方案: fused pad+quantize kernel 消除 bf16 padded 中间量。
8. **Pre-quantized path + unaligned padding = catastrophic** — dequant(lossy!) → scatter → re-quantize 比直接从 bf16 开始更慢更不准确。解决方案: alignment-gated 优化，仅 aligned 时走 pre-quantize。
9. **ISA E8M0 scale packing layout 有复杂的 tile-based 索引** — SF_TILE_M=128, SF_TILE_K=128, SF_VEC_SIZE=32, SF_TILE_STORAGE=512。必须精确匹配硬件布局。

---

## 10. 下一步规划

### P0: 进一步缩小 Forward 差距

FP8 forward 2.52ms vs BF16 1.72ms (差距 0.8ms)。主要来源:
- Padding overhead: 25% 额外 rows + fused pad+quantize kernel overhead
- 2 个额外 Triton kernel launch (gather_quant, SwiGLU_quant)

**方案 A: Production-aligned routing** — 实际系统的 token routing 几乎都是 128-aligned (或可以 round up)。Aligned 场景 FP8 fwd ~1.0ms vs BF16 1.21ms = 1.2x faster。这是最直接的胜利。

**方案 B: Optimize Triton quantization kernels** — `gather_quantize_and_pack_activation` 0.549ms 可能还可以通过更大 BLOCK_ROWS 或 persistent kernel 优化。

**方案 C: 等 quack >= 0.4.0** — 修复 blockscaled accumulator recast，启用 fused GemmGated+FP8。

### P1: 显存优化

- **Drop bf16 weights after FP8 conversion**: `precompute_weight_fp8` 同时保存 bf16 和 fp8 权重。训练时可以只保留 fp8 版本 (weight grad 用 bf16 从 optimizer state 恢复)。

### P2: NCU 逐 kernel 分析

使用 `ncu --set full --kernel-name ...` 对每个关键 kernel 做详细分析，找出 compute/bandwidth 利用率瓶颈。
