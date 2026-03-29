# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-07-26, Session 7+8 (1D-grid kernel optimization, SwiGLU grid opt, memory reduction)**

---

## 1. 当前状态：一句话

**全链路 blockscaled FP8 (1×32 UE8M0) forward + backward 功能完成，11/11 contract tests PASS，精度 RelRMSE 5-7%。Session 7: quantize-and-pack kernels 通过 2D→1D grid 变换获得 4.4x 加速，E2E FP8 total 6.60ms vs BF16 7.46ms = **1.13x faster**。Session 8: SwiGLU fused quant kernels 同样完成 2D→1D grid 变换 (消除 backward atomic contention)，weight cache 8→2 entries (节省 ~7.2GB)，quantize_flat BLOCK_ROWS 4→32。GPU 高度争抢无法获得干净 benchmark，但所有精度测试通过。**

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

## 3. 性能（Session 7+8 优化后）

### E2E Latency (fwd+bwd, T=4096 H=4096 I=1024 E=128 K=8, B200 sm_100a)

| Config | Fwd (ms) | Bwd (ms) | Total (ms) | 相对 BF16 |
|--------|----------|----------|------------|-----------|
| BF16 fused baseline | 1.76 | 5.70 | 7.46 | 1.00x |
| FP8 Session 5 (old) | 3.38 | 5.66 | 9.04 | 0.82x |
| FP8 Session 6 (fused pad+quant) | 2.52 | 4.68 | 7.20 | 1.03x |
| **FP8 Session 7 (1D-grid quant, z-fp8 on)** | **2.26** | **4.34** | **6.60** | **1.13x** |
| **FP8 Session 7 (z-fp8 off, estimated)** | **~2.08** | **~4.14** | **~6.22** | **~1.20x** |
| FP8 aligned (production routing) | ~1.0 | ~3.5 (est.) | ~4.5 | **~1.65x** |

### Session 7 改进幅度 (quantize kernel 4.4x acceleration)

| 优化 | 影响 |
|------|------|
| quantize_and_pack 2D→1D grid (BLOCK_ROWS 4→32) | 0.550→0.126ms (**4.4x**) |
| gather_quantize_and_pack 同上 | 0.564→0.125ms (**4.5x**) |
| pad_quantize_and_pack 同上 | 0.685→0.127ms (**5.4x**) |
| **E2E FP8 fwd** | 2.52→2.26ms (**-10%**) |
| **E2E FP8 bwd** | 4.68→4.34ms (**-7%**) |
| **Total: FP8 1.13x faster than BF16** | |

### Session 8 改进 (SwiGLU grid opt + memory)

| 优化 | 说明 |
|------|------|
| SwiGLU fwd_quant_pack 2D→1D grid | 524K→8K blocks (64x reduction), loop over groups internally |
| SwiGLU bwd_quant_pack 2D→1D grid | 1M→8K blocks (128x reduction), **atomics eliminated** (ds accumulated in registers) |
| quantize_flat_blockscaled BLOCK_ROWS 4→32 | 16K→2K blocks (8x reduction) |
| _WEIGHT_CACHE limit 8→2 | Saves ~3.6GB |
| _FUSED_WEIGHT_CACHE limit 8→2 (×3 functions) | Saves ~3.6GB |
| **Total cache memory reduction** | **~7.2GB** |

**Note**: Session 8 performance delta could not be measured due to GPU contention (other users using 130/183GB). Contract tests 11/11 PASS. ds precision confirmed: max_rel_err < 0.0001%.

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

### Session 7 优化 (quantize kernel breakthrough)

| 优化 | 文件 | 说明 |
|------|------|------|
| quantize_and_pack 2D→1D grid | `blockscaled_fp8_gemm.py` | BLOCK_ROWS 4→32, 1D grid + `tl.range` loop over groups。8K blocks (was 524K) → **4.4x faster** |
| gather_quantize_and_pack 同上 | `blockscaled_fp8_gemm.py` | 同上模式 → **4.5x faster** |
| pad_quantize_and_pack 同上 | `blockscaled_fp8_gemm.py` | 同上模式 → **5.4x faster** |

### Session 8 优化 (SwiGLU grid + memory)

| 优化 | 文件 | 说明 |
|------|------|------|
| SwiGLU fwd_quant_pack 2D→1D grid | `swiglu_triton.py` | 524K→8K blocks (64x), NUM_GROUPS loop, hoisted ISA computations |
| SwiGLU bwd_quant_pack 2D→1D grid | `swiglu_triton.py` | 1M→8K blocks (128x), **atomic_add→register accumulation** (ds), no atomics needed |
| quantize_flat BLOCK_ROWS 4→32 | `blockscaled_fp8_gemm.py` | 16K→2K blocks for `quantize_activation_blockscaled_fast` |
| Weight cache size 8→2 | `blockscaled_fp8_gemm.py` | `_WEIGHT_CACHE`, `_FUSED_WEIGHT_CACHE` (×3 functions) — saves ~7.2GB |

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
| `tools/_bench_quantize.py` | Quantize kernel BLOCK_ROWS parametric benchmark |

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
10. **2D grid per-group-block is catastrophic for Triton kernels** — 对于 blockscaled quantize/ISA-pack 操作，每个 scale group 一个 block 导致 500K-1M blocks。改用 1D grid + 内部循环可获得 4-5x 加速。
11. **SwiGLU kernels 不能用大 BLOCK_ROWS** — compute-heavy kernels (sigmoid, backward formula) 在 BLOCK_ROWS=32 时 register pressure 导致 spill。BLOCK_ROWS=4 + 1D grid loop 是最优组合。
12. **Atomic elimination 是 1D grid 的最大收益** — backward SwiGLU kernel 的 ds atomic_add 从 1M 次降为 0 次，ds 精度从 float32 atomic 变为精确 register accumulation。
13. **Weight cache 不需要 8 entries** — MoE layer 只有 w1/w2，cache > 2 entries 浪费显存 (~600MB/entry)。

---

## 10. 下一步规划

### P0: 性能验证 (需要干净 GPU)

当前 GPU 争抢严重 (130/183GB used)，无法获得可靠 benchmark。需要:
1. **重新跑 `_benchmark_split.py`** 验证 Session 8 SwiGLU 1D-grid 优化的 E2E 影响
2. **SwiGLU BLOCK_ROWS 调优** — 尝试 BLOCK_ROWS=8,16 看是否比 4 更快 (需低争抢环境)
3. **多 shape benchmark** — T=4096,8192,16384,32768 验证优化效果一致

### P1: 进一步内核优化

| 优化 | 预期收益 | 复杂度 |
|------|----------|--------|
| quantize kernel BLOCK_ROWS=64/128 | 10-30% (当前 0.126ms vs 理论 0.051ms) | Low |
| Extract/unpad + downstream fusion | 0.1-0.2ms | Medium |
| CUDA Graph for kernel sequence | 0.1-0.3ms (launch overhead) | Medium |

### P2: 显存进一步优化

- **Bound `_FP8_WEIGHT_CACHE` / `_FP8_ORIG_CACHE`** (当前无上限!) — 加 max 2 entries + LRU
- **Drop bf16 weights after FP8 conversion**: 训练时只保留 fp8 版本 (weight grad 用 bf16 从 optimizer state 恢复)

### P2: NCU 逐 kernel 分析

使用 `ncu --set full --kernel-name ...` 对每个关键 kernel 做详细分析，找出 compute/bandwidth 利用率瓶颈。
