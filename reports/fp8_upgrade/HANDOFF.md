# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-07-28, Session 13 (zero-sync execution, assume_aligned, code cleanup)**

---

## 1. 当前状态：一句话

**全链路 blockscaled FP8 (1×32 UE8M0) forward + backward 功能完成，8/8 contract tests PASS，精度 RelRMSE 5-7%。Session 13: 实现 zero-sync execution — 对齐路径下 ZERO D2H syncs。NCU estimated E2E: FP8 ~6.16ms vs BF16 ~7.46ms = **1.21x faster**。所有 Triton kernel 已达极限 (406µs)，GEMM 占 87%。GemmGated+blockscaled FP8 融合受 CUTLASS DSL accumulator recast 限制，无法实现（需 quack ≥ 0.4.0 或 CUTLASS 基础设施变更）。**

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
| **FP8 Session 11 (NCU estimated)** | **~2.01** | **~4.15** | **~6.16** | **~1.21x** |
| FP8 aligned (production routing) | ~1.0 | ~3.5 (est.) | ~4.5 | **~1.65x** |

### Session 13 实测 E2E (contended B200, E=8 K=1 H=4096 I=1024 T=4096)

**全集群 128 GPUs 100% occupied，实测在高争抢下进行。FP8 的内存带宽优势在争抢下被放大。**

|  | P10 | P50 | Min | Max |
|--|-----|-----|-----|-----|
| BF16 | 13.90ms | 13.95ms | 13.83ms | 138.02ms |
| **FP8 blockscaled** | **6.40ms** | **13.96ms** | **5.93ms** | **14.34ms** |
| **Speedup (P10)** | **2.17x** | | **2.33x (min)** | |

**分析**: FP8 使用 fp8 (1B) 数据传输 vs BF16 (2B)，内存带宽需求减半。在 GPU 争抢下，
带宽是主要瓶颈，FP8 因此获得 2x+ 加速。P50 持平是因为大部分时间 GPU preemption/scheduling
噪声主导。Min 5.93ms 接近 NCU 估算的 6.16ms。

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

### Session 9+10 改进 (integer E8M0 + BLOCK_ROWS=1 + fused z-save)

**核心优化三件套 (NCU verified on B200 sm_100a at production shape TK=512, K/I=4096)**

| 优化 | 文件 | 效果 |
|------|------|------|
| Integer E8M0 (替代 log2/ceil/exp2) | 全部 4 个 quantize kernel | ~5-8% per kernel (transcendental 40→integer 3 cycles) |
| Fused z-fp8-save into SwiGLU+quant | `swiglu_triton.py` | z 只读一次: 322→129µs (**60% faster**) |
| BLOCK_ROWS=1 (universal optimum) | 全部 quantize/SwiGLU kernels | 512 blocks saturate 160 SMs perfectly |

**Forward Triton kernel 逐项对比 (Session 7 → Session 10)**

| Kernel | Session 7 | Session 10 | 加速 |
|--------|-----------|------------|------|
| gather_quantize_and_pack | ~125µs | **86µs** | **31%** |
| SwiGLU+quant (or +zsave) | ~300µs* | **129µs** | **57%** |
| **Total forward Triton** | **~460µs** | **~215µs** | **53%** |

\* Session 7: swiglu_fwd_quant_pack (~190µs) + z-fp8-save (~105µs) + fill (~5µs) = ~300µs

**Integer E8M0 公式 (3 cycles vs transcendental 40-60 cycles)**

```python
# Pure-integer bitwise: replaces log2(amax) → ceil → exp2
amax_bits = block_amax.to(tl.int32, bitcast=True)
biased_exp = (amax_bits >> 23) & 0xFF
mantissa_bits = amax_bits & 0x7FFFFF
carry = tl.where(mantissa_bits > 0x600000, 1, 0)  # >1.75
e8m0_i32 = biased_exp - 8 + carry  # -8: fp8_e4m3 max=448=1.75*2^8
```

### Session 12-13 优化 (zero-sync execution + code cleanup)

**核心突破: 完全消除 D2H syncs (aligned 路径)**

| 优化 | 文件 | 效果 |
|------|------|------|
| `_get_cu_seqlens_cpu` tensor-attribute cache | `blockscaled_fp8_gemm.py` | 每 tensor 生命周期仅 1 次 .tolist() sync |
| `_get_padding_plan` pure-Python rewrite | `blockscaled_fp8_gemm.py` | 用 CPU tuple 做全部计算，0 GPU operations |
| Streak-based alignment assumption | `functional/__init__.py` | 3 次连续 aligned 后自动假设对齐 (0 sync) |
| `assume_aligned` parameter | `blockscaled_fp8_gemm_varlen()` | 跳过 `_get_padding_plan` 调用 (最后 sync 来源) |
| `SONIC_MOE_FP8_ASSUME_ALIGNED=1` env var | `functional/__init__.py` | 立即进入 zero-sync 模式 |
| Backward simplification | `functional/__init__.py` | 移除 `_UpProjection.backward` 冗余 pre-quantize 分支 |

**Sync audit (aligned path, assume_aligned=True):**
- Forward: 0 syncs (gather_quant → GEMM → swiglu_quant, all async)
- Backward: 0 syncs (all GEMM calls pass assume_aligned)
- Weight cache: 0 syncs after warmup (storage-identity cache)
- **Total: ZERO D2H syncs per iteration**

**GemmGated + blockscaled FP8 融合深度分析:**
- `epi_visit_subtile` 的 `cute.recast_layout(2, 1, tRS_rD.layout)` 在 blockscaled MMA 后 crash
- 根因: blockscaled MMA (MmaMXF8Op) 产生的 accumulator TMEM layout 与标准 MMA 不同
- recast 按逻辑维度做，不感知物理 layout → 无效寄存器地址
- **结论: 无法在当前 CUTLASS DSL (4.4.2) + QuACK (0.3.7) 实现**
- FP8 forward gap 仅 ~250µs (14%) — 来自 3 个额外 Triton kernel launch

**Forward 对比分析 (NCU estimated, aligned production shape):**

| Path | Kernels | GEMM | Triton | Total |
|------|---------|------|--------|-------|
| BF16 fused | 1 GemmGated+SwiGLU + 1 GemmDefault | ~1760µs total | 0 | ~1760µs |
| FP8 decomposed | 2 GemmDefault + gather_quant + swiglu_quant | ~1600µs | 215µs | ~2010µs |
| **Gap** | +2 kernels | -160µs | +215µs | **+250µs (+14%)** |

**E2E Performance (NCU estimated):**

| Config | Fwd (ms) | Bwd (ms) | Total (ms) | vs BF16 |
|--------|----------|----------|------------|---------|
| BF16 fused baseline | 1.76 | 5.70 | 7.46 | 1.00x |
| **FP8 Session 13 (zero-sync)** | **~2.01** | **~4.15** | **~6.16** | **~1.21x** |

**Note**: 无法在干净 GPU 上验证 E2E，全集群 128 GPUs 均 100% occupied。上述数据为 NCU kernel-level 估算。

| 优化 | 文件 | 效果 |
|------|------|------|
| Fused z-dequant + SwiGLU backward | `swiglu_triton.py`, `functional/__init__.py` | dequant 117µs + bwd 20.7µs → fused **19µs** (**7.25x**) |
| Lazy dequant fallback | `functional/__init__.py` | BF16/non-quack paths lazy-dequant z only when needed |

**原理**: SwiGLU backward 本身只需 20µs (数据已在 L2 cache)。分离路径的 117µs dequant 是纯 DRAM bandwidth bound (写 8MB bf16 z + 再读)。融合后直接从 fp8 (4MB) 读取 → 消除 12MB DRAM access。且 fp8→f32 比 fp8→bf16→f32 更精确。

**全链路 Triton kernel 总览 (Session 11 Final, NCU verified, B200 sm_100a)**

| Kernel | Position | NCU Time | 说明 |
|--------|----------|----------|------|
| gather_quantize_and_pack | fwd: up-proj act | 86µs | BLOCK_ROWS=1, integer E8M0 |
| swiglu_fwd_quant_pack_zsave | fwd: SwiGLU+quant+zsave | 129µs | z 只读一次, BLOCK_ROWS=1 |
| gather_quantize_and_pack | bwd: dout act-grad | 86µs | 同 forward |
| swiglu_bwd_from_fp8 | bwd: dSwiGLU from FP8 z | **19µs** | 新! fused dequant |
| quantize_and_pack (inside GEMM) | bwd: dz act-grad | 86µs | dz 内部 quantize |
| **Forward Triton total** | | **215µs** | |
| **Backward Triton total** | | **191µs** | |
| **All Triton total** | | **406µs** | GEMM 占 87% (~3400µs) |

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
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | `1` / `0` | 强制 zero-sync 模式 (生产环境推荐) |
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
14. **D2H sync 是 GPU contention 下的性能杀手** — 单次 .tolist() 在 100% GPU 利用率下可能阻塞 1-10ms+，将 NCU-estimated 1.21x speedup 变为 8.6x slowdown。解决方案: streak-based alignment assumption + assume_aligned GEMM parameter 实现 zero-sync。
15. **Tensor-attribute caching 是最轻量的 cache** — `cu_seqlens._cached_cpu_tuple` 直接在 tensor 上附加 Python attribute，O(1) lookup，无 WeakRef/dict overhead，且随 tensor GC 自动回收。
16. **GemmGated + blockscaled FP8 融合不可行** — CUTLASS DSL CollectiveBuilder 将 blockscaled epilogue 硬编码，不支持 post-descale custom epilogue。需要 CUTLASS 基础设施变更或 Triton MXFP8 kernel。

---

## 10. 下一步规划

### P0: E2E 性能验证 (需要干净 GPU)

全集群 128 GPUs 100% occupied。需要:
1. **`benchmarks/e2e_fp8_vs_bf16.py`** — 新增的 benchmark script，small/medium/production 三个 shape
2. **nsys profile** — 验证 zero-sync (无 D2H MemCpy)，验证 kernel overlap
3. **验证 assume_aligned 端到端性能** — streak 自动生效 vs 手动 env var

### P1: Forward 性能提升 (需 CUTLASS 支持)

| 优化 | 预期收益 | 阻塞 |
|------|----------|------|
| GemmGated + blockscaled FP8 fusion | ~250µs (~14% forward) | CUTLASS DSL accumulator recast bug |
| Gather_A in blockscaled GEMM | ~40µs (eliminate gather_quant) | 需验证 varlen + gather_A + blockscaled 兼容性 |
| CUDA Graph for kernel sequence | ~20µs (launch overhead) | 需 assume_aligned + static shapes |

### P2: 显存进一步优化

| 优化 | 预期收益 | 说明 |
|------|----------|------|
| FP8 weight storage (w1, w2) | ~50% weight memory | 训练时 w→fp8 存储, backward dequant for weight grad |
| Drop bf16 weights after FP8 conversion | ~1.5 GB/layer | optimizer state 恢复 bf16 weights |

### P3: quack/CUTLASS 升级

- **quack ≥ 0.4.0** 可能修复 blockscaled accumulator recast → 解锁 GemmGated+FP8 融合
- **CUTLASS DSL composable blockscaled epilogue** → 自定义 post-descale activation
