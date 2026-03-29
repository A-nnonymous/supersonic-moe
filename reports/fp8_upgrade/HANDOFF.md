# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-03-29, Session 5 (decomposed full-chain + fused bug diagnosis)**

---

## 1. 当前状态：一句话

**全链路 blockscaled FP8 (1×32 UE8M0) forward + backward 功能完成，11/11 contract tests PASS，精度 RelRMSE 5-7%。性能上 FP8 decomposed 路径比 BF16 fused 慢 19% (E2E 9.03ms vs 7.29ms)，原因是 fused GemmGated+blockscaled 有 CUTLASS DSL 深层 codegen bug 无法使用，被迫用 2-kernel decomposed 替代。性能优化是下一步唯一重点。**

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

## 3. 性能（未达标，核心瓶颈）

### E2E Latency (fwd+bwd, T=4096 H=4096 I=1024 E=128 K=8, B200 sm_100a)

| Config | Median (ms) | 相对 BF16 |
|--------|------------|-----------|
| BF16 fused baseline | 7.29 | 1.00x |
| FP8 decomposed (当前) | 9.03 | 0.81x (慢 19%) |
| FP8 理论极限 (2x compute) | ~3.65 | 2.00x |

### 分项拆解

| 阶段 | BF16 (ms) | FP8 (ms) | 差距来源 |
|------|-----------|----------|----------|
| Forward | 1.74 | 3.38 | **+1.64ms**：BF16 用 1 个 fused kernel (gemm_gated)，FP8 用 2 个 (GEMM + SwiGLU) |
| Backward | 5.70 | 5.66 | 持平 |

**Forward 是唯一瓶颈**，backward 已经做到持平。

### 多 Shape 性能数据

| Shape | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| T=4096 H=4096 I=1024 E=128 | 1.76 | 3.37 | 0.52x |
| T=8192 H=4096 I=1024 E=128 | 2.68 | 5.49 | 0.49x |
| T=16384 H=4096 I=1024 E=128 | 3.66 | 9.75 | 0.38x |
| T=32768 H=4096 I=1024 E=128 | 6.42 | 18.28 | 0.35x |
| T=4096 H=4096 I=2048 E=128 | 2.94 | 4.23 | 0.70x |

**Insight**: 越大的 T (更多 tokens)，FP8 decomposed 的劣势越大，因为 SwiGLU Triton kernel 是 element-wise 的纯 bandwidth-bound 算子，随 T 线性增长。I 越大 (更大中间维度) FP8 优势越明显 (0.70x vs 0.52x)，因为 GEMM 计算密度更高。

### 显存

| Config | Peak Memory |
|--------|-------------|
| BF16 baseline | 5.06 GB |
| FP8 decomposed | 7.43 GB (1.47x) |

FP8 显存更大因为：decomposed 路径需要 z(TK, 2I) bf16 中间量 + FP8 weight cache + activation quantization buffers。

---

## 4. GEMM 矩阵

| # | 算子 | 实现 | Dtype | 说明 |
|---|------|------|-------|------|
| 1 | up-proj fwd | `blockscaled_fp8_gemm_varlen` + `swiglu_forward_triton` | FP8→bf16 | 2 kernels |
| 2 | down-proj fwd | `blockscaled_fp8_gemm_varlen` | FP8 | 1 kernel |
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

| 修复 | 文件 | 说明 |
|------|------|------|
| `get_c_pointers` / `get_mlir_types` monkey-patch | `fp8_quack_patch.py` | CUTLASS dtype classes (BFloat16 等) 作为 Constexpr 字段时，`get_c_pointers` 递归遍历调用 `__c_pointers__()` 作为 unbound method → TypeError。Fix: 检测 `isinstance(obj, type)` 跳过。 |
| `fma_packed_f32x2` API 迁移 | `gemm_dgated.py` | quack 0.3.7 将 `utils.fma_packed_f32x2` 移至 `cute.arch.fma_packed_f32x2` |
| Decomposed FP8 forward | `functional/__init__.py:468-485` | `blockscaled_fp8_gemm_varlen` + `swiglu_forward_triton` 替代 fused `gemm_gated` |
| Decomposed FP8 backward | `functional/__init__.py:788-822` | `blockscaled_fp8_gemm_varlen` + `swiglu_backward_triton` 替代 fused `gemm_dgated` |
| 消除 `dz.copy_` 冗余 | `functional/__init__.py:775` | FP8 路径不再 pre-allocate dz，直接用 Triton 返回值 |
| `epi_setup_postact` override | `gemm_gated.py`, `gemm_dgated.py` | CopyUniversalOp fallback (功能正确但不足以修复 fused crash，保留为文档) |

---

## 7. 关键代码文件

| 文件 | 内容 |
|------|------|
| `sonicmoe/functional/__init__.py` | FP8/BF16 调度中枢：`_UpProjection.forward` (~468), `_DownProjection.backward` (~788) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | `blockscaled_fp8_gemm_varlen` (~1281), `precompute_weight_fp8` (~1142), `quantize_and_pack_activation_varlen` (~1072) |
| `sonicmoe/quack_utils/swiglu_triton.py` | `swiglu_forward_triton` (~52), `swiglu_backward_triton` (~220), fused quant variants (~123, ~342) |
| `sonicmoe/quack_utils/gemm_gated.py` | `GemmGatedSm100` + `epi_setup_postact` override |
| `sonicmoe/quack_utils/gemm_dgated.py` | `GemmDGatedSm100` + `epi_setup_postact` override + `fma_packed_f32x2` fix |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | `get_c_pointers`/`get_mlir_types` monkey-patch |
| `sonicmoe/quack_utils/gemm_interface.py` | `gemm_gated` / `gemm_dgated` tuned wrappers (转发 `a_scales`/`b_scales`) |
| `tests/fp8_large_project_contract_test.py` | 11 contract tests |
| `tools/_smoke_test.py` | 快速 FP8 vs BF16 forward+backward 精度验证 |
| `tools/_benchmark.py` | E2E latency benchmark |
| `tools/_benchmark_split.py` | Forward-only / backward-only 分拆 benchmark |

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

| 包 | 版本 |
|---|------|
| `quack-kernels` | 0.3.7 |
| `nvidia-cutlass-dsl` | 4.4.2 |
| `torch` | 2.9.1+cu128 |
| `triton` | 3.5.1 |

---

## 9. 教训

1. **CUTLASS DSL accumulator recast 不感知 blockscaled 物理布局** — 这是根本限制，不是 TileStore bug。之前多轮排查（5个隔离测试 + CopyUniversalOp override + postact disable）最终定位。
2. **quack 0.3.7 有大量 undocumented API break** — `fma_packed_f32x2` 位移、`get_c_pointers` 对 dtype class 的处理、`ArgumentsBase` 移除。
3. **QuACK interleaved layout** — `w1(2I, H, E)` 的行是 `[gate0, val0, gate1, val1, ...]`，不是 `[gate0...gateI, val0...valI]`。非 fused path 必须用 `z[:,0::2]`/`z[:,1::2]` 而非 `z.chunk(2)`。
4. **Triton `tl.dot_scaled` 在 SM100a + Triton 3.5.1 不可用** — `NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'`。需要 Triton >= 3.6.0 或手写 PTX。
5. **blockscaled_fp8_gemm_varlen K 维度必须是 128 的倍数** — 否则报错 `K must be multiple of 128`。
6. **Decomposed backward 与 BF16 fused 性能持平** — 因为 backward 的 fused `gemm_dgated` 实际上也是 bandwidth-bound（dSwiGLU 操作在 epilogue 中占比小）。性能瓶颈只在 forward。

---

## 10. 下一步规划

### P0: Forward 性能优化（唯一重大瓶颈）

Forward FP8 = 3.38ms vs BF16 fused = 1.74ms。差距 1.64ms 来自：
- blockscaled GEMM 内部 activation quantization (~0.3ms est.)
- SwiGLU Triton kernel 独立 launch (~0.5ms est.)
- 额外 HBM roundtrip: GEMM 写 z 到 HBM → SwiGLU 读 z 从 HBM (~0.8ms est.)

**方案 A: CUDA Graph** — 将 quantize + GEMM + SwiGLU 捕获为一个 graph，消除 CPU launch overhead。预期提升 ~0.2ms。

**方案 B: Fused SwiGLU+Quantize Triton kernel** — `swiglu_forward_quant_triton` 已存在，输出 y1_fp8 + scales。通过 `_PREQUANTIZED_SCALES` 传递给 down-proj 避免 down-proj 内重复量化。需要适配 varlen ISA scale packing。预期消除 1 个 quantize kernel + 减少 HBM traffic。

**方案 C: 写 Triton MXFP8 fused GEMM+SwiGLU** — 需要 Triton >= 3.6.0 的 `tl.dot_scaled` 支持，或手写 `tcgen05.mma` PTX inline asm。收益最大（理论上可达 BF16 fused 的 2x），但工程量最大。

**方案 D: 等 quack >= 0.4.0** — 可能修复 blockscaled accumulator recast。风险：不确定时间线。

### P1: 显存优化

- **Save z in FP8**: 目前 z(TK, 2I) bf16 占 ~TK*2I*2 bytes。用 `swiglu_forward_quant_triton` 的变体量化后存储 z_fp8 + scales，backward 中 `swiglu_backward_triton` 改为接受 FP8 输入并 inline dequant。预期节省 ~50% z 存储。
- **Fused gather+quantize**: `gather_quantize_and_pack_activation` 已存在，消除 bf16 `x_gathered` 中间量。

### P2: NCU Profiling

每个 kernel 的精细分析，找出计算密度 / bandwidth 利用率 / launch overhead 的具体占比。
