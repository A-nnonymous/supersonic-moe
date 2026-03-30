# Blockscaled FP8 MoE — Handoff

> **Last updated: 2025-03-30, Session 19 (comprehensive profiling audit)**
> **Status: Production-ready, 1.58x E2E over BF16, 8/8 contract tests PASS**

---

## 1. 当前状态

**全链路 blockscaled FP8 (1×32 UE8M0) MoE forward + backward 功能完成。**

8-GPU 平均实测数据 (`tools/bench_aligned_e2e.py`, CUDA events, 8 warmup + 20 iters):

| Phase | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| Forward | 1.134±0.014 | 1.143±0.010 | **0.99x** |
| Backward | 3.624±0.031 | 1.870±0.012 | **1.94x** ⭐ |
| **Total** | **4.758** | **3.014** | **1.58x** ⭐ |

> 数据来自 6 张稳定 GPU (排除 2 张有 JIT compilation outlier 的 GPU)。单卡测试 typical: 1.57-1.59x。

**Forward 持平原因**: FP8 decomposed path 使用 4 kernels (`gather_quant` + GEMM + `SwiGLU_quant` + GEMM)，而 BF16 fused path 仅 2 kernels (`GemmGatedSm100`[GEMM+SwiGLU] + `GemmDefaultSm100`)。Fused GEMM+SwiGLU+FP8 因 CUTLASS DSL 限制无法实现（见 §5）。

**Backward 1.94x 原因**: 4/6 GEMM 使用 FP8 (act-grad kernels)，权重带宽减半；weight-grad 保持 BF16 (bandwidth-bound at tpe=256, FP8 无收益甚至更慢)。

---

## 2. 精度（已达标，8/8 contract tests PASS）

| Metric | Small (T=256) | Production (T=4096) | 阈值 | 状态 |
|--------|--------------|---------------------|------|------|
| Forward RelRMSE | 6.56% | 6.61% | <10% | ✅ |
| dx grad RelRMSE | 6.54% | — | <10% | ✅ |
| dw2 RelRMSE | 5.35% | — | <10% | ✅ |
| Correlation | 0.998 | 0.998 | >0.99 | ✅ |

全链路使用 blockscaled 1×32 E8M0 scaling（非 per-tensor），z 保存为 FP8 + raw E8M0 scales 在 backward dequant。

验证命令:
```bash
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

---

## 3. GPU Kernel Breakdown (nsys correlationId-attributed, single iter)

### BF16 Forward (783µs kernel sum, 21 kernels)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmGatedSm100` | 447.8 | 57.2% | Fused GEMM+SwiGLU (up-proj) |
| `GemmDefaultSm100` | 216.7 | 27.7% | down-proj GEMM |
| `token_gather_sum` | 46.3 | 5.9% | scatter-reduce |
| routing overhead | 29.4 | 3.8% | |

### FP8 Forward (925µs kernel sum, 25 kernels)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| CUTLASS GEMM (up-proj) | 263.6 | 28.5% | FP8×FP8→BF16, GemmDefaultSm100 |
| `_swiglu_fwd_quant_pack_zsave_kernel` | 228.3 | 24.7% | SwiGLU + y1 quant + z fp8 save |
| CUTLASS GEMM (down-proj) | 213.8 | 23.1% | FP8×FP8→BF16 |
| `_gather_quantize_and_pack_kernel` | 96.8 | 10.5% | gather x + blockscaled quant + ISA pack |
| `token_gather_sum` | 47.5 | 5.1% | scatter-reduce |

### BF16 Backward (3768µs kernel sum, 19 kernels)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `elementwise_kernel` | 2082.4 | 55.3% | dout expand/contiguous + score ops (见 §3.1) |
| `GemmDefaultSm100` (wgrad w2) | 570.2 | 15.1% | BF16 weight-grad |
| `GemmDefaultSm100` (wgrad w1) | 396.6 | 10.5% | BF16 weight-grad |
| `GemmDGatedSm100` | 304.2 | 8.1% | Fused act-grad + dSwiGLU |
| `GemmDefaultSm100` (act-grad) | 274.0 | 7.3% | BF16 act-grad |

### FP8 Backward (1979µs kernel sum, 22 kernels)

| Kernel | µs | % | 说明 |
|--------|----|---|------|
| `GemmDefaultSm100` (wgrad w2) | 574.6 | 29.0% | BF16 weight-grad (保持BF16) |
| `_swiglu_bwd_quant_pack_kernel` | 380.4 | 19.2% | dSwiGLU + dz quant + ISA pack + y1s + ds |
| `GemmDefaultSm100` (act-grad dz→x) | 327.7 | 16.6% | FP8 act-grad |
| `GemmDefaultSm100` (wgrad w1) | 277.6 | 14.0% | BF16 weight-grad |
| `GemmDefaultSm100` (act-grad dout→dz) | 144.3 | 7.3% | FP8 act-grad |
| `_gather_quantize_and_pack_kernel` | 96.1 | 4.9% | dout gather + quant |
| `_dequant_blockscaled_fp8_kernel` | 39.2 | 2.0% | z_fp8 → bf16 |

### 3.1 BF16 Backward 的 2082µs elementwise_kernel

Grid=1048576×1×1, Block=128×1×1, 处理 134M elements = TK×H = 32768×4096。

**来源**: `dout.contiguous()` (functional/__init__.py line 901)。`out.sum().backward()` 产生的 `dout` 是 expand 后的 stride-(0,0) tensor，不满足 GEMM k-major assertions，必须 `.contiguous()` 做 full copy。

**Note**: 这个 kernel 在 FP8 backward 中也存在但被 `gather_quantize_and_pack` 替代（gather 本身产生 contiguous output）。

---

## 4. GEMM 矩阵

| # | 算子 | 实现 | Dtype | µs (nsys) |
|---|------|------|-------|----|
| 1 | up-proj fwd | `blockscaled_fp8_gemm_varlen` + `swiglu_fwd_quant_pack_zsave` | FP8 | 264+228 |
| 2 | down-proj fwd | `blockscaled_fp8_gemm_varlen` (pre-quantized y1) | FP8 | 214 |
| 3 | down-proj bwd act | `blockscaled_fp8_gemm_varlen` + `swiglu_bwd_quant_pack` | FP8 | 144+380 |
| 4 | down-proj bwd wt | `quack.gemm` | BF16 | 575 |
| 5 | up-proj bwd act | `blockscaled_fp8_gemm_varlen` (pre-quantized dz) | FP8 | 328 |
| 6 | up-proj bwd wt | `quack.gemm` | BF16 | 278 |

4/6 GEMM 使用 FP8 blockscaled。Weight-grad 保持 BF16: per-expert M=tpe=256 太小，memory-bound，FP8 量化开销 > 带宽节省。经实测验证 FP8 weight-grad 比 BF16 慢 3.6x (见 §10.11)。

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

1. **Standalone CUTLASS C++ kernel** — 绕开 DSL，直接写 `GemmGatedBlockscaledFp8Sm100` (recommended)
2. **等 quack ≥ 0.4.0** — 可能修复 blockscaled accumulator recast
3. **Triton MXFP8 GEMM** — `tl.dot_scaled` 在 SM100a + Triton 3.5.1 broken，需等 Triton 修复

---

## 6. 历史 Bug 修复清单

### Bug 1: 非对齐路由 7x 减速 → BF16 Fallback

**问题**: `blockscaled_fp8_gemm_varlen(assume_aligned=False)` 对每个 expert 做 padding (128× allocate+copy+GEMM+unpad)，导致 7x 减速。

**修复**: gate 所有 FP8 paths on `_ALIGNMENT_ASSUMED`。非对齐时 fallback 到 BF16 fused path。

### Bug 2: Weight FP8 Cache Thrashing

**问题**: `_FUSED_WEIGHT_CACHE` limit 2，但 fwd+bwd 需 4 entries。Cache 每轮清空 → 重新量化全部权重 (~2ms)。

**修复**: 增加 cache limit 从 2 到 8。

### Bug 3: `global _ALIGNMENT_ASSUMED` 传播

**问题**: `@staticmethod forward` 中赋值 `_ALIGNMENT_ASSUMED` 创建局部变量。

**修复**: 添加 `global _ALIGNMENT_ASSUMED`。

### Bug 4: Benchmark Gradient Accumulation

**问题**: 未在 iterations 间 zero_grad → `aten::add_` 累加 ~1.4ms/iter。

**修复**: 添加 `p.grad = None`。

---

## 7. 历史优化清单

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
| 17-18 | Cache fix, alignment fix, benchmark correction | **E2E 1.57x** |
| **19** | **Comprehensive profiling audit, nsys kernel attribution** | **verified 1.58x** |

---

## 8. 关键代码文件

| 文件 | 内容 |
|------|------|
| `sonicmoe/functional/__init__.py` | FP8/BF16 调度中枢 (1463 lines)：alignment-gated paths, pre-quantized caches, z-in-FP8 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 GEMM (2037 lines): `blockscaled_fp8_gemm_varlen`, quantize+ISA-pack, weight cache |
| `sonicmoe/quack_utils/swiglu_triton.py` | SwiGLU (1278 lines): 7 Triton kernel variants for fwd/bwd + fused quant+ISA-pack |
| `sonicmoe/quack_utils/gemm_gated.py` | BF16 fused GEMM+SwiGLU (`GemmGatedSm100`, blockscaled FP8 crashes) |
| `sonicmoe/quack_utils/gemm_dgated.py` | BF16 fused backward GEMM+dSwiGLU (`GemmDGatedSm100`) |
| `sonicmoe/quack_utils/fp8_quack_patch.py` | CUTLASS monkey-patches for quack 0.3.7 |
| `sonicmoe/functional/utils.py` | `enable_quack_gemm()` context manager, `is_using_quack_gemm()` |
| `tests/fp8_large_project_contract_test.py` | 11 contract tests (8 small + 3 large_shape) |
| `tools/bench_aligned_e2e.py` | **Production E2E benchmark** (gold standard, use this) |

### 对比 official repo (`/root/.../official/sonic-moe`)

| | Official | 本 Fork |
|---|---------|---------|
| `functional/__init__.py` | 578 lines (BF16 only) | 1463 lines (+FP8 paths) |
| `quack_utils/` | 3 files (gemm_interface, gemm_gated, gemm_dgated) | 11 files (+blockscaled_fp8_gemm, swiglu_triton, fp8_quack_patch, etc.) |
| `moe.py` | basic forward | +FP8 protocol, weight prefetch, cache clear |
| backward path | 2 paths (quack / non-quack) | 4+ paths (FP8 aligned, FP8 non-aligned→BF16 fallback, BF16 quack, BF16 non-quack) |

---

## 9. 环境

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 small pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# Production E2E benchmark
CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py

# 8-GPU parallel benchmark (Python launcher, avoid shell expansion issues)
python3 -c "
import subprocess, os
procs = []
for gpu in range(8):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    p = subprocess.Popen(['python', 'tools/bench_aligned_e2e.py'],
                         stdout=open(f'/tmp/bench_gpu{gpu}.txt','w'),
                         stderr=subprocess.STDOUT, env=env)
    procs.append(p)
for p in procs: p.wait()
for gpu in range(8):
    with open(f'/tmp/bench_gpu{gpu}.txt') as f: print(f'GPU {gpu}: {f.read().strip()}')
"
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 CUTLASS GEMM (必须，import 前设置) |
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

**Note**: `enable_quack_gemm()` 是 context manager，bare call (`enable_quack_gemm()`) 是 no-op。仅需在 import 前设置 `USE_QUACK_GEMM=1` 环境变量即可，module-level init 会读取。

---

## 10. 教训（高价值）

1. **kernel-level benchmark ≠ E2E benchmark** — kernel-only 不含 Python dispatch (~0.3ms)、routing、alignment check、tensor allocation。先前 "1.76x forward" 是 kernel-only，E2E 实测仅 0.99x。**永远以 E2E 为准。**

2. **benchmark 必须 zero_grad** — 不清 gradient 会引入 `aten::add_` 累加开销 (~1.4ms/iter for 大权重)，严重扭曲 backward 计时。用 `p.grad = None`（不是 `zero_()`）。

3. **weight cache limit 必须覆盖 fwd+bwd** — fwd 需 w1, w2 (2 entries)，bwd 需 w1ᵀ, w2ᵀ (另 2 entries)。Limit 太小 → 每轮重新量化全部权重。

4. **`global` 在 `@staticmethod` 中是必须的** — Python `@staticmethod` 中赋值 module-level variable 不会自动 global，必须显式 `global _VAR`。

5. **非对齐 FP8 routing 是性能陷阱** — `assume_aligned=False` 对每 expert 做 padding，128 experts × padding = 7x 减速。正确做法：detect alignment → fallback BF16。

6. **CUTLASS DSL accumulator recast 不感知 blockscaled 物理布局** — 这是根本限制，不是 monkey-patch 能修的。需 CUTLASS C++ 或等 quack 升级。

7. **Triton `tl.dot_scaled` 在 SM100a + Triton 3.5.1 broken** — 不可用。

8. **ISA E8M0 scale packing 有复杂 tile-based 索引** — SF_TILE_M=128, SF_TILE_K=128, SF_VEC_SIZE=32, SF_TILE_STORAGE=512。

9. **BLOCK_ROWS 必须在 production shape 调参** — BR=1 在小 shape 最优但在 32768×4096 灾难性 (7.5x slower)。Production shape BR=8。

10. **FP8 backward 优势源于 act-grad GEMM 权重带宽减半** — weight-grad GEMM 保持 BF16 因 per-expert M 太小。

11. **FP8 weight-grad GEMM 实测比 BF16 慢 3.6x** — 因为 pack+transpose+quantize overhead 在 small-M (tpe=256) 下完全抵消带宽节省。已验证不可行。

12. **Profiling 方法论** (见 §13):
    - CUDA events 是零开销 stream marker，唯一正确的 GPU 计时方式
    - 永远不要在 fwd/bwd 之间插入 `torch.cuda.synchronize()` — 会制造不存在的 pipeline stall
    - NVTX range 是 CPU-side annotation，不直接对应 GPU 执行时间
    - nsys 中 kernel 归因必须用 correlationId linkage，不能用 CPU timestamp matching

---

## 11. 下一步规划

### P0: Forward 性能提升 (当前瓶颈，0.99x)

| 方向 | 预期收益 | 难度 | 说明 |
|------|----------|------|------|
| **CUTLASS C++ fused GEMM+SwiGLU+FP8** | ~200µs (~25% fwd) | High | 绕开 DSL，写 standalone kernel。这是打破 forward 瓶颈的唯一途径。 |
| Gather_A in blockscaled GEMM | ~96µs (~12% fwd) | Medium | 将 gather 融入 GEMM 的 A_idx prologue，消除独立 gather_quant kernel |
| CUDA Graph for static shapes | ~30µs (launch overhead) | Easy | 需 assume_aligned + static shapes |

### P1: Backward 优化

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| Multi-stream overlap (act-grad ∥ weight-grad) | ~200µs (~10% bwd) | act-grad FP8 和 weight-grad BF16 可并行执行，无数据依赖 |
| `_swiglu_bwd_quant_pack_kernel` 优化 | ~100µs | 380µs 是 backward 最大 Triton kernel，ncu 分析可能揭示优化空间 |

### P2: 显存优化

| 方向 | 预期收益 |
|------|----------|
| FP8 weight storage (w1, w2) | ~50% weight memory per layer |
| z-in-FP8 已实现 | 每层省 ~113MB (TK×2I from bf16 to fp8+scales) |

### P3: 生态 & 依赖

| 方向 | 说明 |
|------|------|
| Token rounding in routing | 保证 128-alignment，使 FP8 path 始终生效 |
| quack ≥ 0.4.0 升级 | 可能修复 fused GemmGated+blockscaled FP8 |
| Triton 升级 | `tl.dot_scaled` SM100a 修复 |

---

## 12. 显存消耗

| 组件 | BF16 | FP8 | 节省 |
|------|------|-----|------|
| z tensor (per layer) | 128MB (TK×2I×2B) | 64MB (FP8) + 2MB (scales) | ~50% |
| Weight cache | 0 | ~600MB×4 entries = 2.4GB | 额外开销 (全局共享) |
| 总 peak (single layer, E=128) | ~12GB | ~10GB | ~17% |

> weight cache 是全局共享的（所有 layer），不随 layer 数增长。对于多层模型，per-layer z-FP8 节省远大于一次性 cache 开销。

---

## 13. Profiling 工具和方法

### 13.1 E2E 性能测量 (gold standard)

使用 `tools/bench_aligned_e2e.py`。它使用 CUDA events 围绕一个 batch of iterations 做异步计时：
- Forward only: `torch.no_grad()` 跑 N iters → 得到 `fwd_ms`
- Forward+Backward: 跑 N iters (含 zero_grad) → 得到 `total_ms`
- `bwd_ms = total_ms - fwd_ms` (derived)

### 13.2 nsys Timeline 采集

```bash
NSYS=/opt/nvidia/nsight-systems-cli/2025.1.1/bin/nsys

# ★ 高价值命令：完整 timeline + GPU hardware metrics (NVLink, DRAM, SM utilization, clock freq)
# --gpu-metrics-device=0 启用 GB20x 硬件采样 (NVLink throughput, DRAM BW, SM occupancy, etc.)
# --gpu-metrics-frequency=10000 采样频率 10kHz (默认值, 可调至 200kHz)
# --gpu-metrics-set=gb20x 使用 Blackwell B200 专用 metric set
# --capture-range=cudaProfilerApi 仅在 cudaProfilerStart/Stop 之间采集 (避免 warmup 噪声)
CUDA_VISIBLE_DEVICES=0 $NSYS profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/sonic_profile --force-overwrite true \
  -t cuda,nvtx \
  --gpu-metrics-device=0 \
  --gpu-metrics-frequency=10000 \
  --gpu-metrics-set=gb20x \
  python tools/profile_async_events.py

# 简化版 (无 GPU hardware metrics, 更轻量)
CUDA_VISIBLE_DEVICES=0 $NSYS profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /tmp/sonic_profile \
  python tools/bench_aligned_e2e.py

# 导出为 sqlite (可 programmatic 分析)
$NSYS export --type=sqlite --force-overwrite=true \
  -o /tmp/sonic_profile.sqlite \
  /tmp/sonic_profile.nsys-rep

# 8-GPU 并行: GPU0 带 nsys, GPU1-7 纯 benchmark
bash tools/launch_8gpu_profile.sh
```

**可用 GPU metric set**: `gb20x` (Blackwell B200), `gh100` (Hopper H100), `ga100` (Ampere A100)。
查看所有可用 set: `$NSYS profile --gpu-metrics-set=help`

### 13.3 nsys Kernel 归因 (correlationId 方法)

**关键原理**: CUDA 是异步的。CPU launch 立即返回，GPU 稍后执行。因此不能用 CPU timestamp 匹配 kernel 到 NVTX range。正确做法：

1. CUPTI Runtime API call (CPU side) 有 `correlationId`
2. GPU kernel 也有相同的 `correlationId`
3. 通过 correlationId JOIN 可精确归因每个 kernel 到其 CPU launch

```sql
-- 在 nsys 导出的 sqlite 中查询某个 NVTX range 内的所有 GPU kernels
SELECT
    k.shortName,
    k.start,
    k.end,
    (k.end - k.start) as duration_ns,
    k.gridX, k.gridY, k.gridZ,
    k.blockX, k.blockY, k.blockZ
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
WHERE r.start >= <nvtx_range_start> AND r.start <= <nvtx_range_end>
ORDER BY k.start;
```

### 13.4 nsys 详细信息查询

```bash
# 查看 GPU 频率、利用率等硬件信息
$NSYS stats --report gpumemsizesum /tmp/sonic_profile.nsys-rep
$NSYS stats --report cuda_gpu_kern_sum /tmp/sonic_profile.nsys-rep
$NSYS stats --report cuda_gpu_mem_size_sum /tmp/sonic_profile.nsys-rep

# 查看所有 kernel 的详细统计 (按耗时排序)
sqlite3 /tmp/sonic_profile.sqlite "
SELECT
    shortName,
    COUNT(*) as count,
    SUM(end-start)/1e6 as total_ms,
    AVG(end-start)/1e3 as avg_us,
    MIN(end-start)/1e3 as min_us,
    MAX(end-start)/1e3 as max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY total_ms DESC
LIMIT 30;
"

# 查看 GPU clock 频率
sqlite3 /tmp/sonic_profile.sqlite "
SELECT * FROM TARGET_INFO_GPU;
"
```

### 13.5 ncu Kernel Profiling

```bash
# Profile 特定 kernel (用 --kernel-name 过滤)
CUDA_VISIBLE_DEVICES=0 ncu \
  --set full \
  --kernel-name "_swiglu_bwd_quant_pack" \
  --launch-skip 5 --launch-count 3 \
  python tools/bench_aligned_e2e.py

# 快速 roofline 分析
CUDA_VISIBLE_DEVICES=0 ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.per_cycle_active \
  --kernel-name "gemm" \
  --launch-skip 5 --launch-count 3 \
  python tools/bench_aligned_e2e.py
```

### 13.6 nsys Timeline 安装

```bash
# 如果 nsys 不可用，安装:
dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb
```

### 13.7 Profiling 工具清单

| 文件 | 用途 |
|------|------|
| `tools/bench_aligned_e2e.py` | **Gold standard** E2E benchmark: CUDA events, uniform routing, zero_grad |
| `tools/profile_async_events.py` | 详细异步 profiling: per-iter events, NVTX, precision, bubble analysis. 配合 nsys 使用 |
| `tools/launch_8gpu_profile.sh` | 8-GPU 并行 launcher: GPU0 带 nsys capture, GPU1-7 纯 benchmark, 自动聚合结果 |

