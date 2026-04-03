# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-03 (Session 35 — native FP8 + SonicMoE alignment analysis)
> **Branch:** `native-fp8-exploration`
> **Status:** Native FP8 is **1.082× faster** (GPU projection, same-node A/B). But **521µs FP8 overhead (14.2%) 全部来自非融合的独立 HBM kernel，违背 SonicMoE IO-aware 融合原则**。下一步：epilogue quant fusion。

---

## 0. One-screen summary

**What works today (frontier, `fork-main-sync`):**
- Zero-materialization FP8 forward via `GemmGatedSm100ZeroMat` (auto-selected in `gemm_gated()`)
- Zero-materialization FP8 backward via `GemmDGatedSm100ZeroMat` (auto-selected in `gemm_dgated()`)
- **Fused z+y1 2D-grid quant kernel** — single launch replaces 2 separate kernels (168µs combined)
- **Stream parallelism** — z-dequant overlaps with dout-quant + s.float() + scale_gather on separate CUDA streams
- Triton weight quantization — single kernel replaces 8-op eager path
- Z saved as FP8 (saves ~186 MiB at Ernie shape)
- Weight cache retention (fwd→bwd reuse, auto-invalidation via `w._version`)
- **31/31** tests pass (11 FP8LargeProjectContractTest + 20 FP8AlignedContractTest)

**What was explored (native FP8, `native-fp8-exploration`):**
- `enable_native_fp8()` context manager + forward path routing — **functionally works but is NOT truly native**
- `compute_scales_from_fp8_and_pack` kernel — **fundamentally broken**, removed (see Lesson #11)
- PostAct FP8 output from GemmGated — works but produces raw FP8 without blockscaled ISA scales
- FP8 wgrad — **blocked** (`blockscaled_fp8_gemm_varlen` only supports `cu_seqlens_m`, wgrad needs `cu_seqlens_k + A_idx`)
- **12/12** native FP8 tests pass (run separately from frontier tests)

**Minimal flags for training (frontier):**
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 python train.py
```
Or programmatically:
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True), enable_fp8():
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Performance

### GPU Projection (nsys, same node 0267 GPU0, back-to-back, Session 35)

Ernie shape T=8192 H=3072 I=1536 E=8 K=8, 10 warmup + 5 profiled:

| Metric | Official BF16 (quack 0.2.5) | Native FP8 (quack 0.3.7) | Ratio |
|--------|---------------------------|--------------------------|-------|
| **GPU total/iter** | **3969µs** | **3669µs** | **1.082× faster** |
| GEMM total | 3648µs | 2831µs | **1.289× faster** |
| FP8 overhead | 0 | 521µs | — |

### Kernel Breakdown (Session 35, node 0267 GPU0)

| Category | BF16 µs | FP8 µs | Delta | FP8 N/iter |
|----------|---------|--------|-------|------------|
| GemmDefault (BF16 wgrad+varlen) | 2390 | 1903 | -487 | 4 |
| Gated/DGated BF16 fused | 1259 | — | -1259 | — |
| Gated/DGated FP8 ZeroMat | — | 929 | +929 | 2 |
| quantize_and_pack (dout+dz) | — | 168 | +168 | 2 |
| fused_z_save_y1_quant | — | 168 | +168 | 1 |
| dequant_blockscaled_fp8 (z) | — | 129 | +129 | 1 |
| gather_isa_packed_scales | — | 55 | +55 | 2 |
| token_gather_sum | 143 | 143 | 0 | 2 |
| Other | 177 | 174 | -3 | — |
| **TOTAL** | **3969** | **3669** | **-300** | — |

GEMM savings: 817µs (22.4%). FP8 overhead: 521µs. Net: **296µs (7.6% faster)**.

x-quant kernel eliminated: `quantize_and_pack` dropped from 3→2 calls/iter (dout + dz only).

### Memory

| Metric | Official BF16 | Native FP8 | Delta |
|--------|--------------|-------------|-------|
| **FWD+BWD Peak** | 1658 MiB | 2130 MiB | **+472 MiB** |
| After BWD | 641 MiB | 888 MiB | +247 MiB |

FP8 peak > BF16 due to: FP8 weight buffers (~108 MiB), z FP8 save (~213 MiB), y1 pre-quant (~96 MiB). BF16 master weights kept for wgrad.

### Precision (Official BF16 gold vs Native FP8, production shape, same seed)

| Variable | Shape | RRMSE | Cosine |
|----------|-------|-------|--------|
| output | 8192×3072 | 6.51% | 1.0001 |
| dx | 8192×3072 | 7.04% | 0.9996 |
| d_c_fc.weight | 8×3072×3072 | 5.94% | 1.0115 |
| d_c_proj.weight | 8×3072×1536 | 6.51% | 1.0024 |
| d_router.weight | 8×3072 | 7.57% | 0.9971 |

MaxRelErr is large (>1e6) but **only at near-zero gold values** (|gold|<1e-8). For |gold|≥1e-4, MaxRelErr < 0.35. This is standard FP8 quantization behavior — not an anomaly.

### Previous Session 33 data (for reference)
Frontier FP8 on node 0342: 3690µs/iter vs BF16 3932µs/iter = 1.066×.

---

## 2. Cumulative Changes

### Session 34: Native FP8 exploration (current branch: `native-fp8-exploration`)

**Goal**: Full-chain FP8 params — x arrives as FP8, weights stored as FP8, no quantization inside MoE.

**What was implemented:**
- `enable_native_fp8()` context manager in `sonicmoe/functional/utils.py` — activates native mode + FP8 + QuACK GEMM
- `_native_fp8_gated_forward()` in `sonicmoe/functional/__init__.py` — dedicated forward path
- Forward routing in `_UpProjection.forward` line 712 — branches on `_native_fp8_enabled()`
- `tools/profile_native_fp8.py` — benchmark script comparing frontier vs native FP8
- `tests/fp8_native_params_test.py` — 12 precision/functional tests

**Critical finding: current native path is functionally identical to frontier FP8.**
The native path still calls `quantize_and_pack_activation(x)` (simulating upstream x-quant) and `precompute_weight_fp8_for_fused_gated(w1)` (cached weights). The same GEMM kernels fire. Timing comparison on fully idle node 0263 (30 warmup, 3 trials × 30 iters):

| Config | µs/iter (min) | vs BF16 |
|--------|--------------|---------|
| BF16 QuACK | 3877 | — |
| FP8 Frontier | 3935 | 0.985× |
| Native FP8 | 3962 | 0.979× |

Native ≈ Frontier within noise (0.993×). No meaningful latency or memory improvement.

**Why the initial approach was wrong:**
1. `compute_scales_from_fp8_and_pack` kernel was fundamentally broken — E8M0 scales encode the original BF16 magnitude range, not the clamped FP8 range. Computing scales from FP8 data produces scales off by ~12.8 bytes on average → total numerical garbage. **Removed.**
2. PostAct FP8 from GemmGated: CUTLASS clamp-casts float32→FP8 without producing blockscaled ISA scales. The missing scales made the y1 FP8 unusable for blockscaled GEMM.
3. x is still BF16 in MoE.forward() — we quantize inside MoE, not upstream.
4. Weights still stored as BF16 nn.Parameters — `precompute_weight_fp8*` caches are populated, not bypassed.

**True native FP8 requires (plan in `.claude/plans/shiny-beaming-knuth.md`):**
1. `NativeFP8Params` dataclass holding pre-computed FP8 weight views for all 4 GEMM paths
2. `MoE.prepare_native_fp8()` + `MoE.refresh_native_fp8()` for weight buffer lifecycle
3. `MoE.forward(x_fp8_data=..., x_fp8_scales=...)` API for pre-quantized x input
4. Thread params through call chain, skip all `precompute_weight_fp8*` and `quantize_and_pack_activation(x)` calls
5. Estimated savings: **~60µs** (x-quant) latency, **~480 MiB** memory (eliminate weight cache overhead)

### Session 33: Authoritative benchmarks on fully idle node

- **Benchmark node**: 0342 (10.51.203.75, 8/8 GPUs idle, 0% utilization)
- **Corrected performance**: BF16 3932µs/iter → FP8 3690µs/iter = **1.066× faster**
  - Previous 0344 data (6609 vs 6290µs) was inflated by GPU contention
- **Corrected memory**: FP8 peak 1913.8 MiB > BF16 peak 1411.8 MiB (+502 MiB)
  - Weight caches are the main cause; Z FP8 save alone saves 186 MiB
  - Previous claim "FP8 peak ≤ BF16" was incorrect at production Ernie shape
- **nsys artifacts**: `output/sonic-moe-profiling/session33_0342/`
- **Profiling runner**: `tools/_profiling_runner.sh` now supports `nsys_official_bf16`, `nsys_fp8_frontier`, `mem_fp8`, `mem_bf16` modes

### Session 32: Fused quant + stream parallel + official BF16 baseline

1. **Fused z+y1 2D-grid quant kernel** (`blockscaled_fp8_gemm.py`):
   - Rewrote `_fused_z_save_y1_quant_kernel` from 1D grid (2048 blocks) → 2D grid (2048, 20) = 40960 blocks
   - `pid_1 < z_col_blocks` → z path (raw E8M0 scales); `pid_1 >= z_col_blocks` → y1 path (ISA-packed)
   - Packed int32 z-scale writes: uncoalesced excess **24% → 6%**, DRAM throughput **49% → 67%**
   - Saves ~10µs/iter vs separate kernels
   - Integrated into `_UpProjection.forward()` line 639 (when `_save_z_fp8()` is true)

2. **s.float() pre-cast optimization** (`functional/__init__.py`):
   - Moved `s.float()` cast from GemmDGated call site to stream overlap window
   - 28µs elementwise kernel now fully hidden behind z-dequant side-stream

3. **Official BF16 baseline** (`tools/profile_both.py`):
   - Fixed: official `MoE.forward()` doesn't accept `use_fp8=` → use `moe(x)` directly
   - Fixed: `z.sum().backward()` produces non-contiguous dout → use `z.backward(dout)`
   - Both envs can now nsys profile on the same idle node in parallel

4. **5 new precision tests** (31 total):
   - `test_fused_z_save_y1_quant_matches_separate_multi_shape` — bit-exact at 3 shapes
   - `test_fused_z_save_y1_quant_roundtrip_precision` — quant→dequant RRMSE <5%
   - `test_fused_quant_full_moe_forward_backward_multi_shape` — full fwd+bwd at T=1024,8192
   - `test_stream_parallel_backward_deterministic` — 5 runs bit-identical dx
   - `test_backward_s_float_precast_precision` — precision + determinism at 2 shapes

### Session 31: z-quant tuning + cache retention + dead code cleanup

- Z quant grid: BR=32, GPB=12 (was BR=16, GPB=128) → 44µs faster
- Weight cache retention: removed eager eviction (saves ~89µs in benchmarks)
- Dead code: `_fp8_lean`, `_WGRAD_DW2_STREAM`, unused imports

### Sessions 25-30: Zero-mat kernels, Triton weight quant, memory optimizations

- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: zero-materialization FP8 CUTLASS
- Triton weight quant: 8-op eager → single kernel (eliminated 3136µs/iter overhead)
- z FP8 save, y1 pre-quant, three-step gather pipeline

---

## 3. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad** | +0.9ms net | Colwise quant SM contention + layout permute ~637µs |
| **FP8 wgrad via varlen** | Blocked | `blockscaled_fp8_gemm_varlen` only supports `cu_seqlens_m`, wgrad needs `cu_seqlens_k + A_idx` |
| **Eliminate z-dequant by feeding FP8 z to GemmDGated** | Blocked | CUTLASS asserts `PreAct.element_size()==2` (bf16 required) |
| **Fuse dz-quant into GemmDGated epilogue** | Blocked | Same CUTLASS PreAct constraint |
| **TMA for quant kernel scale writes** | 2.3× slower | Descriptor setup overhead > 6MB data volume |
| **TMA for quant kernel bf16 loads** | 2.4× slower | Block shape must be power-of-2; per-group overhead |
| **Fused z+y1 quant 1D grid** | +5µs regression | Only 2048 blocks → poor SM utilization |
| **FP8 down-proj at I=1536** | No net win | Quant cost ≈ GEMM savings at small I |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch storage bounds check |
| **Rowwise quant + strided view** | Not possible | HW requires contiguous K groups |
| **compute_scales_from_fp8_and_pack** | Numerical garbage | E8M0 scales encode BF16 magnitude, not FP8 magnitude. Mean error ~12.8 bytes. |
| **PostAct FP8 for blockscaled down-proj** | Scale mismatch | CUTLASS PostAct FP8 clamp-casts without producing ISA-packed blockscaled scales |

---

## 4. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, stream parallelism, native FP8 branch |
| `sonicmoe/functional/utils.py` | `enable_fp8()`, `enable_native_fp8()`, `enable_quack_gemm()` context managers |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant/dequant kernels, fused z+y1 kernel, weight caches |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat CUTLASS kernel classes |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated dispatch (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated dispatch (auto ZeroMat selection) |
| `sonicmoe/quack_utils/swiglu_triton.py` | Dequant kernel, SwiGLU fused variants |
| `sonicmoe/moe.py` | MoE class, `prefetch_all_fp8_weights()`, `forward()` |
| `tests/fp8_large_project_contract_test.py` | **31 tests** (11 project + 20 aligned) |
| `tests/fp8_native_params_test.py` | **12 tests** for native FP8 (run separately) |
| `tools/profile_both.py` | Dual-env nsys profiling (official BF16 + FP8 fork) |
| `tools/profile_native_fp8.py` | Native FP8 vs frontier benchmark |
| `tools/_profiling_runner.sh` | Unified SSH profiling runner (nsys, memory, tests) |
| `tools/cluster_idle_launch.py` | Find idle GPU nodes for clean benchmarks |

---

## 5. Validation

**31/31 frontier tests + 12/12 native tests** (run separately, ~50s each on B200):
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Frontier tests (31 tests)
CUDA_VISIBLE_DEVICES=<gpu> USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Native FP8 tests (12 tests — MUST run separately to avoid global state leak)
CUDA_VISIBLE_DEVICES=<gpu> USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_native_params_test.py -v --tb=short
```

**WARNING:** Do NOT run both test files in the same pytest invocation. The `enable_native_fp8()` context manager modifies process-global state (`_IS_NATIVE_FP8`, `_IS_FP8_ACTIVE`, `_IS_USING_QUACK_GEMM`) that leaks between test classes.

---

## 6. Insights for Next Agent

### Architecture insight
The FP8 pipeline has a fundamental constraint: **CUTLASS GemmDGated requires bf16 PreAct** (line 283 of `gemm_dgated.py`: `assert PreAct.element_size() == 2`). This means z must be dequantized from FP8 to bf16 before the backward GEMM, costing 130µs + 582MB I/O. This is the single largest blocked optimization. A CUTLASS epilogue that reads FP8 z with on-the-fly dequant would eliminate this, but requires non-trivial CUTLASS DSL work.

### Performance insight
77% of FP8 GPU time is GEMMs (down from 92% in BF16 because of quant overhead). The wgrad GEMMs are BF16 (FP8 wgrad validated as net-negative). Further significant speedups require either:
1. FP8 wgrad with zero layout overhead (needs CUTLASS kernel accepting non-contiguous K groups)
2. Larger I (FP8 advantage scales with GEMM size — I=2048 gets ~2.35×)
3. Custom CUTLASS epilogue for FP8 z dequant (saves 130µs/iter)

### Native FP8 insight (Session 34)
**The "simulate native by quantizing inside MoE" approach provides zero value.** Both frontier and native paths fire identical kernels. The only real savings come from:
1. **Pre-quantizing x upstream** (outside MoE) — saves ~60µs x-quant kernel per forward
2. **Pre-computing weight FP8 buffers at init** — saves ~480 MiB weight cache memory
3. Neither requires a new code path in MoE forward — just (a) accept pre-quantized x as input and (b) use persistent FP8 weight buffers instead of dynamic caches

**Key technical discovery:** E8M0 blockscaled scales encode the **original BF16 magnitude** used during quantization, NOT the magnitude of the resulting FP8 data. You cannot reconstruct correct scales from FP8 data alone — the scale must be computed from the BF16 source and stored alongside the FP8 data. This is why `compute_scales_from_fp8_and_pack` was fundamentally broken (every byte off by ~12.8 on average).

**GEMM kernel interface:** All CUTLASS kernels (`gemm_gated`, `gemm_dgated`, `blockscaled_fp8_gemm_varlen`) accept pre-quantized FP8 weights + ISA-packed scales directly. The `precompute_weight_fp8*` functions are just convenience wrappers. If you store weights natively as FP8 + ISA-packed scales, you can bypass all weight quantization completely.

### Weight flow (critical for native FP8 implementation)
```
Storage: c_fc.weight (E, 2I, H) bf16, c_proj.weight (E, H, I) bf16
Permute: w1 = c_fc.weight.permute(1,2,0) → (2I, H, E) view
         w2 = c_proj.weight.permute(1,2,0) → (H, I, E) view

Forward fused gated (gemm_gated):
  precompute_weight_fp8_for_fused_gated(w1):
    (2I,H,E) → permute(2,1,0).mT.contiguous() → (E, 2I, H) contiguous
    quantize → fp8 (E,2I,H) + ISA scales along K=H
    return .mT view: (E,H,2I) fp8 + scales  [cached in _FUSED_WEIGHT_CACHE]

Forward down-proj (blockscaled_fp8_gemm_varlen):
  precompute_weight_fp8(w2):
    (H,I,E) → permute(2,0,1).contiguous() → (E,H,I) contiguous
    quantize → fp8 (E,H,I) + ISA scales along K=I  [cached in _VARLEN_WEIGHT_CACHE]

Backward actgrad (blockscaled_fp8_gemm_varlen):
  precompute_weight_fp8(w1.permute(1,0,2)):
    (H,2I,E) → permute(2,0,1).contiguous() → (E,H,2I) contiguous
    quantize → fp8 (E,H,2I) + ISA scales  [cached]

Backward dgated (gemm_dgated_kernel):
  precompute_weight_fp8_for_direct_fused_dgated(w2):
    (H,I,E) → (E,I,H) contiguous
    quantize → fp8 (E,I,H) + ISA scales along K=H  [cached]

Wgrad: ALWAYS BF16. Uses original bf16 parameters via quack.gemm().
```

### Memory insight
**FP8 uses more peak memory than BF16 at Ernie production shape** (+502 MiB). The Z FP8 save reduces activation memory by 186 MiB, but weight caches (3 caches × 3 weights × ~72MB each) add ~650 MiB. The true native FP8 approach would replace dynamic caches with persistent buffers (~216 MiB total), saving ~434 MiB.

### Profiling insight
- **Always use nsys GPU projection, not wall-clock** — CPU dispatch overhead is 40-60% on contested nodes
- **Use `tools/cluster_idle_launch.py scan`** to find idle nodes with 8/8 free GPUs
- **Official BF16 baseline needs `z.backward(dout)`**, not `z.sum().backward()` (non-contiguous gradient assertion)
- **GPU contention invalidates profiling data** — node 0344 (4/8 idle) showed 6609µs BF16; same workload on fully idle 0342 was 3932µs (1.68× inflation). Always use fully idle nodes.
- **nsys install**: `dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb` (required once per node restart)

---

## 7. SonicMoE 对齐分析 & 下一步

### 当前 FP8 与 SonicMoE 原则的差距

SonicMoE 核心：**所有操作融合进 GEMM kernel，零独立 HBM kernel。** 当前 FP8 有 5 个独立 HBM kernel（521µs/iter, 14.2%），是"附加层"而非"融合层"。

| FP8 操作 | SonicMoE 理想 | 当前实际 | IO 浪费 |
|---------|-------------|----------|---------|
| x quant | GEMM prologue on-the-fly | 已通过 native x 消除 ✓ | 0 |
| z+y1 quant | GEMM epilogue on-the-fly | **独立 fused_z_y1 kernel** 168µs | 582 MB |
| z dequant | GEMM C-load on-the-fly | **独立 dequant kernel** 129µs | 582 MB |
| scale gather | GEMM TMA co-load | **独立 gather kernel** 55µs | ~2 MB |
| dout/dz quant | 上下游传递预量化 gradient | **独立 kernel ×2** 168µs | 96 MB |

**唯二符合 SonicMoE 的设计：** ZeroMat A_idx gather 融合 + stream 并行 z-dequant||dout-quant。

### 融合路线图（按可行性排序）

| 目标 | 消除 kernel | 省延迟 | 省 IO | 方法 | 难度 |
|------|-----------|--------|-------|------|------|
| **epilogue quant** | fused_z_y1_quant (168µs) | **168µs** | 582 MB | 扩展 GemmGated epilogue visitor：compute z/y1 后直接 blockscaled quant + ISA pack in registers | **中 — GemmGated epilogue 已有 SwiGLU 融合先例** |
| prologue dequant | dequant_fp8 (129µs) | 129µs | 582 MB | 新 GemmDGated kernel: C-load FP8+scales, register dequant | 高 — 需要新 kernel class |
| gradient pre-quant | quant ×2 (168µs) | 168µs | 96 MB | 系统级 native FP8 gradient | 高 — 框架级 |

### 下一步：Epilogue Quant Fusion

**目标：** 将 `fused_z_save_y1_quant` (168µs) 融合进 GemmGated epilogue。

**原理：** GemmGated 的 epilogue 已经计算了 SwiGLU(z) → y1。此时 z 和 y1 都在 registers 中。如果在 epilogue 中直接做：
1. 对 z 的 register 值计算 per-group amax → E8M0 scale → FP8 clamp-cast → ISA pack
2. 对 y1 的 register 值做同样操作
3. 将 FP8 data 写入 epilogue D tensor (替代 bf16 write)
4. 将 ISA-packed scales 写入单独的 scale buffer

就能消除 168µs 的独立 kernel + 582 MB 的 HBM 读（当前 fused kernel 需要从 HBM 重读 z 和 y1）。

**关键文件：**
- `sonicmoe/quack_utils/gemm_gated.py` — GemmGatedMixin.epi_visit_subtile
- `quack/gemm_sm100.py` — GemmSm100 epilogue pipeline
- `quack/epi_ops.py` — TileStore composable epilogue ops

**可行性：** GemmGated epilogue 的 `epi_visit_subtile` 已经在 register 中持有 z (via tRS_rD) 和 y1 (via postact computation)。增加一个 blockscaled quant + ISA pack step 是同类工作——核心难度在于 ISA tile layout 的 register→smem→HBM 写路径。

---

## 8. Environment

```
GPU: NVIDIA B200 (SM100a, 183GB HBM, ~8TB/s DRAM BW)
CUDA: 12.8, PyTorch: 2.9.1+cu128
FP8 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer (quack 0.3.7)
BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16 (quack 0.2.5)
FP8 codebase: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
BF16 codebase: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe
Profiling output: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/sonic-moe-profiling/
Session 33 data: .../session33_0342/ (nsys BF16+FP8, mem BF16+FP8, test results)
Session 34 data: .../session33_native/ (nsys native FP8, timing, precision, memory)
```

### Idle node discovery
```bash
# Always use this before profiling
python tools/cluster_idle_launch.py scan
# Pick nodes with 8/8 idle GPUs (0% utilization)
# Verified idle nodes as of session 34: 0263 (10.95.252.148), 0267 (10.51.203.76), 0350 (10.51.195.19)
```

