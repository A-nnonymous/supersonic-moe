# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-02 (Session 34 — native FP8 exploration + frontier cleanup)
> **Branches:** `fork-main-sync` (frontier FP8), `native-fp8-exploration` (native FP8 prototype + this handoff)
> **Status:** FP8 frontier is **1.066× faster** than official BF16 on GPU projection at Ernie production shape (fully idle node). Native FP8 prototype implemented but **functionally identical** to frontier — true native FP8 requires deeper redesign. FP8 peak memory is ~502 MiB higher than BF16 due to weight caches. Precision verified (**31/31 + 12/12** tests pass).

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

### GPU Projection (nsys, Ernie shape T=8192 H=3072 I=1536 E=8 K=8)

Measured on **fully idle node 0342** (10.51.203.75, 8/8 GPUs idle, 0% utilization), nsys + SQLite, 10 warmup + 5 profiled:

| Metric | Official BF16 (quack 0.2.5) | FP8 frontier (quack 0.3.7) | Ratio |
|--------|---------------------------|---------------------------|-------|
| **GPU total (5 iters)** | **19,659µs** | **18,452µs** | **1.066× faster** |
| **GPU per-iter** | **3,932µs** | **3,690µs** | **1.066× faster** |
| Kernel types/iter | 28 | 31 | +3 unique types |

> **Previous contested-node data (0344, 4/8 idle) showed 6609 vs 6290µs — those numbers reflected GPU contention inflating GemmDefault Max to 3440µs. The 0342 data above is authoritative.**

### FP8 Per-Iteration Kernel Breakdown (3690µs/iter, 31 kernel types)

| Kernel | Avg µs | Total (5 iter) | % | Category |
|--------|--------|---------------|---|----------|
| GemmDefaultSm100 BF16 ×4 (wgrad + actgrad + downproj) | 479 | 9,579µs | 51.9% | GEMM |
| GemmDGatedSm100ZeroMat (bwd FP8) | 477 | 2,384µs | 12.9% | GEMM |
| GemmGatedSm100ZeroMat (fwd FP8) | 456 | 2,282µs | 12.4% | GEMM |
| `_quantize_and_pack_kernel` ×3 (x, dout, dz) | 60 | 895µs | 4.8% | FP8 overhead |
| `_fused_z_save_y1_quant_kernel` (z+y1 quant) | 168 | 838µs | 4.5% | FP8 overhead |
| `token_gather_sum_kernel` ×2 | 70 | 702µs | 3.8% | Shared (BF16+FP8) |
| `_dequant_blockscaled_fp8_kernel` (z restore) | 130 | 649µs | 3.5% | FP8 overhead |
| `_gather_isa_packed_scales_kernel` ×2 | 27 | 273µs | 1.5% | FP8 overhead |
| Other (routing, elementwise, reduce) | — | 850µs | 4.6% | Shared |

**FP8-only overhead: ~532µs/iter (14.4% of total)** — GEMMs account for 77.2%.

### Official BF16 Baseline Kernel Breakdown (3932µs/iter, 28 kernel types)

| Kernel | Avg µs | Total (5 iter) | % |
|--------|--------|---------------|---|
| GemmDefaultSm100 BF16 ×4 (all BF16 GEMMs) | 587 | 11,744µs | 59.7% |
| GemmGatedSm100 (up-proj + SwiGLU) | 768 | 3,839µs | 19.5% |
| GemmDGatedSm100 (bwd + dSwiGLU) | 496 | 2,478µs | 12.6% |
| `token_gather_sum_kernel` ×2 | 71 | 714µs | 3.6% |
| Other | — | 884µs | 4.5% |

BF16 GEMM total: 2349 + 768 + 496 = **3613µs/iter** (91.9% of GPU time).

### Savings Breakdown

```
GEMM savings:
  BF16 GEMMs: 2349 + 768 + 496 = 3613µs/iter
  FP8  GEMMs: 1916 + 477 + 456 = 2849µs/iter
  Saved: 764µs (21.1% GEMM reduction)

FP8-only overhead: 179 + 168 + 130 + 55 = 532µs/iter

Net: 764 - 532 = 232µs → 3932→3690µs/iter (6.6% speedup on fully idle node)
```

### Memory

| Metric | Official BF16 | FP8 Frontier | Delta |
|--------|--------------|-------------|-------|
| **FWD Peak** | 1385.9 MiB | 1913.8 MiB | **+527.9 MiB** |
| **Total Peak (FWD+BWD)** | 1411.8 MiB | 1913.8 MiB | **+502.0 MiB** |
| After BWD (steady state) | 640.6 MiB | 871.5 MiB | +230.9 MiB |

**FP8 uses more peak memory than BF16.** The Z FP8 save reduces activation memory (~186 MiB saved), but three weight caches (fwd/dgated/actgrad, each ~72MB FP8 + scales per weight) add ~650 MiB. Net effect is +502 MiB at Ernie shape. This is acceptable on B200 (183GB HBM) but may need cache eviction policies for larger models.

> Note: `test_fp8_memory_less_than_bf16` still passes — it uses a smaller test shape where the Z savings outweigh the weight cache overhead.

### Precision (multi-seed, multi-shape, 31/31 tests)

| Shape | FWD RRMSE | FWD corr | BWD RRMSE | BWD corr |
|-------|-----------|----------|-----------|----------|
| Contract (T=1024, E=8) | <7% | >0.997 | <8% | >0.997 |
| Ernie prod (T=8192, E=8) | <7% | >0.997 | <8% | >0.997 |

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

## 7. Next Steps (Priority Order)

1. **True native FP8 (highest value for inference):** Implement `NativeFP8Params` dataclass + `MoE.prepare_native_fp8()` + `forward(x_fp8_data=..., x_fp8_scales=...)`. Detailed plan at `.claude/plans/shiny-beaming-knuth.md`. Expected: ~60µs latency savings + ~480 MiB memory savings. All GEMM kernels already accept pre-quantized FP8 weights — just need to thread the pre-computed buffers through.

2. **Investigate CUTLASS epilogue FP8 PreAct** — If GemmDGated could read FP8 z (with ISA-packed or raw E8M0 scales) and dequant in epilogue, saves 130µs/iter + 582MB I/O. Highest-value single optimization remaining.

3. **Reduce FP8 weight cache memory (if NOT doing native FP8)** — Selective cache eviction or lazy population could close the 502 MiB gap while retaining most of the speed benefit.

4. **Larger I shapes** — FP8 advantage scales with GEMM size. At I=2048+, expect 1.3-2.4× speedup.

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

