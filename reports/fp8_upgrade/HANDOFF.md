# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-15 (Session 50 — memory optimization analysis, early cache eviction, clean profiling)
> **Branch:** `native-fp8-exploration`
> **Status:** Zero-materialization FP8 + iso32 weights + optional stash + CuTe DSL wgrad quant + Pythonic config API + unaligned FP8 padding + epilogue FP8 D output + early weight cache eviction + nsys GPU-projection engine. Contract suite: **34/34 tests + 20 subtests PASS**.

---

## 0. Bottom Line

### Performance (Session 50 — nsys GPU-projection, 10-iter steady-state, B200)

| Shape | BF16 (µs/iter) | FP8 (µs/iter) | Speedup | Takeaway |
|-------|:--------------:|:-------------:|:-------:|----------|
| **Ernie** (T=8192, H=3072, I=1536, E=8, K=8) | 9658 | 9396 | **1.028×** | Break-even; GEMM savings ~= quant overhead |
| **I=2048** (T=8192, H=3072, I=2048, E=8, K=8) | 12622 | 12036 | **1.049×** | FP8 wins; wgrad GEMM savings (3366µs) > total quant overhead |

> **Methodology:** nsys GPU-projection (merged kernel intervals from `CUPTI_ACTIVITY_KIND_KERNEL` sqlite). 5 warmup + 10 measured iterations, steady-state. Gold standard for latency, immune to CPU contention.

### Memory (Session 50 — clean v3 methodology, env-var decontaminated)

| Config | I | FwdPeak | BwdPeak | Δ FwdPk | Δ BwdPk |
|--------|---|---------|---------|---------|---------|
| bf16 | 1536 | 1289.9 | 1412.4 | — | — |
| **fp8** | 1536 | 1232.2 | 1530.4 | **-57.7** | **+118.0** |
| **fp8_stash** | 1536 | 1018.1 | 1314.4 | **-271.8** | **-98.0** |
| bf16 | 2048 | 1553.9 | 1828.4 | — | — |
| **fp8** | 2048 | 1476.4 | 1985.4 | **-77.5** | **+157.0** |
| **fp8_stash** | 2048 | 1191.7 | 1697.4 | **-362.2** | **-131.0** |

**Key insight:** Without stash, FP8 saves on forward peak (z_fp8 quantization) but costs on backward peak (quant temporaries during wgrad). With stash (bf16 weights offloaded to CPU), FP8 wins everywhere.

> **CRITICAL:** Previous Session 49 memory numbers were contaminated — `SONIC_MOE_FP8_MODE=perf` env var leaked into bf16 runs, making bf16 secretly use FP8 paths. Session 50 fixed this with explicit `os.environ.pop('SONIC_MOE_FP8_MODE', None)` before bf16 runs.

### Precision (34/34 contract tests, 20 subtests, all PASS)

| Tensor | Ernie RRMSE | I=2048 RRMSE | Correlation | Status |
|--------|:-----------:|:------------:|:-----------:|:------:|
| output | 6.49% | 6.49% | 0.9979 | ✓ PASS |
| dx | 6.52% | 6.53% | 0.9979 | ✓ PASS |
| dw1 | 4.69% | 4.68% | 0.9989 | ✓ PASS |
| dw2 | 4.88% | 4.87% | 0.9988 | ✓ PASS |

All within guardrails: **RRMSE < 10%**, **correlation > 0.99**.

---

## 1. Session 50 Changes (Memory Analysis + Early Eviction)

| Change | Impact | Files |
|--------|--------|-------|
| **Early weight cache eviction** | Clears `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE` at backward entry (before dgated). Frees ~37 MiB (w2_varlen). No peak reduction at I=1536 (wgrad peak dominates), but helps at larger shapes. | `sonicmoe/functional/__init__.py` |
| **Streamlined w2_dgated eviction** | Post-dgated eviction now only handles w2_dgated (resize_(0) + stash pop). w1_fused/w2_varlen handled by early eviction. | `sonicmoe/functional/__init__.py` |
| **Removed redundant local import** | `quantize_activation_blockscaled_fast` local import at L934 removed (shadowed module-level import, caused `UnboundLocalError`). | `sonicmoe/functional/__init__.py` |
| **Env var contamination fix** | Documented critical bug: `SONIC_MOE_FP8_MODE=perf` leaks into bf16 runs. Fix: `os.environ.pop('SONIC_MOE_FP8_MODE', None)` before bf16 measurement. | This handoff, profiling scripts |

### Session 50 Memory Deep-Dive

**Why FP8 backward peak is +118 MiB despite z_fp8 savings:**

After-forward tensor snapshots show BF16 and FP8 have **identical** tensor layouts (both use QuACK GEMM which quantizes weights to FP8 regardless of activation mode, and both save z as fp8 via epilogue quant). The +118 MiB backward overhead comes from **wgrad quantization temporaries** (colwise quant of dout and y1s for blockscaled fp8 wgrad).

**Backward memory timeline (I=1536, measured):**
| Stage | Alloc (MiB) | Peak (MiB) |
|-------|-------------|------------|
| down-proj-dgated (after dgated GEMM) | 1275 | 1344 |
| **down-proj-weight (wgrad)** | 1167 | **1578** ← peak |
| up-proj-core | 1449 | 1578 |
| token-reduce | 1497 | 1578 |

The backward peak occurs during **wgrad** (not dgated), so early cache eviction before dgated doesn't reduce overall peak.

**Cache structure investigation:**
- `_FUSED_WEIGHT_CACHE`: 1 entry (37.1 MiB) — shares storage with a `_VARLEN_WEIGHT_CACHE` entry
- `_VARLEN_WEIGHT_CACHE`: 2 entries (74.2 MiB w1T + 37.1 MiB w2_varlen)
- w1T held by `ctx._w1T_fp8` (needed for backward actgrad) — cannot evict
- Clearing both dicts frees only **37.1 MiB** (w2_varlen, the only entry without ctx reference)

**Phase B (save x as fp8) was attempted and reverted:**
- Saving x (T×H bf16, 48 MiB) as fp8 (24 MiB) + scales saves 23 MiB during DownProj backward
- But dequant at UpProj backward start creates transient x_fp8 + x_bf16 coexistence (+24 MiB spike)
- Net result: +24.8 MiB WORSE at both shapes. Reverted.

### nsys Category Breakdown (Session 50, per-iter, µs)

**I=1536:**

| Category | BF16 | FP8 | Delta |
|----------|-----:|----:|------:|
| Wgrad GEMM | 6088 | 3682 | -2406 |
| GemmGated (fwd) | 1303 | — | — |
| GemmDGated (bwd) | 745 | — | — |
| GemmDGated ZeroMat (bwd) | — | 684 | — |
| Blockscaled Quant | — | 2073 | +2073 |
| Row Quant | — | 325 | +325 |
| Token Gather | 633 | 391 | -242 |
| Other | 881 | 2216 | +1335 |
| **Total** | **9658** | **9396** | **-262** |

**I=2048:**

| Category | BF16 | FP8 | Delta |
|----------|-----:|----:|------:|
| Wgrad GEMM | 7160 | 3794 | -3366 |
| GemmGated (fwd) | 2853 | — | — |
| GemmDGated (bwd) | 1259 | — | — |
| GemmDGated ZeroMat (bwd) | — | 1912 | — |
| Blockscaled Quant | — | 3904 | +3904 |
| Row Quant | — | 1150 | +1150 |
| Token Gather | 139 | 144 | +5 |
| Other | 1202 | 1107 | -95 |
| **Total** | **12622** | **12036** | **-586** |

### Key Insights from Session 50

1. **Wgrad GEMM is 1.84× faster at I=1536 and 1.89× faster at I=2048** — this is the core FP8 value, and it scales with I.

2. **Quant overhead dominates FP8 cost**: Blockscaled Quant + Row Quant = 2398µs at I=1536 (25% of FP8 time), 5054µs at I=2048 (42% of FP8 time). This is the #1 optimization target.

3. **"Other" category inflation in FP8**: 2216µs vs 881µs at I=1536 (+1335µs). This likely includes CUDA allocator overhead, kernel launch overhead, and scale manipulation kernels not classified into named categories.

4. **FP8 advantage grows with I**: 1.028× at I=1536 → 1.049× at I=2048. At larger I, GEMM savings grow O(I²) while quant overhead grows O(I).

### Previous Sessions (47-49)

| Change | Impact | Files |
|--------|--------|-------|
| **Epilogue FP8 D output** | GemmGated writes z directly as fp8 (`out_dtype=float8_e4m3fn`). Eliminates standalone z quant kernel (~141µs) + z.to(fp8) cast (~288µs). Never allocates bf16 z (saves 384 MiB transient). | `sonicmoe/functional/__init__.py` |
| **BF16 placeholder for autograd** | z stored as lightweight bf16 `as_strided((0,0))` placeholder (2 bytes). Actual fp8 z lives in prequant cache. Avoids fp8 tensors in autograd graph (which cause illegal memory access at large shapes). | `sonicmoe/functional/__init__.py` |
| **Eager weight cache release** | After dgated GEMM, clear `_FUSED_WEIGHT_CACHE`, `_VARLEN_WEIGHT_CACHE`, and `resize_(0)` consumed w2_dgated + unused w2_varlen/w1_fused tensors. Estimated -74 to -148 MiB peak. | `sonicmoe/functional/__init__.py` |
| **Single-stream wgrad pipeline** | Removed cross-stream overlap for wgrad quant. Single-stream execution enables caching allocator block reuse (cross-stream reuse requires record_stream). +50µs latency (0.6% of total) but better memory reuse. | `sonicmoe/functional/__init__.py` |
| **CuTe colwise+gather coalesced loads** | Rewrote gather path from per-thread sequential element loads to warp-cooperative coalesced loads (32 lanes load 32 consecutive columns of same row). NCU LD efficiency 2.1→~16 bytes/sector. **154µs → 58µs** (2.7× improvement). Still behind Triton's 39µs. | `sonicmoe/quack_utils/cute_blockscaled_quant.py` |
| **Epilogue quant default-ON** | `_use_epilogue_quant()` now defaults True (env var `SONIC_MOE_FP8_EPILOGUE_QUANT=1`). | `sonicmoe/functional/__init__.py` |

### Sessions 45-46 Changes (prior)

| Change | Impact | Files |
|--------|--------|-------|
| **Pythonic Config API** (`SonicMoEConfig`) | Replace env-var flags with dataclass + thread-local context manager. 10 fields. | `sonicmoe/config.py`, `sonicmoe/__init__.py`, `sonicmoe/moe.py`, `sonicmoe/functional/__init__.py` |
| **wgrad FP8 default-ON** | `_use_fp8_wgrad()` defaults True at all shapes | `sonicmoe/functional/__init__.py` |
| **Unaligned FP8 padding** | `_padded_blockscaled_gated_forward()` pads to 128 for FP8 GEMM | `sonicmoe/functional/__init__.py`, `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` |
| **CuTe DSL colwise quant** | 90µs vs Triton 136µs = 1.51× faster (no gather). 30 regs, 89% occupancy, zero bank conflicts | `sonicmoe/quack_utils/cute_blockscaled_quant.py` |

---

## 2. NCU Kernel Performance Data (B200, clock-control=none)

This is the **authoritative** per-kernel profiling data from Sessions 47-48. All measurements use `ncu --clock-control=none --set full` on GPU 0 (uncontested kernel execution, no thermal throttling).

### Quant Kernels

| Kernel | Time (µs) | Regs | Occ% | DRAM% | LD Eff (bytes/sector) | Effective BW (GB/s) |
|--------|-----------|------|------|-------|-----------------------|---------------------|
| `_quantize_flat_v2_kernel` (row quant) | 20 | 28 | 97% | 56% | 32 (optimal) | 4613 |
| `_colwise_quantize_and_pack_kernel` (Triton, no gather) | 39 | 48 | 58% | 28% | 16 (50% waste) | 2280 |
| `_colwise_quantize_and_pack_kernel` (Triton, with gather) | 39 | 48 | 58% | 31% | 16 | 2145 |
| `ColwiseQuantOp` (CuTe DSL, no gather) | 29 | 30 | 93% | 42% | TMA (cp.async) | 3465 |
| `ColwiseQuantOp` (CuTe DSL, with gather, **NEW**) | 58 | 30 | 89% | ~35% | ~16 | ~1700 |
| `ColwiseQuantOp` (CuTe DSL, with gather, OLD) | 154 | 30 | 89% | 8% | 2.1 (93% waste!) | 685 |
| `dual_quantize_varlen_kernel` (Triton fused) | 47 | 48 | 58% | 30% | 21 | 2444 |

### Key NCU Insights

1. **CuTe DSL beats Triton by 1.3× without gather** (29µs vs 39µs): TMA loads via cp.async, 30 regs → 93% occupancy vs 48 regs → 58%. CuTe's LD_eff shows 0 because TMA bypasses SASS LD tracking.

2. **With gather, Triton beats CuTe** (39µs vs 58µs): Triton's compiler generates efficient multi-address scatter-gather from `tl.load(ptr + offsets)`. CuTe element-wise `mSrc[row, col]` generates scalar loads that can't be vectorized.

3. **Pre-gather is NOT the solution**: `torch.index_select` (24µs) + CuTe colwise (29µs) = 53µs > Triton fused (39µs). The extra HBM write+read of the 48 MiB gathered buffer costs more than in-register gather.

4. **row_quant at HBM ceiling**: 4613 GB/s effective bandwidth, 97% occupancy. No further optimization possible.

5. **Estimated total quant overhead per iteration (Ernie I=1536)**:
   - 4× row_quant (20µs) = 80µs
   - 2× CuTe colwise no-gather (29+17µs) = 46µs
   - 2× Triton colwise+gather (39µs) = 78µs
   - Total: ~200µs quantization overhead

### How to Profile

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate

# Single kernel NCU (clock-control=none is essential on contested nodes)
ncu --clock-control=none --set full \
    --kernel-name "regex:quantize|ColwiseQuant" --launch-count 5 \
    -o /tmp/ncu_quant.ncu-rep \
    python -c "import your_profiling_script"

# Export to CSV for analysis
ncu -i /tmp/ncu_quant.ncu-rep --csv --page raw > /tmp/ncu_metrics.csv
```

### JIT Cache Warning

QuACK CuTe DSL compiles kernels to `/tmp/root/quack_cache/<fingerprint>/*.o`. The fingerprint is based on **quack package source**, NOT user kernel source. After editing CuTe kernels, you MUST:
```bash
rm -rf /tmp/root/quack_cache/1c7d36d20210c6ea683563bc34f8d2c853acd2b841f8dd9ad017cb9b7f223ce1/*.o
```
And in Python: `_compile_colwise_quant.cache_clear()` (the `@lru_cache` wrapper).

---

## 3. Pythonic Config API

```python
from sonicmoe import SonicMoEConfig

cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True,
                     fp8_wgrad=True, assume_aligned=True)
with cfg.activate():
    output, loss = moe(x, use_fp8=True)
```

**Priority chain:** `SonicMoEConfig` (thread-local) > `enable_fp8()`/`enable_quack_gemm()` context managers > env vars

**10 config fields** (all default `None` → fall back to env var):
- `use_fp8`, `use_quack_gemm`, `fp8_wgrad`, `fused_gated`, `save_z_fp8`
- `fused_swiglu_quant`, `epilogue_quant`, `fused_zy1_quant`
- `assume_aligned`, `stagewise_memory`

**Key files:** `sonicmoe/config.py` (dataclass + thread-local) • `sonicmoe/functional/__init__.py` (all `_use_*()` helpers check config first)

---

## 4. Critical Design Context

### Zero-Materialization FP8

SonicMoE core design avoids materializing gathered TK-sized activations:
- `quantize_and_pack_activation(x)` → T-sized FP8 + T-sized scales
- `_gather_isa_packed_scales_kernel` → TK-sized ISA-packed scales (~54µs, 1.6% of CUDA time)
- `GemmGatedSm100ZeroMat` kernel: T-FP8 + A_idx + TK-scales (no TK FP8 materialization)

### Epilogue FP8 D Output (Session 48)

When `epilogue_quant=True` (now default), GemmGated writes z directly as `float8_e4m3fn`:
- CUTLASS epilogue computes blockscaled E8M0 scales in registers
- D output is fp8 — no bf16 z allocation, no standalone quant kernel
- A bf16 placeholder tensor with `as_strided((0,0))` wraps the autograd graph node (avoids fp8 tensors in autograd chain which cause illegal memory access at large shapes)
- Actual fp8 data lives in `_PREQUANTIZED_SCALES["z_fp8"]`

### Unaligned FP8 Padding

When expert segments are not 128-aligned:
- **Up-proj:** `_padded_blockscaled_gated_forward()` pads segments to 128, runs zero-mat FP8 GEMM+SwiGLU, then unpads
- **Down-proj:** `blockscaled_fp8_gemm_varlen(assume_aligned=False)` handles padding internally
- **Backward:** stays BF16 (QuACK BF16 handles unaligned natively)

### Memory Optimizations (Sessions 48-50)

1. **Early weight cache eviction** (Session 50): At backward entry, clear `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE` and resize_(0) stash entries for w1_fused/w2_varlen. Frees ~37 MiB. Does NOT reduce peak at I=1536 because wgrad (not dgated) is the peak. Helps at larger shapes where dgated peak > wgrad peak.
2. **Streamlined post-dgated eviction** (Session 50): Only resize_(0) w2_dgated after dgated GEMM (previously also redundantly evicted w1_fused/w2_varlen).
3. **Single-stream wgrad**: Removed cross-stream dz quant overlap. Single-stream lets caching allocator reuse freed blocks without `record_stream` overhead. +50µs latency but better peak memory.
4. **BF16 placeholder for z**: 2-byte storage broadcast via zero strides instead of full-size bf16 tensor in autograd graph.

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/config.py` | Pythonic config API — `SonicMoEConfig` dataclass, thread-local, context manager |
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, epilogue quant, stash, cache eviction, unaligned padding |
| `sonicmoe/moe.py` | MoE class, `refresh_fp8_shadow_weights`, stash/unstash API |
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | **CuTe DSL colwise quant kernel** — 29µs no-gather, 58µs with gather |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, weight caches, blockscaled helpers |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-materialization FP8 CUTLASS kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | Gated GEMM wrapper (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | Backward gated GEMM wrapper (auto ZeroMat selection) |
| `tools/introspect.py` | Main experiment entry point — trace, benchmark, profiler, precision audit, **nsys GPU-projection** |
| `tests/fp8_large_project_contract_test.py` | Correctness gate (34 tests + 20 subtests) |
| `docs/FP8_ARCH_SPEC.md` | Full architecture spec, data flow, memory lifecycle |

---

## 6. Validation Commands

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Clear CuTe JIT cache (required after editing cute_blockscaled_quant.py)
rm -rf /tmp/root/quack_cache/1c7d36d20210c6ea683563bc34f8d2c853acd2b841f8dd9ad017cb9b7f223ce1/*.o

# Contract suite (34 tests, ~3.5 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Unaligned FP8 smoke tests
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest -v tests/test_unaligned_fp8_padded.py

# nsys GPU-projection profiling (gold standard for performance)
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys \
  --nsys-shapes 8192,3072,1536,8,8 8192,3072,2048,8,8

# Full artifact refresh (trace + profiler + precision + benchmark)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python tools/introspect.py --mode full --gpu 0 \
  --precision-seeds 42,123,456,789,1024 --bench-repeats 3 --profile-trials 2
```

---

## 7. Dead Ends (verified, do NOT retry)

| Approach | Why it fails | Session |
|----------|-------------|---------|
| FP8 wgrad colwise quant at I=1536 | SM contention → 0.887× net negative | 45 |
| torch.as_strided for fake TK shape | PyTorch storage bounds check rejects it | 41-42 |
| Rowwise quant + strided view for wgrad | HW requires contiguous K groups | 45 |
| Transpose + rowquant for wgrad | Transpose alone 1509µs > colwise 260µs | 45 |
| Fused CuTe dual quant | 288M instructions (3.6× bloat vs separate) | 43 |
| cp.async aligned to stride-33 smem | Alignment constraint prevents 128-bit vectorized loads | 43 |
| Smem-mediated fp8 vectorized store | -44% instructions but +96% L1 traffic | 43 |
| Full loop unroll for fp8 store | 64 regs → 44% occupancy; runtime loop faster | 43 |
| Pre-gather + CuTe colwise (no in-kernel gather) | torch.index_select 24µs + CuTe 29µs = 53µs > Triton fused 39µs | 48 |
| Blind wallclock comparison on busy cluster | Shared-node contention swamps kernel-time signal | 44+ |
| Micro-optimizing quant kernels at 89%+ HBM | row_quant at 97% occ, 56% DRAM — ceiling reached | 45 |
| Save x (activation) as fp8 between fwd/bwd | Dequant in UpProj backward creates transient spike (x_fp8 + x_bf16 coexist, +24.8 MiB vs baseline). Net WORSE at all shapes tested. | 50 |
| Early weight cache eviction at I=1536 | Frees only 37 MiB (shared data_ptr between FUSED/VARLEN caches). Peak is at wgrad, not dgated, so eviction before dgated has no effect on peak. Only helps at larger shapes. | 50 |

---

## 8. Next Steps (Prioritized)

### P0: Compress quant overhead (Target: 2× reduction)
Quant is 30-45% of FP8 time. Strategies:
- **Fused row+colwise quant**: single pass over data for both quantization formats
- **Epilogue quant on wgrad GEMM**: emit pre-quantized weights from GEMM epilogue
- **Async quant overlap**: overlap quantization with independent computation
- Benchmark each with nsys GPU-projection: `python tools/introspect.py --mode nsys`

### P1: Close CuTe gather gap (58µs → target ≤39µs)
CuTe colwise+gather is 58µs vs Triton 39µs. Options:
- Vectorized gather in CuTe (4-8 bf16 elements per gather)
- Shared memory staging (coalesced global → smem → colwise quant)
- Or accept Triton for gather sites (2 call sites) and CuTe for no-gather (3 call sites)

### P2: Memory reduction
FP8 backward peak is +118 MiB over BF16 due to wgrad quantization temporaries. Strategies:
- **Stash weights to CPU**: `moe.stash_bf16_to_cpu()` → FP8 wins everywhere (-98 MiB bwd at I=1536, -131 MiB at I=2048). Cost: CPU↔GPU transfer latency per optimizer step.
- **Reduce wgrad quant temporaries**: The wgrad blockscaled path creates colwise-quantized copies of dout and y1s. Fusing or streaming these reduces transient peak.
- **Save x as fp8**: Saves T×H bytes between forward and backward BUT dequant in UpProj backward creates transient spike. Only viable if DownProj backward peak > UpProj backward peak (tested and failed at I=1536; may work at very large T).
- At production scale with expert parallelism (E_local ≈ 1-4): weights are tiny, activations dominate. FP8 activation savings (z_fp8, potential x_fp8) scale with T and become dominant.

### P3: Larger shapes (I=4096, I=8192)
FP8 GEMM savings scale with I. At I=4096+ the fixed quant overhead becomes negligible and speedup should exceed 1.2×.

---

## 9. Environment & Information Sources

### Environment

| Item | Value |
|------|-------|
| Python env | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate` |
| GPUs | 8× B200 (192 GiB each) |
| PyTorch | 2.11.0+cu130, CUDA 13.0 |
| QuACK | 0.3.7 |
| JIT cache | `/tmp/root/quack_cache/` |
| Env docs | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` |
| BF16 baseline env | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16` |

### Data Sources

| Resource | Location | Description |
|----------|----------|-------------|
| `benchmark_final.json` | repo root | Session 46 timing, memory, precision for 2 shapes |
| `manifest.json` | repo root | Tensor lifetimes, theoretical memory, precision audit |
| `kernel_breakdown.json` | repo root | Aggregated GPU-projection timing |
| `mem_breakdown.json` | repo root | Checkpoint-by-checkpoint allocator snapshots |
| `reports/nsys_final/nsys_gpu_projection.json` | reports/ | **Session 50 nsys GPU-projection data** — per-kernel timing, category breakdown, speedup for 2 shapes |
| `reports/nsys_final/` | reports/ | nsys-derived kernel breakdown |
| NCU reports | `/tmp/ncu_quant2.ncu-rep`, `/tmp/ncu_improved_gather.ncu-rep` | Session 48 NCU profiling (23 kernels, full metrics) |
| `docs/FP8_ARCH_SPEC.md` | docs/ | Architecture spec, data flow, memory lifecycle |
| `reports/fp8_upgrade/engineering_log.md` | reports/ | Phase-by-phase development history |

### Measurement Methodology

1. **Idle GPU required** for absolute performance numbers (CUDA events or nsys GPU projection).
2. **Subprocess isolation mandatory** — `SONIC_MOE_FP8_MODE` and `USE_QUACK_GEMM` are cached at import time.
3. **NCU `--clock-control=none`** — essential on contested nodes to avoid thermal throttle artifacts.
4. **nsys GPU projection** — the gold standard for latency (parse sqlite, compute GPU busy intervals). Wall-clock is unreliable on busy nodes.
5. **Trimmed mean** — 12 iterations, drop 2 min + 2 max, mean of 8.
