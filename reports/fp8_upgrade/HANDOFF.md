# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-15 (Session 49 — nsys GPU-projection profiling, introspect.py nsys mode, kernel classifier fix)
> **Branch:** `native-fp8-exploration`
> **Status:** Zero-materialization FP8 + iso32 weights + optional stash + CuTe DSL wgrad quant + Pythonic config API + unaligned FP8 padding + epilogue FP8 D output + eager weight cache eviction + **nsys GPU-projection engine in introspect.py**. Contract suite: **34/34 tests + 20 subtests PASS**.

---

## 0. Bottom Line

### Performance (Session 49 — nsys GPU-projection, 8-iter steady-state, B200)

| Shape | BF16 (µs/iter) | FP8 (µs/iter) | Speedup | Takeaway |
|-------|:--------------:|:-------------:|:-------:|----------|
| **Ernie** (T=8192, H=3072, I=1536, E=8, K=8) | 9434 | 8735 | **1.08×** | FP8 wins; GEMM savings (2741µs) > quant overhead (2271µs) |
| **I=2048** (T=8192, H=3072, I=2048, E=8, K=8) | 12252 | 11443 | **1.07×** | FP8 wins; GEMM savings (2208µs) ≈ quant overhead (4130µs) + routing savings offset |

> **Methodology:** nsys GPU-projection (merged kernel intervals from `CUPTI_ACTIVITY_KIND_KERNEL` sqlite). Steady-state: 5 warmup + 8 measured iterations with no FP8 state reset between iterations. This is the gold standard for latency measurement, immune to CPU contention. Run via `python tools/introspect.py --mode nsys`.

### Memory (Session 49, subprocess-isolated peak, post-warmup)

| Shape | BF16 Peak | FP8 Peak | Delta |
|-------|-----------|----------|-------|
| **Ernie** | 1412 MiB | 1530 MiB | +118 MiB (+8.4%) |
| **I=2048** | 1828 MiB | 1986 MiB | +158 MiB (+8.6%) |

Session 48's eager weight cache eviction + single-stream wgrad reduced FP8 memory overhead from +391 MiB (+27%) to +118 MiB (+8.4%) at I=1536.

### Precision (5 seeds × 2 shapes, all PASS)

| Tensor | Ernie RRMSE | I=2048 RRMSE | Correlation | Status |
|--------|:-----------:|:------------:|:-----------:|:------:|
| output | 6.49% | 6.49% | 0.9979 | ✓ PASS |
| dx | 6.52% | 6.53% | 0.9979 | ✓ PASS |
| dw1 | 4.69% | 4.68% | 0.9989 | ✓ PASS |
| dw2 | 4.88% | 4.87% | 0.9988 | ✓ PASS |

All within guardrails: **RRMSE < 10%**, **correlation > 0.99**. Stable across seeds (std < 0.001%).

---

## 1. Session 49 Changes (nsys GPU-Projection Engine)

| Change | Impact | Files |
|--------|--------|-------|
| **nsys GPU-projection mode** | Gold-standard profiling: generates steady-state workload, runs `nsys profile`, parses sqlite for merged GPU busy time. Immune to CPU contention. | `tools/introspect.py` |
| **Kernel classifier fix** | CuTe-compiled quant kernels (name contains "cutlass") were misclassified as GEMM. Now check quant patterns BEFORE GEMM patterns. | `tools/introspect.py` |
| **Multi-shape nsys comparison** | `--nsys-shapes` flag allows profiling multiple I sizes in one run with per-shape category breakdown. | `tools/introspect.py` |
| **Steady-state measurement** | No FP8 state reset between measured iterations — weight caches, alignment streak preserved. Previous sessions had cold-start artifacts. | `tools/introspect.py` |

### nsys Category Breakdown (per-iter, µs)

**I=1536:**

| Category | BF16 | FP8 | Delta |
|----------|-----:|----:|------:|
| Wgrad GEMM | 6018 | 3277 | -2741 |
| GemmGated (fwd) | 1438 | — | — |
| GemmDGated (bwd) | 817 | — | — |
| GemmDGated ZeroMat (bwd) | — | 750 | — |
| Blockscaled Quant | — | 2271 | +2271 |
| Row Quant | — | 325 | +325 |
| ISA Scale Gather | — | 17 | +17 |
| Token Gather | 448 | 757 | +309 |
| Other | 704 | 1331 | +627 |
| **Total** | **9434** | **8735** | **-699** |

**I=2048:**

| Category | BF16 | FP8 | Delta |
|----------|-----:|----:|------:|
| Wgrad GEMM | 5879 | 3671 | -2208 |
| GemmGated (fwd) | 2254 | — | — |
| GemmDGated (bwd) | 726 | — | — |
| GemmDGated ZeroMat (bwd) | — | 1575 | — |
| Blockscaled Quant | — | 4130 | +4130 |
| Row Quant | — | 1036 | +1036 |
| ISA Scale Gather | — | 17 | +17 |
| Token Gather | 2307 | 142 | -2165 |
| Other | 1403 | 863 | -540 |
| **Total** | **12252** | **11443** | **-809** |

### Key Insights from Session 49

1. **FP8 wins at BOTH shapes** (1.08× and 1.07×) — previous Session 46 showed 0.993× at I=1536 because: (a) CUDA events instead of GPU-projection, (b) CPU contention on busy node, (c) cold-start artifacts from resetting FP8 state each iteration.

2. **Quant overhead is the dominant FP8 cost**: Blockscaled Quant + Row Quant = 2596µs at I=1536 (30% of FP8 time), 5166µs at I=2048 (45%). Compressing this is the main path to further speedup.

3. **Token Gather anomaly at I=2048**: BF16 uses 2307µs but FP8 only 142µs. The zero-materialization approach eliminates most token gathering for the FP8 forward pass. At I=1536, BF16 is 448µs and FP8 is 757µs — the balance depends on routing patterns and tensor sizes.

4. **Wgrad GEMM halved by FP8**: 6018→3277µs at I=1536 (1.84×), 5879→3671µs at I=2048 (1.60×). This is the core FP8 value proposition.

### Sessions 47-48 Changes (prior)

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

### Memory Optimizations (Session 48)

1. **Eager weight cache eviction**: After dgated GEMM consumes w2_dgated, immediately clear `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE` and `resize_(0)` consumed tensors. Only w1T_varlen (needed for UpProj actgrad) is kept.
2. **Single-stream wgrad**: Removed cross-stream dz quant overlap. Single-stream lets caching allocator reuse freed blocks without `record_stream` overhead. +50µs latency but better peak memory.
3. **BF16 placeholder for z**: 2-byte storage broadcast via zero strides instead of full-size bf16 tensor in autograd graph.

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
FP8 uses +118-158 MiB more than BF16 (down from +391-455 MiB). Strategies:
- **Stash weights to CPU**: `moe.stash_bf16()` → save ~216 MiB, measure CPU↔GPU transfer cost
- **Lazy weight cache**: only quantize weights on first use, don't pre-cache all experts
- Larger I shapes (weight cache is fixed cost, amortizes better)

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
| `reports/nsys_final/nsys_gpu_projection.json` | reports/ | **Session 49 nsys GPU-projection data** — per-kernel timing, category breakdown, speedup for 2 shapes |
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
