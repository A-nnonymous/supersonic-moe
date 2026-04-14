# SonicMoE FP8 Blockscaled Optimization — Complete Handoff

> **Branch:** `native-fp8-exploration`
> **Date:** 2026-04-14
> **Environment:** quack-kernels 0.3.7, torch 2.11.0+cu130, 8× NVIDIA B30Z (275 GiB), Python 3.13
> **Python binary:** `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python`
> **Activate:** `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`

---

## 1. Current State — What Works

FP8 blockscaled (E4M3 data + E8M0 scales, 1×32 blocks) training is **fully functional** for forward + backward on Blackwell SM100a. All 27 shapes in the test grid pass performance and precision gates.

- **Speedup:** 1.29× – 1.70×, mean **1.53×** vs official BF16 (nsys GPU-projection)
- **Memory overhead:** +5% to +10% peak backward (FP8 shadow weight caches)
- **Precision:** all RRMSE < 7%, cosine > 0.997 (3 seeds, multi-shape)
- **Tests:** 34/34 contract tests + 20 subtests PASS

---

## 2. Performance Data (nsys GPU-Projection, 12 iters after 3 warmup)

**BF16 baseline = official SonicMoE** (`/lab/official/sonic-moe`, env `official_bf16`).

Our branch BF16 was verified within <1% of official. The gap was caused by a single `.contiguous()` call after `B.mT` in `gemm_interface.py` (~600µs elementwise copy per iter) — removed, now matches.

**Full 27-shape grid** (3T × 3E × 3I, H=3072, K=8):

| T | I | E | BF16 µs | FP8 µs | Speedup | FP8 Bwd MiB | MemΔ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| 8192 | 1536 | 8 | 3644 | 2715 | **1.34×** | 1547 | +6% |
| 8192 | 2048 | 8 | 4958 | 3387 | **1.46×** | 1992 | +6% |
| 8192 | 3072 | 8 | 8110 | 4774 | **1.70×** | 2884 | +7% |
| 8192 | 1536 | 32 | 3844 | 2922 | **1.32×** | 2909 | +8% |
| 8192 | 2048 | 32 | 5263 | 3709 | **1.42×** | 3678 | +8% |
| 8192 | 3072 | 32 | 8124 | 5318 | **1.53×** | 5218 | +8% |
| 8192 | 1536 | 128 | 5009 | 3897 | **1.29×** | 8700 | +10% |
| 8192 | 2048 | 128 | 6967 | 4995 | **1.39×** | 11385 | +10% |
| 8192 | 3072 | 128 | 10839 | 7267 | **1.49×** | 16756 | +10% |
| 16384 | 1536 | 8 | 7953 | 5227 | **1.52×** | 2819 | +8% |
| 16384 | 2048 | 8 | 10832 | 6765 | **1.60×** | 3622 | +8% |
| 16384 | 3072 | 8 | 16172 | 10065 | **1.61×** | 5232 | +9% |
| 16384 | 1536 | 32 | 8129 | 5432 | **1.50×** | 3891 | +6% |
| 16384 | 2048 | 32 | 10860 | 7039 | **1.54×** | 4794 | +7% |
| 16384 | 3072 | 32 | 16558 | 10166 | **1.63×** | 6863 | +5% |
| 16384 | 1536 | 128 | 9099 | 6360 | **1.43×** | 9688 | +9% |
| 16384 | 2048 | 128 | 12348 | 8198 | **1.51×** | 12506 | +10% |
| 16384 | 3072 | 128 | 19216 | 11862 | **1.62×** | 18142 | +10% |
| 32768 | 1536 | 8 | 16287 | 10652 | **1.53×** | 5359 | +9% |
| 32768 | 2048 | 8 | 21753 | 13587 | **1.60×** | 6882 | +9% |
| 32768 | 3072 | 8 | 33278 | 20010 | **1.66×** | 9927 | +10% |
| 32768 | 1536 | 32 | 16829 | 10753 | **1.56×** | 6176 | +7% |
| 32768 | 2048 | 32 | 22584 | 13967 | **1.62×** | 7965 | +7% |
| 32768 | 3072 | 32 | 33504 | 19761 | **1.70×** | 11549 | +7% |
| 32768 | 1536 | 128 | 17635 | 11509 | **1.53×** | 11669 | +8% |
| 32768 | 2048 | 128 | 23312 | 14956 | **1.56×** | 14751 | +9% |
| 32768 | 3072 | 128 | 35627 | 22026 | **1.62×** | 20919 | +9% |

### Scaling Rules

- **I scaling dominates**: I=1536→I=3072 adds ~0.3× speedup. FP8 GEMM savings ∝ O(I²), quant overhead ∝ O(I).
- **T scaling significant**: T=8k→T=32k adds ~0.2× speedup. Larger T amortizes per-iter fixed overhead.
- **E scaling minimal**: E=8 vs E=128 differ by <0.15× at fixed T×I. E affects only routing and cache size, not GEMM shape.
- **Memory overhead**: +5-10% backward peak (4 FP8 weight caches: w1_fused, w2_varlen, w2_dgated, w1T_varlen).

### Budget Breakdown (T=8192, E=8, I=1536 — representative)

Where FP8 gains and loses time:

| Category | BF16 µs | FP8 µs | Delta | Note |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 2078 | 1148 | **-930** | FP8 wgrad CUTLASS wins big |
| GemmGated (fwd) | 700 | 0 → ZeroMat 451 | **-249** | Zero-materialization kernel |
| GemmDGated (bwd) | 439 | 0 → ZeroMat 391 | **-48** | Zero-materialization kernel |
| Row Quant | 0 | 77 | +77 | Activation quantization |
| Dual Quant | 0 | 153 | +153 | Row+Col quant (single HBM read) |
| Blockscaled Quant | 0 | 235 | +235 | Weight quantization (cached after 1st iter) |
| ISA Scale Gather | 0 | 16 | +16 | Scale tile gather for CUTLASS |
| **NET** | **3644** | **2715** | **-929** | FP8 wins |

### Precision

| T | E | output | dx | dw1 | dw2 | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 8192 | 8 | 6.52% | 6.53% | 4.71% | 4.90% | PASS |
| 8192 | 32 | 6.52% | 6.51% | 5.47% | 5.88% | PASS |
| 8192 | 128 | 6.52% | 6.52% | 6.01% | 6.50% | PASS |
| 32768 | 8 | 6.55% | 6.55% | 4.12% | 4.20% | PASS |
| 32768 | 32 | 6.55% | 6.54% | 4.60% | 4.84% | PASS |
| 32768 | 128 | 6.55% | 6.55% | 5.40% | 5.81% | PASS |

Guard: RRMSE < 10%, cosine > 0.99. output/dx RRMSE ~6.5% independent of shape; dw1/dw2 scales with E.

---

## 3. Architecture

### FP8 Data Format
- **E4M3** data + **E8M0** scales, **1×32 block granularity**
- ISA-packed scale tiles (128×128 layout) for CUTLASS SM100a consumption
- **32×32 isotropic weight quantization** for weight caches

### Default FP8 Path (max performance)
```
Setup:   refresh_fp8_shadow_weights()    # bf16 → 4 FP8 weight caches
         stash_bf16_to_cpu()             # bf16 master weights → CPU pinned
Each iter: fwd(fp8) → bwd(fp8) → zero_grad
Optimizer: unstash → Adam.step → refresh → re-stash
```
All 4 FP8 weight caches retained across iterations (version-keyed auto-invalidation at optimizer step).

### Zero-Materialization
No TK-sized FP8 activation is ever stored in HBM. CUTLASS `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat` gather A rows internally via `A_idx`.

### Token Rounding (E>8)
FP8 CUTLASS requires **128-aligned expert segments** (SM100 ISA scale tile hardware constraint). For E>8, tokens-per-expert are not naturally 128-aligned. Solution: official `forward_token_choice_rounding(Mtile=128)` → `ceil(freq/128)*128`. This wastes at most 1 extra tile per expert (verified: matches paper assumption). For E≤8, all tokens go to all experts → segment = T (always 128-aligned in practice).

Non-aligned FP8 backward raises `RuntimeError` explicitly (`functional/__init__.py:~1878`).

### Optional CPU Optimizer (for memory-constrained scenarios)
```python
moe.setup_cpu_optimizer(torch.optim.Adam, lr=1e-3)
# Each iter: fwd(fp8) → bwd(fp8) → moe.cpu_optimizer_step()
# Saves ~3.4 GB base at E=128 (bf16 weights + Adam states → CPU)
# Costs ~500µs/iter from CPU↔GPU transfer + weight re-quantization
```

---

## 4. Key Code Changes and Bug Fixes

### Performance-Critical Fixes

| Fix | File | Impact | Details |
|-----|------|--------|---------|
| **VARLEN weight cache preservation** | `functional/__init__.py:~1629` | **+11pp** (1.03→1.14×) | `_VARLEN_WEIGHT_CACHE.clear()` at DownProj backward forced re-quantization every iter (~360µs). Cache is version-keyed, auto-invalidates at optimizer step. |
| **FUSED weight cache preservation** | `functional/__init__.py:~1629,~2249` | Eliminates re-quant at large E | Same pattern: removed `_FUSED_WEIGHT_CACHE.clear()` between forward and backward |
| **B.mT.contiguous() removal** | `gemm_interface.py` | +600µs saved | Our branch added `.contiguous()` after `B.mT`; official has just `B.mT`. The copy was a ~600µs elementwise kernel. Fix limited to BF16 path to avoid affecting FP8. |

### Correctness Fixes

| Fix | File | Impact | Details |
|-----|------|--------|---------|
| **Cache corruption fix** | `functional/__init__.py:~1741` | Fixes E>8 crash | `ctx._w2_dgated_fp8.untyped_storage().resize_(0)` freed tensor aliased in `_FUSED_WEIGHT_CACHE` → "storage of size 0" crash on next forward. |
| **INT32 pointer overflow** | `blockscaled_fp8_gemm.py`, `backward.py` | Silent data corruption fix | Triton int32 pointer arithmetic wraps when `row × stride > 2³¹-1`. Dual-path dispatch: `SAFE_INT64` branch compiled only when needed. See Section 7 for full kernel audit. |
| **Non-aligned RuntimeError** | `functional/__init__.py:~1878` | Prevents silent precision loss | FP8+non-aligned raises explicit error; callers must use token rounding. |

### Infrastructure Fixes

| Fix | File | Impact |
|-----|------|--------|
| GPU contention in parallel runs | `tools/introspect.py` | `_subprocess_env_for_gpu()` respects shell-level `CUDA_VISIBLE_DEVICES` |
| Missing `return env` in subprocess builder | `tools/introspect.py` | Without it, subprocess had `env=None` → no `USE_QUACK_GEMM` |
| Grid `--nsys-shapes` argparse fix | `tools/introspect.py` | Multiple shapes as `nargs="+"` + `*shape_strs` unpacking |
| Wgrad threshold=0 | `functional/__init__.py:~622` | FP8 wgrad profitable at all I values after cache fixes |

---

## 5. Measurement Methodology

### Gold Standard: nsys GPU-Projection
The ONLY trusted performance metric. Merges overlapping kernel intervals from nsys sqlite trace, immune to CPU contention.

**Rules:**
1. GPUs must be idle (`nvidia-smi` util=0%) before measurement
2. Each shape×mode runs in isolated subprocess (prevents CUTLASS JIT cache cross-contamination)
3. BF16 uses `moe_TC_softmax_topk_layer` (same API as official benchmark, verified <1% gap)
4. FP8 E≤8 uses stash mode; E>8 uses token rounding + `moe_general_routing_inputs`
5. Expert segments must be 128-aligned (SM100 ISA constraint)
6. E=7,K=8 is invalid (K>E is a topk constraint violation, not an FP8 bug)

### Why Not Other Metrics
- **Wall-clock**: includes 40-60% CPU overhead on contested nodes. Node 0344 (4/8 idle) showed 6609µs BF16; fully idle was 3932µs (1.68× inflation).
- **CUDA events same-process**: reliable for A/B comparison under identical contention, but measures different thing than kernel-level GPU busy time.
- **nsys with `retain_graph=True`**: produces incomplete captures (small sqlite). Must use `out.sum().backward()` or similar.

### Reproduce

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Full 27-shape grid (8 GPUs, ~15 min)
python tools/introspect.py --mode grid --gpu 8 --nsys-warmup 3 --nsys-iters 12

# Single shape
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --gpu 0 \
  --nsys-iters 12 --nsys-warmup 3 --nsys-shapes 8192,3072,1536,8,8

# Multi-shape on one GPU
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --gpu 0 \
  --nsys-iters 12 --nsys-warmup 3 \
  --nsys-shapes 8192,3072,1536,8,8 8192,3072,2048,8,8

# Precision audit
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode precision --gpu 0 \
  --nsys-shapes 8192,3072,1536,8,8 --precision-seeds 42,123,777
```

---

## 6. File Map

| File | Role | Key Lines |
|------|------|-----------|
| `sonicmoe/functional/__init__.py` | Core FP8 fwd/bwd orchestration | ~622 (wgrad threshold), ~1629 (cache preservation), ~1741 (no resize_), ~1878 (non-aligned error) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, FP8 weight caches, CUTLASS wrappers | `_FUSED_WEIGHT_CACHE`, `_VARLEN_WEIGHT_CACHE`, dual quant, ISA scale packing |
| `sonicmoe/quack_utils/gemm_interface.py` | BF16 GEMM wrapper | `B.mT` (no .contiguous()) |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-materialization FP8 GEMM | `_GemmSm100ZeroMatMixin`, SFA layout fix, `gather_A=False` |
| `sonicmoe/moe.py` | MoE module | `stash_bf16_to_cpu()`, `unstash_bf16()`, `refresh_fp8_shadow_weights()`, `setup_cpu_optimizer()`, `cpu_optimizer_step()`, `optimizer_step_stashed()` |
| `tools/introspect.py` | All-in-one profiling tool | `--mode nsys/grid/precision/report/quant-bench/wgrad-bench/ncu-bench` |
| `reports/grid_session53/session53_grid_full.json` | Full 27-shape grid raw data | Performance + memory per shape |
| `reports/session53_breakdown.md` | Final performance/memory data table | |
| `reports/fp8_upgrade/engineering_log.md` | Chronological development log (Phases 1-17) | Historical reference, not authoritative for current state |
| `reports/fp8_upgrade/HANDOFF.md` | **STALE** — Session 52 data, superseded by this file | Do not use for current numbers |

---

## 7. INT32 Pointer Overflow in Triton — Full Kernel Audit

> **Severity**: Silent data corruption / illegal-memory-access crash
> **Discovered by user**: Session 53, diagnosed via `compute-sanitizer --tool memcheck`

### Root Cause
Triton compiles pointer offset arithmetic as **int32 by default**. When `row_index × stride > 2³¹-1` (2,147,483,647), the multiplication wraps to a negative value → GPU reads/writes ~1.8 GB before tensor base.

### Trigger
Any tensor where `(rows-1) × stride_row > INT32_MAX`:
- Weight W1 reshaped to `(E×2I, H)`: E=128, I=3072 → `(786432, 3072)`, max offset = 2.4B > INT32_MAX
- Activation `(T, K)` with T > ~700K and K ≥ 3072
- Backward `(TK, 2I)` with TK > ~350K and I ≥ 3072

### Fix — Dual-Path Dispatch
Hot-path quant kernels: `SAFE_INT64: tl.constexpr` parameter. Dispatch checks `(rows-1) * max_stride > INT32_MAX`, compiles int64 branch only when needed. Zero overhead for normal shapes.

Backward reduction kernels: always int64 for index arrays (negligible cost).

### Audit Table

| Kernel | File | Status |
|--------|------|--------|
| `_quantize_and_pack_iso32_kernel` | blockscaled_fp8_gemm.py | **Fixed** (always int64) |
| `_quantize_and_pack_kernel` | blockscaled_fp8_gemm.py | **Fixed** (SAFE_INT64 dispatch) |
| `_gather_quantize_and_pack_kernel` | blockscaled_fp8_gemm.py | **Fixed** (SAFE_INT64 dispatch) |
| `_fused_z_save_y1_quant_kernel` | blockscaled_fp8_gemm.py | **Fixed** (SAFE_INT64 dispatch) |
| `db2_and_ds_kernel` | backward.py | **Fixed** (always int64) |
| `db1_kernel` | backward.py | **Fixed** (always int64) |
| `_dual_quantize_kernel` | blockscaled_fp8_gemm.py | Safe (uses int64) |
| `_fused_transpose_quantize_kernel` | blockscaled_fp8_gemm.py | Safe (int64) |
| `_warp32x32_transpose_quant_kernel` | blockscaled_fp8_gemm.py | Safe (int64) |
| `_dual_varlen_quantize_kernel` | blockscaled_fp8_gemm.py | Safe (int64) |
| All SwiGLU kernels (×10) | swiglu_triton.py | Safe (TK×stride < 2³¹) |
| CuTe/CUTLASS kernels | gemm_sm100_fp8_zeromat.py | Safe (C++ size_t) |

---

## 8. Hard-Won Lessons

### Performance Measurement
1. **nsys GPU-projection is the ONLY trustworthy metric.** Wall-clock includes 40-60% CPU overhead on shared nodes.
2. **GPU contention invalidates everything.** Same workload: 3932µs on idle node, 6609µs on 50% busy node (1.68× inflation).
3. **CUTLASS JIT cache cross-contamination across shapes.** Different shapes compile to incompatible kernels; running multiple shapes in the same process → ILLEGAL_INSTRUCTION. Each shape must use an isolated subprocess.
4. **QuACK JIT cache is source-fingerprint-based.** Fingerprint covers `quack/` package, NOT user kernel source. Must clear `/tmp/root/quack_cache/<hash>/*.o` after editing CuTe kernels.

### FP8 Implementation
5. **Weight cache invalidation must be version-keyed, not eager.** Clearing caches every backward costs 300-980µs/iter re-quantization. Caches keyed on `weight._version` auto-invalidate at optimizer step.
6. **Never `resize_(0)` tensors that may be aliased in caches.** Context tensors saved by autograd may share storage with cache entries. Freeing via ctx reference corrupts the cache.
7. **B.mT.contiguous() is a silent 600µs penalty.** `B.mT` returns a view (free); `.contiguous()` forces a full elementwise copy. Official SonicMoE uses just `B.mT`.
8. **FP8 weight caches dominate memory overhead.** 4 caches × weights × ~72MB each ≈ 650 MiB at E=8. This is inherent to blockscaled quantization (different scale dimensions per layout).
9. **E8M0 scales encode BF16 magnitude, NOT FP8 magnitude.** You cannot reconstruct scales from FP8 data. Scales must be computed from the original BF16 source.
10. **Epilogue FP8 D output saves 2 kernel launches.** GemmGated writes z as fp8 directly; store bf16 placeholder with `as_strided((0,0))` for autograd (fp8 tensors in autograd graph → segfault at large shapes).

### Triton/CUTLASS
11. **num_warps=1 is dramatically better for bandwidth-bound Triton kernels.** Fewer warps/block → more blocks in-flight per SM → better utilization. NCU diagnostic: "% cycles with no eligible warp". Gave 2.3× speedup on colwise quant.
12. **Fused dual quant (row+col) saves 1 HBM read.** Single pass produces both row-major and col-major FP8 = ~80µs savings vs separate kernels.
13. **CUTLASS PreAct constraint** — `assert PreAct.element_size() == 2` blocks feeding FP8 z directly to GemmDGated. Would save ~130µs (z-dequant). Requires a new kernel class, not a config change.
14. **Token rounding overhead is negligible.** `ceil(freq/128)*128` wastes at most 1 tile per expert. `moe_general_routing_inputs` is the official API that accepts pre-rounded routing.

### Process/Env
15. **`_IS_FP8_ACTIVE` is cached at import from env var.** Same-process BF16/FP8 comparison with env var set produces fake results. All comparisons must use separate subprocesses.
16. **Weight view refcount prevents stash memory savings.** Creating `w1_p = moe.c_fc.weight.permute(1,2,0)` BEFORE stash holds a Python reference to bf16 storage → GC can't free it. Create views AFTER stash.
17. **`gather_A` + SFA**: Our code uses `gather_A=False` everywhere. The ZeroMat mixin handles gather via `A_idx` with correct SFA layout fix. QuACK 0.3.7 loads SFA correctly in this path.

---

## 9. Insights and Strategic Analysis

### Where FP8 Speedup Comes From
The dominant source is **Wgrad GEMM** (~930µs saved at Ernie shape), followed by **GemmGated/GemmDGated replacement** with zero-materialization kernels (~300µs). The FP8 overhead is **quantization** (~480µs total: dual quant + blockscaled quant + row quant). The net is positive when GEMM savings > quant overhead, which is always true for I≥1536.

### Why E Has Minimal Impact
E only affects routing metadata and cache size, not the core GEMM shapes. At E=128, each expert processes fewer tokens (T/E), but the GEMM shape per expert is identical. The extra FP8 cache memory (~3.5 GB for 4 layouts at E=128) doesn't impact compute.

### Why Larger T and I Are Better
- **T↑**: Larger T amortizes per-iteration fixed overhead (router, softmax, gather/scatter). GEMM time grows linearly while overhead stays constant.
- **I↑**: GEMM savings grow ∝ O(I²) (matrix multiply FLOPs), quant overhead grows ∝ O(I) (linear scan for quantization). The quadratic-vs-linear scaling makes FP8 increasingly profitable at larger I.

### Memory: The Trade-Off
FP8 uses 5-10% MORE peak memory than BF16 (4 weight caches). The stash saves bf16 master weights to CPU (~216 MiB at E=8, ~3.4 GB at E=128), but the FP8 caches add back more at small E. At E=128, CPU optimizer offload saves 3.4 GB base by moving master weights + Adam states to CPU, at a cost of ~500µs/iter.

---

## 10. Next Steps (Prioritized)

### High Value
1. **FP8 native parameters**: Store weights as fp8 natively (eliminate 4 shadow weight caches → save 50% param memory AND remove 5-10% memory overhead). This is the single highest-ROI optimization remaining.
2. **Fuse dual quant into dSwiGLU epilogue**: Saves 1 HBM read (~80µs). The SwiGLU backward output is immediately quantized — fusing avoids the intermediate bf16 materialization.
3. **CUTLASS PreAct FP8 z**: Eliminate z-dequant (~130µs). Requires new CUTLASS kernel class that accepts fp8 PreAct with separate scale TMA. Hard but high value.

### Medium Value
4. **8-bit Adam / CPU optimizer integration with training loop**: Reduce optimizer state memory by 4× or offload entirely.
5. **Stream overlap for quant kernels**: Quant on parallel stream while GEMM runs (~50µs hidden). Requires careful `record_stream` for caching allocator.
6. **Multi-node distributed training integration**: Current work is single-node. Need to validate FP8 with tensor/expert parallelism.

### Research
7. **Dynamic FP8 scaling**: Per-tensor adaptive scaling instead of per-32-block. Could improve precision at the cost of CUTLASS compatibility.
8. **Training convergence validation**: Current precision audit is single-step RRMSE. Need multi-step convergence study on real training runs.

---

## 11. Data Sources

| Resource | Path | Notes |
|----------|------|-------|
| Grid benchmark data | `reports/grid_session53/session53_grid_full.json` | 27-shape, authoritative |
| Per-GPU grid data | `reports/grid_session53/gpu{0-7}.json` | Individual GPU results + kernel breakdowns |
| Grid logs | `reports/grid_session53/logs/gpu{0-7}.log` | Subprocess stdout |
| Breakdown | `reports/session53_breakdown.md` | Performance/memory table + scaling rules |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Historical Phases 1-17 (chronological) |
| nsys-rep files | `/root/paddlejob/.../panzhaowu/output/nsys/*.nsys-rep` | Open in Nsight Systems GUI |
| Quant bench | `reports/quant_bench_final.json` | Per-kernel CUDA event benchmarks |
| Wgrad bench | `reports/wgrad_bench.json` | Wgrad FP8 vs BF16 per-shape |
| Official BF16 | `/root/paddlejob/.../panzhaowu/lab/official/sonic-moe` | Env: `official_bf16` |
| Environment docs | `/root/paddlejob/.../panzhaowu/env.md` | Machine setup, compilation |
