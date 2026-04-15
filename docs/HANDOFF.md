# SonicMoE FP8 Blockscaled Optimization — Complete Handoff

> **Branch:** `native-fp8-exploration`
> **Date:** 2026-04-15 (Session 54 — MoE module test suite + 0-size audit)
> **Environment:** quack-kernels 0.3.7, torch 2.11.0+cu130, 8× NVIDIA B30Z (275 GiB), Python 3.13
> **Python binary:** `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python`
> **Activate:** `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`

---

## 1. Current State — What Works

FP8 blockscaled (E4M3 data + E8M0 scales, 1×32 blocks) training is **fully functional** for forward + backward on Blackwell SM100a. All 27 shapes in the test grid pass performance and precision gates.

- **Speedup:** 1.29× – 1.70×, mean **1.53×** vs official BF16 (nsys GPU-projection)
- **Memory overhead:** +5% to +10% peak backward (FP8 shadow weight caches)
- **Precision:** all RRMSE < 7%, cosine > 0.997 (3 seeds, multi-shape)
- **Tests:** 59 MoE module tests + ~480 op-level tests, all PASS

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

### Immediate — ernie-core FP8 Migration
1. **Migrate FP8 frontier into ernie-core as optional switch.** The `test_moe_module.py` gold reference validates the full MoE pipeline (permute→up-proj→SwiGLU→down-proj→unpermute) at both split-half (ERNIE) and interleaved (SonicMoE) conventions with weight conversion verified. Key architectural difference: SonicMoE uses **interleaved SwiGLU** (`gate=z[:,0::2], up=z[:,1::2]`), ERNIE-core uses **split-half SwiGLU** (`gate=z[:,:I], up=z[:,I:]`). Weight conversion between conventions is implemented and tested in `tests/ops/test_moe_module.py::split_to_interleaved` / `interleaved_to_split`.
2. **Cross-framework Paddle↔PyTorch precision validation.** Use `dlpack` or `numpy` as tensor intermediary between frameworks. Run ERNIE-core's `ExpertsGroupGemmContiguousNode` (Paddle, `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`) and SonicMoE's blockscaled CUTLASS path (PyTorch) with identical weights+routing, compare output/gradients. The ERNIE-core MoE forward is analyzed in detail in the Session 54 agent transcript (key file: `ernie-core/src/ernie_core/models/moe/token_dispatcher/fp8_utils.py`).

### High Value
3. **FP8 native parameters**: Store weights as fp8 natively (eliminate 4 shadow weight caches → save 50% param memory AND remove 5-10% memory overhead). This is the single highest-ROI optimization remaining.
4. **Fuse dual quant into dSwiGLU epilogue**: Saves 1 HBM read (~80µs). The SwiGLU backward output is immediately quantized — fusing avoids the intermediate bf16 materialization.
5. **CUTLASS PreAct FP8 z**: Eliminate z-dequant (~130µs). Requires new CUTLASS kernel class that accepts fp8 PreAct with separate scale TMA. Hard but high value.

### Medium Value
6. **8-bit Adam / CPU optimizer integration with training loop**: Reduce optimizer state memory by 4× or offload entirely.
7. **Stream overlap for quant kernels**: Quant on parallel stream while GEMM runs (~50µs hidden). Requires careful `record_stream` for caching allocator.
8. **Multi-node distributed training integration**: Current work is single-node. Need to validate FP8 with tensor/expert parallelism.

### Research
9. **Dynamic FP8 scaling**: Per-tensor adaptive scaling instead of per-32-block. Could improve precision at the cost of CUTLASS compatibility.
10. **Training convergence validation**: Current precision audit is single-step RRMSE. Need multi-step convergence study on real training runs.

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

---

## 12. Op-Level Unit Test Suite (`tests/ops/`)

> **Date:** 2026-04-15
> **Test count:** 59 MoE module + ~480 op-level parametrized cases (12 test files)

### Overview

Comprehensive 3-way cross-validation suite for every FP8 frontier operator:
- **Category B (Quant):** 6 files testing quantization kernels (byte-exact against torch gold E8M0)
- **Category A (GEMM/SwiGLU):** 5 files testing fused operators (torch ↔ BF16 ↔ FP8)

Every comparison reports RRMSE, cosine similarity, max/mean absolute error.

### Test Files

| File | Operators | Tests | Verification |
|------|-----------|-------|--------------|
| `test_rowwise_quant.py` | `quantize_and_pack_activation` | fp8/scales byte-exact, roundtrip | torch gold E8M0 |
| `test_colwise_quant.py` | `colwise_quantize_and_pack`, CuTe variant | fp8/scales vs gold, CuTe=Triton | byte-exact |
| `test_dual_quant.py` | `dual_quantize_varlen` | row/col match separate | byte-exact |
| `test_fused_zy1_quant.py` | `fused_z_save_y1_quant` | z/y1 match separate | byte-exact |
| `test_weight_quant.py` | `quantize_and_pack_weight_iso32` | fp8/scales vs gold | byte-exact |
| `test_dequant.py` | `dequantize_blockscaled_fp8` | dequant vs gold, roundtrip | RRMSE < 0.05 |
| `test_swiglu.py` | SwiGLU fwd/bwd (BF16+FP8) | 6 tests: 3-way fwd + 3-way bwd | BF16: atol 2e-2, FP8: RRMSE<10% |
| `test_gemm_gated.py` | `gemm_gated` (BF16+FP8) | 3-way: torch/BF16/FP8 | BF16: atol 1.4e-2, FP8: RRMSE<10% |
| `test_gemm_dgated.py` | `gemm_dgated` (BF16 only) | torch/BF16 + determinism + postact | BF16: atol 1.4e-2 |
| `test_varlen_gemm.py` | `blockscaled_fp8_gemm_varlen` | 3-way (subprocess-isolated) | FP8: RRMSE<10%, cos>0.99 |
| `test_wgrad_gemm.py` | `blockscaled_fp8_weight_grad_gemm` | 3-way: torch/BF16/FP8 | FP8: RRMSE<10%, cos>0.99 |

### Running

```bash
# Full suite
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/ -v --tb=short

# Smoke only (~100 cases)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/ -v -k smoke

# Without quack (all skipped — verifies guard logic)
python -m pytest tests/ops/ -v
```

---

## 13. CUTLASS Blockscaled GEMM Workspace Corruption Bug (quack-kernels 0.3.7)

> **Date discovered:** 2026-04-15
> **Severity:** Test-only (production NOT affected — see analysis below)
> **Root cause:** CUTLASS DSL runtime retains internal state tied to GPU memory addresses

### Symptoms

When `blockscaled_fp8_gemm_varlen` is called multiple times in the same process with
**different `total_M`** values (but same `(K, H, E)`), subsequent calls produce completely
uncorrelated garbage output:
- RRMSE ≈ √2 (1.413, the theoretical value for two independent random vectors)
- Cosine similarity ≈ 0
- max_abs_err >> expected (e.g., 0.13 instead of 0.003)

### Reproduction

```python
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_gemm_varlen, quantize_and_pack_activation, precompute_weight_fp8
)
# Call with shape A (M=2048, K=1536, H=3072, E=8) — OK
# Call with shape B (M=65536, K=1536, H=3072, E=8) — OK
# Call with shape A again — GARBAGE (RRMSE ≈ √2)
```

### Root Cause Analysis

The CUTLASS DSL (`cute.compile`) produces compiled CUDA kernels whose internal state
(likely JIT-compiled module workspace for persistent tile scheduling) retains references
to GPU memory addresses from prior kernel launches. When PyTorch's CUDA allocator
reclaims and reuses that memory for different tensor allocations, the stale references
cause the kernel's tile scheduler to read corrupted data.

**Key findings:**
1. **NOT** a Python-level cache bug — clearing `_GEMM_FAST_PATH`, `_COMPILE_CACHE`, and calling `torch.cuda.empty_cache()` does NOT fully fix it
2. The corruption is **deterministic** per process (same shapes always fail)
3. Running the same call in a **fresh subprocess** always succeeds
4. The CUTLASS DSL has no public API to clear its internal compiled-module state

### Why Production Training Is SAFE

1. **`total_M = T × K` is constant** across ALL forward/backward calls within a step and across steps (T = batch tokens, K = top-k, both fixed per config)
2. Only `cu_seqlens_m` (expert routing distribution) varies between steps — and `varlen_args` is recreated fresh each call (line 3626)
3. **`tile_count_semaphore=None`** in all production paths — no GPU-side semaphore workspace exists
4. **`_GEMM_FAST_PATH` key includes `total_M`** — different total_M values never share cached state

### Mitigation in Tests

`tests/ops/test_varlen_gemm.py` uses **subprocess isolation** for FP8 tests:
each `blockscaled_fp8_gemm_varlen` call runs in a fresh Python subprocess via
`subprocess.run()`, ensuring a clean CUTLASS DSL state. This adds ~3s/test overhead
but guarantees correctness.

The `conftest.py` also includes an `_isolate_cuda_memory` autouse fixture that clears
Python-level CUTLASS caches + CUDA allocator between all tests in `tests/ops/`.

### Recommendation

- **Do NOT call `blockscaled_fp8_gemm_varlen` with varying `total_M`** in the same process without restarting the CUDA context
- If variable `total_M` is needed (e.g., dynamic batching), pad `total_M` to a fixed value
- File a bug with quack-kernels team: compiled CUTLASS DSL kernels should not retain stale internal workspace references across launches

---

## 14. MoE Module-Level Test Suite (`tests/ops/test_moe_module.py`)

> **Date:** 2026-04-15 (Session 54)
> **Test count:** 59 parametrized cases, all PASS

### Purpose

Validates the **full MoE forward/backward pipeline** (permute → up-gate projection → SwiGLU → down projection → unpermute) — not just individual ops. Cross-validates SonicMoE's BF16 and FP8 paths against a pure-torch float32 gold reference using ERNIE-core's split-half SwiGLU convention.

### Architecture

- **Gold reference** (`_torch_moe_gold`): Pure float32 per-expert matmul with split-half SwiGLU (`gate=z[:,:I], up=z[:,I:]`). Also has a manual backward (`_torch_moe_gold_backward`) verified against `torch.autograd.grad`.
- **Weight conversion**: `split_to_interleaved()` / `interleaved_to_split()` convert between ERNIE split-half and SonicMoE interleaved layouts per expert. Round-trip is verified bit-exact.
- **Deterministic routing**: `_make_deterministic_routing` generates round-robin topk_indices with softmax scores, eliminating routing randomness.
- **SonicMoE runners**: `_run_sonicmoe_bf16` and `_run_sonicmoe_fp8` call `_UpProjection.apply` + `_DownProjection.apply` directly with pre-computed routing metadata from `TC_topk_router_metadata_triton`.

### Module-Level Precision (Session 54)

| Comparison | Fwd RRMSE | Fwd Cosine | dW RRMSE |
|-----------|:---------:|:----------:|:--------:|
| BF16 vs Gold | 0.0044 | 0.99999 | 0.004 |
| FP8 vs Gold | 0.065 | 0.998 | — |
| BF16 vs FP8 | 0.065 | 0.998 | — |

### Test Breakdown (59 tests)

| Category | Tests | Key Checks |
|----------|-------|------------|
| Gold self-consistency | 6 | Per-element manual verification |
| BF16 vs Gold | 6 | RRMSE < 1%, cosine > 0.999 |
| FP8 vs Gold (subprocess) | 6 | RRMSE < 10%, cosine > 0.99 |
| BF16 vs FP8 (subprocess) | 6 | Cross-check within FP8 tolerance |
| ERNIE split-half vs Gold | 6 | Exact match + weight round-trip |
| Gold backward | 1 | autograd vs manual backward match |
| Empty experts BF16 fwd | 4 | 1-7 empty experts, output correct |
| Empty experts BF16 bwd | 4 | Zero grads for unused experts |
| Empty experts FP8 (subprocess) | 4 | 128-aligned, fwd+bwd correct |
| Deterministic BF16 | 2 | Bit-exact repeated runs |
| Large tensor (T=4096) | 2 | BF16 + FP8 at production scale |
| Weight conversion round-trip | 3 | split↔interleaved identity |
| Routing metadata correctness | 3 | cu_seqlens, gather_idx, s_reverse verified |
| Routing metadata empty experts | 2 | freq=0, equal cu_seqlens for inactive |
| BF16 backward vs Gold | 1 | dw1/dw2 RRMSE < 2% |
| Gold all-same-expert | 1 | No NaN/Inf with degenerate routing |
| Stability (scale 0.5/2.0) | 2 | No NaN/Inf with varied activation scale |

### Running

```bash
# Full MoE module suite (59 tests, ~4 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/test_moe_module.py -v --tb=short

# Only gold/BF16 (no FP8 subprocess overhead, ~3 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/test_moe_module.py -v -k "gold or bf16"

# Only edge cases (empty experts, deterministic, stability)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/test_moe_module.py -v -k "empty or deterministic or stability or routing"
```

---

## 15. 0-Size Expert Audit (Session 54)

### Scenario

When MoE routing assigns 0 tokens to some experts (common with large E, non-uniform workloads, or dropped experts), `cu_seqlens` has consecutive equal values for those experts.

### Audit Results

| Component | Handles 0-size? | Notes |
|-----------|:---:|-------|
| `TC_topk_router_metadata_triton` | **YES** | Correctly produces `cu_seqlens[e]==cu_seqlens[e+1]`, `freq[e]=0` |
| `_all_segments_128_aligned` | **YES** | `0 % 128 == 0`, empty segments pass alignment check |
| `gemm_gated` (BF16 path) | **YES** | CUTLASS varlen scheduler emits 0 rows for empty segments |
| `blockscaled_fp8_gemm_varlen` | **YES** | Same — 0-length segments handled by CUTLASS varlen |
| `quantize_and_pack_activation` | **YES** | `torch.empty(0, K)` + 0-grid Triton launch = no-op |
| `swiglu_forward_triton` | **YES** | Grid `(0,)` = no-op in Triton |
| `_DownProjection.forward` | **YES** | Router scatter handles empty expert output correctly |
| BF16 backward (all paths) | **YES** | Zero gradients for unused experts (verified) |
| FP8 backward (128-aligned) | **YES** | Empty experts produce zero grads (verified) |
| FP8 backward (non-aligned) | **EXPECTED FAIL** | `RuntimeError` at line ~1878 (by design — requires token rounding) |

### Key Insight

The FP8 path is safe with 0-token experts **as long as all non-empty expert segments are 128-aligned**. The alignment check correctly treats 0 as aligned. The only failure mode is non-aligned non-empty segments, which raises an explicit error (not a silent bug).

---

## 16. ERNIE-Core Cross-Framework Reference (Session 54)

### ERNIE-core MoE Architecture (analyzed from source)

The ERNIE-core MoE layer (`ernie-core/src/ernie_core/models/moe/`) uses:

- **SwiGLU**: Split-half (`paddle.chunk(x, 2, axis=-1)` → `silu(first) * second`), NOT interleaved
- **Weight layout**: `up_gate_proj.weight` is `[H, 2*I]` per expert (fused gate+up), `down_proj.weight` is `[I, H]`
- **Prob scaling**: Applied **AFTER SwiGLU, BEFORE down-proj**: `o2 = swiglu(o1) * probs` then `o3 = o2 @ W2`
- **FP8**: `float8_e4m3fn` with 128-element block-wise scaling via `kitchen_quant` + `deep_gemm` grouped GEMM
- **GEMM API**: `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` for forward, `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked` for backward
- **Key file**: `ernie_core/models/moe/token_dispatcher/fp8_utils.py` — contains `ExpertsGroupGemmContiguousNode` with full fwd/bwd

### Differences from SonicMoE

| Aspect | SonicMoE | ERNIE-core |
|--------|----------|------------|
| SwiGLU layout | Interleaved (gate=even, up=odd) | Split-half (gate=first, up=second) |
| Framework | PyTorch + CUTLASS/QuACK | PaddlePaddle + deep_gemm |
| FP8 quant | E8M0 1×32 blockscaled (ISA-packed) | 1×128 blockscaled (kitchen_quant) |
| Weight storage | `(E, 2I, H)` → `.permute(1,2,0)` → `(2I, H, E)` | `[E, H, 2*I]` stacked, per-expert views |
| Prob scaling | Inside `_router_forward` after down-proj | After SwiGLU, before down-proj |

### Cross-Framework Validation Path

To validate FP8 migration fidelity:
1. Generate shared test data (weights, activations, routing) in numpy/dlpack
2. Run SonicMoE FP8 forward (PyTorch) → extract output
3. Run ERNIE-core FP8 forward (Paddle) → extract output
4. Compare outputs at both module level (full MoE) and op level (individual GEMMs)

The `_torch_moe_gold` function in `test_moe_module.py` serves as the framework-independent float32 reference for both.
