# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-10 (Session 42 final — iso32 weights, all optimizations)
> **Branch:** `native-fp8-exploration`  **Latest commit:** `a2cb7b0`
> **Status:** ✅ FP8 + iso32 weights + stash. 1.12× speedup, −8.3%/−12.3% memory, 40/40 PASS.

---

## 0. Bottom Line (idle B200, subprocess-isolated, 3 seeds × 3 repeats, std=0)

| Metric | BF16 | FP8 | FP8 + Stash | Stash vs BF16 |
|--------|:---:|:---:|:---:|:---:|
| **nsys GPU Projection** | **3993 µs** | **3564 µs** | 3564 µs* | **1.12× faster** |
| **Forward peak** | **1386 MiB** | 1440 MiB | **1271 MiB** | **−115 MiB (−8.3%)** |
| **Backward peak** | **1412 MiB** | 1492 MiB | **1239 MiB** | **−173 MiB (−12.3%)** |
| Output RRMSE | — | 6.60% | 6.60% | PASS (<10%) |
| dx RRMSE | — | 7.47% | 7.47% | PASS (<10%) |
| Test suite | — | 40/40 PASS | 40/40 PASS | ✅ |

> *Stash mode uses identical GPU kernels. Memory-only difference.

### Quick Start
```python
moe.refresh_fp8_shadow_weights()  # bf16 → iso32 fp8 caches
moe.stash_bf16_to_cpu()           # −216 MiB GPU
with enable_fp8():
    out, aux = moe(x, use_fp8=True)
out.backward(dout)
moe.unstash_bf16()                # +216 MiB GPU, grads ready
```

---

## 1. Session 42 Changes (this session)

| Change | Impact | Commit |
|--------|--------|--------|
| **32×32 isotropic weight quant** | Same precision as 1×32 (RRMSE ratio=1.000×). Enables transpose without re-quant. | `06a35cd` |
| **stash_bf16_to_cpu / unstash_bf16** | −216 MiB during fwd+bwd. Async pin_memory D2H/H2D. | `124f69f` |
| **Weight decoupling from save_for_backward** | w1/w2 not in ctx.saved_tensors in FP8+aligned mode. Enables stash. | `124f69f` |
| **dw2_base deferred allocation** | −72 MiB dgated phase peak. | `124f69f` |
| **Epilogue z-quant** (reverted to OFF) | Was +48µs net regression: z.to(fp8) cast = 170µs > standalone quant = 122µs. | `aaef9c3`→`a2cb7b0` |
| **stash cache leak fix** | unstash clears FP8 caches (data_ptr changes). Prevents 112 MiB/iter leak. | `aaef9c3` |
| **FP8 wgrad (opt-in)** | `dual_quantize_varlen` kernel: single HBM read → row+col fp8. dw1 RRMSE=3.77%. | `6bbbac2` |
| **w2 cache eviction removed** | Keeping w2 in cache avoids 87µs/iter iso32 re-quant. +37 MiB cache. | `a2cb7b0` |
| **71 dead files deleted** | −9714 lines. 12 test files, 15 tool files, 8 viz files retained. | `17f4eec` |
| **FP8_ARCH_SPEC.md** | Comprehensive architecture specification for next agent. | `aaef9c3` |
| **Subprocess-isolated tests** | Memory + precision tests use separate processes to avoid FP8_MODE contamination. | `124f69f` |

---

## 2. Kernel-Level GPU Time Breakdown (nsys, idle B200, 3564 µs/iter)

| # | Phase | Kernel | µs | % |
|---|-------|--------|:---:|:---:|
| 1 | FWD: x quant+gather | iso32(45) + pack(16) + gather(27) | **88** | 2.5% |
| 2 | FWD: GemmGated | fp8 zero-mat | **465** | 13.0% |
| 3 | FWD: z→z_fp8 | flat blockscaled | **123** | 3.4% |
| 4 | FWD: y1→y1_fp8 | pack ISA | **56** | 1.6% |
| 5 | FWD: DownProj | fp8 varlen | **232** | 6.5% |
| 6 | BWD: dout quant+gather | pack(16) + gather(27) | **43** | 1.2% |
| 7 | BWD: DGated | fp8 zero-mat + FP8CLoad | **414** | 11.6% |
| 8 | BWD: dw2 wgrad | **bf16 GEMM** | **369** | 10.3% |
| 9 | BWD: dz prequant | pack ISA | **109** | 3.0% |
| 10 | BWD: dw1 wgrad | **bf16 GEMM** | **756** | 21.1% |
| 11 | BWD: actgrad | fp8 varlen | **438** | 12.2% |
| 12 | Routing | various | **160** | 4.5% |
| 13 | Other | elementwise | **311** | 8.7% |

**FP8 quant overhead: 440 µs (12.3%). BF16 wgrad: 1125 µs (31.4%).**

---

## 3. Optimization Opportunities (ordered by impact)

### P0: FP8 Wgrad (−300+ µs at I≥2048, memory win at any I)
- Infrastructure ready: `dual_quantize_varlen`, `blockscaled_fp8_wgrad_varlen_k`, decomposed UpProj path
- **Blocker at I=1536**: col_fp8 transient (+192 MiB) exceeds stash savings. dw1 colwise quant ~150µs.
- **Viable at I≥2048**: GEMM savings exceed quant overhead. Auto-enable via `SONIC_MOE_FP8_WGRAD=1`.
- **Key insight**: Use `dual_quantize_varlen(dz)` in DownProj to fuse row+col quant (single HBM read). UpProj uses pre-computed col_fp8 from cache. Free dz_bf16 before wgrad.

### P1: Quant Kernel Fusion (−50+ µs)
- x quant: 3 kernels (45+16+27=88µs) → 1 fused kernel
- dout quant: 2 kernels (16+27=43µs) → 1 fused kernel
- These are memory-bandwidth-bound. Fusion reduces launch overhead.

### P2: CUTLASS Epilogue FP8 Output (blocked by ISA)
- z.to(fp8) cast = 170µs. Standalone quant = 122µs. Epilogue quant computes scales for free but cast is slower.
- **Root cause**: `tcgen05.mma` D output must be ≥16-bit. No fp8 D support in SM100.
- **Workaround**: Write a fused Triton kernel that reads pre-scaled bf16 z and writes fp8 z (skip PyTorch `to()` overhead). Expected: ~50µs (vs 170µs cast or 122µs standalone quant).

### P3: 32×32 Weight Cache Consolidation (−79 MiB)
- Iso32 already enabled. Next: lazy transpose cache (1 fp8 data + 2 ISA scale packs per weight).
- Transpose = fp8 memcpy + scale re-pack (no re-quantization needed).
- Reduces 4 weight cache entries → 2 (original + transposed), sharing fp8 data.

---

## 4. Critical Insights (for next agent)

### Process Contamination
`SONIC_MOE_FP8_MODE` is cached at import time. **BF16 vs FP8 comparison MUST use separate subprocesses.** Same-process comparison gives 0% RRMSE (both use FP8).

### Epilogue Quant is a Trap
Looks like a win (scale compute is "free" in registers) but the separate `z.to(fp8)` cast (170µs) is slower than the standalone Triton quant kernel (122µs) which does quant+cast+scale in one pass. **Net: +48µs regression.** Only worth enabling once D output can be fp8 natively.

### stash_bf16_to_cpu Cache Invalidation
`unstash_bf16()` changes `data_ptr` → old FP8 cache entries become stale. **Must call `clear_all_fp8_weight_caches()` in unstash.** Without this: 112 MiB/iter leak.

### 32×32 Isotropic Weight Quant
- Hardware only supports 1×32. Trick: compute amax over 32×32 tile, write same scale to all 32 rows.
- Round-trip precision = 1×32 for normal weight distributions (ratio 1.000×).
- **Adversarial case**: 10^6× magnitude diff between rows → small rows underflow to 0. Not triggered by real weights.
- Enables transpose without re-quantization (scale values are same in both directions).

### FP8 Wgrad Dual-Quant
`dual_quantize_varlen(dz)`: single HBM read of dz_bf16 → row_fp8 (for actgrad) + col_fp8 (for wgrad). Eliminates the separate `colwise_quantize_and_pack(dz)` call in UpProj backward.

### Memory Lifecycle Critical Path
Forward peak is at GemmGated output (z_bf16 + y1_bf16 coexist). Backward peak is at DownProj postact-release (dz + dz_fp8 + dw2 coexist). With stash, backward peak drops to 1239 MiB.

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, stash, decoupling |
| `sonicmoe/moe.py` | MoE class, refresh_fp8, stash_bf16_to_cpu, unstash_bf16 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | All FP8 quant kernels (1×32, iso32, dual_varlen, colwise), weight caches, GEMM wrappers |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated CUTLASS DSL + BlockscaledScaleStore EpiOp |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated CUTLASS DSL + FP8PreActLoad EpiOp |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-materialization GEMM classes |
| `docs/FP8_ARCH_SPEC.md` | **Start here** — full architecture spec, data flow, memory lifecycle |
| `tests/fp8_large_project_contract_test.py` | 34-test contract suite (incl subprocess memory test) |
| `tests/fp8_frontier_strict_test.py` | 6-test strict suite (incl subprocess precision test) |
| `tools/rigorous_benchmark_s42.py` | Authoritative benchmark: 3 modes × 3 seeds × 3 repeats |
| `tools/nsys_benchmark.py` | nsys-compatible profiling (bf16/fp8/fp8_stash modes) |

---

## 6. Validation Commands
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Test suite (40 tests, ~4 min on B200)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py tests/fp8_frontier_strict_test.py -v --tb=short

# Rigorous benchmark (3 modes × 3 seeds × 3 repeats, ~8 min)
CUDA_VISIBLE_DEVICES=0 python tools/rigorous_benchmark_s42.py --gpu 0

# nsys GPU Projection
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda --capture-range=cudaProfilerApi \
  --cuda-event-trace=false -o /tmp/fp8_profile \
  python tools/nsys_benchmark.py --mode fp8 --gpu 0 --warmup 5 --iters 10

# Idle node scan
python tools/cluster_idle_launch.py scan
```

---

## 7. Dead Ends (verified, do NOT retry)

| Approach | Why it fails | Session |
|----------|-------------|---------|
| Epilogue z-quant default ON | z.to(fp8) cast (170µs) > standalone quant (122µs). Net +48µs. | 42 |
| FP8 wgrad at I=1536 default ON | col_fp8 transient +192 MiB > stash savings. Peak 24 MiB above BF16. | 42 |
| w2 cache eager eviction + iso32 | Re-quant cost 87µs/iter. Keep cache (+37 MiB) is better. | 42 |
| dz prequant ∥ wgrad (stream overlap) | y1s + dz_fp8 coexist → +192 MiB peak. | 41, 42 |
| 4-layout fp8 ≈ bf16 params | 222 MiB (4 layouts) > 216 MiB (bf16 params). | 41 |
| `w.data = fp8` for param offload | autograd internal behavior change, gradient corruption. | 41 |
| `torch.empty(0).as_strided()` | PyTorch bounds check rejects. | 41, 42 |
| FP8 wgrad transpose-quant at I=1536 | 634µs overhead > 259µs GEMM savings. Viable at I≥2048. | 41 |
