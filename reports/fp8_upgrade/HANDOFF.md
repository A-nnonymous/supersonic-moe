# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-08 (Session 41 — Memory optimization + kernel fusion + adapter skip)
> **Branch:** `native-fp8-exploration`
> **Status:** ✅ FP8 fully functional. **1.13× GPU compute speedup (nsys)**, precision PASS, memory optimized.

---

## 0. Current Bottom Line

> All data measured via **nsys GPU Projection on idle B200 nodes** (0% util, <5 MiB VRAM).

| Metric | BF16 Baseline | FP8 Frontier | Delta |
|--------|--------------|--------------|-------|
| **GPU Projection total** | **3966 µs** | **3512 µs** | **1.13× faster** |
| Output RRMSE | — | 6.60% ✓ | <10% threshold |
| Worst grad RRMSE | — | 7.48% (dx) ✓ | <10% threshold |
| All cosines | — | ≥0.9972 ✓ | >0.99 threshold |
| Test suite | — | **31/31 PASS** | ✓ |

Shape: T=8192, H=3072, I=1536, E=8, K=8 (Ernie).  GPU: B200 (SM 10.0).

### Usage (two env vars only)
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 python train.py
```
```python
from sonicmoe.functional.utils import enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

For best forward performance (trades +96 MiB peak for +64µs/iter):
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 SONIC_MOE_FP8_FUSED_ZY1_QUANT=1 python train.py
```

---

## 1. nsys Kernel Breakdown (idle B200, 10 profiled iterations)

### BF16 baseline: 3966 µs/iter

| Kernel | µs/iter | % |
|--------|---------|---|
| GemmDefault (wgrad×3 + downproj) | 2257 | 56.9% |
| GemmGatedSm100 (up-proj fwd) | 737 | 18.6% |
| GemmDGatedSm100 (bwd) | 482 | 12.2% |
| elementwise (SwiGLU bwd etc) | 157 | 4.0% |
| token_gather_sum | 141 | 3.6% |
| Other | 192 | 4.8% |

### FP8 frontier (with fused_zy1_quant): 3512 µs/iter

| Kernel | µs/iter | % | vs BF16 |
|--------|---------|---|---------|
| GemmDefault (wgrad BF16) | 1764 | 50.2% | 1.28× faster |
| GemmGatedSm100ZeroMat (FP8 fwd) | 454 | 12.9% | 1.62× faster |
| GemmDGatedFP8CLoadSm100ZeroMat (FP8 bwd) | 415 | 11.8% | 1.16× faster |
| _fused_z_save_y1_quant_kernel | 166 | 4.7% | FP8 overhead |
| _quantize_and_pack_kernel (×6) | 222 | 6.3% | FP8 overhead |
| elementwise (SwiGLU bwd) | 156 | 4.4% | ~same |
| token_gather_sum | 144 | 4.1% | ~same |
| _gather_isa_packed_scales_kernel (×2) | 55 | 1.6% | FP8 overhead |
| Other | 136 | 3.9% | — |

**GEMM savings: +942µs. FP8 overhead: −488µs. Net: +454µs (1.13×).**

---

## 2. Precision (subprocess-isolated, 3 seeds)

| Tensor | Seed 42 | Seed 123 | Seed 777 | Cosine |
|--------|---------|----------|----------|--------|
| output RRMSE | 6.60% | 6.60% | 6.60% | 0.9978 |
| dx RRMSE | 7.47% | 7.48% | 7.48% | 0.9972 |

wgrad norm relative error: c_fc <0.33%, c_proj <0.38%, router <0.55%.
31/31 contract tests PASS.

---

## 3. Memory — Official BF16 vs FP8 Frontier Breakdown

> **Methodology**: subprocess-isolated, idle B200 (0% util, 0 MiB), 2 warmup + 1 measured iter.
> **BF16 baseline**: official upstream code (`lab/official/sonic-moe`, `envs/official_bf16`).
> **FP8 frontier**: `native-fp8-exploration` branch (`lab/sonic-moe`, `envs/xfer`).
> **Reproducibility**: BF16 3 runs (GPU 3,4,5), FP8 2 runs (GPU 5,6) — all identical (deterministic).

### 3.1 Per-Checkpoint Comparison

| Checkpoint | Official BF16 (MiB) | FP8 Frontier (MiB) | Delta | Note |
|-----------|:---:|:---:|:---:|------|
| 00_empty | 0.0 | 0.0 | — | bare CUDA context |
| 01_model | 216.1 | 216.1 | 0 | same params (bf16 storage) |
| 02_input | 312.1 | 312.1 | 0 | x(48M) + dout(48M) + model |
| 03_post_warmup | 376.3 | 487.7 | **+111.4** | FP8 weight caches from warmup |
| **04_fwd_peak** | **1385.7** | **1337.7** | **−48.0 (−3.5%)** | FP8 saves on z/y1 storage |
| 05_post_fwd | 761.6 | 649.8 | **−111.8** | FP8 freed z/y1 bf16 via resize_(0) |
| **06_bwd_peak** | **1411.6** | **1366.9** | **−44.6 (−3.2%)** | FP8 z stored as fp8 (−192M), but dz+dz_fp8 coexist |
| 07_post_bwd | 640.3 | 751.7 | +111.4 | FP8 backward weight caches |
| 08_cleanup | 328.3 | 439.7 | +111.4 | FP8 caches persist (invalidated at optimizer.step) |

### 3.2 Key Metrics

| Metric | Official BF16 | FP8 Frontier | Delta |
|--------|:---:|:---:|:---:|
| **Forward peak** | **1385.7 MiB** | **1337.7 MiB** | **−48.0 MiB (−3.5%)** |
| **Backward peak** | **1411.6 MiB** | **1366.9 MiB** | **−44.6 MiB (−3.2%)** |
| Net forward activation (peak − base) | 1009.4 MiB | 850.0 MiB | **−159.4 MiB (−15.8%)** |
| Net backward activation (peak − post_fwd) | 650.0 MiB | 717.1 MiB | +67.1 MiB (+10.3%) |
| Post-cleanup residual | 328.3 MiB | 439.7 MiB | +111.4 MiB |

### 3.3 Analysis

**Forward peak (−48 MiB / −3.5%)**:
- FP8 saves ~159 MiB on z/y1 (resize_(0) frees bf16 after fp8 quant)
- FP8 warmup base is +111 MiB higher (weight caches from previous iter)
- Net: 159 − 111 = **−48 MiB absolute peak reduction**

**Backward peak (−44.6 MiB / −3.2%)**:
- FP8 saves 192 MiB by storing z as fp8 (vs bf16 in BF16 path)
- FP8 spends +198 MiB on dz_fp8 prequant + weight caches
- z_fp8 released after dgated GEMM (−198 MiB)
- Net: modest reduction due to offsetting effects

**Post-cleanup residual (+111.4 MiB)**:
- Entirely from FP8 weight caches: w1T VARLEN (74.3 MiB) + w2 FUSED (37.1 MiB)
- These auto-invalidate at `optimizer.step()` via `w._version` increment
- In training with optimizer: **effective residual delta = 0**

### 3.4 Weight cache lifecycle

| Cache | Size (MiB) | Forward | Backward | Post-step |
|-------|:---------:|---------|----------|-----------|
| w1 fused (FUSED_CACHE) | 74.3 | populated → cleared | — | invalidated |
| w2 varlen (VARLEN_CACHE) | 37.1 | populated → evicted | — | invalidated |
| w1T varlen (VARLEN_CACHE) | 74.3 | — | populated → persists | invalidated |
| w2 fused (FUSED_CACHE) | 37.1 | — | populated → persists | invalidated |

### 3.5 Optimizations applied (Session 41)
- z_fp8/z_raw_scales released immediately after dgated GEMM (−198 MiB backward peak)
- w2 varlen evicted after down-proj forward (−37 MiB during ctx save)
- w1 fused cleared before down-proj (−74 MiB during down-proj)
- Redundant FP8 protocol adapter eliminated (−250µs when fp8_protocol is not None)

---

## 4. Architecture — Zero-Materialization FP8

SonicMoE's core principle: **no TK-sized FP8 activation materialized in HBM**.

### Forward path (fused_gated, aligned)
1. `quantize_and_pack_activation(x)` → T-sized FP8 + T-sized ISA scales
2. `_gather_isa_packed_scales_kernel` → TK-sized ISA scales (~27µs)
3. `GemmGatedSm100ZeroMat`: T-FP8 + A_idx + TK-scales → z(bf16) + y1(bf16)
4. `fused_z_save_y1_quant(z, y1)` or split quant → z_fp8 + y1_fp8

### Backward path (fused_gated, aligned)
1. `quantize_and_pack_activation(dout)` + scale_gather → dout FP8 + TK scales
2. `GemmDGatedFP8CLoadSm100ZeroMat`: dout_FP8 + w2T_FP8 + z_fp8(preact) → dz + y1s
3. wgrad: `gemm(dout.T, y1s, out=dw2)`; `gemm(x.T, dz, out=dw1)`
4. actgrad: `blockscaled_fp8_gemm_varlen(dz_fp8, w1T_fp8)` → dx

---

## 5. FP8 Flag Reference

### Required
| Flag | Default | Set to |
|------|---------|--------|
| `SONIC_MOE_FP8_MODE` | off | **perf** |
| `USE_QUACK_GEMM` | 0 | **1** |

### Optimal defaults (do NOT change)
| Flag | Default | Effect |
|------|---------|--------|
| `SONIC_MOE_FP8_FUSED_GATED` | 1 | Fused GEMM+SwiGLU+descale |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | 1 | z stored as FP8 (−192 MiB) |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | 1 | Fused SwiGLU+quant (fallback path) |

### Optional
| Flag | Default | Effect | Cost |
|------|---------|--------|------|
| `SONIC_MOE_FP8_FUSED_ZY1_QUANT` | 0 | Fused z+y1 quant (1 launch) | +96 MiB fwd peak, saves 64µs |
| `SONIC_MOE_FP8_EPILOGUE_QUANT` | 0 | GEMM epilogue z-scale compute | +JIT time, marginal gain |
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | 0 | Skip 128-alignment check | Crashes if misaligned |

---

## 6. Bugs & Lessons

### Bug 1: GemmDGatedFP8CLoadSm100 SFA layout — ✅ FIXED
Root cause: SFA used T rows instead of TK rows for scale layout with gather_A.
Fix: `GemmDGatedFP8CLoadSm100ZeroMat` uses `mD.shape[0]` for SFA M-dimension.

### Bug 2: dw2_base permutation — ✅ FIXED
Fix: `out=dw2.permute(2, 0, 1)` → correct layout.

### Process contamination ⚠️ CRITICAL
`SONIC_MOE_FP8_MODE` is cached at import time. **Any BF16 vs FP8 comparison MUST use separate subprocesses.** Same-process gives fake bit-identical results.

### Measurement methodology
- **CUDA events** include Python dispatch overhead (~0.5-1ms). Use for trends, not absolute numbers.
- **nsys GPU Projection** is ground truth for kernel compute time. Always use idle GPUs (0% util, <5 MiB).
- **ncu** uses base clock by default (`--clock-control=base`); nsys uses boost clock. Don't compare durations across tools.
- Weight cache effects: benchmarks without optimizer.step() see cache hits from iter 2+; training sees miss every step.

---

## 7. Dead Ends (do NOT retry)

| Attempt | Why it failed |
|---------|--------------|
| FP8 wgrad | Colwise quant overhead > GEMM savings at all Ernie shapes |
| `torch.as_strided` for fake TK shape | PyTorch storage bounds check rejects |
| Aggressive VARLEN cache eviction in backward | Causes 308µs permute+contiguous re-quant every iter |
| FP8 down-proj at I=1536 | Quant cost ≈ GEMM savings, net zero |
| Same-process FP8/BF16 comparison | Process contamination gives fake results |

---

## 8. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 forward/backward logic, `_FP8Config`, weight cache management |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches, `fused_z_save_y1_quant` |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | ZeroMat kernel classes (SFA layout fix) |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated + epilogue quant mixin |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated + FP8CLoad mixin |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU+quant Triton kernels |
| `tests/fp8_large_project_contract_test.py` | 31-test precision+correctness gate |
| `tools/nsys_benchmark.py` | nsys-compatible benchmark script |
| `tools/precision_audit.py` | Subprocess-isolated precision audit |
| `benchmarks/nsys_run/nsys_s41_*.sqlite` | Session 41 nsys data |

---

## 9. Next Steps (prioritized)

### P0: GEMM Epilogue FP8 Output
**Potential: −177µs forward quant overhead → 1.18× total speedup.**

Current `GemmGatedBlockscaledQuantMixin` computes E8M0 scales in the epilogue but still writes D as bf16. If D could be written as `float8_e4m3fn` directly, it eliminates:
- `_fused_z_save_y1_quant_kernel` (166µs) or split quant z+y1 (177µs)
- The standalone `.to(fp8)` cast

**Feasibility assessment**: CUTLASS CuTeDSL supports `Float8E4M3FN` as a dtype, and TMA stores for 8-bit elements exist. The challenge is:
1. D and PostAct have different dtypes (z=fp8, y1=fp8) — need two TMA store atoms
2. ISA scale packing must happen in the epilogue (per-tile, not per-subtile)
3. SM100 MMA register-to-thread mapping must be understood for warp-level amax reduction

**Recommendation**: Start with PostAct (y1) → fp8 in epilogue (simpler, single output), then extend to D (z).

### P1: End-to-end Training Validation
Run actual training loop with optimizer.step() and compare FP8 vs BF16 loss curves. Current data is forward+backward only.

### P2: Shape Scaling
Test T=4096, T=16384, I=2048+ shapes. At larger I, down-proj FP8 becomes beneficial.

### P3: Multi-node Expert Parallelism
Validate FP8 path with distributed expert parallelism.

---

## 10. Session 41 Changelog

| Change | Impact | Files |
|--------|--------|-------|
| `_FP8Config` object | Eliminates ~15 os.getenv/iter | `__init__.py` |
| Adapter skip | −250µs (when fp8_protocol≠None) | `__init__.py` L1783-1788 |
| z_fp8 early release | −198 MiB backward peak | `__init__.py` L1332-1334 |
| w2 forward eviction | −37 MiB during ctx save | `__init__.py` L1086-1089 |
| w1 fused clear | −74 MiB during down-proj | `__init__.py` L1828-1830 |
| `SONIC_MOE_FP8_FUSED_ZY1_QUANT` | −64µs/iter, +96 MiB fwd peak | `__init__.py` L722-740 |
| nsys benchmark tooling | Reproducible profiling | `tools/nsys_benchmark.py` |

---

## 11. Quick Validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Test suite (31 tests, ~100s)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Precision audit (subprocess-isolated, 3 seeds)
CUDA_VISIBLE_DEVICES=0 python tools/precision_audit.py --gpu 0 --seeds 42,123,777

# nsys GPU Projection
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  --sample=none -o /tmp/sonic_fp8 \
  python tools/nsys_benchmark.py --mode fp8 --gpu 0 --warmup 5 --iters 10
nsys export --type=sqlite --output=/tmp/sonic_fp8.sqlite /tmp/sonic_fp8.nsys-rep
python tools/nsys_full_breakdown.py /tmp/sonic_bf16.sqlite /tmp/sonic_fp8.sqlite --labels bf16 fp8
```
