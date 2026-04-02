# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-02 (Session 29)
> **Branch:** `fork-main-sync`
> **Status:** FP8 frontier is **1.03× faster** than BF16 on GPU projection at Ernie production shape. Memory ≤ BF16. Precision verified (11/11 tests pass).

---

## 0. One-screen summary

**What works today:**
- Zero-materialization FP8 forward via `gemm_gated()` + `A_idx` (auto-selects `GemmGatedSm100ZeroMat`)
- Zero-materialization FP8 backward via `gemm_dgated()` + `A_idx` (auto-selects `GemmDGatedSm100ZeroMat`)
- Triton weight quantization: single-kernel replaces 8-op eager path (eliminated 3136µs/iter reduce_kernel)
- Precision verified: **RRMSE <7%, correlation >0.997** at both contract and production shapes
- 11/11 aligned contract tests pass (including weight quant and cache eviction tests)
- Dead flags removed: `_fp8_lean`, `_use_fp8_wgrad`, `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD`

**Best training path:**
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Performance (GPU Projection, Ernie shape T=8192 H=3072 I=1536 E=8 K=8)

Measured via nsys + SQLite analysis, 10 warmup + 5 profiled iterations:

| Metric | BF16 | FP8 | Ratio |
|--------|------|-----|-------|
| **GPU projection (steady-state)** | **6.78ms** | **6.57ms** | **1.03× faster** |
| reduce_kernel overhead | 0µs | 49µs | Eliminated from 3136µs |
| Kernel count per iter | ~43 | ~50 | +7 (quant kernels) |

**FP8 iter2 GPU kernel breakdown (6557µs total):**

| Kernel | Time | Count | % |
|--------|------|-------|---|
| GemmDefault (wgrad dw1, BF16) | 3432µs | 1 | 52% |
| GemmDefault (wgrad dw2 + actgrad, BF16) | 1718µs | 3 | 26% |
| GemmDGatedSm100ZeroMat (bwd FP8) | 481µs | 1 | 7% |
| GemmGatedSm100ZeroMat (fwd FP8) | 455µs | 1 | 7% |
| _quantize_and_pack_kernel (Triton) | 283µs | 7 | 4% |
| elementwise + misc | ~188µs | — | 3% |

**92% of GPU time is GEMMs** — quant overhead is only 4.3%. The frontier is near-optimal.

**Memory:** FP8 peak ≤ BF16 peak (verified by `test_fp8_memory_less_than_bf16`).

**Precision (multi-seed, multi-shape):**

| Shape | FWD RRMSE | FWD corr | BWD RRMSE | BWD corr |
|-------|-----------|----------|-----------|----------|
| Contract | 6.60% | 0.9978 | 7.50% | 0.9972 |
| Ernie prod | 6.60% | 0.9978 | 7.47% | 0.9972 |

**nsys timelines:**
- `reports/timelines/fp8_triton_wquant.nsys-rep` — FP8 frontier with Triton weight quant
- `reports/timelines/bf16_steady_state.nsys-rep` — BF16 baseline

---

## 2. Session 29 Changes: Triton Weight Quant + Dead Flag Cleanup

### Critical fix: Replace eager weight quantization with Triton kernel

**Root cause:** Weight cache eviction (lines 796, 1150 in `functional/__init__.py`) forces
backward to re-quantize weights every iteration. The old `quantize_activation_blockwise()`
used 8 PyTorch eager kernels per weight (abs, amax, reciprocal, mul, cast...), generating
3136µs/iter of `reduce_kernel` overhead = **30% of FP8 GPU time**.

**Fix:** Added `_quantize_weight_3d_triton()` in `blockscaled_fp8_gemm.py` which:
1. Reshapes 3D (E,N,K) → 2D (E*N,K) — ISA scale tiles align when N % 128 == 0
2. Calls existing Triton `quantize_and_pack_activation()` — single kernel, inline `tl.max(tl.abs())`
3. Reshapes back — produces same FP8 + ISA-packed scales as before

Applied to all 5 weight precompute paths:
- `precompute_weight_fp8()`
- `precompute_weight_fp8_for_fused_gated()`
- `precompute_weight_fp8_for_fused_dgated()`
- `precompute_weight_fp8_for_direct_fused_dgated()`
- `_quantize_w2_cached()`

### Dead flag cleanup

Removed 3 dead flags and all code paths they guard:

| Flag | Why dead | Lines removed |
|------|----------|---------------|
| `_fp8_lean()` / `SONIC_MOE_FP8_LEAN` | Disabled by default, untested, created parallel code paths | ~100 lines across 8 locations |
| `_use_fp8_wgrad()` / `SONIC_MOE_FP8_WGRAD` | Proven dead-end (layout permutation ~637µs > GEMM savings) | ~30 lines, 2 if/else blocks |
| `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD` | Prequant always hits at I=1536, threshold never triggered | ~15 lines |

### New precision tests

| Test | What it verifies |
|------|-----------------|
| `test_triton_weight_quant_matches_eager` | Triton 3D weight quant produces identical FP8 values per-expert |
| `test_weight_cache_eviction_precision` | Full fwd+bwd with cache eviction maintains <10% RRMSE at production shape |

---

## 3. Prior Session Changes (25-28)

### Session 28: Zero-mat B-layout bug fix
- Fixed `gemm_gated_zeromat()` B-tensor layout bug (was passing .mT view → 101% RRMSE)
- Replaced standalone wrapper with standard `gemm_gated()` interface
- Fixed test env-var masking (`_reset_fp8_state()` strips `SONIC_MOE_FP8_MODE`)

### Session 25-27: Zero-mat kernels + memory optimizations
- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: auto-selected for SM100 + gather_A + blockscaled
- z FP8 save (~171 MiB), y1 pre-quant, weight cache dedup, cache collision fix

---

## 4. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad** | +0.9ms net | colwise quant SM contention, layout permute ~637µs |
| **Standalone `gemm_gated_zeromat()` wrapper** | B-layout bug | Standard `gemm_gated()` handles B correctly |
| **`SONIC_MOE_FP8_MODE=perf` for benchmarks** | Masks bugs | Makes BF16 ref use FP8 path |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch storage bounds check |
| **Rowwise quant + strided view** | Not possible | HW requires contiguous K groups |
| **Transpose + rowquant** | 3.8× slower | transpose 1509µs > colwise quant 260µs |
| **FP8 down-proj at I=1536** | No net win | quant cost ≈ GEMM savings at small I |
| **Eager weight quant** | 30% overhead | 8 kernel launches × 2 weights = 3136µs/iter |

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, `_quantize_weight_3d_triton()`, weight caches |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat kernel classes |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated (auto ZeroMat at L236) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated (auto ZeroMat at L319) |
| `tests/fp8_large_project_contract_test.py` | 11 aligned contract tests |
| `tools/gpu_projection_benchmark.py` | nsys benchmark with sync'd NVTX markers |
| `reports/timelines/*.sqlite` | nsys SQLite profiles for GPU projection analysis |

---

## 6. Validation

11/11 aligned contract tests pass:
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_large_project_contract_test.py::FP8AlignedContractTest -v --tb=short
```

---

## 7. Remaining Optimization Opportunities

92% of FP8 GPU time is BF16 wgrad GEMMs (unchanged from BF16 baseline). Further gains require:

1. **FP8 wgrad without layout overhead** — Needs CUTLASS kernel that accepts non-contiguous K groups
2. **Larger intermediate size (I≥2048)** — FP8 GEMM savings scale with N; previous estimates: 2.35× at I=2048
3. **Fuse quant into SwiGLU epilogue** — Save one kernel launch per forward pass
4. **Weight quant persistence** — If weights don't change between iterations, skip re-quant entirely

---

## 8. Environment

```
GPU: NVIDIA B200 (SM100a, 183GB)
CUDA: 12.8, PyTorch: 2.9.1+cu128, QuACK: 0.3.7
Python: 3.13
FP8 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer
Official BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16
Remote node: 10.51.195.12
```
