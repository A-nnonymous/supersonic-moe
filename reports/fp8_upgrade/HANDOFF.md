# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-13 (Session 46 — Pythonic config, unaligned padding, TILE_ROWS tuning, idle-GPU benchmarks)
> **Branch:** `native-fp8-exploration`
> **Status:** Zero-materialization FP8 + iso32 weights + optional stash + CuTe DSL wgrad quant + **Pythonic config API** + **unaligned FP8 padding**. Contract suite: **33/34 tests + 20 subtests PASS** (1 memory test expected — FP8 trades memory for speed).

---

## 0. Bottom Line

Session 46 measurements are from an **idle B200** (0% util, 182 GiB free) using **CUDA events** with subprocess isolation. Two shapes tested: Ernie (I=1536) and I=2048.

### Performance (CUDA events, 12 runs, drop 2 min + 2 max, trimmed mean)

| Shape | BF16 | FP8 | Speedup | Takeaway |
|-------|:----:|:---:|:-------:|----------|
| **Ernie** (T=8192, H=3072, I=1536, E=8, K=8) | 4.97 +/- 0.02 ms | 5.00 +/- 0.03 ms | **0.993x** | Break-even; I=1536 GEMM too small for FP8 gains to offset quant overhead |
| **I=2048** (T=8192, H=3072, I=2048, E=8, K=8) | 6.56 +/- 0.01 ms | **5.82 +/- 0.01 ms** | **1.127x** | Meaningful speedup at larger intermediate size |

### Memory (subprocess-isolated peak)

| Shape | BF16 Peak | FP8 Peak | Delta |
|-------|-----------|----------|-------|
| **Ernie** | 1460 MiB | 1851 MiB | **+391 MiB (+27%)** |
| **I=2048** | 1876 MiB | 2331 MiB | **+455 MiB (+24%)** |

FP8 uses **more** memory (weight caches + FP8 shadow weights). Use **FP8 + stash** to trade CPU memory for GPU savings.

### Precision (5 seeds x 2 shapes, all PASS)

| Tensor | Ernie RRMSE | I=2048 RRMSE | Correlation | Status |
|--------|:-----------:|:------------:|:-----------:|:------:|
| output | 6.51% | 6.51% | 0.9979 | PASS |
| dx | 6.52% | 6.54% | -- | PASS |
| dw1 | 4.69% | 4.72% | -- | PASS |
| dw2 | 4.84% | 4.88% | -- | PASS |

All well within guardrails: **RRMSE < 10%**, **correlation > 0.99**. Extremely stable across seeds (std < 0.001%).

---

## 1. Sessions 45-46 Changes

| Change | Impact | Files |
|--------|--------|-------|
| **Pythonic Config API** (`SonicMoEConfig`) | Replace env-var flags with a `SonicMoEConfig` dataclass + thread-local context manager. 10 fields, priority chain: config > context manager > env var | `sonicmoe/config.py` (NEW), `sonicmoe/__init__.py`, `sonicmoe/moe.py`, `sonicmoe/functional/__init__.py`, `sonicmoe/functional/utils.py` |
| **wgrad FP8 default-ON** | `_use_fp8_wgrad()` now defaults True at all shapes. Stream-overlapped quant pipeline in up-proj backward | `sonicmoe/functional/__init__.py` |
| **NCU-driven quant analysis** | All 4 hot quant kernels at 89-99% HBM bandwidth utilization -> limited kernel-level optimization ROI | Analysis only (no code changes needed) |
| **Unaligned FP8 padding** | `_padded_blockscaled_gated_forward()` pads expert segments to 128 for FP8 GEMM, then unpads. Down-proj `elif cfg.enabled:` branch. Non-aligned shapes now get FP8 forward + BF16 backward | `sonicmoe/functional/__init__.py`, `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` |
| **TILE_ROWS tuning** | `_quantize_flat_v2_kernel` 32->128, `_gather_isa_packed_scales_kernel` 32->128 (3 sites), `_pad_quantize_and_pack_kernel` 16->32 | `sonicmoe/functional/__init__.py`, `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` |
| **Import fix** | `_padded_blockscaled_gated_forward` was importing low-level kernel wrapper -> fixed to high-level `gemm_gated` from `quack_utils.__init__` | `sonicmoe/functional/__init__.py` |
| **Idle-GPU benchmarks** | Clean CUDA-event numbers on truly idle B200 (0% util). Both Ernie and I=2048 shapes | `benchmark_final.json` |
| **5-seed precision audit** | 5 seeds (42, 123, 456, 789, 1024) for both shapes. All <7% RRMSE, >0.997 correlation | `benchmark_final.json` |

---

## 2. Measurement Methodology and Caveats

1. **Idle GPU confirmed:** All 8 local GPUs showed 0% util and 182 GiB free at measurement time.
2. **Subprocess isolation mandatory:** `SONIC_MOE_FP8_MODE` and `USE_QUACK_GEMM` are cached at import time. BF16 and FP8 measurements MUST run in separate subprocesses.
3. **JIT warmup:** Triton JIT compilation takes ~2ms extra on first few FP8 iterations. 15+ warmup iterations required for stable CUDA-event timing.
4. **CUDA events, not wallclock:** CUDA events measure GPU-side time accurately regardless of CPU load.
5. **Trimmed mean:** 12 iterations, drop 2 min + 2 max, mean of remaining 8. Typical std < 0.03ms on idle GPU.
6. **Do not compare BF16 and FP8 in-process.** Separate subprocesses remain mandatory due to import-time caching.

---

## 3. Pythonic Config API

### Design

```python
from sonicmoe import SonicMoEConfig

# Full control -- no env vars needed
cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True,
                     fp8_wgrad=True, assume_aligned=True)
with cfg.activate():
    output, loss = moe(x, use_fp8=True)
```

**Priority chain:** `SonicMoEConfig` (thread-local) > `enable_fp8()`/`enable_quack_gemm()` context managers > env vars

**10 config fields** (all default `None` -> fall back to env var):
- `use_fp8`, `use_quack_gemm`, `fp8_wgrad`, `fused_gated`, `save_z_fp8`
- `fused_swiglu_quant`, `epilogue_quant`, `fused_zy1_quant`
- `assume_aligned`, `stagewise_memory`

**Thread safety:** Uses `threading.local()` for thread-local active config.

**Auto-enable:** Setting `use_fp8=True` automatically sets `use_quack_gemm=True` in `__post_init__`.

### Key files
- `sonicmoe/config.py` -- `SonicMoEConfig` dataclass + `get_active_config()`/`set_active_config()`
- All `_use_*()` helpers in `functional/__init__.py` now check config first

---

## 4. Critical Design Context

### Zero-Materialization FP8

SonicMoE core design avoids materializing gathered TK-sized activations:
- `quantize_and_pack_activation(x)` -> T-sized FP8 + T-sized scales
- `_gather_isa_packed_scales_kernel` -> TK-sized ISA-packed scales (~54us, 1.6% of CUDA time)
- `GemmGatedSm100ZeroMat` kernel: T-FP8 + A_idx + TK-scales (no TK FP8 materialization)

### Unaligned FP8 Padding

When expert segments are not 128-aligned:
- **Up-proj:** `_padded_blockscaled_gated_forward()` pads segments to 128, runs zero-mat FP8 GEMM+SwiGLU, then unpads
- **Down-proj:** `blockscaled_fp8_gemm_varlen(assume_aligned=False)` handles padding internally
- **Backward:** stays BF16 (QuACK BF16 handles unaligned natively)
- Padding rows gather from index 0 (safe, output discarded after unpad)
- `_get_padding_plan()` is cached by cpu_tuple key

### Why FP8 Uses More Memory

FP8 weight caches (`iso32_fp8_w1`, `iso32_fp8_w2`, associated scales) persist across iterations for reuse in backward. At Ernie shape this adds ~391 MiB. The **stash** path moves bf16 master weights to CPU during fwd+bwd, recovering this cost.

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/config.py` | **Pythonic config API** -- `SonicMoEConfig` dataclass, thread-local storage, context manager |
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, stash flow, cache invalidation, unaligned padding |
| `sonicmoe/moe.py` | MoE class, `refresh_fp8_shadow_weights`, stash/unstash API, `config` parameter |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, weight caches, blockscaled helpers, TILE_ROWS tuning |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-materialization FP8 CUTLASS kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | Gated GEMM wrapper with zero-mat auto-selection |
| `sonicmoe/quack_utils/gemm_dgated.py` | Backward gated GEMM wrapper with zero-mat auto-selection |
| `tools/introspect.py` | Main experiment entry point -- trace, repeated benchmark, profiler, precision audit |
| `tests/fp8_large_project_contract_test.py` | Correctness gate (34 tests + 20 subtests) |
| `tests/test_unaligned_fp8_padded.py` | Unaligned FP8 padding smoke tests (3 tests) |
| `docs/FP8_ARCH_SPEC.md` | Full architecture spec, data flow, memory lifecycle |

---

## 6. Validation / Refresh Commands

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract suite
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf   python -m pytest -q tests/fp8_large_project_contract_test.py

# Unaligned FP8 smoke tests
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf   python -m pytest -v tests/test_unaligned_fp8_padded.py

# Full artifact refresh (local GPU)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf   python tools/introspect.py --mode full --gpu 0   --precision-seeds 42,123,456,789,1024 --bench-repeats 3 --profile-trials 2
```

---

## 7. Dead Ends (verified, do NOT retry)

| Approach | Why it fails | Session |
|----------|-------------|---------|
| FP8 wgrad colwise quant at I=1536 | SM contention -> 0.887x net negative, saves 384 MiB but slower | 45 |
| torch.as_strided for fake TK shape | PyTorch storage bounds check rejects it | 41-42 |
| Rowwise quant + strided view for wgrad | HW requires contiguous K groups | 45 |
| Transpose + rowquant for wgrad | Transpose alone 1509us > colwise 260us | 45 |
| Micro-optimizing quant kernels at 89%+ HBM | All 4 hot kernels at 89-99% bandwidth utilization -> ceiling reached | 45 |
| Fused CuTe dual quant | 288M instructions (3.6x bloat) | 43 |
| cp.async aligned to stride-33 smem | Alignment constraint prevents it | 43 |
| Smem-mediated fp8 vectorized store | -44% instructions but +96% L1 traffic | 43 |
| Full loop unroll for fp8 store | 64 regs -> 44% occupancy; runtime loop faster | 43 |
| Blind wallclock comparison on busy cluster | Shared-node contention swamps kernel-time signal | 44 |

---

## 8. Next Steps

### P0: Larger shapes (I=4096, I=8192)
FP8 speedup scales with intermediate size. At I=2048 we see 1.127x. Testing at I=4096+ should show even stronger gains.

### P1: Stash benchmarks
FP8+stash path needs CUDA-event benchmarks on idle GPU to quantify the CPU<->GPU transfer cost.

### P2: Visualization refresh
Run `python -m visualization` to regenerate figures from updated JSON artifacts.

### P3: Harden `cluster_idle_launch.py`
Make the scan resilient to bad hosts so "find me an idle GPU" becomes reliable.

---

## 9. Information Sources

| Resource | Location | Use |
|----------|----------|-----|
| `benchmark_final.json` | repo root | **Session 46 authoritative data**: timing, memory, precision for 2 shapes |
| `manifest.json` | repo root | Tensor lifetimes, theoretical memory, precision audit |
| `kernel_breakdown.json` | repo root | Aggregated GPU-projection timing |
| `mem_breakdown.json` | repo root | Checkpoint-by-checkpoint allocator snapshots |
| `reports/nsys_final/kernel_breakdown.json` | reports/nsys_final/ | Executive-summary kernel breakdown |
| QuACK 0.3.7 CuTe DSL kernels | envs/xfer/lib/.../quack/ | Reference patterns |
| FP8 architecture spec | `docs/FP8_ARCH_SPEC.md` | Full data flow, memory lifecycle, weight-cache system |
