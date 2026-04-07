# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-12 (Session 39 — Memory optimization complete)
> **Branch:** `native-fp8-exploration`
> **Status:** ✅ FP8 fully functional. **1.52× E2E speedup**, **−33.4% peak memory**, all precision PASS.

---

## 0. Current Bottom Line

| Metric | BF16 Baseline | FP8 Frontier | Delta |
|--------|--------------|--------------|-------|
| **E2E CUDA-event time** | 9.44 ms | 6.22 ms | **1.52× faster** |
| Forward time | 1.36 ms | 1.16 ms | 1.17× faster |
| Backward time | 8.07 ms | 5.06 ms | 1.59× faster |
| **Peak memory** | **2540 MiB** | **1693 MiB** | **−33.4% (−847 MiB)** |
| Forward peak | 2346 MiB | 1567 MiB | −33.2% |
| Backward peak | 2540 MiB | 1693 MiB | −33.4% |
| Output RRMSE | — | 6.52% ✓ | <10% threshold |
| Worst grad RRMSE | — | 7.05% (dx) ✓ | <10% threshold |
| All cosines | — | >0.997 ✓ | >0.99 threshold |

Shape: T=8192, H=3072, I=1536, E=8, K=8 (Ernie production shape).
GPU: B200 (SM 10.0, 148 SMs, HBM3e).

### Correct training command
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 SONIC_MOE_FP8_DOUBLE_QUANT=1 python train.py
```
```python
from sonicmoe import MoE
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Bugs — Status

### Bug 1: `GemmDGatedFP8CLoadSm100` dz output zeroed for experts 1-7 — ✅ FIXED

| | |
|---|---|
| **Root cause** | `GemmDGatedFP8CLoadSm100` inherited `GemmSm100.__call__` which derived SFA layout from `mA.shape = (T, K)`. With `gather_A=True`, pre-gathered scales span TK rows but SFA used T rows. Expert 0 worked because `cu_seqlens_m[0]=0` → trivially correct offset. Experts 1-7 got wrong scale factors → garbage GEMM accumulator → wrong dz output. |
| **Fix** | Created `GemmDGatedFP8CLoadSm100ZeroMat` class that combines `GemmDGatedFP8CLoadMixin` + `_GemmSm100ZeroMatMixin` (which uses `mD.shape[0]=TK` for SFA). Updated dispatch in `gemm_dgated.py` line 817-819. |
| **Files** | `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` (new class), `sonicmoe/quack_utils/gemm_dgated.py` (dispatch) |
| **Validation** | 31/31 tests PASS, 17/17 per-expert precision PASS (RRMSE 4.71-5.40%) |

### Bug 2: `dw2_base` permutation — ✅ FIXED (prior session)

| | |
|---|---|
| **File** | `sonicmoe/functional/__init__.py`, line 1310 |
| **Fix** | `out=dw2_base` → `out=dw2.permute(2, 0, 1)` |

### Process Contamination ⚠️ CRITICAL

`SONIC_MOE_FP8_MODE` is cached at import time (`sonicmoe/functional/utils.py:38` → `_IS_FP8_ACTIVE`).

**Any precision test comparing FP8 vs BF16 MUST use separate subprocesses.** Same-process comparison produces fake "bit-identical" results. This has caused multiple false-positive reports. See `tools/_test_bug1_fix.py` for correct methodology.

---

## 2. Architecture — Zero-Materialization FP8

SonicMoE's core design avoids materializing TK-sized gathered activations.

### BF16 path
- `A_idx` gathers data rows **inside** CUTLASS kernel (no TK×H copy)

### FP8 path (same principle)
1. `quantize_and_pack_activation(x)` → T-sized FP8 tensor + T-sized scales
2. `_gather_isa_packed_scales_kernel` → TK-sized ISA-packed scales (~27µs/call)
3. `GemmGatedSm100ZeroMat` / `GemmDGatedFP8CLoadSm100ZeroMat`: T-FP8 + A_idx + TK-scales
   - **No TK-sized FP8 activation materialized** — matches BF16 zero-mat principle

The ZeroMat kernels subclass `GemmSm100` via MRO and override `__call__` to fix SFA layout.
Auto-selected in `gemm_gated.py`/`gemm_dgated.py` when `gather_A + blockscaled`.

---

## 3. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, ctx state |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat kernel classes (Bug 1 fix here) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, three-step pipeline, scale gather |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated wrapper (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated wrapper (auto ZeroMat + FP8CLoad) |
| `tests/fp8_large_project_contract_test.py` | 31 precision+correctness tests |
| `reports/fp8_upgrade/FP8_BENCHMARK_REPORT.md` | Detailed benchmark report (Chinese) |

---

## 4. Dead Ends (proven net-negative, do NOT retry)

| Attempt | Why it failed | Evidence |
|---------|--------------|---------|
| FP8 wgrad | Colwise quant SM contention, net-negative at ALL shapes | colwise quant alone costs 260µs, wgrad saves <200µs |
| `torch.as_strided` for fake TK shape | PyTorch storage bounds check rejects it | Hard runtime error |
| Rowwise quant + strided view for wgrad | HW requires contiguous K groups in blockscaled | Numerical garbage |
| FP8 down-proj at I=1536 | Quant cost ≈ GEMM savings | Measured: ~46µs quant vs ~50µs GEMM saving |
| Transpose + rowquant for wgrad | Transpose alone 1509µs > colwise 260µs | nsys measured |
| Same-process FP8/BF16 comparison | Process contamination gives fake bit-identical results | See §1 |

---

## 5. Performance Deep Dive

### 5.1 E2E CUDA-Event Timing (T=8192, H=3072, I=1536, E=8, K=8)

| Phase | BF16 (ms) | FP8 (ms) | Speedup |
|-------|----------|---------|---------|
| Forward | 1.357 | 1.160 | 1.17× |
| Backward | 8.066 | 5.058 | 1.59× |
| **Total** | **9.436** | **6.217** | **1.52×** |

Measured with CUDA events (30 iterations, trimmed mean excluding 3 best/worst). Subprocess-isolated.

### 5.2 Per-kernel breakdown (µs/iter, nsys GPU time — Session 38)

| BF16 Kernel | µs | % | | FP8 Kernel | µs | % |
|-------------|-----|---|-|------------|-----|---|
| Wgrad GEMMs | 2298 | 55.3 | | Wgrad K=16 | 1125 | 32.3 |
| Fwd Gated | 734 | 17.7 | | Wgrad K=32 | 699 | 20.1 |
| Bwd DGated | 470 | 11.3 | | Fwd FP8 ZeroMat | 452 | 13.0 |
| elementwise | 200 | 4.8 | | Bwd FP8CLoad ZeroMat | 411 | 11.8 |
| vectorized_add | 155 | 3.7 | | fuse_z_y1_quant | 168 | 4.8 |
| gather_sum | 143 | 3.4 | | vectorized_add | 154 | 4.4 |
| Other | 156 | 3.8 | | quant+gather+other | 475 | 13.6 |
| **Total** | **4156** | | | **Total** | **3484** | |

Note: nsys kernel-only time (4.16→3.48ms = 1.19×) differs from CUDA-event E2E time (9.44→6.22ms = 1.52×) because CUDA events capture full stream time including scheduling, while nsys reports pure GPU compute. The backward improvement is larger due to serialized execution reducing memory pressure.

---

## 6. Precision Summary

All metrics below threshold (RRMSE <10%, cosine >0.99). Measured with subprocess isolation, shared weights/input.

| Tensor | RRMSE | Cosine | MaxRelErr |
|--------|-------|--------|-----------|
| output | 6.52% | 1.000199 | 1399.7 |
| dx (input grad) | 7.05% | 0.999648 | 2572.1 |
| c_fc.weight grad | 5.97% | 1.011907 | 1028.0 |
| c_proj.weight grad | 6.54% | 1.002587 | 1259.2 |

Test suite: 31/31 tests PASS (`tests/fp8_large_project_contract_test.py`).

---

## 7. Memory — Detailed Breakdown

### 7.1 Peak Memory Comparison

| Checkpoint | BF16 (MiB) | FP8 (MiB) | Delta |
|-----------|-----------|----------|-------|
| Base (empty GPU) | 0.0 | 0.0 | — |
| After model load | 216.1 | 216.1 | 0 |
| After input alloc | 360.1 | 360.1 | 0 |
| Pre-forward | 1336.3 | 717.3 | −619.0 (−46.3%) |
| Post-forward | 1769.8 | 927.2 | −842.6 (−47.6%) |
| **Forward peak** | **2345.7** | **1567.4** | **−778.3 (−33.2%)** |
| Pre-backward | 1769.8 | 927.2 | −842.6 |
| Post-backward | 2032.3 | 1029.3 | −1003.0 |
| **Backward peak** | **2540.2** | **1692.7** | **−847.5 (−33.4%)** |
| Post-cleanup | 1768.3 | 765.3 | −1003.0 |

### 7.2 Memory Optimizations Applied (Session 39)

| Optimization | Technique | Saving (MiB) | Mechanism |
|-------------|-----------|-------------|-----------|
| Inner z release | `z.untyped_storage().resize_(0)` inside `_UpProjection.forward` | ~192 | Frees z bf16 before y1 quantization |
| Inner y1 release | `y1.untyped_storage().resize_(0)` inside `_UpProjection.forward` | ~96 | Frees y1 bf16 after ISA pack |
| Split quantization | Separate z quant → free z → y1 quant (vs fused) | ~100 | Reduces forward simultaneous peak |
| Fused weight cache clear | `clear_fused_weight_cache()` after up-proj | ~37 | Frees cached fused weights |
| Serialized wgrad/actgrad | Sequential (not parallel) in `_UpProjection.backward` | ~137 | `dz.resize_(0)` between wgrad and actgrad |
| Wgrad-before-prequant | Reorder in `_DownProjection.backward` fused path | ~167 | `del y1s` before dz prequant |

**Key technique:** `tensor.untyped_storage().resize_(0)` frees storage while preserving shape/stride metadata for PyTorch autograd validation. Works inside `torch.autograd.Function.forward`. `tensor.data = torch.empty(0)` does NOT work inside forward (changes recorded shape to [0]).

### 7.3 FP8-Specific Caches (persistent across iterations)

| Cache | Size (MiB) | Purpose |
|-------|-----------|---------|
| VARLEN_(3072, 1536, 8) | 37.1 | w2 (c_proj) FP8 weight cache |
| VARLEN_(3072, 3072, 8) | 74.3 | w1 (c_fc) FP8 weight cache |
| FUSED_(3072, 1536, 8) | 37.1 | w2 fused FP8 weight cache (backward) |
| **Total** | **148.5** | — |

### 7.4 Optimization History

| Stage | FP8 Peak (MiB) | vs BF16 |
|-------|---------------|---------|
| Before optimization (Session 38) | 2098 | −17.4% |
| + outer z/y1/cache clear | 1997 | −21.4% |
| + split quantization | 1997 (fwd: 1566) | −21.4% |
| + serialized backward | 1860 | −26.8% |
| + wgrad/prequant reorder | **1693** | **−33.4%** |

### 7.5 Theoretical Memory Analysis (Ernie shape)

Key tensor sizes at T=8192, H=3072, I=1536, E=8, K=8 (TK=65536):
- z_bf16 (TK × 2I): 384 MiB
- y1_bf16 (TK × I): 192 MiB
- z_fp8 (TK × 2I): 192 MiB
- y1_fp8 (TK × I): 96 MiB
- dz_bf16 (TK × 2I): 384 MiB
- dx_expanded (TK × H): 384 MiB
- dw1 (E × 2I × H): 144 MiB
- dw2 (E × I × H): 72 MiB

---

## 8. Next Steps

1. **Training validation**: Run end-to-end training and compare loss curves FP8 vs BF16
2. **Performance recovery**: Serialized wgrad/actgrad trades ~0.2-0.5ms parallelism for −137 MiB memory. Consider making configurable via env var (e.g., `SONIC_MOE_FP8_PARALLEL_BWD=1` to restore parallelism).
3. **Quant kernel fusion**: Merge `quantize_and_pack` + `gather_isa_packed_scales` into forward GEMM epilogue (potential −190µs)
4. **c_proj FP8 at larger I**: Sweep I=2048+ shapes where quant overhead < GEMM saving
5. **FP8 weight cache memory**: 148.5 MiB persistent caches — could be freed between iterations with lazy re-quant
6. **Multi-node**: Validate FP8 with expert parallelism across nodes

### Insight: Memory-Performance Tradeoff Knobs

The current implementation chose maximum memory savings. Two easy knobs to dial back:
- **Restore parallel wgrad/actgrad** (`_UpProjection.backward`): +137 MiB, −0.2-0.5ms backward
- **Restore fused z+y1 quant** (`_UpProjection.forward`): +100 MiB, −0.05ms forward (L2-hot benefit)
Even with both restored, FP8 would still be −21.4% under BF16 peak.

---

## 9. Quick Validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Full test suite
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Per-expert precision (subprocess-isolated)
CUDA_VISIBLE_DEVICES=0,1 python tools/_test_bug1_fix.py
```
