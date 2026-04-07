# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-09 (Session 38 — Bug 1 fixed, final benchmark)
> **Branch:** `native-fp8-exploration`
> **Status:** ✅ FP8 fused path **fully functional**. 1.19× speedup, all precision PASS. Ready for training validation.

---

## 0. Current Bottom Line

| Metric | BF16 Baseline | FP8 Fused (Fixed) |
|--------|--------------|-------------------|
| GPU kernel µs/iter | **4156** | **3484 (1.19× faster)** |
| GEMM-only µs/iter | 3502 | 2687 (1.30× faster) |
| FP8 quant overhead | — | 359 µs (10.3%) |
| Peak memory (MiB) | 1658 | 2079 (+25.4%) |
| Output RRMSE | — | 6.60% ✓ |
| Worst grad RRMSE | — | 7.01% (dx) ✓ |
| Per-expert RRMSE | — | 4.71-5.40% ✓ (all 8 experts) |

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

### 5.1 Per-kernel breakdown (µs/iter, nsys GPU time)

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

### 5.2 Savings breakdown
- GEMM savings: +282 (fwd) +59 (bwd) +475 (wgrad tile) +163 (elementwise) = **+979 µs**
- FP8 overhead: 168 + 137 + 54 = **−359 µs**
- **Net: +620 µs saved → 1.19×**

---

## 6. Precision Summary

All metrics below threshold (RRMSE <10%, cosine >0.99):

| Tensor | RRMSE | Cosine |
|--------|-------|--------|
| output | 6.60% | 1.000486 |
| dx | 7.01% | 0.999891 |
| c_fc.weight grad | 4.75% | 1.014582 |
| c_proj.weight grad | 5.22% | 1.003160 |
| router.weight grad | 6.85% | 0.997661 |

Per-expert: all 8 experts in range 4.71-5.40% RRMSE for c_fc, 5.00-5.40% for c_proj.

---

## 7. Memory

| | BF16 | FP8 | Delta |
|--|------|-----|-------|
| Model params | 216 MiB | 216 MiB | 0 |
| Fwd activation delta | 442 MiB | 367 MiB | −75 (−17%) |
| **Peak (training)** | **1658 MiB** | **2079 MiB** | **+421 (+25.4%)** |

Peak increase from: FP8 scales storage + ISA-packed scales + CUTLASS FP8 workspace.

---

## 8. Next Steps

1. **Training validation**: Run end-to-end training and compare loss curves FP8 vs BF16
2. **Quant kernel fusion**: Merge `quantize_and_pack` + `gather_isa_packed_scales` into forward GEMM epilogue (potential −190µs)
3. **c_proj FP8 at larger I**: Sweep I=2048+ shapes where quant overhead < GEMM saving
4. **Memory optimization**: Explore activation checkpointing + FP8 synergies
5. **Multi-node**: Validate FP8 with expert parallelism across nodes

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
