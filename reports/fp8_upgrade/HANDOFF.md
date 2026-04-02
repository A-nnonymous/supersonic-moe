# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-02 (Session 28)
> **Branch:** `fork-main-sync`
> **Status:** Zero-mat kernel precision bug fixed. 20/20 contract tests pass (BF16 vs FP8, no env-var masking).
> **Reality check:** FP8 is currently **slower** than BF16 (0.82× at Ernie shape). Precision is correct (<10% RRMSE, >0.99 corr). See §1 for details.

---

## 0. One-screen summary

**What works today:**
- Zero-materialization FP8 forward via standard `gemm_gated()` + `A_idx` (auto-selects `GemmGatedSm100ZeroMat`)
- Zero-materialization FP8 backward via `gemm_dgated()` + `A_idx` (auto-selects `GemmDGatedSm100ZeroMat`)
- Precision verified: **RRMSE <7%, correlation >0.997** at both contract and production shapes
- 20/20 contract tests pass WITHOUT `SONIC_MOE_FP8_MODE=perf` env var
- `use_fp8=True` API on `MoE.forward()` and `enable_fp8()` context manager

**Critical finding (Session 28):**
The standalone `gemm_gated_zeromat()` wrapper had a B-tensor layout bug (passed `.mT` view instead of contiguous `(E,N,K)`). This caused 101% RRMSE. The fix removes the standalone wrapper entirely and routes through the standard `gemm_gated()` interface which handles B-layout correctly.

Previous HANDOFF claims of 1.30× speedup and 30% memory savings were invalid — they were measured with `SONIC_MOE_FP8_MODE=perf` which made both BF16 and FP8 runs use the same (broken) FP8 path.

**Best training path:**
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Performance & Memory (measured 2026-04-02, GPU 0 on 10.51.195.12)

**Wall-clock (20 iterations, 5 warmup, `CUDA_VISIBLE_DEVICES=0`):**

| Shape | BF16 (ms/iter) | FP8 (ms/iter) | Speedup |
|-------|---------------|---------------|---------|
| Contract (T=1024, H=3072, I=1536, E=8, K=8) | 3.10 | 6.66 | **0.46×** (slower) |
| **Ernie prod (T=8192, H=3072, I=1536, E=8, K=8)** | **9.72** | **11.81** | **0.82×** (slower) |

**Peak memory:**

| Shape | BF16 (MB) | FP8 (MB) | Ratio |
|-------|-----------|----------|-------|
| Contract | 1942 | 2276 | 1.17× (more) |
| Ernie prod | 17504 | 18691 | 1.07× (more) |

**Precision (verified correct):**

| Shape | FWD RRMSE | FWD corr | BWD RRMSE | BWD corr |
|-------|-----------|----------|-----------|----------|
| Contract | 6.60% | 0.9978 | 7.50% | 0.9972 |
| Ernie prod | 6.60% | 0.9978 | 7.47% | 0.9972 |

**nsys timelines** (NVTX-annotated, 3 warmup + 3 profiled iterations at Ernie shape):
- `reports/timelines/bf16_fork.nsys-rep`
- `reports/timelines/fp8_frontier.nsys-rep`

---

## 2. Root Causes of FP8 Slowdown

FP8's overhead comes from quantization + scale handling that is NOT offset by GEMM speedup at I=1536:

1. **Quantization overhead**: `quantize_and_pack_activation()` runs 3× per fwd+bwd. Each call does element-wise FP8 cast + scale computation + ISA packing.

2. **Scale gather**: `_gather_isa_packed_scales_kernel` gathers T-sized scales to TK-sized.

3. **Context manager overhead**: `enable_fp8()` and `enable_quack_gemm()` Python overhead per call.

4. **FP8 GEMM not faster enough**: At I=1536 (N=3072), BF16 GEMM is already compute-bound. FP8 provides 2× peak flops but loses throughput to blockscale descaling.

**Likely high-ROI optimization targets** (inspect `fp8_frontier.nsys-rep` to verify):
- Fuse quantize_and_pack into preceding kernel (SwiGLU epilogue, or attention output)
- Amortize context manager overhead (enter once for full training step, not per-MoE-call)
- At larger I (e.g., I=2048), FP8 should win because GEMM savings outweigh quant overhead

---

## 3. What Was Fixed in Session 28

### Critical: `gemm_gated_zeromat()` B-tensor layout bug

**File:** `sonicmoe/functional/__init__.py` (lines 160-175)

**Bug:** `_fused_blockscaled_gated_forward()` called the standalone `gemm_gated_zeromat()` wrapper which expected B as `(E, N, K)` contiguous. But `precompute_weight_fp8_for_fused_gated()` returns B as `(E, K, N)` — a `.mT` view of contiguous `(E, N, K)`. The wrapper's `validate_and_prepare_tensors` misinterpreted the layout, producing scrambled output (101% RRMSE, 0 correlation).

**Fix:** Replaced the standalone wrapper call with the standard `gemm_gated()` interface from `gemm_interface.py`:
```python
z, y1 = gemm_gated(
    x_fp8, w1_fp8,
    activation="swiglu",
    cu_seqlens_m=expert_frequency_offset,
    A_idx=x_gather_idx,          # triggers auto-selection of ZeroMat kernel
    a_scales=x_scales_tk_e8m0,
    b_scales=w1_scales,
    ...
)
```

The standard interface does `B.mT.contiguous()` which correctly recovers the `(E, N, K)` contiguous layout. It then auto-selects `GemmGatedSm100ZeroMat` when `gather_A + blockscaled` on SM100.

### Contract test env-var bug

**File:** `tests/fp8_large_project_contract_test.py`

**Bug:** Tests passed when run with `SONIC_MOE_FP8_MODE=perf` because this env var made the BF16 reference ALSO use the FP8 path → comparing broken FP8 vs broken FP8 → RRMSE=0.

**Fix:** `_reset_fp8_state()` now strips `SONIC_MOE_FP8_MODE` from environment. Tests pass without any env-var interference.

---

## 4. Prior Optimizations (Sessions 25-27)

### Session 25: Zero-materialization FP8 kernels
Custom `GemmGatedSm100ZeroMat` and `GemmDGatedSm100ZeroMat` subclass GemmSm100 via MRO, overriding only `__call__`. Auto-selected in `gemm_gated.py` L236 and `gemm_dgated.py` L319 when `SM100 + gather_A + blockscaled`.

### Session 26: Memory optimizations
- z FP8 save in fused gated path (~171 MiB savings)
- z pre-quantization while hot in L2
- Weight cache deduplication
- Aggressive FP8 weight cache eviction
- y1 pre-quantization for FP8 down-proj at I=1536
- Bug fix: prequant format mismatch (2-tuple → 3-tuple)

### Session 27: Weight cache collision fix
- Separated `_VARLEN_WEIGHT_CACHE` from `_FUSED_WEIGHT_CACHE` (incompatible layouts)
- `_PREQUANT_HIT_COUNT` → `defaultdict(int)`
- Added `test_multi_iteration_cache_consistency`

---

## 5. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **Standalone `gemm_gated_zeromat()` wrapper** | B-layout bug | Standard `gemm_gated()` interface handles B correctly |
| **FP8 wgrad** | +0.9ms net | colwise quant SM contention |
| **`SONIC_MOE_FP8_MODE=perf` for benchmarks** | Masks bugs | Makes BF16 ref use FP8 path |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch storage bounds check |
| **Rowwise quant + strided view for wgrad** | Not possible | HW requires contiguous K groups |
| **Transpose + rowquant** | 3.8× slower | transpose 1509µs > colwise quant 260µs |

---

## 6. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat kernel classes (used via auto-selection, not directly) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches, scale gather |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated wrapper (auto ZeroMat at L236) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated wrapper (auto ZeroMat at L319) |
| `sonicmoe/quack_utils/gemm_interface.py` | High-level interface with B-layout handling |
| `tests/fp8_large_project_contract_test.py` | 20 precision tests |
| `tools/final_benchmark.py` | Reproducible perf+precision benchmark |
| `tools/nsys_timeline_bf16_fp8.py` | nsys timeline generator with NVTX markers |
| `reports/timelines/*.nsys-rep` | nsys timeline files |

---

## 7. Correctness

20/20 contract tests pass (no env-var tricks):
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

---

## 8. Next Steps (ordered by ROI)

1. **Analyze nsys timelines** — Open `bf16_fork.nsys-rep` and `fp8_frontier.nsys-rep` in Nsight Systems GUI. Identify where FP8 overhead lives.

2. **Reduce quantization overhead** — 3× `quantize_and_pack_activation()` is the primary suspect. Options:
   - Fuse quant into SwiGLU epilogue
   - Cache activation FP8 across forward+backward
   - Skip quant for down-proj at I=1536

3. **Reduce Python overhead** — `enable_fp8()` per MoE call has overhead. Consider persistent mode.

4. **Test at I=2048** — FP8 GEMM savings scale with N. Previous estimates: 2.35× at I=2048.

5. **Official BF16 baseline** — `/envs/official_bf16` cannot run on SM100a. Need SM90 node.

---

## 9. Environment

```
GPU: NVIDIA B200 (SM100a, 183GB)
CUDA: 12.8, PyTorch: 2.9.1+cu128, QuACK: 0.3.7
Python: 3.13
FP8 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer
Official BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16
Remote node: 10.51.195.12 (idle GPUs)
```
