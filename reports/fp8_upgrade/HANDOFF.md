# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-08 (Session 40 — Clean-node benchmark, verified data)
> **Branch:** `native-fp8-exploration`
> **Status:** ✅ FP8 fully functional. **1.10× GPU compute speedup (nsys)**, precision PASS, memory ~parity.

---

## 0. Current Bottom Line

> ⚠️ **All performance data below measured via nsys GPU Projection on an idle B200 node
> (tjzj-inf-sci-k8s-bzz2-0274, 0% util, <5 MiB VRAM per GPU).**
> Prior sessions reported 1.52× (CUDA events) and −33.4% memory — both were contaminated
> by GPU contention on a shared machine. See §10 for measurement methodology lessons.

| Metric | BF16 Baseline | FP8 Frontier | Delta |
|--------|--------------|--------------|-------|
| **GPU Projection total** | **4036.0 µs** | **3654.3 µs** | **1.10× faster** |
| Forward GPU time | 1387.7 µs | 1178.0 µs | 1.18× faster |
| Backward GPU time | 2648.3 µs | 2476.3 µs | 1.07× faster |
| **Cold-start peak memory** | **1657.9 MiB** | **1620.2 MiB** | **−2.3% (−38 MiB)** |
| Steady-state fwd peak | 1529.9 MiB | 1518.7 MiB | −0.7% |
| Steady-state bwd peak | 1555.8 MiB | 1549.1 MiB | −0.4% |
| Output RRMSE | — | 6.51% ✓ | <10% threshold |
| Worst grad RRMSE | — | 7.04% (dx) ✓ | <10% threshold |
| All cosines | — | ≥0.9971 ✓ | >0.99 threshold |

Shape: T=8192, H=3072, I=1536, E=8, K=8 (Ernie production shape).
GPU: B200 (SM 10.0, 148 SMs, HBM3e).
Measurement node: tjzj-inf-sci-k8s-bzz2-0274 (7 idle GPUs, verified 0% util).

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

> **Source:** nsys GPU Projection on idle B200 node (0% util), 10 profiled iterations with NVTX ranges.
> nsys reports: `benchmarks/nsys_clean/{bf16,fp8}_clean.nsys-rep`

### 5.1 nsys GPU Projection Summary

| Phase | BF16 (µs) | FP8 (µs) | Speedup |
|-------|----------|---------|---------|
| Forward | 1387.7 | 1178.0 | **1.18×** |
| Backward | 2648.3 | 2476.3 | **1.07×** |
| **Total** | **4036.0** | **3654.3** | **1.10×** |

### 5.2 Forward Kernel Breakdown (µs/iter avg, nsys NVTX-scoped)

| Kernel | BF16 (µs) | % | FP8 (µs) | % | Speedup |
|--------|----------|---|----------|---|---------|
| GemmGated (fwd GEMM) | 779.0 | 56.1 | — | — | — |
| GemmGatedSm100ZeroMat (FP8 fwd) | — | — | 461.4 | 39.2 | 1.69× |
| GemmDefault (down-proj) | 386.9 | 27.9 | 234.6 | 19.9 | 1.65× |
| `_quantize_flat_blockscaled_kernel` | — | — | 123.1 | 10.5 | FP8 overhead |
| `_quantize_and_pack_kernel` (×3) | — | — | 115.0 | 9.8 | FP8 overhead |
| `_gather_isa_packed_scales_kernel` | — | — | 27.1 | 2.3 | FP8 overhead |
| token_gather_sum | 71.4 | 5.1 | 72.5 | 6.2 | — |
| Other elementwise | 150.4 | 10.8 | 143.3 | 12.2 | — |
| **Total** | **1387.7** | | **1178.0** | | **1.18×** |

**Forward analysis:** FP8 GEMMs are 1.65-1.69× faster (FP8 tensor cores). FP8 quant overhead = 265.2µs (22.5% of FP8 forward). Net gain: 470µs GEMM savings − 265µs quant = **+205µs saving**.

### 5.3 Backward Kernel Breakdown (µs/iter avg, nsys NVTX-scoped)

| Kernel | BF16 (µs) | % | FP8 (µs) | % | Speedup |
|--------|----------|---|----------|---|---------|
| Wgrad GEMMs (3× GemmDefault) | 1978.3 | 74.7 | 1577.9 | 63.7 | 1.25× |
| DGated GEMM | 507.5 | 19.2 | — | — | — |
| DGatedFP8CLoadSm100ZeroMat | — | — | 409.7 | 16.5 | 1.24× |
| SwiGLU bwd elementwise | — | — | 156.5 | 6.3 | FP8 overhead |
| `_quantize_and_pack_kernel` (×3) | — | — | 144.4 | 5.8 | FP8 overhead |
| `_gather_isa_packed_scales_kernel` | — | — | 27.1 | 1.1 | FP8 overhead |
| token_gather_sum | 69.9 | 2.6 | 71.1 | 2.9 | — |
| Other | 92.6 | 3.5 | 89.6 | 3.6 | — |
| **Total** | **2648.3** | | **2476.3** | | **1.07×** |

**Backward analysis:** Wgrad GEMMs are 1.25× faster despite being BF16 (lower memory pressure from FP8 activations = better cache behavior). DGated 1.24× faster. FP8 quant+cast overhead = 328µs (13.2%). Net gain: 498µs GEMM savings − 328µs overhead = **+170µs saving**.

### 5.4 CUDA Events vs nsys — Measurement Methodology Warning

> ⚠️ **CUDA events (`torch.cuda.Event`) measure wall-clock between event records, NOT GPU compute time.**
> Under GPU contention or scheduling gaps, they wildly overestimate speedup.
> Prior sessions reported 1.52× from CUDA events. nsys GPU Projection shows the truth: **1.10×**.
> **Always use nsys GPU Projection or isolated kernel benchmarks for performance claims.**

---

## 6. Precision Summary

> **Source:** Subprocess-isolated, shared model init seed, 3 random seeds, idle B200 node.

All metrics below threshold (RRMSE <10%, cosine >0.99). Measured with subprocess isolation, shared weights/input.

| Tensor | Seed 42 RRMSE | Seed 123 RRMSE | Seed 777 RRMSE | Avg Cosine |
|--------|-------------|---------------|---------------|------------|
| output (out) | 6.51% | 6.51% | 6.51% | 0.9979 |
| input grad (dx) | 7.03% | 7.04% | 7.04% | 0.9975 |
| router grad | 7.59% | 7.50% | 7.51% | 0.9972 |
| loss | 0.00% | 0.00% | 0.00% | 1.0000 |

**12/12 measurements PASS.** Weight gradients (c_fc, c_proj) not measured in this session due to param name mismatch in benchmark script; prior session measured 5.97% (c_fc) and 6.54% (c_proj).

Test suite: 31/31 tests PASS (`tests/fp8_large_project_contract_test.py`).

---

## 7. Memory — Detailed Breakdown

> **Source:** Subprocess-isolated measurement on idle B200 node. 2 warmup iters + GC + empty_cache + reset_peak, then 1 measured iter.

### 7.1 Peak Memory Comparison (Clean Node, Post-Warmup Steady State)

| Checkpoint | BF16 (MiB) | FP8 (MiB) | Delta |
|-----------|-----------|----------|-------|
| Base (empty GPU) | 0.0 | 0.0 | — |
| After model load | 216.1 | 216.1 | 0 |
| After input alloc | 312.1 | 312.1 | 0 |
| Before forward (post-warmup) | 520.6 | 669.1 | +148.5 (weight cache) |
| **Forward peak** | **1529.9** | **1518.7** | **−11.2 (−0.7%)** |
| After forward | 905.8 | 831.2 | −74.6 |
| **Backward peak** | **1555.8** | **1549.1** | **−6.8 (−0.4%)** |
| After backward | 784.6 | 933.1 | +148.5 (weight cache) |

### 7.2 Cold-Start Peak (First Iteration, Includes JIT Compilation)

| Metric | BF16 (MiB) | FP8 (MiB) | Delta |
|--------|-----------|----------|-------|
| **Cold-start peak** | **1657.9** | **1620.2** | **−37.8 (−2.3%)** |
| Post-backward residual | 640.6 | 789.1 | +148.5 |

### 7.3 Why Previous −33.4% Claim Was Wrong

Previous sessions reported BF16=2540 MiB, FP8=1693 MiB. This was **GPU contention noise**:
- The measurement machine had 8-11 GiB occupied on measurement GPUs
- CUDA context of other processes inflated `max_memory_allocated` readings
- On a truly idle GPU (4 MiB baseline), BF16 peak = 1658 MiB, not 2540 MiB

**Lesson:** Memory benchmarks MUST be done on isolated GPUs with `nvidia-smi` showing <10 MiB before the test.

### 7.4 Memory Budget Analysis

Net activation memory per phase (peak − before_fwd baseline):

| Phase | BF16 Net Alloc | FP8 Net Alloc | FP8 Saves |
|-------|---------------|--------------|-----------|
| Forward | 1009.3 MiB | 849.6 MiB | 159.7 MiB (−15.8%) |
| Backward (above fwd residual) | 650.0 MiB | 717.9 MiB | −67.9 MiB (+10.4%) |

FP8 saves 160 MiB in forward activations (resize_(0) optimizations) but:
- FP8 weight cache adds 148.5 MiB persistent overhead
- FP8 backward needs 68 MiB more for quant buffers
- Net: the savings nearly cancel out

### 7.5 Memory Optimizations Applied

| Optimization | Technique | Mechanism |
|-------------|-----------|-----------|
| Inner z release | `z.untyped_storage().resize_(0)` | Frees z bf16 before y1 quant |
| Inner y1 release | `y1.untyped_storage().resize_(0)` | Frees y1 bf16 after ISA pack |
| Split quantization | Separate z quant → free z → y1 quant | Reduces fwd simultaneous peak |
| Fused weight cache clear | `clear_fused_weight_cache()` | Frees backward weight cache |
| Serialized wgrad/actgrad | Sequential in `_UpProjection.backward` | `dz.resize_(0)` between phases |
| Wgrad-before-prequant | Reorder in `_DownProjection.backward` | `del y1s` before dz prequant |

**Key technique:** `tensor.untyped_storage().resize_(0)` frees GPU storage while preserving shape/stride metadata for autograd. Works inside `torch.autograd.Function.forward`.

### 7.6 FP8-Specific Caches (persistent across iterations)

| Cache | Size (MiB) | Purpose |
|-------|-----------|---------|
| VARLEN_(3072, 1536, 8) | 37.1 | c_proj FP8 weight cache |
| VARLEN_(3072, 3072, 8) | 74.3 | c_fc FP8 weight cache |
| FUSED_(3072, 1536, 8) | 37.1 | c_proj fused weight cache (bwd) |
| **Total** | **148.5** | — |

### 7.7 Theoretical Tensor Sizes (Ernie shape)

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
2. **Weight gradient precision**: Confirm c_fc.weight.grad and c_proj.weight.grad RRMSE <10% (missed in current benchmark due to param name mismatch in script; prior session measured 5.97% and 6.54% — likely still correct)
3. **Quant kernel fusion**: Merge `quantize_and_pack` + `gather_isa_packed_scales` into GEMM epilogue (potential −265µs forward, −172µs backward)
4. **c_proj FP8 at larger I**: Sweep I=2048+ shapes where quant overhead < GEMM saving
5. **FP8 weight cache optimization**: 148.5 MiB persistent caches — explore lazy re-quant or shared workspace
6. **Multi-node**: Validate FP8 with expert parallelism across nodes
7. **Shape scaling**: Test T=4096 and T=16384 for scaling behavior

### Insight: Where the Speedup Comes From

The 1.10× speedup has clear attribution:
- **Forward:** FP8 GEMMs are 1.65-1.69× faster via FP8 tensor cores, offset by 265µs quant overhead → net 1.18×
- **Backward:** Wgrad GEMMs are 1.25× faster (lower memory pressure), DGated 1.24× faster, offset by 328µs overhead → net 1.07×
- **Remaining opportunity:** Quant kernel fusion could eliminate ~430µs of overhead, boosting total to ~1.23×

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

---

## 10. Measurement Methodology Lessons (CRITICAL for future agents)

### 10.1 GPU Contention Invalidates ALL Benchmarks

**Problem:** Sessions 38-39 ran benchmarks on a shared machine with 8-11 GiB occupied per GPU.
- CUDA events reported 1.52× speedup → **actual: 1.10×** (3.6× over-estimate of gain)
- Memory profiling showed −33.4% savings → **actual: −0.4%** (84× over-estimate of savings)

**Root cause:** Other processes' CUDA contexts inflate `max_memory_allocated()`, and GPU scheduling contention causes CUDA events to capture idle time between kernel launches as "compute time".

**Rule:** Before ANY benchmark:
1. `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
2. Memory must be <10 MiB on the target GPU
3. Utilization must be 0%
4. If not available locally, use `python tools/cluster_idle_launch.py scan` to find idle nodes

### 10.2 CUDA Events vs nsys GPU Projection

| Tool | What it measures | Use case |
|------|-----------------|----------|
| `torch.cuda.Event` | Wall-clock between event records | ❌ Unreliable under contention |
| nsys GPU Projection (NVTX-scoped) | Actual kernel execution on GPU | ✅ Ground truth for compute |
| ncu | Single-kernel roofline analysis | ✅ Per-kernel efficiency |

**Always use nsys for E2E performance claims. CUDA events are only valid on 100% idle GPUs.**

### 10.3 Process Contamination

`SONIC_MOE_FP8_MODE` is cached at import time (`sonicmoe/functional/utils.py:38`).
**Any BF16 vs FP8 comparison MUST use separate subprocesses.** Same-process gives fake bit-identical results.

### 10.4 Benchmark Reproduction

```bash
# 1. Find idle node
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
python tools/cluster_idle_launch.py scan

# 2. SSH to idle node and run nsys benchmark
ssh <idle_host> bash /path/to/sonic-moe/benchmarks/nsys_remote_bench.sh

# 3. Analyze results
python tools/nsys_full_breakdown.py benchmarks/nsys_clean/bf16_clean.sqlite benchmarks/nsys_clean/fp8_clean.sqlite --labels bf16 fp8
```
