# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-01 (Session 26 final)
> **Branch:** `fork-main-sync`
> **Status:** Full FP8 forward+backward with zero-materialization kernels. 19/19 precision tests pass.
> FP8 path achieves **1.18x GPU projection speedup** (3324µs vs 3937µs CUDA kernel time).
> FP8 **peak memory 0.610× of BF16** (39% savings) at Ernie production shape.

---

## 0. One-screen summary

**What works today:**
- Zero-materialization FP8 GemmGated forward (T-quant + scale_gather + A_idx GEMM)
- Zero-materialization FP8 GemmDGated backward (same pattern)
- Custom `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat` kernel classes
- Auto-selection of zero-mat kernels when `gather_A + blockscaled`
- Three-step gather pipeline (`fast_gather_quantize_and_pack_activation`)
- Backward FP8 state persistence (ctx flags survive autograd context exit)
- FP8 down-proj with pre-quantized y1 (zero-overhead at I≥1536 via L2-hot pre-quantization)
- z FP8 save in fused gated path (saves ~171MB at Ernie shape)
- Weight cache deduplication (direct_fused_dgated shares fused_dgated cache)
- Aggressive FP8 weight cache eviction (w1 after up-proj, w2 after down-proj)
- `use_fp8=True` API on `MoE.forward()` and `enable_fp8()` context manager
- 19/19 contract tests pass including production shape (T=8192, H=3072, I=1536, E=8, K=8)

**Best training path:**
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Performance (Ernie production shape, idle GPU)

| Config | Wall-clock (ms/iter) | vs BF16 |
|--------|---------------------|---------|
| Fork BF16 | 4.45 | baseline |
| **Fork FP8 frontier** | **3.42** | **1.30× faster** |

**CUDA GPU projection (torch.profiler):**

| Kernel | FP8 (µs) | % |
|--------|----------|---|
| BF16 GemmDefault ×3 (wgrad dw1) | 1239 | 37.3% |
| **FP8 GemmGated ZeroMat (fwd up-proj)** | **547** | **16.5%** |
| **FP8 GemmDGated ZeroMat (bwd dgated)** | **481** | **14.5%** |
| BF16 GemmDefault (wgrad dw2) | 383 | 11.5% |
| BF16 GemmDefault (down-proj fwd) | 336 | 10.1% |
| quantize_and_pack ×3 | 137 | 4.1% |
| token_gather_sum ×2 | 143 | 4.3% |
| **gather_isa_packed_scales ×2** | **54** | **1.6%** |
| **Total FP8 CUDA** | **3324** | — |
| Official BF16 CUDA (reference) | 3937 | — |

**Multi-shape wall-clock (idle GPU, uniform 128-aligned routing):**

| Shape | BF16 | FP8 | Speedup |
|-------|------|-----|---------|
| I=1024 (E=128, K=8) | 4.90ms | 3.77ms | 1.30× |
| I=2048 (E=128, K=8) | 9.83ms | 4.18ms | **2.35×** |
| Ernie (I=1536, E=8, K=8) | 4.45ms | 3.42ms | 1.30× |

---

## 2. Optimizations Applied

### Session 25: Zero-materialization FP8 kernels
**File:** `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py`

Custom `GemmGatedSm100ZeroMat` and `GemmDGatedSm100ZeroMat` subclass GemmSm100 via MRO, overriding only `__call__` (decorated with `@cute.jit`). The single core fix:

```python
if const_expr(self.gather_A):
    sfa_layout = _tile_atom_to_shape_SF_rank_aware(mSFA.shape, self.sf_vec_size)
else:
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(mA.shape, self.sf_vec_size)
```

**Why this works:** When `gather_A=True`, `mA` is T-sized but `cu_seqlens_m` offsets reach TK. The upstream `tile_atom_to_shape_SF(mA.shape)` creates a T-row SFA layout, causing TMA to read out-of-bounds. Using `mSFA.shape` (TK-sized, from pre-gathered scales) fixes the layout.

**Auto-selection** in `gemm_gated.py` and `gemm_dgated.py`:
```python
if device_capacity[0] > 9 and gather_A and blockscaled_runtime:
    GemmCls = GemmGatedSm100ZeroMat  # or GemmDGatedSm100ZeroMat
```

### Session 25: Three-step gather pipeline
**File:** `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`

`fast_gather_quantize_and_pack_activation(x, gather_idx)`:
1. `quantize_and_pack_activation(x)` — T-sized rowwise FP8 quant (~3-8µs)
2. `x_fp8.index_select(0, gather_idx)` — FP8 data gather (~15-25µs, L2-resident)
3. `_gather_isa_packed_scales_kernel` — ISA-packed scale gather (~5µs)

Total ~25µs vs 96µs for monolithic `gather_quantize_and_pack_activation`.

### Session 25: Backward FP8 state persistence
**File:** `sonicmoe/functional/__init__.py`

**Root cause (Session 24 bug):** `_fp8_enabled()` checks `is_fp8_active()` which is a thread-local flag set by `enable_fp8()` context manager. Backward runs via autograd OUTSIDE this context → `_fp8_enabled()` returns False → backward falls into BF16 path → `B.mT.contiguous()` copies 72MB of weight data (156µs).

**Fix:** Save FP8 flags on `ctx` during forward:
```python
ctx._fp8_enabled = _fp8_enabled()
ctx._fp8_lean = _fp8_lean()
ctx._alignment_assumed = _ALIGNMENT_ASSUMED
ctx._use_fused_blockscaled_gated_flag = _use_fused_blockscaled_gated()
```
Backward uses `ctx._fp8_enabled` instead of calling `_fp8_enabled()`.

### Session 26: Memory optimizations (FP8 0.765× of BF16)

**Problem:** FP8 used 1.38-1.48× MORE memory than BF16 due to redundant weight caches and BF16 activation saves.

**Optimizations applied (all in `sonicmoe/functional/__init__.py` and `blockscaled_fp8_gemm.py`):**

1. **z FP8 save in fused gated path** — Removed `and not _use_fused_blockscaled_gated()` condition on z_is_fp8 flag. z(TK, 2I) saved as FP8 instead of BF16 in fused gated path, saving ~171 MiB at Ernie shape.

2. **z pre-quantization while hot in L2** — Added `quantize_activation_blockscaled_fast(z)` right after `_fused_blockscaled_gated_forward()` returns. Avoids cold-cache penalty in `_DownProjection.forward`.

3. **Weight cache deduplication** — Merged `_DIRECT_FUSED_DGATED_WEIGHT_CACHE` into `_FUSED_WEIGHT_CACHE`. `precompute_weight_fp8_for_direct_fused_dgated()` now checks shared cache first, eliminating one full FP8 weight copy.

4. **Aggressive FP8 weight cache eviction** — w1 FP8 cache evicted after `_UpProjection.forward`, w2 FP8 cache evicted after `_DownProjection.forward`. Training always re-quantizes on next forward (w._version increments), so eviction is free.

5. **Early z deallocation in backward** — Added `del z, colvec_reduce_partial` after dgated kernel in fused gated backward, freeing 384 MiB before dz pre-quantization.

6. **y1 pre-quantization + FP8 down-proj at I=1536** — Pre-quantize y1 while hot in L2 after SwiGLU, store in `_PREQUANTIZED_SCALES["fwd"]` as 3-tuple `(bf16_ref, fp8_data, scales)`. Down-proj uses pre-quantized y1 (zero quant overhead), enabling FP8 GEMM at I=1536 where inline quant was previously net-zero.

7. **Bug fix: prequant format mismatch** — The original 2-tuple format `(y1_fp8, y1_packed_scales)` caused `_matches_prequant_tensor` to compare FP8 dtype vs BF16 dtype, always failing. FP8 down-proj was silently falling back to BF16. Fixed by storing `(y1_bf16_ref, y1_fp8, y1_packed_scales)` 3-tuple, matching the backward format. Added `_PREQUANT_HIT_COUNT` diagnostics counter.

**Memory measurement (isolated process, single fwd+bwd, Ernie shape T=8192 H=3072 I=1536 E=8 K=8):**

| Metric | BF16 | FP8 | Ratio |
|--------|------|-----|-------|
| Peak memory | 2828 MiB | 1726 MiB | **0.610×** |
| After forward | 2106 MiB | 581 MiB | 0.276× |
| Before (model+caches) | 1529 MiB | 382 MiB | 0.250× |

---

## 3. Critical Architectural Insights

### 3.1 SFA (Scale Factor A) indexing with gather_A
**BOTH GemmGated and GemmDGated have the same SFA bug** in upstream QuACK. The SFA TMA uses `varlen_manager.offset_batch_A(mSFA, batch_idx)` which offsets by `cu_seqlens_m[batch_idx]`. When `gather_A=True`:
- `mA` is T-rows, SFA layout derived from `mA.shape` → T-row layout
- `cu_seqlens_m` offsets go up to TK → TMA reads past T-row SFA buffer

The backward dgated "appeared to work" because the production code path used `gather_quantize_and_pack_activation` (TK-sized) without A_idx, avoiding the bug entirely.

**Fix:** Zero-mat kernels use `mSFA.shape` (TK-rows) for SFA layout.

### 3.2 TMA limitations
TMA requires contiguous, regularly-strided memory regions. Cannot do gathered/scattered reads. This means:
- `gather_A` for data uses cp.async (not TMA) in dedicated gather warps
- SFA for gathered A must be pre-arranged in TK order (scale gather) since TMA can't scatter-read
- Blockscaled FP8 scale groups must be along contiguous memory dimension

### 3.3 FP8 wgrad is net-negative
At Ernie shape (K_per_expert=8192):
- `colwise_quantize_and_pack` ×4 = 916µs (49152 blocks, saturates 128 SMs)
- FP8 GEMM ×3 = 907µs (faster than BF16's 3311µs)
- **But** colwise quant's SM saturation throttles parallel BF16 wgrad on side stream by 8%
- Wall-clock: +0.9ms net slower with FP8 wgrad

Alternatives explored and rejected:
- Transpose + rowwise quant: transpose alone costs 1509µs (more than entire colwise quant)
- Rowwise quant + strided view: HW requires K-dimension to be contiguous for blockscaled
- Async dw2 on side stream: SM contention unchanged

### 3.4 SonicMoE design principle: zero materialization
SonicMoE's core design avoids TK-sized activation intermediates:
- BF16 path: A_idx gathers during CUTLASS G2S (no TK copy in HBM)
- FP8 path now: T-quant + scale_gather + A_idx GEMM (only TK-sized scales materialized, ~3% of data)

---

## 4. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad** | +0.9ms net | colwise quant SM contention breaks stream parallelism |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch checks storage bounds |
| **Rowwise quant + strided view for wgrad** | Not possible | HW requires contiguous K groups in blockscaled |
| **FP8 down-proj at I=1536 (inline quant)** | Net-zero | quant overhead ≈ GEMM savings — solved via L2-hot pre-quant in Session 26 |
| **Transpose + rowquant** | 3.8× slower | transpose 1509µs > colwise quant 260µs |
| **Eager FP8 weight cache eviction (precompute)** | +2.6 GiB | precompute temp > cache — solved via evict-after-use pattern in Session 26 |
| **Stream overlap (quant ∥ wgrad)** | No effect | Both saturate 128 SMs → bandwidth contention |

---

## 5. Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `USE_QUACK_GEMM=1` | off | **Required** — Blackwell CUTLASS kernels |
| `SONIC_MOE_FP8_MODE=perf` | off | Enables FP8 path (or use `enable_fp8()` API) |
| `SONIC_MOE_FP8_FUSED_GATED` | **1** | Fused GemmGated+SwiGLU (default on) |
| `SONIC_MOE_FP8_WGRAD` | **0** | FP8 wgrad — keep 0 (proven net-negative) |
| `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD` | 2048 | I threshold for FP8 down-proj (bypassed when y1 pre-quant available) |
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | 0 | Skip alignment check (set 1 for benchmarks) |

---

## 6. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/functional/utils.py` | `enable_fp8()` context manager |
| `sonicmoe/moe.py` | `MoE.forward(use_fp8=True)` entry point |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | **NEW** Zero-mat kernel classes + wrapper |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches, three-step pipeline |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated wrapper (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated wrapper (auto ZeroMat selection) |
| `tests/fp8_large_project_contract_test.py` | 19 precision tests (11 original + 4 aligned + 4 memory/opt) |

---

## 7. Correctness

19/19 contract tests pass:
- 11 `FP8LargeProjectContractTest` (T=4096, H=4096, E=128, K=8)
- 3 `FP8AlignedContractTest` multi-seed + context manager (T=1024, H=3072, I=1536, E=8, K=8)
- 1 production shape (T=8192, H=3072, I=1536, E=8, K=8) fwd+bwd
- 1 z FP8 save precision (fused gated path round-trip)
- 1 FP8 peak memory < BF16 peak memory (isolated process)
- 1 weight cache deduplication (direct_fused_dgated reuses fused_dgated)
- 1 FP8 down-proj pre-quantization precision

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

---

## 8. Next Steps (ordered by ROI)

1. **Nsys profiling on idle GPU** — Cluster GPUs are at 100% utilization. When available, run nsys to get post-Session-26 CUDA kernel timing (FP8 down-proj saves ~86µs at Ernie shape). Expected CUDA time: ~3240µs (vs 3324µs pre-Session-26, vs 3937µs BF16).

2. **Custom CUTLASS kernel for SFA cp.async gather** — Eliminate scale pre-gathering entirely. Modify GemmSm100's kernel to load SFA via cp.async alongside A data in the gather warp group. Removes `_gather_isa_packed_scales_kernel` (54µs, 1.6% of CUDA time) and TK-sized scale buffer.

3. **Reduce BF16 wgrad bottleneck** — 1239µs BF16 wgrad (37.3% of CUDA time) is the dominant remaining cost. FP8 wgrad is proven net-negative. Future GPU architectures with better multi-stream SM sharing could enable FP8 wgrad.

4. **Dead code cleanup** — `_fused_blockscaled_dgated_backward()` (line 192 in `functional/__init__.py`) is defined but never called; the actual logic is inline in `_DownProjection.backward`. Can be removed.

---

## 9. Environment

```
GPU: 8× NVIDIA B200 (SM100, 183GB)
CUDA: 12.8, PyTorch: 2.9.1+cu128
QuACK: 0.3.7, CUTLASS DSL: latest
Python: 3.13
Env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer
Official BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16
```
