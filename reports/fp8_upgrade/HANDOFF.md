# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-01 (Session 24 final)
> **Branch:** `fork-main-sync`
> **Status:** FP8 forward+backward works with fused GemmGated+SwiGLU. 12/12 precision tests pass.
> The forward path still materializes the gathered TK-sized activation (via `gather_quantize_and_pack_activation`).
> **The #1 next optimization is eliminating this materialization** — quantize at T-size and let CUTLASS gather via A_idx, matching the backward pattern.

---

## 0. One-screen summary

**What works today:**
- Fused GemmGated (GEMM + SwiGLU + FP8 descale) in a single CUTLASS kernel
- Fused GemmDGated backward with A_idx (T-sized quant, no gather materialization)
- Adaptive FP8 down-proj (enabled when I ≥ 2048)
- `use_fp8=True` API on `MoE.forward()` and `enable_fp8()` context manager
- 12/12 contract tests pass including production shape (T=8192, H=3072, I=1536, E=8, K=8)

**Best training path (programmatic):**
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

**Best training path (env vars, legacy):**
```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf
```
(Fused gated defaults to enabled. WGRAD defaults to disabled.)

**Performance (old shapes, T=4096, H=4096, E=128, K=8):**

| Shape | BF16 NSYS (µs) | FP8 NSYS (µs) | Speedup | Wall-clock ratio |
|-------|----------------|----------------|---------|------------------|
| I=1024 | 2511 | 2137 | **14.9%** | **1.66×** |
| I=2048 | 7654 | 4403 | **42.5%** | **2.15×** |
| I=4096 | 20711 | 10479 | **49.4%** | **2.37×** |

**Performance (Ernie production shape, T=8192, H=3072, I=1536, E=8, K=8):**

| Config | NSYS GPU projection (µs/iter) |
|--------|-------------------------------|
| Official Sonic BF16 | 3937.4 |
| Sonic FP8 frontier | 3786.6 |
| Ernie MOELayer BF16 | 15325.8 |

> ⚠️ At the Ernie shape, FP8 barely wins (1.04×). The `gather_quantize_and_pack` overhead (~95µs/iter) and 156µs elementwise copies eat most of the GEMM savings. **Eliminating the forward gather materialization is critical.**

**Memory:** FP8 uses 1.38–1.48× MORE than BF16 (3 weight caches). Not yet a win.
**Correctness:** 12/12 tests pass. RelRMSE <10%, correlation >0.99.

---

## 1. SonicMoE Core Design (Critical Context)

SonicMoE's memory efficiency comes from **avoiding materialization of gathered activations**:

1. **Gather fusion via A_idx:** Instead of `x_gathered = x[gather_idx]` (creates TK-sized copy), pass A_idx to the CUTLASS kernel which gathers rows on-the-fly during G2S (Global-to-Shared) copy. No TK-sized intermediate is ever written to HBM.

2. **No Y/Xe caching in backward:** Traditional MoE caches the gathered input Xe (TK×d) and down-proj output Y (TK×d). SonicMoE redesigns the backward path using dS=⟨dA', A⟩ so neither needs to be cached. Memory is constant with respect to expert granularity.

3. **FP8 complication:** In BF16, A_idx gathers data rows — no issue. In FP8, each row has associated **ISA-packed blockscaled scales** (1×32 UE8M0). When CUTLASS gathers data row A_idx[j], it must also gather the matching scale row. **Both data and scales must be gathered simultaneously** because they are row-correlated.

**Current state of A_idx in FP8:**
- ✅ **Backward dgated:** Uses T-sized `quantize_and_pack_activation(dout)` + A_idx. CUTLASS correctly gathers both data and scales. **Works, passes precision.**
- ❌ **Forward GemmGated:** Attempted T-sized quant + A_idx → **93.5% RRMSE (garbage)**. Reverted to `gather_quantize_and_pack_activation` which materializes TK rows. See §8.1 for root cause analysis.

---

## 2. NSYS Measurements

### 2.1 Methodology
- **Authoritative metric:** NSYS NVTX GPU projection (merged GPU-busy time within `forward`/`backward` NVTX ranges of `iter_2`)
- **BF16 baseline caveat:** Exclude `elementwise_kernel` from BF16 numbers (QuACK 0.3.7 layout bug adds ~2082µs of spurious work)
- **Official BF16 env:** `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16` + quack 0.2.5 + `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe`
- **FP8 env:** `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer` + quack 0.3.7

### 2.2 Ernie Shape Kernel Breakdown (T=8192, H=3072, I=1536, E=8, K=8)

This is the production-relevant shape. TK = T×K = 65536.

**FP8 per-iteration breakdown (42 kernels, ~3786µs total):**
| Kernel | Time (µs) | Notes |
|--------|-----------|-------|
| GemmGatedSm100 FP8 (fwd) | 446.6 | Fused GEMM+SwiGLU+FP8 descale |
| GemmDefaultSm100 BF16 (down-proj fwd) | 378.7 | BF16 because I=1536 < threshold |
| token_gather_sum (fwd) | 71.3 | |
| **elementwise direct_copy** | **156.2** | **Unexplained dtype conversion before dgated** |
| GemmDGatedSm100 FP8 (bwd dgated) | 478.0 | With A_idx, T-quant |
| GemmDefaultSm100 BF16 (bwd wgrad dw2) | 389.1 | |
| GemmDefaultSm100 BF16 (bwd wgrad/actgrad) | 780.7 + 770.9 | **Largest kernels** |
| token_gather_sum (bwd) | 68.4 | |
| gather_quant_pack (Triton, for next iter) | 96.2 | **This is the overhead to eliminate** |

**Key overhead sources:**
1. `gather_quantize_and_pack` in forward: ~96µs (quantizes TK=65536 rows instead of T=8192)
2. `elementwise direct_copy`: ~156µs × ~10 instances = 1558µs total across 10 iters
3. BF16 wgrad: ~1551µs combined (largest component of backward)

### 2.3 Old Shape Kernel Breakdown (T=4096, H=4096, I=2048, E=128, K=8)

**FP8 Forward (1017µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| GemmGatedSm100 FP8 | 562µs | vs BF16 1063µs → **47% faster** |
| GemmDefaultSm100 FP8 (down-proj) | 315µs | vs BF16 424µs → **26% faster** |
| quant_y1 | 35µs | for down-proj FP8 input |
| token_gather_sum | 46µs | |

**FP8 Backward (3386µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| wgrad_w1 BF16 | 1727µs | **51% of backward** |
| actgrad FP8 | 514µs | |
| wgrad_w2 BF16 | 511µs | |
| dgated FP8 | 488µs | |
| quant_dz | 71µs | |
| quant_dout (A_idx) | 12µs | |

---

## 3. Optimizations Applied

### 3.1 Fused GemmGated + SwiGLU (CUTLASS epilogue)
- Single kernel: FP8 GEMM + SwiGLU activation + blockscaled descale
- Eliminates separate SwiGLU kernel and intermediate z buffer read/write
- Now default (`SONIC_MOE_FP8_FUSED_GATED` defaults to `"1"`)

### 3.2 A_idx for backward dgated — commit `0bcb474`
- Quantize only T rows (~8µs) instead of TK rows (~96µs)
- CUTLASS gathers via A_idx during GEMM
- Backward passes precision tests

### 3.3 2D grid quant kernel — commit `2aaa278`
- ISA-packed scale writes: 1 byte → 4-byte uint32 packing
- quant_x: 58→8µs (at T=4096 with A_idx)

### 3.4 Adaptive FP8 down-proj — commit `9c95c14`
- Enabled at I≥2048 (configurable via `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD`)
- At I=1024: quant cost > GEMM savings → BF16

### 3.5 `use_fp8=True` API + `enable_fp8()` context manager (uncommitted)
- Clean programmatic API: `moe(x, use_fp8=True)` or `with enable_fp8(): moe(x)`
- Auto-enables QuACK GEMM, sets optimal defaults
- No env vars needed

---

## 4. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad (WGRAD=1)** | +400µs net at I=1024 | K_per_expert too small; colwise quant cost > GEMM savings |
| **Eager FP8 weight cache eviction** | Peak memory +2.6 GiB | `precompute_weight_fp8_*` creates BF16 temp copies > cached FP8 |
| **FP8 down-proj at I=1024** | -10µs net | quant 19µs > GEMM savings 9µs |
| **Stream overlap (quant ∥ wgrad)** | +7.8µs | Both saturate 128 SMs → bandwidth contention |
| **Transpose+rowquant for wgrad** | 8-12× slower | Terrible cache behavior |

---

## 5. Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `USE_QUACK_GEMM=1` | off | **Required** — Blackwell CUTLASS kernels |
| `SONIC_MOE_FP8_MODE=perf` | off | Enables FP8 path (or use `enable_fp8()` API) |
| `SONIC_MOE_FP8_FUSED_GATED` | **1** | Fused GemmGated+SwiGLU (default on) |
| `SONIC_MOE_FP8_WGRAD` | 0 | FP8 wgrad — keep 0 at standard shapes |
| `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD` | 2048 | I threshold for FP8 down-proj |

---

## 6. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | **Main FP8 logic** — forward, backward, all decisions |
| `sonicmoe/functional/utils.py` | `enable_fp8()` context manager, `enable_quack_gemm()` |
| `sonicmoe/moe.py` | `MoE.forward(use_fp8=True)` entry point |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches, ISA scale packing |
| `sonicmoe/quack_utils/gemm_gated.py` | Low-level CUTLASS GemmGated |
| `sonicmoe/quack_utils/gemm_dgated.py` | Low-level CUTLASS GemmDGated |
| `sonicmoe/quack_utils/gemm_interface.py` | High-level GEMM wrappers |
| `tests/fp8_large_project_contract_test.py` | 12 precision tests (8 original + 4 aligned) |

### 6.1 Architecture Notes

- **Weight caches:** 3 caches keyed on `(data_ptr(), _version, shape, stride)`:
  - `_FUSED_WEIGHT_CACHE` → forward GemmGated
  - `_DIRECT_FUSED_DGATED_WEIGHT_CACHE` → backward dgated
  - `_WEIGHT_CACHE` → backward actgrad
  - `clear_all_fp8_weight_caches()` for manual clearing

- **gemm_dgated_kernel** is called directly (not through `gemm_dgated_tuned`) because the tuned wrapper does `B.mT.contiguous()` which corrupts pre-quantized FP8 weight layout.

- **ISA Scale Layout:**
  - `_SF_TILE_M=128`, `_SF_TILE_K=128`, `_SF_VEC_SIZE=32`, `_SF_TILE_STORAGE=512 bytes`
  - Scale storage: `ceil(rows/128) × ceil(cols/128) × 512` bytes
  - Row position encoded: `row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4`

---

## 7. Correctness

12/12 contract tests pass:
- 8 original `FP8LargeProjectContractTest` (T=4096, H=4096, E=128, K=8)
- 3 aligned `FP8AlignedContractTest` (T=1024, H=3072, I=1536, E=8, K=8) — multi-seed forward/backward + context manager
- 1 production shape (T=8192, H=3072, I=1536, E=8, K=8) — fwd+bwd

```bash
# Quick validation
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape" --tb=short
```

---

## 8. Critical Findings and Insights for Next Agent

### 8.1 Forward A_idx + T-quant fails: root cause and fix path

**What was tried:** Replace `gather_quantize_and_pack_activation(x, gather_idx)` (TK rows, ~96µs) with `quantize_and_pack_activation(x)` (T rows, ~12µs) + pass `A_idx=gather_idx` to `gemm_gated()`.

**Result:** 93.5% RRMSE (complete garbage).

**Root cause:** When `A_idx` is passed, `GemmWrapperBase.validate_and_prepare_tensors` sets `M = A_idx.shape[0] = TK`. But the ISA-packed scale tensor was computed for T rows. The CUTLASS kernel's scale tile lookup uses M=TK to compute tile offsets → reads beyond the T-row scale tensor → garbage scales.

**Why backward dgated works:** The backward calls `gemm_dgated_kernel` **directly** (line 1225 of `__init__.py`), which also goes through `validate_and_prepare_tensors` with M=TK. BUT the CUTLASS GemmDGated kernel appears to correctly use `A_idx[j]` (not `j`) for scale indexing. The forward GemmGated kernel apparently does NOT.

**The user's key insight:** "A和Scale需要同时被gather" — in FP8, data and scales are row-correlated. Both must be gathered together. The solution is NOT to fall back to separate gather+quant (碎算子), but to fix the fused path where CUTLASS gathers both data and scales via A_idx.

**Recommended fix paths:**
1. **Debug GemmGated vs GemmDGated scale indexing:** Both inherit from `GemmSm100`, but the GatedMixin/DGatedMixin epilogues may differ in how they index `a_scales`. Compare `GemmGatedSm100` and `GemmDGatedSm100` in quack's CUTLASS DSL code to find the discrepancy.
2. **Gather scales alongside data:** Create `gather_scales_by_aidx(scales_T, A_idx)` that remaps ISA-packed scales from T-layout to TK-layout. Then pass TK-sized scales without A_idx. This preserves correctness but still has some quant overhead (though much less than full gather_quantize).
3. **Modify CUTLASS kernel:** Ensure GemmGated scale indexing uses `A_idx[row]` not `row` for scale tile lookups, matching what GemmDGated does.

### 8.2 The 156µs elementwise copy mystery

10 instances of `elementwise_kernel<128,4> direct_copy` at ~156µs each appear in backward, right before GemmDGated. Pattern per iter:
```
unroll_copy(4µs) → index_gather(7µs) → BIG copy(156µs) → dgated(481µs)
```

**NOT z dequantization** (fused gated path sets `z_is_fp8=False`). Likely candidates:
- `s = topk_scores[s_scatter_idx]` score gathering (line ~1170)
- Implicit dtype conversion in dgated setup
- Needs sync-barrier instrumentation to isolate

### 8.3 BF16 wgrad is the irreducible floor

At Ernie shape: wgrad_w1 + wgrad_w2 ≈ 1551µs = ~41% of total. Making this FP8 requires viable colwise quantization at K_per_expert = T*K/E = 8192*8/8 = 8192 rows — this is large enough that FP8 wgrad MIGHT be net-positive (unlike T=4096/E=128 where K_per_expert=256 was too small).

### 8.4 Wall-clock vs GPU projection divergence

FP8 wins much bigger in wall-clock (1.66-2.37× at old shapes) than in GPU projection (14.9-49.4%) because FP8 has fewer kernel launches → less CPU dispatch overhead (~2ms). For real-world throughput, wall-clock matters more.

### 8.5 Ernie comparison context

Ernie MOELayer (BF16, sequential expert loop) is **3.89× slower** than Sonic BF16 at the same shape. Ernie's FP8 path uses `moe_permute` + `spaq` (fused SwiGLU+routing+quant) operators. Ernie FP8 NSYS benchmark not yet collected — the script is at `tools/_nsys_ernie_v2.sh` but needs Ernie's FP8 config flags enabled.

**Ernie FP8 source code:** `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/tests/reference_layers/standalone_moe_layer/`:
- `moe/deep_ep_moe_layer.py` (1903 lines)
- `token_dispatcher/fp8_utils.py` (2000+ lines) — spaq, FP8 GEMM
- Enable FP8: `config.fp8 = "e4m3"`, `config.fp8_wgrad = True`, `config.use_fuse_node = True`
- PaddlePaddle env: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/dev_b`

### 8.6 Reference implementations

Located at `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/operator-incubator/cutify/ops/cute/`:
- `fused_weighted_swiglu_act_quant_cute()`: SwiGLU + FP8 quant fusion
- `fused_act_dequant_cute()`: FP8 dequant fusion
- `fused_swiglu_weighted_bwd_cute()`: SwiGLU backward fusion

### 8.7 NSYS timeline files

All at `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/`:
- `bf16_kernsum_cuda_gpu_kern_sum.csv`, `fp8_kernsum_cuda_gpu_kern_sum.csv` — kernel summaries
- `fp8_trace_cuda_gpu_trace.csv` — full kernel trace
- `ernie_v2_kernsum_cuda_gpu_kern_sum.csv` — Ernie BF16 kernel summary
- `nsys_fp8_nvtx2.nsys-rep` — FP8 with NVTX (full trace)
- `nsys_ernie_v2.nsys-rep` — Ernie full trace

---

## 9. Next Steps (ordered by ROI)

1. **🔴 Fix forward A_idx + T-quant** (biggest win, see §8.1)
   - Saves ~84µs/iter at Ernie shape (TK→T quant: 96→12µs)
   - Saves ~160MB activation memory (TK-sized FP8 data not materialized)
   - Aligns with SonicMoE's core "no materialization" design
   - Approach: debug GemmGated vs GemmDGated scale indexing in CUTLASS

2. **🟡 Eliminate 156µs elementwise copy** (see §8.2)
   - Saves ~156µs/iter if it's a single avoidable dtype conversion
   - Needs sync-barrier instrumentation to identify exact source

3. **🟡 Collect Ernie FP8 NSYS benchmark** (see §8.5)
   - Complete three-way comparison: Sonic BF16 / Sonic FP8 / Ernie FP8
   - Script exists but needs Ernie FP8 config

4. **🟢 FP8 wgrad at Ernie shape** (K_per_expert=8192, may be viable)
   - At old shapes K_per_expert=256 was net-negative
   - At Ernie shape K_per_expert=8192 — colwise quant ratio much better

5. **🟢 Memory optimization** (strided weight quant to eliminate cache copies)
