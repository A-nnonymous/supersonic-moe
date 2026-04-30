# dz iso32 epilogue-fusion — Phase 0 audit + Phase 1 design

## TL;DR

**Phase 0 (precision audit) PASSED** on real Ernie-shape (TK=65536, 2I=3072)
dz tensors across 3 routing distributions. iso32 (32×32) quant produces
**identical** downstream-GEMM RRMSE to the production 1×32 quant
(ratio = 1.000× for both dx and dw1 GEMMs) under FP8 e4m3.

→ iso32 is **precision-safe** for dz under FP8 e4m3 — Phase 1A (kernel work)
is unblocked from a numerical standpoint.

## Method

### 1. Capture real dz tensors

`tools/dump_real_dz.py` monkey-patches
`sonicmoe.functional.gemm_dgated_kernel`, runs warmup + capture iters via
`SonicMoEMlpNode`, and saves the third positional-arg dz (the BF16 output
of the dGated kernel) to numpy at full precision.  Captured 6 tensors:
`{none, skew, extreme} × {iter4, iter5}`, each shape `(65536, 3072)` bf16,
amax ≈ 3.3 – 3.9 (realistic gradient magnitude — earlier audits with the
0.02-scaled fixture collapsed everything into a single FP8 bin and were
not informative).

### 2. Pure-PyTorch quant fidelity

`tools/audit_dz_iso32_quality.py` runs both 1×32 and 32×32 quant→dequant
via `tests/ops/audit_iso32_numerics._quant_dequant_blockscaled` (clean
reference impl: tile amax → e8m0 → e4m3 cast → dequant), then reports
cosine, RRMSE, max_abs, and per-(32,32) dyn-range bits-lost.

### 3. Downstream-GEMM error (the actual training-impact metric)

For each captured dz, generate random `w1 (2I, H)` and `x (TK, H)` with
Ernie-realistic scaling, then compute three versions of:
- `dx_proxy  = dz @ w1ᵀ`
- `dw1_proxy = dzᵀ @ x`

(a) reference using BF16 dz, (b) using 1×32-dq dz, (c) using 32×32-dq dz.
Headline metric is the **iso32/1×32 RRMSE ratio** for each downstream
output — this directly reflects what the next bwd GEMM consumer would see.

## Results (representative — `dz_extreme_iter4.npy`)

| metric                      | 1×32       | 32×32      | ratio |
|-----------------------------|-----------:|-----------:|------:|
| direct cos                  | 0.999646   | 0.999646   | —     |
| direct RRMSE                | 2.659e-2   | 2.659e-2   | 1.000×|
| direct max_abs              | 1.250e-1   | 1.250e-1   | —     |
| dx_proxy RRMSE vs ref       | 2.659e-2   | 2.659e-2   | 1.000×|
| dw1_proxy RRMSE vs ref      | 2.658e-2   | 2.658e-2   | 1.000×|
| dyn-range frac>1bit (advis.)| —          | 0.735      | —     |

Identical numbers across `none/skew/extreme` and across iters — see
`audit.md` for the full per-capture table.

## Why iso32 is a free lunch on FP8 dz

The dyn-range proxy `log2(tile_amax / row_amax)` predicted a large
mantissa loss for iso32 (frac>1bit ≈ 73%), but in practice **1×32 has the
same RRMSE as 32×32**. Reason: e4m3's 3-bit mantissa imposes ~12.5%
per-value rounding noise, which dominates the per-row scale advantage that
1×32 buys. The "extra" mantissa bits 1×32 reserves are below the e4m3
mantissa precision floor and therefore unused.

In contrast, when iso32 was retired for **weights** (PR #15), weights are
much larger and have tighter per-row distributions — the mantissa
advantage of 1×32 there was not wasted.

## Performance ceiling (revised honest numbers)

If Phase 1A lands successfully:

- **Eliminated**: dz BF16 (TK, 2I) ≈ 384 MiB HBM write + 384 MiB read +
  the entire `_dual_varlen_quantize_kernel` launch+compute.
- **Added**: epilogue amax+cast cost (≈ 30–60 µs); FP8 store TMA atom
  (half BF16 bandwidth → cheaper).
- **Net realistic savings**: 100–180 µs / iter (vs 2754 µs busy).
- **MFU lift**: +1.5 to 3.0 pp (from 44.91% baseline → 46.5–48.0%).

## Phase 1A design — kernel change details

### Target

`sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` line 397
(`GemmDGatedFP8CLoadSm100ZeroMat`) and the underlying mixin
`sonicmoe/quack_utils/gemm_dgated.py:510` (`GemmDGatedFP8CLoadMixin`).

### Template

`sonicmoe/quack_utils/gemm_gated.py:221` (`GemmGatedBlockscaledQuantMixin`)
is the **forward** analog — it already does production 1×32 blockscaled
FP8 quant in the epilogue with bit-exact match to Triton. Its
`epi_visit_subtile` (line 244) is the pattern to mirror.

### What the bwd version must do differently

1. **iso32 amax instead of 1×32** — share one e8m0 byte across 32
   M-rows × 32 N-cols. SM100 `Ld32x32bOp` already places exactly one
   M-row per thread; need a warp-level shuffle to reduce amax across the
   32 lanes (M-axis) holding the same N-tile.
2. **D shape is (TK, 2I)** — dGated outputs gate+up packed; the
   epilogue currently writes BF16 (TK, I-as-f32). When we switch D dtype
   to FP8, we get (TK, 2I) FP8 directly. Need to update `d_dtype`,
   `epi_tile`, and TMA store atom selection (8-bit StMatrix instead of
   16-bit). The fwd `GemmGatedBlockscaledQuantMixin` already handles
   this — model after `_make_tma_epi_atoms_and_tensors` chain.
3. **Scale store layout** — use `BlockscaledScaleStore` (already in
   `gemm_gated.py:172`) but with a wider (M, 2I/32) shape rather than
   (M, I/32). Same ISA-packing logic.
4. **Preserve BF16-emit path** — add an optional flag (e.g.,
   `mDZScale: Optional[cute.Tensor] = None`); when None, fall through to
   the existing BF16-emit behavior. This protects all current callers
   and gives us a one-flag rollback.

### Register budget — main risk

S78b confirmed `GemmDGatedFP8CLoadSm100ZeroMat` is register-bound
(every register tweak landed neutral or regressed). The amax + e8m0 +
cast logic adds ≈ 24–32 registers per thread. Mitigations:
- BF16 store → FP8 store frees TMA atom registers (each elem 2B → 1B).
- Reorder `epi_visit_subtile` to release tRS_rD slots before TMA store.
- If unavoidable spill, fall back to **dual-amax 1×32 epilogue** (Phase
  1B) — same structure, more arithmetic, possibly within budget if the
  warp shuffle in iso32 hurts more than the extra amax loop.

### Phase 2 wire-up — Python side

`sonicmoe/functional/__init__.py` line 1967:

```python
# OLD
dz = torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=...)
gemm_dgated_kernel(... dz, ...)
# downstream: fused_dual_colwise_quantize(dz, ...)

# NEW
dz_fp8 = torch.empty((total_m, n * 2), dtype=torch.float8_e4m3fn, device=...)
dz_scales = torch.empty(_storage_per_batch(total_m, 2*n, ...), dtype=torch.uint8, ...)
gemm_dgated_kernel(... dz_fp8, ..., mDZScale=dz_scales)
# downstream dx GEMM uses dz_fp8 + dz_scales directly (rowwise format)
# downstream dw1 GEMM gets a cheap scale-rewrite (~10 µs) into colwise format
```

The colwise scale layout has the same e8m0 bytes (iso32 → byte identical
between formats), just placed at different offsets.  A one-pass byte
shuffle kernel is needed (`sonicmoe/quack_utils/scales_iso_rewrite.py`).

### Phase 3 — CI gates

Must pass without modifying CI:
- `tests/fp8_frontier_determinism_test.py` (byte-identical determinism)
- `tests/fp8_large_project_contract_test.py` (production tolerance)
- `tests/fp8_protocol_test.py`
- `tests/ops/test_gemm_dgated.py` (per-shape numerical)
- `tests/ops/test_dual_quant.py` (downstream consumer)
- `tests/moe_blackwell_test.py` (end-to-end)

Phase 0 audit shows these *should* pass (downstream RRMSE identical to
production 1×32 path), but the bit-exact determinism test will only pass
if the e8m0 encoding matches Triton — model after
`GemmGatedBlockscaledQuantMixin` integer+carry algorithm exactly.

## Files / pointers for next agent

| Purpose | Path |
|---|---|
| Phase 0 capture tool | `tools/dump_real_dz.py` |
| Phase 0 audit tool | `tools/audit_dz_iso32_quality.py` |
| Pure-pytorch quant ref | `tests/ops/audit_iso32_numerics.py:32` |
| Captured dz tensors | `reports/iso32_dz_audit/dz_*.npy` |
| Audit results | `reports/iso32_dz_audit/audit.md` |
| Forward fp8-emit template | `sonicmoe/quack_utils/gemm_gated.py:221` |
| Forward scale store EpiOp | `sonicmoe/quack_utils/gemm_gated.py:172` |
| Bwd target (kernel) | `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py:397` |
| Bwd target (mixin) | `sonicmoe/quack_utils/gemm_dgated.py:510` |
| Skeleton notes (advisory) | `sonicmoe/quack_utils/epi_blockscaled_quant.py` |
| Wire-up call site | `sonicmoe/functional/__init__.py:1967` |
| Storage size helper | `sonicmoe/quack_utils/blockscaled_fp8_gemm.py:199` |
| ISA constants | top of `blockscaled_fp8_gemm.py` |
| Reference iso32 weight quant | `blockscaled_fp8_gemm.py:3465` |

## Status

- [x] Phase 0.1 — dz capture infra
- [x] Phase 0.2 — iso32 vs 1×32 + downstream-GEMM audit (PASS)
- [ ] Phase 1A — Cutlass DSL epilogue change (designed; not yet
      implemented — multi-day CuTe DSL work)
- [ ] Phase 1B — fallback dual-amax epilogue (only if 1A fails register
      budget)
- [ ] Phase 2 — Python wire-up + scales_iso_rewrite kernel
- [ ] Phase 3 — CI + nsys perf
- [ ] Phase 4 — final report
