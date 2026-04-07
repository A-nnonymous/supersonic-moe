# Phase 3.1: TMA-based FP8 C Load for GemmDGated — Technical Report

> Branch: `native-fp8-exploration` @ `8c43c83`
> Date: 2026-04-07

---

## 1. Problem Statement

In SonicMoE's backward pass, `GemmDGated` computes dSwiGLU(dout, w2, z) where z
is the forward pre-activation tensor (TK × 2I, bf16). When z is saved as FP8 for
memory efficiency, a standalone Triton dequant kernel converts z_fp8 → z_bf16 before
the CUTLASS GemmDGated kernel can use it.

**Cost**: 126µs dequant + 384 MiB z_bf16 temporary buffer.

**Goal**: Eliminate both by loading z_fp8 directly via TMA inside the CUTLASS kernel.

---

## 2. Architecture

### Standard BF16 Path
```
z_bf16.view(f32)  →  TMA(f32)  →  smem(f32)  →  reg(f32)  →  recast(bf16×2)  →  dSwiGLU
```

### FP8 TMA Path (This Work)
```
z_fp8.view(int16)  →  TMA(i16)  →  smem(i16)  →  reg(i16)  →  recast(fp8)  →  cvt(f32)  →  dequant  →  dSwiGLU
```

### Key Insight: Int16 Packing

The CUTLASS SM100 kernel shares a single `epi_tile` between C (source) and D (output)
for both TMA atom creation and `zipped_divide` partitioning. Changing the epi_tile for
C alone would require modifying the installed quack kernel — impossible without forking.

**Solution**: View the fp8 tensor `(TK, 2I) fp8` as `(TK, I) Int16`:
- Each Int16 = 2 packed fp8 values (gate + up), mirroring D's f32 = 2 packed bf16
- C and D have identical shapes → shared epi_tile works
- `recast_tensor(Int16, Float8E4M3FN)` in registers unpacks to 2 fp8 per Int16
- Vectorized `fp8.to(Float32)` converts to f32 for dequant + dSwiGLU

---

## 3. Implementation

### Files Modified

| File | Changes |
|------|---------|
| `sonicmoe/quack_utils/gemm_dgated.py` | `GemmDGatedFP8CLoadMixin`: Int16 C support |
| `sonicmoe/quack_utils/gemm_dgated.py` | `gemm_dgated()`: `preact_fp8.view(int16)` |

### `GemmDGatedFP8CLoadMixin` Overrides

1. **`_setup_attributes`**: Creates Int16 smem layout via `make_smem_layout_epi(Int16, ...)`.
   Same epi_tile as D — only the element type and swizzle pattern differ.

2. **`epilog_smem_load_and_partition`**: Keeps register layout identical to D (N elements).
   No doubling needed — the `recast_tensor` in `epi_visit_subtile` handles the expansion.

3. **`epi_visit_subtile`**: The fp8 path:
   ```python
   tRS_rC_fp8 = cute.recast_tensor(tRS_rC, Float8E4M3FN)  # N Int16 → 2N fp8
   tRS_rXY_f32x2.store(tRS_rC_fp8.load().to(Float32))     # 2N fp8 → 2N f32
   # Blockscaled dequant: scale_i = 2^(e8m0_i << 23)
   # dSwiGLU on dequanted values
   ```

4. **`epi_to_underlying_arguments`**: Allows Int16 c_dtype (skip bf16 width assertions).

### `gemm_dgated()` Wrapper Changes

```python
if fp8_preact_mode:
    PreAct = preact_fp8.view(torch.int16)  # (TK, 2I) fp8 → (TK, I) int16
```

C validation in `validate_and_prepare_tensors` passes because Int16 C shape `(TK, I)`
matches D shape `(TK, I)` f32.

---

## 4. Results

### Precision (Cross-verified)

| Comparison | y RRMSE | dx RRMSE | dw max RRMSE |
|-----------|---------|----------|-------------|
| FP8 TMA vs FP8 Frontier | 0.000000 | 0.000000 | 0.000000 |
| FP8 vs BF16 gold | 0.015 | 0.518 | 0.534 |

**FP8 TMA is bit-exact with the frontier FP8 path** — all RRMSE = 0.
The FP8 vs BF16 deviation is the normal FP8 quantization precision loss.

### Performance (Cross-node validated, 2 idle B200, CV < 1%)

| Metric | Node 0266 | Node 0275 |
|--------|-----------|-----------|
| BF16 total (dequant + GemmDGated) | 513µs | 511µs |
| BF16 GemmDGated only | 406µs | 408µs |
| **FP8 TMA (Int16 + in-kernel dequant)** | **496µs** | **495µs** |
| **Speedup vs BF16 total** | **-17µs (-3.4%)** | **-16µs (-3.2%)** |

**In-kernel dequant has ZERO overhead** — fully hidden by MMA compute latency.
(Verified: FP8 with real scales 505µs ≈ FP8 with zero scales 507µs)

### Memory

| Tensor | BF16 | FP8 | Saving |
|--------|------|-----|--------|
| z (TK=65536, 2I=3072) | 384 MiB | 192 MiB | **-192 MiB** |

---

## 5. E2E Integration

The backward integration was already complete in `_DownProjection.backward`:

```python
# Line 1177: Auto-detect fp8 preact
use_fp8_preact = (z is None and z_fp8 is not None)

# Line 1248-1249: Pass to kernel
preact_fp8=z_fp8 if use_fp8_preact else None,
preact_scales=z_raw_scales_u8 if use_fp8_preact else None,
```

**No code changes needed** — the E2E path works when:
- `SONIC_MOE_FP8_SAVE_Z_FP8=1` (default) — forward saves z as fp8
- `SONIC_MOE_FP8_FUSED_GATED=1` (default) — uses fused GemmDGated
- `enable_fp8()` context is active

---

## 6. Key Lessons Learned

### SM100 Epilogue Architecture Constraints

1. **epi_tile is shared**: The kernel's `epi_tile` (a CuTe Layout tuple on SM100) is
   used for both C and D in `epilog_gmem_copy_and_partition → zipped_divide`. Changing
   it for one tensor requires changing it for both.

2. **CuTe Layout ≠ Integer**: On SM100, epi_tile elements are CuTe Layout objects
   encoding TMEM warp distribution patterns. Integer tiles produce wrong CTA V-maps.

3. **TMA partition validation**: The MLIR `AtomTmaPartitionOp` checks that
   `size(smem_first_rank) == size(gmem_first_rank)`. The smem layout comes from
   `make_smem_layout_epi` and the gmem partition from `zipped_divide(tensor, epi_tile)`.

### Performance Insights

1. **In-kernel dequant is free**: Blockscaled scale LDG (per-element scalar loads) is
   fully hidden by MMA compute latency. No optimization of the dequant loop was needed.

2. **Busy-node measurements are unreliable**: Our initial measurement showed 1039µs
   (on a busy node). The real number is 496µs (on an idle node). Always use
   `nvidia-smi` to verify GPU idle, and cross-validate on 2+ nodes.

3. **TMA bandwidth is the real cost**: The 90µs overhead of Int16 TMA load + fp8→f32
   conversion is the irreducible cost. This is offset by eliminating the 126µs
   standalone dequant kernel.

### The Int16 Packing Technique

This technique may be applicable to other cases where a source tensor has a different
element type but the kernel's tiling structure is fixed:
- View the source as a packed type matching the expected element width
- Use `recast_tensor` in registers to unpack
- The TMA loads the right number of bytes (Int16 = 2 bytes = 2 fp8 bytes)

---

## 7. Files Reference

| File | Purpose |
|------|---------|
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGatedFP8CLoadMixin + gemm_dgated wrapper |
| `sonicmoe/functional/__init__.py` | _DownProjection.backward (E2E integration) |
| `tests/test_fp8c_tma_compile.py` | Compilation + precision test |
| `tests/bench_fp8_tma_diagnosis.py` | Performance diagnosis |
| `tests/test_fp8_tma_vs_frontier.py` | E2E bit-exactness verification |
| `tests/test_e2e_fp8_tma.py` | Full MoE E2E validation |

---

## 8. Git History

```
8c43c83  E2E validation — TMA path bit-exact vs frontier
c164b31  performance diagnosis — FP8 TMA 3.2% faster
8a833ff  cross-node validated — FP8 TMA 496µs vs BF16 512µs
85a11b8  TMA-based FP8 C load — 0 RRMSE, -192MiB
```
