# Plan: FP8 Unaligned Token Support (Remove Token Rounding Requirement)

## Goal

Enable SonicMoE FP8 path for E>8 **without token rounding**, supporting arbitrary (non-128-multiple) per-expert token counts. Principle: **minimal changes, maximum performance**.

## Root Cause

128-alignment is **NOT** a raw CUTLASS GEMM requirement — CUTLASS varlen scheduling handles arbitrary segment sizes. The constraint comes from **ISA-packed E8M0 scale tiles** which have fixed 128×128 tiling. Partial tiles need scale bytes pre-filled with the neutral value 127 (= 2^0 = 1.0). The codebase already handles this for forward via `torch.full(..., 127)` initialization.

## Current State

| Component | Aligned (E≤8) | Non-aligned (E>8) | Status |
|-----------|:---:|:---:|--------|
| _UpProjection.forward | FP8 fused gated | FP8 padded (`_padded_blockscaled_gated_forward`) | **WORKS** |
| _DownProjection.forward | FP8 varlen | FP8 varlen (internal padding in `blockscaled_fp8_gemm_varlen`) | **WORKS** |
| _UpProjection.backward | FP8 wgrad + actgrad | BF16 fallback (line 1264) | Works (suboptimal) |
| **_DownProjection.backward** | FP8 dgated + wgrad | **RuntimeError (line 1878)** | **BLOCKER** |

## Alignment Constraint Inventory (9 sites, 2 need fixing)

| # | File:Line | Constraint | Class | Fix? |
|---|-----------|-----------|-------|------|
| 1 | `__init__.py:485` | `_all_segments_128_aligned` routing decision | TRIVIAL | No — already routes correctly |
| 2 | `__init__.py:213-308` | Up-proj forward padding | SOLVED | No |
| 3 | `__init__.py:1398-1411` | Down-proj forward varlen | SOLVED | No |
| **4** | **`__init__.py:1875-1882`** | **Down-proj backward REJECT** | **SOFT** | **YES — primary blocker** |
| **5** | **`__init__.py:1131`** | **Up-proj backward FP8 gate** | **SOFT** | **YES — secondary** |
| 6 | `blockscaled_fp8_gemm.py:190` | ISA scale storage calc | TRIVIAL | No — uses `_div_up` correctly |
| 7 | `blockscaled_fp8_gemm.py:348-387` | `_get_padding_plan` utility | SOLVED | No — reusable for backward |
| 8 | `blockscaled_fp8_gemm.py:2311-3848` | K/H dim checks | HARD (arch) | No — always satisfied by model design |
| 9 | `blockscaled_fp8_gemm.py:3764,128` | Grouped GEMM capacity | SOFT | No — deprecated path |

## Implementation Plan

### Phase 1: Remove RuntimeError — BF16 backward fallback (~15 LoC)

**The BF16 backward code already exists at lines 1883-1917.** The RuntimeError at line 1875 is just a guard that prevents reaching it. Remove the guard.

**Changes:**

1. `_DownProjection.backward` line 1875: Change
   ```python
   elif ctx._fp8_enabled_flag and not ctx._alignment_assumed_flag:
       raise RuntimeError(...)
   ```
   to:
   ```python
   elif ctx._fp8_enabled_flag and not ctx._alignment_assumed_flag:
       # Non-aligned FP8: use BF16 backward (FP8 forward benefits preserved)
       # Fall through to BF16 dgated + BF16 wgrad below
       pass  # (the BF16 path at lines 1883-1917 handles this)
   ```
   Actually, need to verify the BF16 fallback path (lines 1883-1917) doesn't have its own alignment issues. If it does, adjust context flags.

2. `_UpProjection.backward` line 1131: The `elif` for non-aligned already falls through to BF16 at line 1264. Verify this path works.

**Risk**: LOW — uses proven BF16 backward code.
**Performance**: FP8 forward + BF16 backward → ~1.1-1.25× over pure BF16.
**Verification**: Run `test_moe_module.py` with non-128-aligned shapes (e.g., T=300, E=32, K=8 → per-expert ~75 tokens).

### Phase 2: Padded FP8 dgated backward (~80-100 LoC)

Add padding support to the `gemm_dgated_kernel` call in `_DownProjection.backward`:

1. Call `_get_padding_plan(expert_frequency_offset, TK)` (already exists)
2. If `needs_pad`: create padded dout, padded cu_seqlens, padded z_fp8
3. Run `gemm_dgated_kernel` with padded inputs (same as aligned path, lines 1701-1724)
4. Unpad dz, y1s, ds via `dst_idx`
5. Keep BF16 wgrad (no FP8 wgrad for unaligned segments — wgrad varlen_k scheduler has alignment constraint)

**Key insight from agent research**: `gemm_dgated_kernel` with varlen scheduling should work on padded segments because ISA scale tiles already initialize padding bytes to neutral value 127. The dgated GEMM epilogue masks handle partial CTA tiles.

**Risk**: MEDIUM — need extensive numerical validation.
**Performance**: FP8 forward + FP8 dgated backward + BF16 wgrad → ~1.3-1.5× over pure BF16.
**Lines**: ~80-100 new in `__init__.py`.

### Phase 3: Padded FP8 wgrad backward (~120-200 LoC)

The hardest piece. `_run_cutlass_blockscaled_gemm_varlen_k` partitions K (token dim) by expert boundaries. Non-aligned segments mean partial scale tiles at expert boundaries — **this produces wrong gradients** because the varlen_k scheduler's CTA will load scale bytes from the adjacent expert.

**Two options:**

- **(A) Pad activation/gradient segments before varlen_k GEMM**: Scatter quantized FP8 data into padded layout with neutral scale bytes at boundaries. Complex but memory-efficient.
- **(B) Use grouped GEMM with auto-capacity padding**: `blockscaled_fp8_gemm_grouped` with `_auto_capacity` already handles arbitrary sizes via capacity padding. Simpler but allocates max-capacity per expert.

Recommend **(B)** for E≤32 (acceptable memory) and **(A)** for E>32 (memory concern).

**Risk**: HIGH — incorrect scale tile handling silently corrupts gradients.
**Performance**: Full FP8 path → ~1.45-1.65× over pure BF16.
**Lines**: ~120-200.

## Performance Impact Summary

| Phase | Forward | Backward | Speedup vs BF16 | Extra Overhead |
|-------|---------|----------|:---:|---:|
| Current (token rounding) | FP8 | FP8 | 1.3-1.7× | 0-10% wasted tokens |
| **Phase 1** | FP8 | BF16 | **1.1-1.25×** | **0%** |
| **Phase 2** | FP8 | FP8 dgated + BF16 wgrad | **1.3-1.5×** | ~5-25% extra dgated rows |
| **Phase 3** | FP8 | FP8 full | **1.45-1.65×** | ~5-25% extra rows all GEMMs |

Token rounding wastes tokens at routing level (model quality impact). Internal padding wastes compute but preserves all real tokens with correct gradients. For E=32, TK=65536: worst-case padding = 32×127 = 4064 extra rows → 6.2% overhead.

## Verification Plan

For each phase, run:
1. `test_moe_module.py` with shapes `(T=300, H=768, I=384, E=32, K=8)` — per-expert ~75 tokens, NOT 128-aligned
2. Compare output/dx/dw1/dw2 against BF16 gold (RRMSE < 0.10 for FP8, < 0.01 for BF16)
3. Cross-check: FP8-non-aligned output should match FP8-token-rounded output within noise

## Files to Modify

| File | Phase | Lines Added | Lines Modified |
|------|:-----:|--------:|--------:|
| `sonicmoe/functional/__init__.py` | 1 | 5 | 10 |
| `sonicmoe/functional/__init__.py` | 2 | 80-100 | 15 |
| `sonicmoe/functional/__init__.py` | 3 | 120-200 | 20 |
| `tests/ops/test_moe_module.py` | 1-3 | 30 | 0 |
| **Total** | | **235-335** | **45** |
