# SonicMoE FP8 Frontier — Architecture Specification

> Canonical reference for the blockscaled FP8 training path.
> Last updated: 2026-04-10 (Session 42, commit `17f4eec`).

## 1. Shape Convention (Ernie)

```
T = 8192   (tokens per batch)
H = 3072   (hidden dimension)
I = 1536   (intermediate dimension, SwiGLU: gate + value each I, fused 2I)
E = 8      (number of experts)
K = 8      (top-K per token)
TK = T×K = 65536  (tokens after routing, each token sent to K experts)
```

## 2. Forward Data Flow (FP8 + aligned + fused_gated)

```
Input: x (T, H) bf16

Step 1: quantize_and_pack_activation(x)
  → x_fp8 (T, H) fp8 + x_scales_t (ISA-packed, T-sized)

Step 2: _gather_isa_packed_scales_kernel(x_scales_t, x_gather_idx)
  → x_scales_tk (ISA-packed, TK-sized)
  Note: fp8 data NOT gathered — zero-materialization kernel gathers internally.

Step 3: GemmGatedSm100ZeroMat(x_fp8, w1_fp8, A_idx=x_gather_idx)
  → z (TK, 2I) bf16 — pre-activation (gate|value interleaved)
  → y1 (TK, I) bf16 — post-SwiGLU activation
  When EPILOGUE_QUANT=ON:
    → z pre-scaled in registers (×quant_scale), z_scale_out (TK, 2I/32) E8M0 raw

Step 4: z quantization (one of three paths):
  (a) Epilogue quant ON:  z_fp8 = z.to(fp8), scales from epilogue. FREE z_bf16.
  (b) Split quant:        z_fp8 = quantize_blockscaled_fast(z). FREE z_bf16. Then y1 quant.
  (c) Fused z+y1 quant:   Single Triton kernel for both. FREE z_bf16.

Step 5: quantize_and_pack_activation(y1)
  → y1_fp8 (TK, I) fp8 + y1_packed_scales (ISA-packed)
  Store in _PREQUANTIZED_SCALES["fwd"]. FREE y1_bf16.

Step 6: blockscaled_fp8_gemm_varlen(y1_fp8, w2_fp8, cu_seqlens_m)
  → y2 (TK, H) bf16

Step 7: Router scatter → output (T, H) bf16
```

### Memory lifecycle (forward, per step):
| After step | Live tensors | Approx MiB |
|-----------|-------------|:---:|
| 0 (input) | x, params, caches | 377+120 |
| 1 | x, x_fp8, x_scales | +24+3 |
| 3 | z(384), y1(192), x(48), params(216), caches(120) | ~960 |
| 4a (epi) | z_fp8(192), y1(192) — z_bf16 freed | ~576 |
| 5 | z_fp8(192), y1_fp8(96) — y1_bf16 freed | ~480 |
| 6 | y2(384) — y1_fp8 freed | ~580 |

**Forward peak = Step 3** (z+y1+params+caches = ~960 MiB for activations only).

## 3. Backward Data Flow (FP8 + aligned + fused_gated)

### DownProjection backward:
```
Input: dout (T, H) bf16

Step 1: quantize_and_pack_activation(dout) → dout_fp8 + dout_scales_t
Step 2: _gather_isa_packed_scales_kernel → dout_scales_tk

Step 3: GemmDGatedFP8CLoadSm100ZeroMat(dout_fp8, w2_fp8, z_fp8_preact)
  → dz (TK, 2I) bf16     (CUTLASS constraint: D output must be bf16)
  → y1s (TK, I) bf16     (post-SwiGLU derivative, for wgrad)
  Note: z_fp8 loaded via TMA as Int16 C, dequantized in epilogue registers.
  FREE z_fp8, z_raw_scales after dgated.

Step 4: Wgrad dw2 = gemm(dout.T, y1s, cu_seqlens_k, A_idx)  [BF16]
  FREE y1s after wgrad.

Step 5: quantize_and_pack_activation(dz) → dz_fp8 + dz_packed_scales
  Store in _PREQUANTIZED_SCALES["bwd"].
```

### UpProjection backward:
```
Step 6: Pop _PREQUANTIZED_SCALES["bwd"] → (dz_ref, dz_fp8, dz_scales)

Step 7: Wgrad dw1 = gemm(x.T, dz_bf16, cu_seqlens_k, A_idx)  [BF16]
  FREE dz_bf16 via resize_(0).

Step 8: Actgrad dx = blockscaled_fp8_gemm_varlen(dz_fp8, w1T_fp8)
  → dx_expanded (TK, H) bf16

Step 9: Token reduce → dx (T, H) bf16
```

### Backward peak tensors:
| Tensor | MiB | Phase | Freed |
|--------|:---:|-------|-------|
| dz bf16 | 384 | dgated output | After UpProj wgrad (Step 7) |
| y1s bf16 | 192 | dgated output | After DownProj wgrad (Step 4) |
| dz_fp8 | 192 | prequant (Step 5) | After actgrad (Step 8) |
| z_fp8 (ctx) | 192 | from forward | After dgated (Step 3) |
| dw2 | 72 | deferred alloc | returned |
| dw1 | 144 | UpProj alloc | returned |
| dx_expanded | 384 | actgrad (Step 8) | After token reduce |

**Backward peak = DownProj postact-release** (~1314 MiB with stash).

## 4. Weight Cache System

4 distinct FP8 layouts needed (different GEMM kernels require different physical layouts):

| Cache | Variable | Layout | Size (Ernie) | Used by |
|-------|----------|--------|:---:|---------|
| Fused gated | `_FUSED_WEIGHT_CACHE` | (E, K_gated, N_gated) | ~74 MiB | GemmGated forward |
| Varlen w2 | `_VARLEN_WEIGHT_CACHE` | (E, dim0, dim1) | ~37 MiB | DownProj forward |
| Fused dgated | `_FUSED_WEIGHT_CACHE` | (E, N_dgated, K_dgated) | ~37 MiB | GemmDGated backward |
| Varlen w1T | `_VARLEN_WEIGHT_CACHE` | (E, H, 2I) | ~74 MiB | Actgrad backward |

Cache key: `(data_ptr, _version, shape, stride)`. Invalidates on `optimizer.step()` (`_version` increments).

**When stash_bf16_to_cpu() is active**: data_ptr changes → cache misses. Solved by `_STASHED_FP8_WEIGHTS` module-level dict that bypasses global cache lookups.

## 5. FP8 Quantization Kernels

| Kernel | Time (µs) | Operation | Output format |
|--------|:---:|---------|-------|
| `_quantize_and_pack_kernel` (Triton) | 44 | bf16→fp8 + ISA-packed E8M0 | (M,K) fp8 + (1, packed) ISA |
| `_quantize_flat_blockscaled_kernel` (Triton) | 122 | bf16→fp8 + raw E8M0 | (M,K) fp8 + (M, K/32) raw |
| `_gather_isa_packed_scales_kernel` (Triton) | 27 | ISA-scale gather T→TK | (1, packed_TK) ISA |
| `BlockscaledScaleStore` (CUTLASS epilogue) | 0* | E8M0 in-register compute + STG | (M, K/32) raw E8M0 |
| `_fused_z_save_y1_quant_kernel` (Triton) | ~180 | z+y1 dual quant, single launch | z_fp8 + z_raw + y1_fp8 + y1_ISA |

*BlockscaledScaleStore has zero standalone cost — it executes inside the GEMM epilogue registers.

## 6. CUTLASS Epilogue Architecture

### EpiOp Chain (GemmGated variants):

```
GemmDefaultEpiMixin:  Scalar(α) → Scalar(β) → RowVecLoad → ColVecLoad
GemmActMixin:         + TileStore("mPostAct")
GemmGatedMixin:       + TileStore("mPostAct", half-N-tile for SwiGLU)
GemmGatedBlockscaledQuantMixin: + BlockscaledScaleStore("mZScale")
```

### BlockscaledScaleStore mechanics:
1. `begin()`: compute absolute (m_abs, n_base) from tile coordinates
2. In `epi_visit_subtile`: amax reduction → integer+carry E8M0 → quant_scale → multiply tRS_rD in-place → STG scale byte
3. No smem, no TMA — pure register compute + direct global store

### GemmDGated FP8PreAct epilogue:
- `FP8PreActLoad`: LDG-loads z_fp8 bytes + E8M0 scale bytes from global memory
- Dequant in registers: `fp8_val * 2^(scale - 127)`
- dSwiGLU computation: `dz = dout * (σ(gate) * (1 + value * (1 - σ(gate))))`

## 7. Performance Model

`time(T) = α × T + β`

| Mode | α (µs/token) | β (µs) | Asymptotic speedup (T→∞) |
|------|:---:|:---:|:---:|
| BF16 | 0.4556 | 258 | baseline |
| FP8 | 0.4066 | 250 | **1.120×** |

FP8 achieves 10.7% per-token advantage. Speedup monotonically increases from 1.087× (T=1K) to 1.120× (T→∞).

## 8. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, cache management |
| `sonicmoe/moe.py` | MoE class, refresh_fp8, stash_bf16_to_cpu |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 quant kernels, weight caches, GEMM wrappers |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated CUTLASS DSL + BlockscaledScaleStore EpiOp |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated CUTLASS DSL + FP8PreActLoad EpiOp |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-materialization GEMM classes |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU + quant Triton kernels |

## 10. Future: 32×32 Isotropic Weight Quantization

### Problem
4 different FP8 weight cache layouts needed (w1_fused, w2_varlen, w2_dgated, w1T_varlen) = ~120 MiB.
Each uses different physical layout + different 1×32 scale grouping direction.

### Solution
Use 32×32 block quantization for weights (one E8M0 scale per 32×32 tile).
Store as standard 1×32 ISA format with **row broadcast** (32 identical scale bytes per atom row).

**Why this works:**
- 32×32 block: scale is isotropic — covers 32 rows AND 32 columns
- Forward GEMM reads groups along K (dim1): 32-element groups → exactly one 32×32 block's column
- Backward actgrad reads groups along K (dim0 of transposed): 32-element groups → exactly one 32×32 block's row
- Same scale value in both directions → **1 fp8 data + 2 scale packs (original + transposed ISA)**
- Transpose scale re-pack is a pure index operation (no data read, no re-quantize)

**Constraints:**
- Activations MUST use 1×32 (hardware ISA requirement for `tcgen05.mma`)
- Only weights benefit from 32×32 (used in both forward and backward GEMM)
- Quantization precision slightly lower (amax over 1024 elements vs 32)

**Expected savings:**
- 4 weight cache layouts → 1 fp8 data + 2 ISA scale packs
- ~120 MiB cache → ~37 MiB (fp8 data) + ~4 MiB (scales) = ~41 MiB
- Net: **−79 MiB** weight cache reduction

| Approach | Why it fails |
|----------|-------------|
| FP8 wgrad at I=1536 | transpose-quant 634µs > GEMM savings 259µs |
| 4-layout fp8 ≈ bf16 params | 222 MiB (4 layouts) > 216 MiB (bf16 params) |
| dz prequant ∥ wgrad | y1s + dz_fp8 coexist → +192 MiB peak |
| CUTLASS D output as fp8 | d_dtype.width must be ≥16 (bf16 minimum) |
| `torch.empty(0).as_strided()` | PyTorch bounds check rejects |
| `w.data = fp8` for param offload | autograd internal behavior change |
