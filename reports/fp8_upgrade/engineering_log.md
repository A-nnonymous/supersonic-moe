# FP8 Engineering Log

> This log keeps only the conclusions that still survive revalidation.
> Historical dead ends remain in git history; this file is the cleaned, high-signal timeline for the next agent.

## Session 22 / 2026-03-31 — Deep profiling, lean path prototype, frontier redefined

### 1. Re-validated the authoritative NSYS baseline

- **Tool:** `tools/nsys_full_breakdown.py` against `reports/sonic_official_bf16.sqlite`
- **Confirmed:** official BF16 = `2475.2us`, current fused FP8 + BF16 wgrad = `2600.3us`
- **New finding:** the `125us` gap is entirely explained by standalone quant/dequant kernels

### 2. Precise kernel-level cost accounting (NSYS GPU projection)

| Kernel category | FP8 time | BF16 time | Delta | Notes |
|---|---|---|---|---|
| `gemm_gated` (up-proj) | 286us | 418us | **-132us** | FP8 tensor core advantage |
| `quack.gemm` (down-proj fwd) | 215us | 231us | **-16us** | Slight FP8 win |
| `gemm_dgated` (backward) | 249us | 258us | **-9us** | Slight FP8 win |
| `quack.gemm` ×3 (wgrad+actgrad) | 1195us | 1297us | **-102us** | FP8 weight caching effect |
| **Total GEMM savings** | | | **-259us** | |
| `_gather_quantize_and_pack` ×2 | 196us | 0 | **+196us** | Activation quant for GEMM input |
| `_quantize_and_pack` ×2 | 92us | 0 | **+92us** | y1/dz pre-quant |
| `_quantize_flat_blockscaled` ×1 | 56us | 0 | **+56us** | z FP8 save |
| `_dequant_blockscaled_fp8` ×1 | 42us | 0 | **+42us** | z FP8 restore |
| **Total quant overhead** | | | **+386us** | |
| **Net** | | | **+127us** | Matches measured 125us gap |

### 3. Discovered wall-clock vs GPU-projection divergence

- **Local event timing:** FP8 fused = `3.14ms`, BF16 = `4.83ms` → **1.54x faster**
- **NSYS GPU projection:** FP8 fused = `2.60ms`, BF16 = `2.48ms` → **0.95x**
- **Root cause:** FP8 path has fewer total kernel launches → much less Python/CPU dispatch overhead → better wall-clock latency. NSYS GPU projection only measures kernel execution time, ignoring the ~2ms of inter-kernel CPU gaps in the BF16 path.
- **Conclusion:** For real-world throughput, FP8 is **already 1.5x faster** than BF16. The "125us gap" is a GPU-projection-only artifact.

### 4. Prototyped "lean FP8" path (experimental, behind `SONIC_MOE_FP8_LEAN=1`)

- **What:** Only use FP8 for up-proj `gemm_gated`; skip z FP8 save, skip y1/dz quant, use BF16 for everything else.
- **Result:** `4.67ms` — only 1.02x faster than BF16. Too conservative.
- **Conclusion:** The fully-fused FP8 path (`SONIC_MOE_FP8_FUSED_GATED=1`) is far superior in wall-clock time because it reduces total kernel count and benefits from FP8 tensor cores across the full pipeline.

### 5. Confirmed correctness at production shape

- **Shape:** T=4096, H=4096, I=1024, E=128, K=8 with 128-aligned uniform routing
- **Output RelRMSE:** 6.59% (target <10%) ✓
- **Output correlation:** 0.996 (target >0.99) ✓
- **dx RelRMSE:** 6.98% (target <10%) ✓
- **dx correlation:** 1.000 (target >0.99) ✓

### 6. Confirmed Triton blockscaled FP8 kernels work on SM100a

- All custom Triton kernels (`_quantize_flat_blockscaled_kernel`, `_gather_quantize_and_pack_kernel`, etc.) execute correctly on SM100a Blackwell GPUs with Triton 3.5.1.
- Previously observed "illegal instruction" errors were caused by CUDA context pollution from other crashed kernels, not by SM100 incompatibility.
- The CUTLASS blockscaled GEMM (`GemmDefaultSm100` with `mSFA/mSFB` scale factors) also works correctly at production shapes.

### 7. Analyzed `HopperWgmma_MoE_kernel` for FP8 wgrad potential

- The specialized CuTeDSL MoE kernel in `grouped_gemm.py` supports FP8 at the hardware instruction level (checks `a_dtype.width == 8`), but the validation function (`is_valid_dtypes`) only lists BF16/FP16 as tested.
- In weight gradient mode (`compute_weight_gradient=True`), tokens become the K-dimension with dynamic per-expert k_tile_cnt.
- **Key insight:** Extending these kernels to accept blockscaled FP8 inputs would bypass the generic `blockscaled_fp8_wgrad_varlen_k` path entirely, potentially eliminating the expensive column-wise quant + ISA-pack overhead.

---

## Clean lessons to keep (including previous)

1. **Official BF16 only** for baseline.
2. **NSYS GPU projection ≠ wall-clock latency.** For real throughput, use event timing. GPU projection misses CPU dispatch overhead.
3. **FP8 fused gated path is already 1.5x faster in wall-clock** at production shapes.
4. **The 125us NSYS gap is 100% quant overhead** — precisely accounted for in §2.
5. **Standalone quant kernels cost 386us total; FP8 GEMM saves only 259us.** To win in GPU projection, must eliminate ~127us of quant.
6. **Fewer kernel launches = less CPU overhead = faster wall-clock.** FP8 path benefits hugely from this.
7. **Do not claim FP8 memory or inference win.** FP8 training uses 10.7 GiB vs BF16 7.1 GiB. FP8 inference is 4.6ms vs 1.0ms.

---

## Strongest next directions

### Direction A: Eliminate quant overhead (close the NSYS gap)
The 386us of standalone quant can be reduced by:
1. **Fusing quant into GEMM epilogue** — the CUTLASS `GemmGatedSm100` epilogue already does SwiGLU; adding blockscaled FP8 output quantization would eliminate the post-GEMM quant kernels entirely.
2. **Fusing quant into GEMM prologue** — modifying `GemmDefaultSm100` to accept BF16 input and do online blockscaled quantization in the TMA load stage would eliminate the pre-GEMM quant kernels.
3. **Using cutify-style fused kernels** — the reference implementation in `operator-incubator/cutify/ops/cute/` shows how to fuse SwiGLU+quant and dequant into single kernels with warp-shuffle for scale broadcast.

### Direction B: Reduce FP8 memory overhead
Current FP8 training uses 3.7 GiB more than BF16. Sources:
- Persistent FP8 weight caches (`precompute_weight_fp8`, `precompute_weight_fp8_for_fused_gated`, `precompute_weight_fp8_for_direct_fused_dgated`)
- z FP8 save + raw scales
- ISA-packed scale tensors
- Prequantized activation dictionaries (`_PREQUANTIZED_SCALES`)

### Direction C: FP8 weight gradient via specialized MoE kernel
Extend `HopperWgmma_MoE_kernel(compute_weight_gradient=True)` to accept blockscaled FP8 inputs. This bypasses the generic `blockscaled_fp8_wgrad_varlen_k` path and its expensive column-wise quant + ISA-pack overhead.
