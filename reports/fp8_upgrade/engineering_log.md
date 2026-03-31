# FP8 Engineering Log

> This log keeps only the conclusions that still survive revalidation.
> Historical dead ends remain in git history; this file is the cleaned, high-signal timeline for the next agent.

## Session 21 / 2026-03-31 — Clean frontier established

### 1. Re-baselined against the real official BF16 reference

- **Source of truth:** `reports/sonic_official_bf16.sqlite`
- **Tool:** `tools/nsys_full_breakdown.py`
- **Conclusion:** all meaningful performance claims must compare against official BF16, not fork BF16.

### 2. Proved fused blockscaled gated / dgated are viable in practice

- **Files touched:**
  - `sonicmoe/quack_utils/gemm_interface.py`
  - `sonicmoe/functional/__init__.py`
- **What changed:** blockscaled fused gated/dgated now bypass the unsafe autotune path.
- **Conclusion:** the fused path itself is viable; the old problem was wrapper/runtime selection, not the fused math.

### 3. Removed the giant fused backward wrapper copy

- **Files touched:**
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - `sonicmoe/functional/__init__.py`
- **What changed:** added direct fused-dgated weight cache and called low-level `gemm_dgated` directly.
- **Why it mattered:** the wrapper-side `B.mT.contiguous()` copy was the real reason fused backward looked broken.

### 4. Simplified the fused path by deleting low-value standalone prequant work

- **File touched:** `sonicmoe/functional/__init__.py`
- **What changed:** removed explicit standalone `y1` and `dz` prequant from the fused path.
- **Conclusion:** those kernels were not the decisive training bottleneck anymore, and keeping them complicated the timeline more than they helped.

### 5. Fixed the FP8 wgrad autograd-layout copy

- **Files touched:**
  - `sonicmoe/functional/__init__.py`
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- **What changed:** FP8 wgrad now writes into base-layout buffers and returns parameter-compatible grad views.
- **Evidence:** local dense benchmark improved from `16.225ms` to `6.810ms`, and the profiler no longer showed the `[128, 2048, 4096]` giant `aten::copy_`.
- **Conclusion:** the remaining full-chain FP8 wgrad problem is no longer autograd bookkeeping.

### 6. Re-measured the real frontier after the copy fix

- **Authoritative training NSYS:**
  - official BF16: `2475.2us`
  - current fused FP8 + BF16 wgrad: `2600.3us`
  - current fused FP8 + FP8 wgrad: `5650.4us`
- **Conclusion:** the project frontier moved.
  - aligned fused FP8 act-grad is now near baseline
  - FP8 wgrad is the blocker

### 7. Local peak-memory / inference check invalidated another old narrative

- **Tool:** `tools/measure_aligned_perf_memory.py`
- **Observed local aligned numbers (latest verified run; event timings can drift on the shared machine):**
  - training BF16: `4.894ms`, `7.051 GiB`
  - training FP8 + BF16 wgrad: `3.136ms`, `10.746 GiB`
  - training FP8 + FP8 wgrad: `3.325ms`, `10.808 GiB`
  - inference BF16: `1.019ms`, `7.526 GiB`
  - inference FP8: `4.638ms`, `9.760 GiB`
- **Conclusion:** current FP8 does **not** yet win on local peak memory or inference; use NSYS, not local event timings, for authoritative training comparisons.

---

## Clean lessons to keep

1. **Official BF16 only.**
2. **NSYS with NVTX GPU projection + sync barriers beats event-timing narratives.**
3. **A profiler-verified hidden copy can outweigh weeks of kernel tweaking.**
4. **The current training frontier is FP8 wgrad, not forward SwiGLU.**
5. **Do not claim an FP8 memory or inference win on the current branch.**

---

## Strongest next direction

The most promising next step is **not** another small varlen_k wrapper tweak.

The strongest concrete direction is to adapt / extend the specialized weight-grad kernels already present in-tree:

- `sonicmoe/functional/backward.py`
- `sonicmoe/functional/moe_config.py`
- `sonicmoe/functional/grouped_gemm.py`

Look specifically at:

- `HopperWgmma_MoE_Up_proj_WeightGrad_Bwd`
- `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd`
- `HopperWgmma_MoE_kernel(..., compute_weight_gradient=True)`

That is the cleanest frontier left for the next agent.
