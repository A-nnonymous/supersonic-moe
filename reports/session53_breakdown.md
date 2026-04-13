# Session 53 — Performance & Memory Breakdown (Final)

> Date: 2026-04-13
> Method: nsys GPU-projection, 20 measured iters, 5 warmup
> FP8 frontier = stash (E≤8) or no-stash+token-rounding (E>8), wgrad ON, dual quant
> BF16 baseline = official SonicMoE (3767 µs at T=8192 I=1536 E=8)
> nsys-rep files: `panzhaowu/output/nsys/`

## Performance Summary (T=8192, H=3072, I=1536, K=8)

| E | BF16 (µs) | FP8 (µs) | Speedup | Row Quant | Net | BF16 Bwd | FP8 Bwd | MemΔ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 4050 | 2739 | **1.48×** | 76µs | -1312µs | 1460M | 1547M | +6.0% |
| 32 | 4646 | 3008 | **1.54×** | 79µs | -1638µs | 2721M | 3624M | +33.2% |
| 128 | 7906 | 4192 | **1.89×** | 86µs | -3715µs | 7986M | 11562M | +44.8% |

vs Official BF16 baseline (T=8192 I=1536 E=8): 3767µs → FP8 2739µs = **1.38×**

## Critical Fixes (Session 53)

1. **VARLEN weight cache preservation** (line 1629): stop clearing `_VARLEN_WEIGHT_CACHE` at backward
2. **FUSED weight cache preservation** (line 1629/2249): stop clearing `_FUSED_WEIGHT_CACHE`
3. **Cache corruption fix** (line 1741): stop `ctx._w2_dgated_fp8.resize_(0)` which freed tensors aliased in cache
4. **Wgrad threshold=0**: FP8 wgrad profitable at all I values after cache fixes
5. **Non-aligned error**: FP8 raises RuntimeError for non-128-aligned expert segments; callers must use token rounding

Cost: +148 MiB backward peak at E=8 (FP8 weight caches retained vs freed).

## E Scaling Analysis

| Metric | E=8 → E=128 | Explanation |
|---|---|---|
| GEMM savings | -1312 → -3715 µs | Scales well (more expert batches = more FP8 GEMM wins) |
| Row Quant | 76 → 86 µs | Nearly constant (weight cache eliminates re-quant) |
| Memory overhead | +6% → +45% | FP8 shadow weights O(E): 128×(w1_fp8+w2_fp8+scales) |

Key insight: **after cache fix, FP8 overhead is nearly constant while GEMM savings scale with E** → larger E = larger speedup.

## Non-Aligned Routing

FP8 requires 128-aligned per-expert token counts (SM100 ISA scale tile constraint).
For E>8, use official token rounding (`forward_token_choice_rounding` with Mtile=128).
Non-aligned FP8 raises `RuntimeError`.
