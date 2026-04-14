# Session 53 — Performance, Memory & Precision Breakdown

> Branch: `native-fp8-exploration`, commit `5dcb62b`
> Method: nsys GPU-projection, 12 iters after 5 warmup, H=3072, I=1536, K=8
> BF16 baseline: our branch `moe_TC_softmax_topk_layer` (verified <1% of official)
> FP8 frontier: stash (E≤8) + token rounding (E>8) + wgrad ON + dual quant
> Precision: 5 seeds (42,123,777,999,2024), FP8 vs BF16 on identical routing

## Performance + Memory + Precision Summary

| T | E | BF16 µs | FP8 µs | Speedup | BF16 Bwd MiB | FP8 Bwd MiB | MemΔ | out% | dx% | dw1% | dw2% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8192 | 8 | 3524 | 2786 | **1.27×** | 1459 | 1547 | +6% | 6.52 | 6.53 | 4.71 | 4.92 |
| 8192 | 32 | 3774 | 3007 | **1.26×** | 2706 | 3624 | +34% | 6.52 | 6.51 | 5.47 | 5.88 |
| 8192 | 128 | 4924 | 4060 | **1.21×** | 7890 | 11562 | +47% | 6.52 | 6.52 | 6.01 | 6.50 |
| 32768 | 8 | 15900 | 11512 | **1.38×** | 4922 | 5359 | +9% | 6.55 | 6.56 | 4.12 | 4.19 |
| 32768 | 32 | 16893 | 11564 | **1.46×** | 5786 | 6891 | +19% | 6.55 | 6.54 | 4.60 | 4.83 |
| 32768 | 128 | 18478 | 12899 | **1.43×** | 10774 | 14532 | +35% | 6.55 | 6.55 | 5.40 | 5.81 |

## Budget Breakdown: T=8192

### E=8 (BF16=3524µs, FP8=2786µs, 1.27×, NET=-739µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 2078 | 1148 | **-930** | GEMM |
| GemmGated (fwd) | 700 | → ZeroMat 451 | **-249** | GEMM |
| GemmDGated (bwd) | 439 | → ZeroMat 391 | **-48** | GEMM |
| Row Quant | 0 | 77 | +77 | quant |
| Dual Quant | 0 | 153 | +153 | quant |
| Blockscaled Quant | 0 | 235 | +235 | quant |
| ISA Scale Gather | 0 | 16 | +16 | quant |

### E=32 (BF16=3774µs, FP8=3007µs, 1.26×, NET=-772µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 2213 | 1289 | **-924** | GEMM |
| GemmGated (fwd) | 779 | → ZeroMat 495 | **-284** | GEMM |
| GemmDGated (bwd) | 479 | → ZeroMat 428 | **-51** | GEMM |
| Row Quant | 0 | 79 | +79 | quant |
| Dual Quant | 0 | 156 | +156 | quant |
| Blockscaled Quant | 0 | 241 | +241 | quant |

### E=128 (BF16=4924µs, FP8=4060µs, 1.21×, NET=-868µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 2928 | 1969 | **-959** | GEMM |
| GemmGated (fwd) | 1072 | → ZeroMat 704 | **-368** | GEMM |
| GemmDGated (bwd) | 617 | → ZeroMat 539 | **-78** | GEMM |
| Row Quant | 0 | 83 | +83 | quant |
| Dual Quant | 0 | 172 | +172 | quant |
| Blockscaled Quant | 0 | 254 | +254 | quant |

## Budget Breakdown: T=32768

### E=8 (BF16=15900µs, FP8=11512µs, 1.38×, NET=-4390µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 9757 | 4844 | **-4913** | GEMM |
| GemmGated (fwd) | 3138 | → ZeroMat 1999 | **-1139** | GEMM |
| GemmDGated (bwd) | 1936 | → ZeroMat 1649 | **-287** | GEMM |
| Row Quant | 0 | 324 | +324 | quant |
| Dual Quant | 0 | 682 | +682 | quant |
| Blockscaled Quant | 0 | 932 | +932 | quant |

### E=32 (BF16=16893µs, FP8=11564µs, 1.46×, NET=-5334µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 10333 | 5019 | **-5314** | GEMM |
| GemmGated (fwd) | 3424 | → ZeroMat 2040 | **-1384** | GEMM |
| GemmDGated (bwd) | 2059 | → ZeroMat 1646 | **-413** | GEMM |
| Row Quant | 0 | 326 | +326 | quant |
| Dual Quant | 0 | 666 | +666 | quant |
| Blockscaled Quant | 0 | 942 | +942 | quant |

### E=128 (BF16=18478µs, FP8=12899µs, 1.43×, NET=-5585µs)

| Category | BF16 µs | FP8 µs | Delta | Type |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 11444 | 5808 | **-5636** | GEMM |
| GemmGated (fwd) | 3736 | → ZeroMat 2279 | **-1457** | GEMM |
| GemmDGated (bwd) | 2200 | → ZeroMat 1823 | **-377** | GEMM |
| Row Quant | 0 | 336 | +336 | quant |
| Dual Quant | 0 | 708 | +708 | quant |
| Blockscaled Quant | 0 | 988 | +988 | quant |

## Memory Analysis: FP8-Native Training (Custom Backward)

### Current Memory Model

| Component | E=8 | E=32 | E=128 | Scales with |
|---|:---:|:---:|:---:|---|
| BF16 master weights | 216 MiB | 864 MiB | 3456 MiB | O(E) |
| FP8 shadow weights | 111 MiB | 446 MiB | 1782 MiB | O(E) |
| Optimizer states (Adam, fp32) | ~430 MiB | ~1728 MiB | ~6912 MiB | O(E) |
| **Total param overhead** | **757 MiB** | **3038 MiB** | **12150 MiB** | |

### What Custom Backward Could Save

**Approach**: Replace `torch.autograd.Function` with a manual backward
implementation that computes gradients directly in FP8, then applies
optimizer updates to FP8 weights without maintaining bf16 master copies.

| Savings source | E=8 | E=32 | E=128 | Feasibility |
|---|:---:|:---:|:---:|---|
| Eliminate bf16 master weights | 216 MiB | 864 MiB | **3456 MiB** | ★★★ — straightforward |
| Eliminate bf16 optimizer momentum | ~430 MiB | ~1728 MiB | **6912 MiB** | ★★ — needs fp8 Adam |
| Autograd graph overhead | ~20 MiB | ~50 MiB | ~120 MiB | ★★★ — minor |
| **Total potential savings** | **~666 MiB** | **~2642 MiB** | **~10488 MiB** | |

### FP8-Native Training Design

```
Current:  bf16_param → fp8_shadow_cache → FP8 forward → autograd → bf16 grad → Adam(bf16) → bf16_param
Proposed: fp8_param → FP8 forward → manual backward → fp8 grad → FP8-Adam(fp8) → fp8_param
```

**Key challenges:**
1. **Optimizer precision**: Adam momentum/variance need higher precision than E4M3.
   Solution: use FP8E5M2 or BF16 for optimizer states (still 2× smaller than FP32).
2. **Loss scaling**: FP8 forward has limited dynamic range. Need per-tensor loss
   scaling to avoid underflow in backward.
3. **Weight update**: `w_fp8 += lr * fp8_grad` loses precision at small learning rates.
   Solution: stochastic rounding or Kahan summation in fp8.
4. **Gradient accumulation**: accumulating fp8 gradients across micro-batches
   causes catastrophic cancellation. Need fp32 or bf16 accumulator.

**Practical implementation path:**
1. **Phase 1** (easy, ~2000 MiB savings at E=128): Eliminate bf16 master weights by
   using `torch.autograd.Function` with stash that never unstashes. Optimizer runs
   on a separate bf16 copy that's only materialized during optimizer.step().
2. **Phase 2** (moderate): Replace Adam with FP8-aware optimizer (e.g., Galore-style
   low-rank update in FP8 space, or 8-bit Adam from bitsandbytes).
3. **Phase 3** (hard): Full manual backward without autograd. Requires implementing
   `_UpProjection.backward` and `_DownProjection.backward` as standalone functions
   called from a custom training loop.

**Net assessment**: Custom backward yields ~120 MiB from autograd graph elimination
(minor). The big wins come from eliminating bf16 master weights (3.5 GB at E=128)
and fp32 optimizer states (6.9 GB at E=128) — but these require optimizer changes,
not backward changes. **Recommendation: pursue FP8-aware optimizer (Phase 2) rather
than custom backward (Phase 3).**
