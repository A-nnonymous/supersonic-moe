# Session 53 — Performance, Memory & Precision Breakdown (Final)

> Branch: `native-fp8-exploration`, commit `e5a28ce`
> Method: nsys GPU-projection, 12 iters after 3 warmup, H=3072, I=1536, K=8
> BF16 baseline: our branch `moe_TC_softmax_topk_layer` (verified <1% of official)
> FP8 frontier: stash + all weight caches retained (max performance)
> Precision: 3 seeds (42,123,777), FP8 vs BF16 on identical routing

## Performance + Memory + Precision

| T | E | BF16 µs | FP8 µs | Speedup | BF16 Bwd | FP8 Bwd | MemΔ | out% | dx% | dw1% | dw2% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8192 | 8 | 3600 | 2698 | **1.33×** | 1459 | 1547 | +6.0% | 6.52 | 6.53 | 4.71 | 4.90 |
| 8192 | 32 | 3735 | 2917 | **1.28×** | 2706 | 2909 | +7.5% | 6.52 | 6.51 | 5.47 | 5.88 |
| 8192 | 128 | 5056 | 3909 | **1.29×** | 7890 | 8700 | +10.3% | 6.52 | 6.52 | 6.01 | 6.50 |
| 32768 | 8 | 15811 | 10518 | **1.50×** | 4922 | 5359 | +8.9% | 6.55 | 6.55 | 4.12 | 4.20 |
| 32768 | 32 | 16718 | 10530 | **1.59×** | 5786 | 6176 | +6.7% | 6.55 | 6.54 | 4.60 | 4.84 |
| 32768 | 128 | 18382 | 11798 | **1.56×** | 10774 | 11669 | +8.3% | 6.55 | 6.55 | 5.40 | 5.81 |

## Scaling Rules

- **T scaling**: larger T → higher speedup (1.33× at T=8k → 1.50× at T=32k for E=8)
- **E scaling**: E=32 slightly better than E=8 at large T (1.59× vs 1.50× at T=32k)
- **Memory**: FP8 uses 6-10% more peak backward memory (FP8 shadow weight caches)
- **Precision**: output/dx RRMSE ~6.5% across all shapes; dw1/dw2 scales with E (4.7% at E=8 → 6.5% at E=128)

## Budget Breakdown: T=8192 E=8 (representative)

| Category | BF16 µs | FP8 µs | Delta |
|---|:---:|:---:|:---:|
| Wgrad GEMM | 2078 | 1148 | **-930** |
| GemmGated (fwd) | 700 | → ZeroMat 451 | **-249** |
| GemmDGated (bwd) | 439 | → ZeroMat 391 | **-48** |
| Row Quant | 0 | 77 | +77 |
| Dual Quant | 0 | 153 | +153 |
| Blockscaled Quant | 0 | 235 | +235 |
| ISA Scale Gather | 0 | 16 | +16 |
| **NET** | **3600** | **2698** | **-902** |

## Optional Memory Optimization

For memory-constrained scenarios, CPU optimizer offload is available:

```python
moe.setup_cpu_optimizer(torch.optim.Adam, lr=1e-3)
# Saves ~3.4 GB base at E=128 (bf16 weights → CPU)
# Costs ~500µs/iter from CPU↔GPU transfer + weight re-quantization
```

| Mode | E=128 Base | E=128 Bwd Peak |
|---|:---:|:---:|
| Default (max perf) | 7090 MiB | 8700 MiB |
| CPU optimizer | 3634 MiB | 8106 MiB |
| Savings | **3456 MiB (49%)** | **594 MiB (7%)** |
