# Session 53 — Performance, Memory & Precision Breakdown (Final)

> Branch: `native-fp8-exploration`
> Method: nsys GPU-projection, 12 iters after 3 warmup, H=3072, K=8
> BF16 baseline: our branch `moe_TC_softmax_topk_layer` (verified <1% of official)
> FP8 frontier: stash + all weight caches retained (max performance)
> Precision: 3 seeds (42,123,777), FP8 vs BF16 on identical routing

## Full 27-Shape Grid (3T × 3E × 3I)

| T | I | E | BF16 µs | FP8 µs | Speedup | BF16 Bwd | FP8 Bwd | MemΔ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 8192 | 1536 | 8 | 3644 | 2715 | **1.34×** | 1459 | 1547 | +6.0% |
| 8192 | 2048 | 8 | 4958 | 3387 | **1.46×** | 1874 | 1992 | +6.3% |
| 8192 | 3072 | 8 | 8110 | 4774 | **1.70×** | 2707 | 2884 | +6.5% |
| 8192 | 1536 | 32 | 3844 | 2922 | **1.32×** | 2706 | 2909 | +7.5% |
| 8192 | 2048 | 32 | 5263 | 3709 | **1.42×** | 3409 | 3678 | +7.9% |
| 8192 | 3072 | 32 | 8124 | 5318 | **1.53×** | 4818 | 5218 | +8.3% |
| 8192 | 1536 | 128 | 5009 | 3897 | **1.29×** | 7890 | 8700 | +10.3% |
| 8192 | 2048 | 128 | 6967 | 4995 | **1.39×** | 10323 | 11385 | +10.3% |
| 8192 | 3072 | 128 | 10839 | 7267 | **1.49×** | 15194 | 16756 | +10.3% |
| 16384 | 1536 | 8 | 7953 | 5227 | **1.52×** | 2613 | 2819 | +7.9% |
| 16384 | 2048 | 8 | 10832 | 6765 | **1.60×** | 3347 | 3622 | +8.2% |
| 16384 | 3072 | 8 | 16172 | 10065 | **1.61×** | 4821 | 5232 | +8.5% |
| 16384 | 1536 | 32 | 8129 | 5432 | **1.50×** | 3668 | 3891 | +6.1% |
| 16384 | 2048 | 32 | 10860 | 7039 | **1.54×** | 4502 | 4794 | +6.5% |
| 16384 | 3072 | 32 | 16558 | 10166 | **1.63×** | 6551 | 6863 | +4.8% |
| 16384 | 1536 | 128 | 9099 | 6360 | **1.43×** | 8856 | 9688 | +9.4% |
| 16384 | 2048 | 128 | 12348 | 8198 | **1.51×** | 11404 | 12506 | +9.6% |
| 16384 | 3072 | 128 | 19216 | 11862 | **1.62×** | 16443 | 18142 | +10.3% |
| 32768 | 1536 | 8 | 16287 | 10652 | **1.53×** | 4922 | 5359 | +8.9% |
| 32768 | 2048 | 8 | 21753 | 13587 | **1.60×** | 6300 | 6882 | +9.2% |
| 32768 | 3072 | 8 | 33278 | 20010 | **1.66×** | 9050 | 9927 | +9.7% |
| 32768 | 1536 | 32 | 16829 | 10753 | **1.56×** | 5786 | 6176 | +6.7% |
| 32768 | 2048 | 32 | 22584 | 13967 | **1.62×** | 7453 | 7965 | +6.9% |
| 32768 | 3072 | 32 | 33504 | 19761 | **1.70×** | 10798 | 11549 | +7.0% |
| 32768 | 1536 | 128 | 17635 | 11509 | **1.53×** | 10774 | 11669 | +8.3% |
| 32768 | 2048 | 128 | 23312 | 14956 | **1.56×** | 13585 | 14751 | +8.6% |
| 32768 | 3072 | 128 | 35627 | 22026 | **1.62×** | 19222 | 20919 | +8.8% |

**Speedup range: 1.29× – 1.70×, mean 1.53×.**

## Scaling Rules

- **T scaling**: larger T → higher speedup (1.34× at T=8k → 1.53× at T=32k for E=8,I=1536)
- **I scaling**: larger I → higher speedup (1.34× at I=1536 → 1.70× at I=3072 for T=8k,E=8)
- **E scaling**: minimal impact at fixed T×I (E=8 vs E=128 differ by <0.15×)
- **Memory**: FP8 uses 5-10% more peak backward memory (FP8 shadow weight caches)
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
