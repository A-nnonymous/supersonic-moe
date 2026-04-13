# SonicMoE Session 52+ — Performance / Memory / Precision Report

> Date: 2026-04-13
> GPU: NVIDIA B30Z (Blackwell), idle (util=0%)
> Shape: T=8192, H=3072, I=1536, E=8, K=8 (Ernie config)
> Branch: `native-fp8-exploration`
> quack-kernels: 0.3.7, torch: 2.11.0+cu130
> Methodology: nsys GPU-projection (gold standard), repeated measurement, CUDA events

---

## 1. Performance Breakdown

### 1.1 E2E GPU-Projection Time (nsys, merged kernel intervals)

| Mode | GPU-proj µs/iter | Kernel Launches | Speedup |
|------|:---:|:---:|:---:|
| **BF16** | 3722.5 | 504 | baseline |
| **FP8** | 3642.2 | 592 | **1.022×** |

Repeated measurement (12 iters): BF16=3773.8, FP8=3636.1, speedup=**1.038×**. CV < 1.4%.

### 1.2 Category-Level Breakdown

| Category | BF16 (µs) | FP8 (µs) | Delta | % of FP8 |
|---|:---:|:---:|:---:|:---:|
| **Wgrad GEMM** | 2060.8 | 1625.9 | **-434.9** | 44.6% |
| GemmGated (fwd) | 687.1 | 449.3 (ZeroMat) | **-237.8** | 12.3% |
| GemmDGated (bwd) | 436.0 | 392.6 (ZeroMat) | **-43.4** | 10.8% |
| Row Quant | — | 312.9 | +312.9 | 8.6% |
| Other | 390.2 | 691.7 | +301.5 | 19.0% |
| Token Gather | 140.3 | 145.3 | +5.0 | 4.0% |
| ISA Scale Gather | — | 16.3 | +16.3 | 0.4% |
| Softmax | 9.7 | 9.7 | 0.0 | 0.3% |
| TopK Router | 2.4 | 2.4 | 0.0 | 0.1% |
| **TOTAL** | **3722.5** | **3642.2** | **-80.3** | 100% |

### 1.3 FP8 Budget Equation

```
Savings:
  Wgrad GEMM:     -434.9 µs  (BF16 varlen → FP8 blockscaled, 1.27×)
  GemmGated fwd:  -237.8 µs  (BF16 → ZeroMat FP8, 1.53×)
  GemmDGated bwd:  -43.4 µs  (BF16 → FP8 C-load TMA, 1.11×)
  ─────────────────────────────
  Total savings:  -716.1 µs

Overhead:
  Row Quant:      +312.9 µs  (activation quantize + ISA pack)
  Other delta:    +301.5 µs  (elementwise bf16↔fp8 casts, padding, z-save/restore)
  ISA Gather:      +16.3 µs
  Token delta:      +5.0 µs
  ─────────────────────────────
  Total overhead: +635.7 µs

NET: -80.4 µs → FP8 wins by 2.2%
```

### 1.4 Top 10 FP8 Kernels (per-iter)

| # | Kernel | µs/iter | Calls | µs/call | Category |
|---|--------|:---:|:---:|:---:|---|
| 1 | GemmDefaultSm100 (BF16 wgrad, up-proj) | 1020.2 | 16 | 510.1 | Wgrad GEMM |
| 2 | GemmDefaultSm100 (FP8 wgrad, down-proj) | 605.7 | 16 | 302.9 | Wgrad GEMM |
| 3 | elementwise_kernel (bf16↔fp8 casts) | 479.2 | 24 | 159.7 | Other |
| 4 | GemmGatedSm100ZeroMatBlockscaled (fwd) | 449.3 | 8 | 449.3 | GemmGated ZeroMat |
| 5 | GemmDGatedFP8CLoadSm100ZeroMat (bwd) | 392.6 | 8 | 392.6 | GemmDGated ZeroMat |
| 6 | _quantize_and_pack_kernel | 183.1 | 32 | 45.8 | Row Quant |
| 7 | token_gather_sum_kernel | 145.3 | 16 | 72.6 | Token Gather |
| 8 | _quantize_and_pack_iso32_kernel | 129.8 | 32 | 32.4 | Row Quant |
| 9 | vectorized_elementwise_kernel | 22.1 | 16 | 11.1 | Other |
| 10 | reduce_kernel (db) | 17.4 | 8 | 17.4 | Other |

### 1.5 Theoretical Efficiency Analysis

```
Ernie shape: T=8192, H=3072, I=1536, E=8, K=8, TK=65536

Total wgrad FLOPs:
  UpProj:   2 × 3072 × 3072 × 65536 = 1237.0 GFLOP  (dz.T @ x)
  DownProj: 2 × 1536 × 3072 × 65536 =  618.5 GFLOP  (dout.T @ y1s)
  Total:                                1855.4 GFLOP

Theoretical minimum (compute-bound):
  BF16 @ 2250 TFLOPS peak:  824.6 µs
  FP8  @ 4500 TFLOPS peak:  412.3 µs

Actual:
  BF16 wgrad: 2060.8 µs → 39.5% of peak  (memory-bandwidth limited)
  FP8  wgrad: 1625.9 µs → 25.4% of peak  (quant overhead + bandwidth)

Wgrad is deeply memory-bandwidth limited at Ernie shape:
  Arithmetic intensity = 1855.4 GFLOP / (TK*(H+2I+I)*2 bytes) ≈ 42 FLOP/byte
  B30Z balance point ≈ 562 FLOP/byte (4500 TFLOPS / 8000 GB/s)
  Ratio = 42/562 = 7.5% → firmly in bandwidth-limited regime
  (varlen GEMM per-expert M=8192 is too small to fill SM100 tiles efficiently)
```

**Key insight**: At Ernie shape (I=1536), each expert segment has only TK/E=8192 tokens. Wgrad GEMMs are (3072×3072, K=8192) and (1536×3072, K=8192) — too small to saturate Blackwell tensor cores. This is why the FP8 compute speedup (theoretical 2×) only manifests as 1.27×: the bottleneck is HBM bandwidth and tile scheduling overhead, not raw FLOP throughput.

---

## 2. Memory Breakdown

### 2.1 Peak Memory (torch.cuda.max_memory_allocated)

| Checkpoint | BF16 (MiB) | FP8 (MiB) | Delta | Saving |
|---|:---:|:---:|:---:|:---:|
| Baseline (model + caches) | 232.3 | 232.3 | 0.0 | — |
| Pre-Forward | 280.3 | 280.3 | 0.0 | — |
| **Peak Forward** | 1289.7 | 1231.2 | **-58.5** | **4.5%** |
| Pre-Backward | 713.8 | 676.3 | -37.5 | 5.3% |
| **Peak Backward** | 1411.7 | 1305.2 | **-106.5** | **7.5%** |

### 2.2 Activation Delta Analysis

| Metric | BF16 (MiB) | FP8 (MiB) | Delta |
|---|:---:|:---:|:---:|
| Fwd peak above pre-fwd | 1009.4 | 950.9 | **-58.5** |
| Bwd peak above pre-bwd | 697.9 | 628.9 | **-69.0** |

### 2.3 Theoretical Memory Savings

```
FP8 saves memory via:

1. z tensor compression (forward save → backward restore):
   z_bf16: TK × 2I × 2B = 65536 × 3072 × 2 = 384.0 MiB
   z_fp8:  TK × 2I × 1B + scales           ≈ 198.0 MiB
   Saving: 186.0 MiB (48.4%)

2. Early dz_bf16 freeing in wgrad path:
   dz_bf16: TK × 2I × 2B = 384.0 MiB
   Freed before wgrad GEMM (FP8 dz pre-computed from dual quant)

3. FP8 weight caches (persistent, amortized):
   w1_fp8: 2I × H × E × 1B + scales ≈ 37.5 MiB
   w2_fp8: I × H × E × 1B + scales  ≈ 18.8 MiB

Offsetting costs:
   Row+col quant temporaries: ~96-192 MiB (transient, reused by allocator)
   FP8 weight shadow caches: ~56 MiB (persistent)
```

### 2.4 Cross-Validation

Forward saving (-58.5 MiB) < theoretical z compression (186 MiB) because:
- FP8 path creates additional temporaries (quant scales, ISA-packed scales)
- Weight FP8 caches are persistent and counted in pre-fwd baseline
- Some quant temporaries overlap with the forward peak

Backward saving (-106.5 MiB) is driven by early dz_bf16 freeing. The allocator
reclaims 384 MiB of dz_bf16 before wgrad GEMM allocates dw1/dw2 outputs,
reducing the backward memory envelope.

---

## 3. Precision Audit

### 3.1 RRMSE (Relative Root Mean Square Error, FP8 vs BF16 reference)

| Tensor | RRMSE (%) | Std (%) | Cosine Sim | Max Abs Err | Status |
|--------|:---------:|:-------:|:----------:|:-----------:|:------:|
| **output** | 6.5185 | 0.0014 | 0.997879 | 2.5e-05 | PASS |
| **dx** | 6.5302 | 0.0014 | 0.997870 | 1.8e-03 | PASS |
| **dw1** | 4.2673 | 0.0010 | 0.999092 | 4.6e-03 | PASS |
| **dw2** | 4.7148 | 0.0435 | 0.998891 | 9.5e-04 | PASS |

Measured over 5 seeds (42, 123, 777, 999, 2024). All within guardrails:
- **RRMSE < 10%** (max observed: 6.53%)
- **Cosine similarity > 0.99** (min observed: 0.9979)

### 3.2 Theoretical Error Bound Analysis

```
FP8 E4M3 has 3 mantissa bits → 1 ULP = 2^(-3) = 12.5% relative error per value.
Blockscaled (1×32) amortizes a shared scale across 32 elements, adding at most
1 extra bit of error from scale quantization (E8M0 → integer exponent).

Expected RRMSE for random data with independent FP8 quantization errors:
  Per-element relative error ε ~ Uniform(-6.25%, +6.25%)
  RRMSE ≈ sqrt(E[ε²]) = 6.25% / sqrt(3) ≈ 3.6%

Observed RRMSE (6.5%) is higher than single-step theory (3.6%) because:
1. Error accumulates through 2 GEMM passes (fwd: x→z, z→y1→output)
2. SwiGLU activation amplifies errors near zero (sigmoid derivative)
3. Backward pass compounds forward errors via chain rule

The output/dx errors (~6.5%) are higher than dw1/dw2 (~4.3-4.7%) because:
  - Output/dx pass through the full forward+backward chain
  - Weight gradients are averaged over TK=65536 tokens (sqrt(N) noise reduction)
  - Theory: dw RRMSE ≈ single_error / sqrt(TK/E) ≈ 3.6% / sqrt(8192) ≈ 0.04%
    but actual is 4.3% because wgrad uses FP8 quantized inputs, not exact values

Cross-seed std < 0.05% confirms measurement stability.
```

---

## 4. FP8 vs BF16 Trade-off Summary

### Honest Assessment (I=1536, Ernie shape)

| Dimension | Winner | Magnitude | Confidence |
|---|---|---|---|
| **E2E Latency** | FP8 | 2.2-3.8% faster | High (nsys, 2 runs, CV<1.4%) |
| **Peak Forward Memory** | FP8 | 58.5 MiB less (4.5%) | High (torch.cuda, deterministic) |
| **Peak Backward Memory** | FP8 | 106.5 MiB less (7.5%) | High |
| **Forward Precision** | BF16 | 6.5% RRMSE | High (5 seeds, std<0.01%) |
| **Gradient Precision** | BF16 | 4.3-6.5% RRMSE | High (5 seeds, std<0.05%) |

### Where FP8 Excels vs Struggles

**Excels at:**
- Wgrad GEMM compute (1.27× speedup despite bandwidth limitation)
- Forward GemmGated ZeroMat (1.53× via FP8 activations, no materialization)
- Memory pressure reduction (enables larger batch sizes)

**Struggles with:**
- Quant overhead at small I (I=1536 is borderline — 636µs overhead vs 716µs savings)
- elementwise bf16↔fp8 cast kernels (479µs, could be fused)
- Wgrad at small per-expert M (8192 tokens/expert underutilizes SM100 tiles)

### Current Bottleneck Ranking

| # | Bottleneck | µs/iter | Feasible Fix | Est. Gain |
|---|---|:---:|---|:---:|
| 1 | Wgrad GEMM (still BF16 for up-proj) | 1020 | FP8 wgrad for up-proj too | ~300µs |
| 2 | elementwise casts | 479 | Fuse into quant/dequant kernels | ~200µs |
| 3 | Row Quant (_quantize_and_pack) | 313 | Fuse into GEMM epilogue | ~100µs |
| 4 | Token Gather | 145 | Already optimized Triton kernel | — |

---

## 5. Methodology Notes

1. **nsys GPU-Projection** is the gold standard: merges overlapping kernel intervals on the GPU timeline, immune to CPU contention. CUDA events measure wall-clock including kernel launch overhead and are cross-validated against nsys.

2. **Memory measurements** use `torch.cuda.max_memory_allocated()` which tracks the caching allocator's high-water mark. This reflects actual GPU memory pressure but may undercount transient allocations that reuse cached blocks.

3. **Precision audit** runs in a clean subprocess per mode/seed to avoid FP8 state contamination. BF16 is the reference; RRMSE = ||fp8 - bf16|| / ||bf16|| × 100%.

4. **Repeated measurement**: nsys was run twice (8 and 12 measured iterations) with 5 warmup iterations each. The two measurements agree within 1.4% CV, confirming result stability.

5. **Bug fix**: The nsys workload template previously set `SONIC_MOE_FP8_MODE=perf` *after* importing sonicmoe. Since `_IS_FP8_ACTIVE` is evaluated at module load time, FP8 was silently disabled in all prior nsys profiling runs. This was fixed by moving env var setup before the import statement.
