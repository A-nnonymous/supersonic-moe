# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-03-31 (Session 23)
> **Status:** FP8 selective fused gated path is **7.0% faster** GPU projection than BF16, validated across 4 seeds. FP8 wgrad is permanently net-negative (keep WGRAD=0).

---

## 0. One-screen summary

- The current **best training path** is:
  ```
  USE_QUACK_GEMM=1
  SONIC_MOE_FP8_MODE=perf
  SONIC_MOE_FP8_ASSUME_ALIGNED=1
  SONIC_MOE_FP8_FUSED_GATED=1
  SONIC_MOE_FP8_WGRAD=0
  ```
- **NSYS GPU projection (authoritative):** FP8 `2272µs` vs BF16 `2442µs` = **7.0% faster**
- **Wall-clock (clean GPU):** FP8 `9.647ms` vs BF16 `10.064ms` ≈ **4.3% faster** (includes BF16 elementwise_kernel bug from QuACK 0.3.7)
- **Multi-shape:** FP8 advantage **scales with intermediate size** — 18.9% at I=2048, 7.5% at T=8192, 6.9% at E=64
- **Memory:** FP8 `11.0 GiB` vs BF16 `8.9 GiB` — FP8 weight caches
- **Correctness:** 32/32 contract tests pass across seeds 42, 123, 777, 2024 — RelRMSE <10%, correlation >0.99
- **FP8 wgrad:** Proven permanently net-negative (colwise quant 487µs > GEMM savings 118µs)

---

## 1. Authoritative measurements

### 1.1 NSYS GPU projection (iter_2, standard shape T=4096 H=4096 I=1024 E=128 K=8)

| Path | Forward | Backward | Total | vs BF16 |
|------|---------|----------|-------|---------|
| **Official BF16** | `788.6µs` | `1653.3µs` | `2441.9µs` | 1.00x |
| **FP8 Selective (CURRENT BEST)** | `647.7µs` | `1624.3µs` | `2271.9µs` | **1.075x** |

> BF16 baseline excludes QuACK 0.3.7 `elementwise_kernel` bug (2082µs in backward).

### 1.2 Multi-shape wall-clock (clean GPU, first benchmark run)

| Shape | BF16 ms | FP8 ms | Speedup |
|-------|---------|--------|---------|
| T=4096 H=4096 I=1024 E=128 K=8 | 10.064 | 9.647 | **1.043x** |
| T=4096 H=4096 I=2048 E=128 K=8 | 20.358 | 17.124 | **1.189x** |
| T=8192 H=4096 I=1024 E=128 K=8 | 14.440 | 13.431 | **1.075x** |
| T=4096 H=4096 I=1024 E=64 K=8 | 8.327 | 7.792 | **1.069x** |

### 1.3 FP8 Selective Kernel Breakdown (standard shape)

**Forward (647.7µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| _quantize_and_pack_kernel (x) | 54.5µs | Wide-scatter optimized |
| GemmGatedSm100 **FP8** | 272.5µs | vs BF16 464.5µs → saves 192µs |
| GemmDefaultSm100 BF16 (down-proj) | 220.5µs | BF16 optimal at K=1024 |
| token_gather_sum | 46.1µs | |
| misc | 54.1µs | |

**Backward (1624.3µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| GemmDGatedSm100 BF16 | 296.6µs | BF16 dgated saves 53µs vs FP8 |
| _quantize_and_pack_kernel (dz) | 52.6µs | For FP8 actgrad input |
| GemmDefault BF16 (wgrad_w2) | 271.6µs | |
| GemmDefault BF16 (wgrad_w1) | 584.3µs | Largest kernel, 84% utilization |
| GemmDefault **FP8** (actgrad) | 339.9µs | vs BF16 416.1µs → saves 76µs |
| token_gather_sum | 45.7µs | |
| misc | 33.6µs | |

---

## 2. Optimizations Applied (Session 22→23)

### 2.1 A_idx GemmGated optimization (biggest win)
- **Before:** Quantize all TK=32768 gathered rows → `gather_quantize_and_pack_activation` (98µs)
- **After:** Quantize only T=4096 original rows, pass `A_idx` to GemmGated CUTLASS kernel
- **Result:** quant_x 98→55µs, GemmGated handles gather internally
- File: `sonicmoe/functional/__init__.py` `_fused_blockscaled_gated_forward()`

### 2.2 Wide-scatter quant kernel optimization
- **Before:** ISA-packed E8M0 scale writes: 1 byte at a time, 6.25% coalescing
- **After:** Pack 4 scale bytes as uint32 per k-tile, 25% coalescing
- **Result:** quant_x 63→55µs (-13%), quant_dz 57→53µs (-7%)
- File: `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` `_quantize_and_pack_kernel`

### 2.3 BF16 dgated backward (selective FP8)
- **Rationale:** FP8 dgated requires `gather_quant_dout` (100µs) but saves only 47µs in GEMM
- **Change:** Use BF16 `gemm_dgated` with native A_idx support, saves 53µs net
- File: `sonicmoe/functional/__init__.py` `_DownProjection.backward()`

### 2.4 BF16 down-proj forward (selective FP8)
- **Rationale:** At K=1024, FP8 GEMM saves only 10µs but quant costs 29µs
- **Change:** Use BF16 GEMM for down-proj when `fused_gated` is enabled
- File: `sonicmoe/functional/__init__.py` `_DownProjection.forward()`

### 2.5 Proven dead-ends (DO NOT retry)
| Approach | Result | Why |
|----------|--------|-----|
| FP8 wgrad (WGRAD=1) | 4434µs backward (2.7×) | colwise quant 487µs > GEMM savings 118µs |
| Triton quant tuning (num_warps=8) | Regression | More warps = more register pressure |
| Stream overlap (quant_dz ∥ wgrad_w2) | +7.8µs | Both saturate all 128 SMs → bandwidth contention |
| Two-phase quant (natural + repack) | Breakeven | Extra kernel launch offsets write efficiency |
| FP8 down-proj (standard shape) | -19µs net | K=1024 too small for FP8 compute advantage |
| Transpose+rowquant for wgrad | 8-12× slower | Terrible cache behavior vs native colwise |

---

## 3. Code state

### 3.1 Feature flags

| Flag | Default | Recommendation |
|------|---------|----------------|
| `SONIC_MOE_FP8_MODE=perf` | off | **Required** for FP8 |
| `SONIC_MOE_FP8_ASSUME_ALIGNED=1` | off | **Required** for aligned routing |
| `SONIC_MOE_FP8_FUSED_GATED=1` | off | **Required** for best performance |
| `SONIC_MOE_FP8_WGRAD=0` | off | **Keep off** — FP8 wgrad permanently net-negative |
| `SONIC_MOE_FP8_SAVE_Z_FP8=1` | on | On by default with fused gated |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT=1` | on | On by default with fused gated |

### 3.2 Key files modified

| File | Changes |
|------|---------|
| `sonicmoe/functional/__init__.py` | A_idx forward, BF16 dgated/down-proj, inference alignment |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Wide-scatter quant kernel |
| `tools/measure_aligned_perf_memory.py` | Inference routing monkey-patch |

---

## 4. Correctness

- **Contract tests:** 32/32 pass across seeds 42, 123, 777, 2024
- **Shape:** T=4096, H=4096, I=1024, E=128, K=8 with uniform 128-aligned routing
- **Thresholds:** RelRMSE <10%, correlation >0.99 (all met)

---

## 5. Remaining opportunities

### 5.1 CUTLASS epilogue FP8 fusion (~53µs savings → 8.3% total)
- Fuse `quantize_and_pack_activation(dz)` into GemmDGated epilogue
- **Feasibility:** Architecturally possible but 2-3 weeks engineering
- **Blocker:** Cross-thread reduction for amax, TileStore multi-output
- See detailed analysis in Session 23 checkpoint

### 5.2 quant_x further optimization (~20µs possible → 8-9% total)
- Current: 54.5µs at ~15% bandwidth efficiency for [4096, 4096]
- Theoretical minimum: ~6.3µs (50.6 MB at 8 TB/s)

### 5.3 Adaptive FP8 down-proj for larger shapes
- At I≥2048, FP8 down-proj becomes net-positive
- Could add shape-based decision (currently hardcoded BF16)

### 5.4 Memory reduction (target: <9 GiB)
- Current: 11.0 GiB (vs BF16 8.9 GiB)
- Main overhead: 2.4 GiB from w1_fp8 + w1T_fp8 weight caches

---

## 6. Reproduction commands

### 6.1 Contract tests (4 seeds)
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

for seed in 42 123 777 2024; do
  CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
    SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0 SONIC_MOE_FP8_SEED=$seed \
    python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
done
```

### 6.2 NSYS profiling + GPU projection
```bash
CUDA_VISIBLE_DEVICES=0 SONIC_MOE_FP8_FUSED_GATED=1 \
  nsys profile -t cuda,nvtx --cuda-memory-usage=false -f true \
  -o /tmp/sonic_fp8 --export=sqlite \
  python tools/nsys_profile_comprehensive.py --mode fp8
```

---

## 7. Bottom line

FP8 selective path is **7.0% faster** than BF16 in GPU projection (2272µs vs 2442µs). The advantage **scales with intermediate size** — up to **18.9% faster** at I=2048 (production-relevant). The approach uses FP8 only where it provides net benefit: GemmGated (forward) and actgrad (backward), keeping dgated, down-proj, and wgrad in BF16.

Remaining quant overhead (107µs) is the main friction. Eliminating it via CUTLASS epilogue fusion would push the advantage to ~11%+, but requires significant engineering investment.
