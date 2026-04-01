# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-01 (final cleanup)
> **Branch:** `fork-main-sync` at commit `e5d3ca8`
> **Status:** FP8 beats BF16 by **14.9–49.4%** in NSYS GPU projection and **1.66–2.37×** in wall-clock across all tested shapes. 44/44 precision tests pass. WGRAD=0 (BF16 wgrad).

---

## 0. One-screen summary

- The current **best training path** is:
  ```
  USE_QUACK_GEMM=1
  SONIC_MOE_FP8_MODE=perf
  SONIC_MOE_FP8_FUSED_GATED=1
  SONIC_MOE_FP8_WGRAD=0
  ```
  (`SONIC_MOE_FP8_ASSUME_ALIGNED=1` is automatically set when routing is 128-aligned.)

- **NSYS GPU projection (authoritative, iter_2, excluding elementwise_kernel):**

  | Shape | BF16 (µs) | FP8 (µs) | Speedup |
  |-------|-----------|----------|---------|
  | I=1024 (T=4096,H=4096,E=128,K=8) | 2511 | 2137 | **14.9%** |
  | I=2048 (T=4096,H=4096,E=128,K=8) | 7654 | 4403 | **42.5%** |
  | I=4096 (T=4096,H=4096,E=128,K=8) | 20711 | 10479 | **49.4%** |

- **Wall-clock (event-timer, warmup + 10 iters):**

  | Shape | BF16 (ms) | FP8 (ms) | Speedup |
  |-------|-----------|----------|---------|
  | T=4096,H=4096,I=1024 | 8.98 | 5.40 | **1.66×** |
  | T=4096,H=4096,I=2048 | 18.43 | 8.58 | **2.15×** |
  | T=4096,H=4096,I=4096 | 41.50 | 17.48 | **2.37×** |
  | T=8192,H=4096,I=2048 | 31.03 | 15.33 | **2.02×** |
  | T=4096,H=7168,I=2048 | 37.08 | 15.74 | **2.36×** |

- **Memory:** FP8 uses 1.38–1.48× more than BF16 due to FP8 weight caches (~2.68 GiB for 3 caches at I=1024, scales with I). Not yet a win.
- **Inference:** FP8 inference is slower due to re-quantization overhead. Not yet a win.
- **Correctness:** 44/44 contract tests pass across seeds 42, 123, 777, 2024 (all 11 tests × 4 seeds including `large_shape`). RelRMSE <10%, correlation >0.99.

---

## 1. Authoritative NSYS measurements

### 1.1 How to read NSYS GPU projection

The authoritative metric is **NSYS NVTX GPU projection** — the merged GPU-busy time within the `forward` and `backward` NVTX ranges of `iter_2`. BF16 baseline numbers **exclude `elementwise_kernel`** which is a QuACK 0.3.7 layout bug (adds ~2082µs of spurious work to BF16 backward).

### 1.2 I=2048 Kernel Breakdown (most representative production shape)

**FP8 Forward (1017µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| GemmGatedSm100 FP8 | 562µs | vs BF16 1063µs → **47% faster** |
| GemmDefaultSm100 FP8 (down-proj) | 315µs | vs BF16 424µs → **26% faster**, enabled at I≥2048 |
| quant_x (A_idx) | 8µs | quantize T=4096 rows, not TK=32768 |
| quant_y1 | 35µs | for down-proj FP8 input |
| token_gather_sum | 46µs | |

**FP8 Backward (3386µs):**
| Kernel | Time | Notes |
|--------|------|-------|
| wgrad_w1 BF16 | 1727µs | **51% of backward**, largest single kernel |
| actgrad FP8 | 514µs | |
| wgrad_w2 BF16 | 511µs | |
| dgated FP8 | 488µs | |
| quant_dz | 71µs | for actgrad FP8 input |
| quant_dout (A_idx) | 12µs | quantize T=4096 rows |
| token_gather_sum | 45µs | |

### 1.3 Why FP8 advantage scales with I

GemmGated has N=2×I (due to SwiGLU gating). Larger I means larger GEMM → more arithmetic intensity → FP8's 2× TFLOPS advantage dominates quant overhead. At I=1024 the quant overhead (~107µs) is a significant fraction of the GEMM savings (~268µs). At I=4096 the GEMM savings dwarf the quant cost.

---

## 2. Optimizations applied (commit history)

### 2.1 A_idx GemmGated (biggest win) — commit `0bcb474`
- **Before:** `gather_quantize_and_pack_activation` quantizes all TK=32768 gathered rows (98µs)
- **After:** Quantize only T=4096 original rows (~8µs), pass `A_idx` to GemmGated CUTLASS kernel
- **Key insight:** The CUTLASS GemmGated kernel natively supports `A_idx` for row-gathering. Quantizing pre-gather at T-scale is 8× cheaper.
- File: `sonicmoe/functional/__init__.py` → `_fused_blockscaled_gated_forward()`

### 2.2 2D grid quant kernel — commit `2aaa278`
- **Before:** 1D grid, ISA-packed scale writes 1 byte at a time (6.25% coalescing)
- **After:** 2D grid (rows × k-tiles), pack 4 scale bytes as uint32 (25% coalescing)
- **Result:** quant_x 58→8µs (with A_idx), quant_dz 52→35µs
- File: `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` → `_quantize_and_pack_kernel`

### 2.3 Adaptive FP8 down-proj — commit `9c95c14`
- At I≥2048 (configurable via `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD`), FP8 down-proj is net-positive
- At I=1024, FP8 GEMM saves only ~9µs but quant costs 19µs → BF16 wins
- File: `sonicmoe/functional/__init__.py` → `_DownProjection.forward()`

### 2.4 FP8 dgated with A_idx — commit `0bcb474`
- dout quantized at T=4096 scale (~8µs via A_idx) instead of TK=32768 scale
- Combined with native BF16 dgated fallback when FP8 is net-negative
- File: `sonicmoe/functional/__init__.py` → `_DownProjection.backward()`

### 2.5 Misc — commits `60ee3c6`, `e6b78fd`
- `torch.empty` skip-fill when tile-aligned (saves memset overhead)
- Initial selective FP8 framework (GemmGated forward + actgrad backward)

---

## 3. Proven dead-ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad (WGRAD=1)** | +400µs net at I=1024 | K_per_expert=TK/E=256 too small; colwise quant 488µs > GEMM savings 88µs. Only viable at T≥16K+ where K_per_expert≥1024 |
| **Eager FP8 weight cache eviction** | Peak memory increases 10.26→12.87 GiB | `precompute_weight_fp8_*` functions do `w.permute(...).mT.contiguous()` creating BF16 temp copies (~2.15 GB for w1) larger than cached FP8 (~1.07 GB) |
| **FP8 down-proj at I=1024** | -10µs net | quant_y1=19µs > GEMM savings=9µs |
| **FP8 down-proj at I=1024 via A_idx** | N/A | y1 is already at TK scale, no T-scale pre-gather equivalent |
| **Triton quant (num_warps=8)** | Regression | More warps = more register pressure |
| **Stream overlap (quant_dz ∥ wgrad)** | +7.8µs | Both saturate all 128 SMs → bandwidth contention |
| **Transpose+rowquant for wgrad** | 8-12× slower | Terrible cache behavior vs native colwise |
| **Two-phase quant (natural + repack)** | Breakeven | Extra kernel launch offsets write gains |

---

## 4. Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `USE_QUACK_GEMM=1` | off | **Required** — enables Blackwell CUTLASS kernels |
| `SONIC_MOE_FP8_MODE=perf` | off | **Required** — enables FP8 path |
| `SONIC_MOE_FP8_FUSED_GATED=1` | off | **Required** — fused GemmGated with SwiGLU epilogue |
| `SONIC_MOE_FP8_WGRAD=0` | 0 | **Keep 0** — FP8 wgrad net-negative at standard shapes |
| `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD` | 2048 | Intermediate size threshold for FP8 down-proj |
| `SONIC_MOE_FP8_SAVE_Z_FP8=1` | auto-on | z saved as FP8 (eliminates z BF16 save/restore) |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT=1` | auto-on | SwiGLU+quant fused in forward |
| `SONIC_MOE_FP8_SEED` | 42 | Seed override for contract tests |

---

## 5. Key files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | **Main FP8 logic** — forward, backward, all selective FP8 decisions |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, FP8 GEMM wrappers, weight caches |
| `sonicmoe/quack_utils/gemm_dgated.py` | Low-level CUTLASS GemmDGated (accepts A_idx + scales) |
| `sonicmoe/quack_utils/__init__.py` | Exports |
| `tools/nsys_profile_comprehensive.py` | NSYS profiling harness (standard shape) |
| `tools/measure_aligned_perf_memory.py` | Wall-clock + peak memory benchmark |
| `tests/fp8_large_project_contract_test.py` | 11 precision tests across shapes |

### 5.1 Code architecture notes

- **Weight caches:** Three separate caches keyed on `(data_ptr(), _version, shape, stride)`:
  - `_FUSED_WEIGHT_CACHE` → `precompute_weight_fp8_for_fused_gated()` → forward GemmGated
  - `_DIRECT_FUSED_DGATED_WEIGHT_CACHE` → `precompute_weight_fp8_for_direct_fused_dgated()` → backward dgated
  - `_WEIGHT_CACHE` → `precompute_weight_fp8()` → backward actgrad
  - In-place optimizer updates change `_version`, causing natural cache invalidation.
  - `clear_all_fp8_weight_caches()` also available for manual clearing.

- **gemm_dgated low-level kernel** accepts BOTH `A_idx` and `a_scales/b_scales` simultaneously. Cannot use high-level wrapper `gemm_dgated_tuned` because it does `B.mT.contiguous()` which corrupts pre-quantized FP8 weight layout.

- **Profiling script gotcha:** `tools/nsys_profile_comprehensive.py` `--mode fp8` overrides `SONIC_MOE_FP8_WGRAD=0` (line 56). Use `--mode fp8_wgrad` for actual FP8 wgrad testing.

---

## 6. Correctness

- **44/44** contract tests pass across seeds 42, 123, 777, 2024
- Tests include `large_shape` (T=4096, H=7168, I=2048, E=128, K=8)
- Thresholds: RelRMSE <10%, correlation >0.99
- Test file: `tests/fp8_large_project_contract_test.py`

---

## 7. Reproduction commands

### 7.1 Quick validation (8 tests, ~30s)
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

CUDA_VISIBLE_DEVICES=7 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

### 7.2 Full multi-seed validation (44 tests)
```bash
for seed in 42 123 777 2024; do
  CUDA_VISIBLE_DEVICES=7 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
    SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0 SONIC_MOE_FP8_SEED=$seed \
    python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
done
```

### 7.3 Wall-clock + memory benchmark
```bash
CUDA_VISIBLE_DEVICES=7 python tools/measure_aligned_perf_memory.py
```

### 7.4 NSYS profiling (standard shape I=1024)
```bash
CUDA_VISIBLE_DEVICES=7 SONIC_MOE_FP8_FUSED_GATED=1 \
  nsys profile -t cuda,nvtx --cuda-memory-usage=false -f true \
  -o /tmp/sonic_fp8 --export=sqlite \
  python tools/nsys_profile_comprehensive.py --mode fp8
```

### 7.5 GPU projection analysis from NSYS sqlite
```python
import sqlite3

def gpu_projection(db_path, label):
    conn = sqlite3.connect(db_path)
    iters = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE '%iter_2'"
    ).fetchall()
    iter_start = min(r[1] for r in iters)
    iter_end = max(r[2] for r in iters)
    phases = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE start >= ? AND end <= ? AND text IN ('forward','backward') "
        "ORDER BY start", (iter_start, iter_end)
    ).fetchall()
    total = 0
    for pn, ps, pe in phases:
        kernels = conn.execute(
            "SELECT s.value, k.start, k.end, (k.end-k.start)/1000.0 "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id "
            "WHERE k.start<? AND k.end>? "
            "AND s.value NOT LIKE '%%elementwise_kernel%%' "
            "ORDER BY k.start", (pe, ps)
        ).fetchall()
        intervals = sorted([(max(ks,ps), min(ke,pe)) for _,ks,ke,_ in kernels])
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        gpu_us = sum((e-s) for s,e in merged) / 1000.0
        total += gpu_us
        print(f'{label} {pn}: {gpu_us:.1f}µs')
        for name, ks, ke, dur in kernels:
            if dur > 5:
                print(f'  {dur:8.1f}µs {name[:90]}')
    print(f'{label} TOTAL: {total:.1f}µs')
    conn.close()
    return total
```

---

## 8. Insights for next agent

### 8.1 Why wall-clock is much better than GPU projection suggests
FP8 path has **fewer total kernel launches** → less Python/CPU dispatch overhead → better wall-clock latency. NSYS GPU projection only measures kernel execution time, ignoring the ~2ms of inter-kernel CPU gaps in the BF16 path. For real-world throughput, wall-clock is more representative.

### 8.2 The real remaining bottleneck is BF16 wgrad
At I=2048, `wgrad_w1` BF16 alone is 1727µs = **51% of backward time**. This is the irreducible BF16 floor. Making this FP8 would yield huge savings IF the quant overhead can be eliminated (which it can't at K_per_expert=256).

### 8.3 BF16 wgrad anomaly at I=2048
BF16 path's wgrad_w1 takes 3896µs but the *identical* BF16 GEMM in FP8 path takes only 1727µs. Root cause unclear — possibly different CUTLASS kernel compilation due to tensor layout/alignment differences, or memory pressure effects from QuACK 0.3.7 `elementwise_kernel` bug in BF16 path. This amplifies FP8's measured advantage at I=2048.

### 8.4 FP8 wgrad becomes viable at large T
At T≥16384 with E=128, K_per_expert=T/E≥128 which improves the compute/quant ratio. The `blockscaled_fp8_weight_grad_gemm_fast` function (line ~2623 of `blockscaled_fp8_gemm.py`) exists but is untested at large T.

### 8.5 Memory reduction requires architectural change
The 3 FP8 weight caches are the memory culprit. Can't be fixed by eviction (proven worse — see §3). Requires either: (a) in-place quantization without `w.permute(...).mT.contiguous()`, or (b) CUTLASS kernel that accepts strided BF16 weight inputs and quantizes on-the-fly.

### 8.6 cutify reference implementations
Located at `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/operator-incubator/cutify/ops/cute/`:
- `fused_weighted_swiglu_act_quant_cute()`: SwiGLU + FP8 quant element-wise fusion
- `fused_act_dequant_cute()`: FP8 dequant fusion
- `fused_swiglu_weighted_bwd_cute()`: SwiGLU backward fusion
- These operate at element level, not as GEMM epilogues. Not directly applicable to the remaining wgrad bottleneck.

---

## 9. Suggested next steps (ordered by estimated ROI)

1. **Fuse quant_dz into dgated CUTLASS epilogue** (saves 35–71µs depending on I)
   - dgated writes dz as BF16, then separate quant_dz reads+writes FP8
   - Fusing eliminates one full read+write pass (~256 MB bandwidth at I=2048)
   - Requires CUTLASS epilogue modification

2. **FP8 wgrad at large T** (only if production uses T≥16K)
   - Test `blockscaled_fp8_weight_grad_gemm_fast` at T=16384
   - K_per_expert=16384/128=128 may be sufficient

3. **Memory optimization via strided quantization**
   - Modify `precompute_weight_fp8_*` to avoid `.contiguous()` BF16 temp copies
   - Or accept the memory cost — it's a fixed per-layer overhead

4. **Investigate BF16 wgrad anomaly**
   - Understanding why identical GEMMs get different kernel selections could reveal optimization for both paths
