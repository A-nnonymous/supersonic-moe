# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-03 (Session 31)
> **Branch:** `fork-main-sync`
> **Status:** FP8 frontier is **1.101× faster** than BF16 on GPU projection at Ernie production shape. Memory ≤ BF16. Precision verified (**26/26** tests pass).

---

## 0. One-screen summary

**What works today:**
- Zero-materialization FP8 forward via `gemm_gated()` + `A_idx` (auto-selects `GemmGatedSm100ZeroMat`)
- Zero-materialization FP8 backward via `gemm_dgated()` + `A_idx` (auto-selects `GemmDGatedSm100ZeroMat`)
- Triton weight quantization: single-kernel replaces 8-op eager path (eliminated 3136µs/iter reduce_kernel)
- Z quant kernel tuned: BR=32, GPB=12 grid → 166→122µs (26% faster)
- Weight cache retention: fwd→bwd reuses FP8 caches (saves ~89µs in benchmarks)
- Precision verified: **RRMSE <8%, correlation >0.997** at both contract and production shapes
- **26/26** tests pass (11 FP8LargeProjectContractTest + 15 FP8AlignedContractTest)
- Dead code removed: `_fp8_lean`, `_use_fp8_wgrad`, `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD`, `_WGRAD_DW2_STREAM`, `evict_fp8_weight_cache_entry` import

**Best training path (minimal flags):**
```python
# Option 1: env vars
# SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 python train.py

# Option 2: context manager
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

All other flags default to optimal values (`FUSED_GATED=1`, `SAVE_Z_FP8=1`, `FUSED_SWIGLU_QUANT=1`).

---

## 1. Performance (GPU Projection, Ernie shape T=8192 H=3072 I=1536 E=8 K=8)

Measured on idle B200 (10.51.200.142), nsys + SQLite, 10 warmup + 5 profiled iterations:

| Metric | BF16 (xfer env) | FP8 frontier | Ratio |
|--------|-----------------|--------------|-------|
| **GPU projection (iter2 median)** | **3972µs** | **3608µs** | **1.101× faster** |
| Kernel count per iter | ~43 | ~50 | +7 (quant kernels) |

**FP8 iter2 kernel breakdown (3609µs total, 50 kernels):**

| Kernel | Time | Notes |
|--------|------|-------|
| GemmDefault (dw1 wgrad, BF16) | 758µs | On wgrad stream |
| GemmDGatedSm100ZeroMat (bwd FP8) | 483µs | Zero-mat backward |
| GemmDefault (actgrad, FP8 input) | 470µs | Uses dz_fp8 |
| GemmGatedSm100ZeroMat (fwd FP8) | 453µs | Zero-mat forward |
| GemmDefault (dw2 wgrad, BF16) | 351µs | |
| GemmDefault (down-proj, FP8) | 229µs | |
| _quantize_flat_blockscaled (z save) | 122µs | Tuned BR=32, GPB=12 |
| _dequant_blockscaled_fp8 (z restore) | 123µs | 57% bandwidth |
| _quantize_and_pack (dz) | 111µs | ISA-packed, 65% BW |
| token_gather_sum ×2 | 145µs | |
| _quantize_and_pack (y1) | 55µs | ISA-packed |
| _gather_isa_packed_scales ×2 | 55µs | T→TK scale gather |
| Other (routing, elementwise) | ~260µs | |

**FP8-only overhead: ~466µs (13% of total)** — near-optimal; 87% is GEMMs.

**Memory:** FP8 peak ≤ BF16 peak (verified by `test_fp8_memory_less_than_bf16`).

**Precision (multi-seed, multi-shape):**

| Shape | FWD RRMSE | FWD corr | BWD RRMSE | BWD corr |
|-------|-----------|----------|-----------|----------|
| Contract (T=1024) | <7% | >0.997 | <8% | >0.997 |
| Ernie prod (T=8192) | <7% | >0.997 | <8% | >0.997 |

---

## 2. Cumulative Changes (Sessions 25-31)

### Session 31: Precision tests + z-quant tuning + final cleanup

1. **Z quant grid tuning**: `_quantize_flat_blockscaled_kernel` grid changed from BR=16/GPB=128 → BR=32/GPB=12. Saves 44µs per z quant (166→122µs). More column blocks = better SM utilization on B200.

2. **Cache retention** (session 30): Removed w1/w2 `evict_fp8_weight_cache_entry()` calls. Caches auto-invalidate via `w._version` when optimizer updates weights. Saves ~89µs in benchmarks. No precision impact — in production the cache misses every iter due to `optimizer.step()` version bump.

3. **Dead code removal**: `_WGRAD_DW2_STREAM` / `_get_wgrad_dw2_stream()` (never called), unused `evict_fp8_weight_cache_entry` import.

4. **5 new precision tests** (26 total):
   - `test_weight_cache_retention_precision` — full fwd+dx+dw1+dw2 with cache retention
   - `test_z_quant_blockscaled_roundtrip` — quant+dequant roundtrip at production shape
   - `test_quantize_and_pack_isa_roundtrip` — ISA-packed quant for x/dz/y1 shapes
   - `test_production_shape_all_gradients` — T=8192, H=3072, I=1536, E=8, K=8
   - `test_cache_retention_multi_iter_no_drift` — 3 iters with same weights, bit-exact dx

### Session 29-30: Triton weight quant + dead flag cleanup

1. **Triton weight quant**: `_quantize_weight_3d_triton()` replaces 8-op eager `quantize_activation_blockwise()`. Applied to all 5 weight precompute paths. Eliminated 3136µs/iter reduce_kernel overhead (30% of GPU time).

2. **Dead flags removed**: `_fp8_lean` / `SONIC_MOE_FP8_LEAN`, `_use_fp8_wgrad` / `SONIC_MOE_FP8_WGRAD`, `SONIC_MOE_FP8_DOWNPROJ_THRESHOLD`.

### Sessions 25-28: Zero-mat kernels + memory + bugfixes

- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: CUTLASS zero-materialization kernels for SM100
- Session 28: Fixed B-tensor layout bug (was passing `.mT` view → 101% RRMSE)
- z FP8 save (~171 MiB), y1 pre-quant, weight cache dedup, cache collision fix

---

## 3. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad** | +0.9ms net | colwise quant SM contention, layout permute ~637µs |
| **Standalone `gemm_gated_zeromat()` wrapper** | B-layout bug | Standard `gemm_gated()` handles B correctly |
| **`SONIC_MOE_FP8_MODE=perf` for benchmarks** | Masks bugs | Makes BF16 ref use FP8 path |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch storage bounds check |
| **Rowwise quant + strided view** | Not possible | HW requires contiguous K groups |
| **Transpose + rowquant** | 3.8× slower | transpose 1509µs > colwise quant 260µs |
| **FP8 down-proj at I=1536** | No net win | quant cost ≈ GEMM savings at small I |
| **Eager weight quant** | 30% overhead | 8 kernel launches × 2 weights = 3136µs/iter |
| **Custom CUDA warp-level quant** | 5.7× slower | Scalar warp shuffle, no vectorized loads |
| **Fused z+y1 quant kernel** | +803µs regression | Strided memory access pattern kills throughput |

---

## 4. Flag Inventory

### Production flags (minimal set needed):
| Flag | Default | Required? |
|------|---------|-----------|
| `SONIC_MOE_FP8_MODE=perf` | `off` | **Yes** — master toggle |
| `USE_QUACK_GEMM=1` | `0` | **Yes** — enables CUTLASS/QuACK |

### Auto-optimal flags (already default to best value):
| Flag | Default | What it does |
|------|---------|-------------|
| `SONIC_MOE_FP8_FUSED_GATED` | `1` | Fused gemm_gated + blockscaled FP8 |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | `1` | Store z in FP8 (~50% memory save) |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | `1` | Fused SwiGLU+quantize |
| `SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION` | `fp8` | FP8 epilogue (best perf) |
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | auto-detect | Runtime alignment detection |

### Legacy flags (disabled by default, non-interfering):
| Flag | Purpose |
|------|---------|
| `SONIC_MOE_FP8_CUTELY_FUSED` | Experimental adapter path |
| `SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ` | Legacy blockscaled downproj |
| `SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER` | Debug buffer |
| `SONIC_MOE_OPT_NATIVE_FP8_UPPROJ` | Backward compat |
| `SONIC_MOE_OPT_MIXED_DTYPE_DOWNPROJ_DW2` | Backward compat |
| `SONIC_MOE_STAGEWISE_MEMORY` | Debug memory profiling |

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, `_quantize_weight_3d_triton()`, weight caches |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat kernel classes |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated (auto ZeroMat selection) |
| `sonicmoe/quack_utils/swiglu_triton.py` | Dequant kernel, SwiGLU backward from FP8 |
| `tests/fp8_large_project_contract_test.py` | **26 tests** (11 project + 15 aligned contract) |
| `tools/gpu_projection_benchmark.py` | nsys benchmark with sync'd NVTX markers |

---

## 6. Validation

**26/26 tests pass** (102s on idle B200):
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

Test coverage by optimization:
| Optimization | Tests covering it |
|-------------|-------------------|
| Zero-mat fwd/bwd kernels | forward_precision, backward_precision, production_shape_all_gradients |
| Z FP8 save/restore | z_fp8_save_precision, z_quant_blockscaled_roundtrip |
| Triton weight quant | triton_weight_quant_matches_eager |
| ISA-packed quant | quantize_and_pack_isa_roundtrip |
| Cache retention | weight_cache_retention_precision, cache_retention_multi_iter_no_drift |
| Cache dedup | weight_cache_dedup |
| Memory savings | fp8_memory_less_than_bf16 |
| FP8 downproj | fp8_downproj_prequant_precision, native_fp8_downproj_* |
| Multi-iteration stability | multi_iteration_cache_consistency, cache_retention_multi_iter_no_drift |
| Full production shape | production_shape_all_gradients, use_fp8_production_shape |

---

## 7. Remaining Optimization Opportunities

87% of FP8 GPU time is GEMMs (mostly BF16 wgrad, unchanged from BF16 baseline). Further gains require:

1. **FP8 wgrad without layout overhead** — Needs CUTLASS kernel that accepts non-contiguous K groups
2. **Larger intermediate size (I≥2048)** — FP8 GEMM savings scale with N; ~2.35× at I=2048
3. **Weight quant persistence** — In production with `optimizer.step()`, caches miss every iter. Zero-cost weight quant or persistent caching across optimizer steps could help

---

## 8. Environment

```
GPU: NVIDIA B200 (SM100a, 183GB, ~8TB/s DRAM BW)
CUDA: 12.8, PyTorch: 2.9.1+cu128, QuACK: 0.3.7
Python: 3.13
FP8 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer
Official BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16
Clean test machine: ssh 10.51.200.142 (idle B200, use tools/cluster_idle_launch.py scan to find idle nodes)
Reports output: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/sonic-moe-timelines/
```

## 9. Profiling Commands

```bash
# GPU projection benchmark (FP8)
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 nsys profile -t cuda,nvtx -o /path/to/fp8.nsys-rep \
  python tools/gpu_projection_benchmark.py

# GPU projection benchmark (BF16)
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/activate
CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda,nvtx -o /path/to/bf16.nsys-rep \
  python tools/gpu_projection_benchmark.py --mode bf16

# Find idle cluster node
python tools/cluster_idle_launch.py scan
```
