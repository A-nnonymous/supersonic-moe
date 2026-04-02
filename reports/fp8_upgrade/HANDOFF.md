# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-02 (Session 33 — authoritative benchmarks on fully idle node 0342)
> **Branch:** `fork-main-sync`
> **Status:** FP8 frontier is **1.066× faster** than official BF16 on GPU projection at Ernie production shape (fully idle node). FP8 peak memory is ~502 MiB higher than BF16 due to weight caches. Precision verified (**31/31** tests pass).

---

## 0. One-screen summary

**What works today:**
- Zero-materialization FP8 forward via `GemmGatedSm100ZeroMat` (auto-selected in `gemm_gated()`)
- Zero-materialization FP8 backward via `GemmDGatedSm100ZeroMat` (auto-selected in `gemm_dgated()`)
- **Fused z+y1 2D-grid quant kernel** — single launch replaces 2 separate kernels (168µs combined)
- **Stream parallelism** — z-dequant overlaps with dout-quant + s.float() + scale_gather on separate CUDA streams
- Triton weight quantization — single kernel replaces 8-op eager path
- Z saved as FP8 (saves ~186 MiB at Ernie shape)
- Weight cache retention (fwd→bwd reuse, auto-invalidation via `w._version`)
- **31/31** tests pass (11 FP8LargeProjectContractTest + 20 FP8AlignedContractTest)

**Minimal flags for training:**
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 python train.py
```
Or programmatically:
```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True), enable_fp8():
    out, loss = moe(x, use_fp8=True)
```

---

## 1. Performance

### GPU Projection (nsys, Ernie shape T=8192 H=3072 I=1536 E=8 K=8)

Measured on **fully idle node 0342** (10.51.203.75, 8/8 GPUs idle, 0% utilization), nsys + SQLite, 10 warmup + 5 profiled:

| Metric | Official BF16 (quack 0.2.5) | FP8 frontier (quack 0.3.7) | Ratio |
|--------|---------------------------|---------------------------|-------|
| **GPU total (5 iters)** | **19,659µs** | **18,452µs** | **1.066× faster** |
| **GPU per-iter** | **3,932µs** | **3,690µs** | **1.066× faster** |
| Kernel types/iter | 28 | 31 | +3 unique types |

> **Previous contested-node data (0344, 4/8 idle) showed 6609 vs 6290µs — those numbers reflected GPU contention inflating GemmDefault Max to 3440µs. The 0342 data above is authoritative.**

### FP8 Per-Iteration Kernel Breakdown (3690µs/iter, 31 kernel types)

| Kernel | Avg µs | Total (5 iter) | % | Category |
|--------|--------|---------------|---|----------|
| GemmDefaultSm100 BF16 ×4 (wgrad + actgrad + downproj) | 479 | 9,579µs | 51.9% | GEMM |
| GemmDGatedSm100ZeroMat (bwd FP8) | 477 | 2,384µs | 12.9% | GEMM |
| GemmGatedSm100ZeroMat (fwd FP8) | 456 | 2,282µs | 12.4% | GEMM |
| `_quantize_and_pack_kernel` ×3 (x, dout, dz) | 60 | 895µs | 4.8% | FP8 overhead |
| `_fused_z_save_y1_quant_kernel` (z+y1 quant) | 168 | 838µs | 4.5% | FP8 overhead |
| `token_gather_sum_kernel` ×2 | 70 | 702µs | 3.8% | Shared (BF16+FP8) |
| `_dequant_blockscaled_fp8_kernel` (z restore) | 130 | 649µs | 3.5% | FP8 overhead |
| `_gather_isa_packed_scales_kernel` ×2 | 27 | 273µs | 1.5% | FP8 overhead |
| Other (routing, elementwise, reduce) | — | 850µs | 4.6% | Shared |

**FP8-only overhead: ~532µs/iter (14.4% of total)** — GEMMs account for 77.2%.

### Official BF16 Baseline Kernel Breakdown (3932µs/iter, 28 kernel types)

| Kernel | Avg µs | Total (5 iter) | % |
|--------|--------|---------------|---|
| GemmDefaultSm100 BF16 ×4 (all BF16 GEMMs) | 587 | 11,744µs | 59.7% |
| GemmGatedSm100 (up-proj + SwiGLU) | 768 | 3,839µs | 19.5% |
| GemmDGatedSm100 (bwd + dSwiGLU) | 496 | 2,478µs | 12.6% |
| `token_gather_sum_kernel` ×2 | 71 | 714µs | 3.6% |
| Other | — | 884µs | 4.5% |

BF16 GEMM total: 2349 + 768 + 496 = **3613µs/iter** (91.9% of GPU time).

### Savings Breakdown

```
GEMM savings:
  BF16 GEMMs: 2349 + 768 + 496 = 3613µs/iter
  FP8  GEMMs: 1916 + 477 + 456 = 2849µs/iter
  Saved: 764µs (21.1% GEMM reduction)

FP8-only overhead: 179 + 168 + 130 + 55 = 532µs/iter

Net: 764 - 532 = 232µs → 3932→3690µs/iter (6.6% speedup on fully idle node)
```

### Memory

| Metric | Official BF16 | FP8 Frontier | Delta |
|--------|--------------|-------------|-------|
| **FWD Peak** | 1385.9 MiB | 1913.8 MiB | **+527.9 MiB** |
| **Total Peak (FWD+BWD)** | 1411.8 MiB | 1913.8 MiB | **+502.0 MiB** |
| After BWD (steady state) | 640.6 MiB | 871.5 MiB | +230.9 MiB |

**FP8 uses more peak memory than BF16.** The Z FP8 save reduces activation memory (~186 MiB saved), but three weight caches (fwd/dgated/actgrad, each ~72MB FP8 + scales per weight) add ~650 MiB. Net effect is +502 MiB at Ernie shape. This is acceptable on B200 (183GB HBM) but may need cache eviction policies for larger models.

> Note: `test_fp8_memory_less_than_bf16` still passes — it uses a smaller test shape where the Z savings outweigh the weight cache overhead.

### Precision (multi-seed, multi-shape, 31/31 tests)

| Shape | FWD RRMSE | FWD corr | BWD RRMSE | BWD corr |
|-------|-----------|----------|-----------|----------|
| Contract (T=1024, E=8) | <7% | >0.997 | <8% | >0.997 |
| Ernie prod (T=8192, E=8) | <7% | >0.997 | <8% | >0.997 |

---

## 2. Cumulative Changes

### Session 33: Authoritative benchmarks on fully idle node

- **Benchmark node**: 0342 (10.51.203.75, 8/8 GPUs idle, 0% utilization)
- **Corrected performance**: BF16 3932µs/iter → FP8 3690µs/iter = **1.066× faster**
  - Previous 0344 data (6609 vs 6290µs) was inflated by GPU contention
- **Corrected memory**: FP8 peak 1913.8 MiB > BF16 peak 1411.8 MiB (+502 MiB)
  - Weight caches are the main cause; Z FP8 save alone saves 186 MiB
  - Previous claim "FP8 peak ≤ BF16" was incorrect at production Ernie shape
- **nsys artifacts**: `output/sonic-moe-profiling/session33_0342/`
- **Profiling runner**: `tools/_profiling_runner.sh` now supports `nsys_official_bf16`, `nsys_fp8_frontier`, `mem_fp8`, `mem_bf16` modes

### Session 32: Fused quant + stream parallel + official BF16 baseline

1. **Fused z+y1 2D-grid quant kernel** (`blockscaled_fp8_gemm.py`):
   - Rewrote `_fused_z_save_y1_quant_kernel` from 1D grid (2048 blocks) → 2D grid (2048, 20) = 40960 blocks
   - `pid_1 < z_col_blocks` → z path (raw E8M0 scales); `pid_1 >= z_col_blocks` → y1 path (ISA-packed)
   - Packed int32 z-scale writes: uncoalesced excess **24% → 6%**, DRAM throughput **49% → 67%**
   - Saves ~10µs/iter vs separate kernels
   - Integrated into `_UpProjection.forward()` line 639 (when `_save_z_fp8()` is true)

2. **s.float() pre-cast optimization** (`functional/__init__.py`):
   - Moved `s.float()` cast from GemmDGated call site to stream overlap window
   - 28µs elementwise kernel now fully hidden behind z-dequant side-stream

3. **Official BF16 baseline** (`tools/profile_both.py`):
   - Fixed: official `MoE.forward()` doesn't accept `use_fp8=` → use `moe(x)` directly
   - Fixed: `z.sum().backward()` produces non-contiguous dout → use `z.backward(dout)`
   - Both envs can now nsys profile on the same idle node in parallel

4. **5 new precision tests** (31 total):
   - `test_fused_z_save_y1_quant_matches_separate_multi_shape` — bit-exact at 3 shapes
   - `test_fused_z_save_y1_quant_roundtrip_precision` — quant→dequant RRMSE <5%
   - `test_fused_quant_full_moe_forward_backward_multi_shape` — full fwd+bwd at T=1024,8192
   - `test_stream_parallel_backward_deterministic` — 5 runs bit-identical dx
   - `test_backward_s_float_precast_precision` — precision + determinism at 2 shapes

### Session 31: z-quant tuning + cache retention + dead code cleanup

- Z quant grid: BR=32, GPB=12 (was BR=16, GPB=128) → 44µs faster
- Weight cache retention: removed eager eviction (saves ~89µs in benchmarks)
- Dead code: `_fp8_lean`, `_WGRAD_DW2_STREAM`, unused imports

### Sessions 25-30: Zero-mat kernels, Triton weight quant, memory optimizations

- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: zero-materialization FP8 CUTLASS
- Triton weight quant: 8-op eager → single kernel (eliminated 3136µs/iter overhead)
- z FP8 save, y1 pre-quant, three-step gather pipeline

---

## 3. Proven Dead-Ends (DO NOT retry)

| Approach | Result | Root cause |
|----------|--------|------------|
| **FP8 wgrad** | +0.9ms net | Colwise quant SM contention + layout permute ~637µs |
| **Eliminate z-dequant by feeding FP8 z to GemmDGated** | Blocked | CUTLASS asserts `PreAct.element_size()==2` (bf16 required) |
| **Fuse dz-quant into GemmDGated epilogue** | Blocked | Same CUTLASS PreAct constraint |
| **TMA for quant kernel scale writes** | 2.3× slower | Descriptor setup overhead > 6MB data volume |
| **TMA for quant kernel bf16 loads** | 2.4× slower | Block shape must be power-of-2; per-group overhead |
| **Fused z+y1 quant 1D grid** | +5µs regression | Only 2048 blocks → poor SM utilization |
| **FP8 down-proj at I=1536** | No net win | Quant cost ≈ GEMM savings at small I |
| **torch.as_strided for fake TK shape** | RuntimeError | PyTorch storage bounds check |
| **Rowwise quant + strided view** | Not possible | HW requires contiguous K groups |

---

## 4. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, stream parallelism |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant/dequant kernels, fused z+y1 kernel, weight caches |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat CUTLASS kernel classes |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated dispatch (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated dispatch (auto ZeroMat selection) |
| `sonicmoe/quack_utils/swiglu_triton.py` | Dequant kernel, SwiGLU fused variants |
| `tests/fp8_large_project_contract_test.py` | **31 tests** (11 project + 20 aligned) |
| `tools/profile_both.py` | Dual-env nsys profiling (official BF16 + FP8 fork) |
| `tools/_profiling_runner.sh` | Unified SSH profiling runner (nsys, memory, tests) |
| `tools/gpu_projection_benchmark.py` | nsys benchmark with NVTX markers |
| `tools/cluster_idle_launch.py` | Find idle GPU nodes for clean benchmarks |

---

## 5. Validation

**31/31 tests pass** (~285s on B200):
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=<gpu> USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

Or via the profiling runner on a remote idle node:
```bash
ssh <idle_node> "bash tools/_profiling_runner.sh test <gpu_id>"
```

Test coverage by optimization:
| Optimization | Tests |
|-------------|-------|
| Fused z+y1 2D quant | `fused_z_save_y1_quant_matches_separate_multi_shape`, `_roundtrip_precision`, `_full_moe_forward_backward_multi_shape` |
| Stream parallelism | `stream_parallel_backward_deterministic` |
| s.float() pre-cast | `backward_s_float_precast_precision` |
| Zero-mat fwd/bwd | `forward_precision`, `backward_precision`, `production_shape_all_gradients` |
| Z FP8 save/restore | `z_fp8_save_precision`, `z_quant_blockscaled_roundtrip` |
| Weight quant + cache | `triton_weight_quant_matches_eager`, `weight_cache_*`, `cache_retention_*` |
| Memory | `fp8_memory_less_than_bf16` |
| Full production | `production_shape_all_gradients`, `use_fp8_production_shape` |

---

## 6. Insights for Next Agent

### Architecture insight
The FP8 pipeline has a fundamental constraint: **CUTLASS GemmDGated requires bf16 PreAct** (line 283 of `gemm_dgated.py`: `assert PreAct.element_size() == 2`). This means z must be dequantized from FP8 to bf16 before the backward GEMM, costing 130µs + 582MB I/O. This is the single largest blocked optimization. A CUTLASS epilogue that reads FP8 z with on-the-fly dequant would eliminate this, but requires non-trivial CUTLASS DSL work.

### Performance insight
77% of FP8 GPU time is GEMMs (down from 92% in BF16 because of quant overhead). The wgrad GEMMs are BF16 (FP8 wgrad validated as net-negative). Further significant speedups require either:
1. FP8 wgrad with zero layout overhead (needs CUTLASS kernel accepting non-contiguous K groups)
2. Larger I (FP8 advantage scales with GEMM size — I=2048 gets ~2.35×)
3. Custom CUTLASS epilogue for FP8 z dequant (saves 130µs/iter)

### Memory insight
**FP8 uses more peak memory than BF16 at Ernie production shape** (+502 MiB). The Z FP8 save reduces activation memory by 186 MiB, but weight caches (3 caches × 3 weights × ~72MB each) add ~650 MiB. Possible mitigations:
- Selective cache eviction (keep only forward cache, recompute dgated/actgrad caches)
- Cache sharing across experts (if weight structure permits)
- Lazy cache population (only cache on first access per training step)

### Profiling insight
- **Always use nsys GPU projection, not wall-clock** — CPU dispatch overhead is 40-60% on contested nodes
- **Use `tools/cluster_idle_launch.py scan`** to find idle nodes with 4+ free GPUs
- **Official BF16 baseline needs `z.backward(dout)`**, not `z.sum().backward()` (non-contiguous gradient assertion)
- **The fused quant kernel at 167µs operates at 67% DRAM throughput** (55% of B200's 8TB/s peak) — this is near the practical limit for single-pass blockscaled quantization

### ncu insight
- The dominant quant kernel bottleneck is **L1TEX scoreboard stalls** (waiting for DRAM loads)
- Z-scale stores had 24% excess sectors due to stride-96 byte writes — fixed with packed int32 (down to 6%)
- `token_gather_sum_kernel` has **8-16× shared memory bank conflicts** (from `tl.sum(x_vals, axis=0)` on [BLOCK_K, BLOCK_H] tensor) — DO NOT MODIFY per user directive

---

## 7. Next Steps (Priority Order)

1. **Investigate CUTLASS epilogue FP8 PreAct** — If GemmDGated could read FP8 z (with ISA-packed or raw E8M0 scales) and dequant in epilogue, saves 130µs/iter + 582MB I/O. This is the highest-value single optimization remaining.

2. **Reduce FP8 weight cache memory** — FP8 peak is 502 MiB above BF16. Selective cache eviction or lazy population could close this gap while retaining most of the speed benefit.

3. **FP8 wgrad revisited at E=8, K=8** — Previous dead-end was at E=128, K_per_expert=256. At Ernie shape E=8, K_per_expert=8192, the layout overhead might be amortized. Worth re-profiling.

4. **Larger I shapes** — FP8 advantage scales with GEMM size. At I=2048+, expect 1.3-2.4× speedup.

---

## 8. Environment

```
GPU: NVIDIA B200 (SM100a, 183GB HBM, ~8TB/s DRAM BW)
CUDA: 12.8, PyTorch: 2.9.1+cu128
FP8 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer (quack 0.3.7)
BF16 env: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16 (quack 0.2.5)
FP8 codebase: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
BF16 codebase: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe
Profiling output: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/sonic-moe-profiling/
Session 33 data: /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/sonic-moe-profiling/session33_0342/
```

### Profiling commands

```bash
# Install nsys on remote node first (required once per node restart)
ssh <idle_node> "dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb"

# Unified profiling runner (handles env activation + codebase switching)
ssh <idle_node> "bash tools/_profiling_runner.sh nsys_official_bf16 0"  # BF16 on GPU0
ssh <idle_node> "bash tools/_profiling_runner.sh nsys_fp8_frontier 1"  # FP8 on GPU1
ssh <idle_node> "bash tools/_profiling_runner.sh mem_fp8 2"            # FP8 memory
ssh <idle_node> "bash tools/_profiling_runner.sh mem_bf16 3"           # BF16 memory
ssh <idle_node> "bash tools/_profiling_runner.sh test 4"               # 31 tests

# Analysis
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
python tools/nsys_full_breakdown.py <sqlite_file> --labels <label>

# Tests (direct)
CUDA_VISIBLE_DEVICES=<gpu> USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```
