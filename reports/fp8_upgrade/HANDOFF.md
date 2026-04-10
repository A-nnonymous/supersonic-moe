# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-10 (Session 43 — CuTe DSL colwise quant, dual quant split, wgrad integration, full breakdown)
> **Branch:** `native-fp8-exploration`  **Latest commit:** `8920c10`
> **Status:** ✅ FP8 + iso32 weights + stash + CuTe DSL wgrad quant. 40/40 PASS.

---

## 0. Bottom Line

### Forward+Backward Performance (idle B200, subprocess-isolated)

| Metric | BF16 | FP8 | FP8 + Stash | Stash vs BF16 |
|--------|:---:|:---:|:---:|:---:|
| **nsys GPU Projection** | **3993 µs** | **3564 µs** | 3564 µs* | **1.12× faster** |
| **Forward peak** | **1386 MiB** | 1440 MiB | **1271 MiB** | **−115 MiB (−8.3%)** |
| **Backward peak** | **1412 MiB** | 1492 MiB | **1239 MiB** | **−173 MiB (−12.3%)** |
| Output RRMSE | — | 6.60% | 6.60% | PASS (<10%) |
| dx RRMSE | — | 7.47% | 7.47% | PASS (<10%) |

> *Stash mode uses identical GPU kernels; memory-only difference.

### CuTe DSL Colwise Quant (Session 43, NCU clock-control=none)

| Metric | Triton `colwise_quantize_and_pack` | CuTe DSL `colwise_quantize_cute` |
|--------|:---:|:---:|
| **GPU time** (65536×1536) | 136 µs | **90 µs (1.51×)** |
| **GPU time** (65536×3072) | ~270 µs | **~180 µs (1.50×)** |
| Registers/thread | 48 | **30** |
| Occupancy | 60% | **89%** |
| Smem bank conflicts | 11.1M (84%) | **110K (1.7%)** |
| 100% bit-exact | — | ✅ fp8 AND ISA-packed scales |

---

## 1. Session 43 Changes

| Change | Impact | Commit |
|--------|--------|--------|
| **CuTe DSL colwise quant kernel** | 1.51× faster than Triton, 30 regs, 89% occ | `2cab236` |
| **abs.f32 PTX optimization** | -5.9% instructions (74.8M→72.9M then later refinement to 85µs) | `e94f3d4` |
| **rcp.approx E8M0** | Bit-exact vs integer bitops. Fewer instructions. | `e94f3d4` |
| **Coalesced (num_groups, dim) scale store** | L1 store traffic -48% (201→104 MB) | `e94f3d4` |
| **ISA-packed scale output** | Drop-in compatible with CUTLASS GEMM | `7bea4ac` |
| **gather_idx support** | x operand indirect addressing for wgrad | `dd7fb5e` |
| **Integrated into wgrad path** | functional/__init__.py uses CuTe colwise for both dz and x | `dd7fb5e` |
| **Smem fp8 store (dead end)** | -44% instructions but +96% L1 store traffic → reverted | `297f7a4` |
| **Dual quant split strategy** | CuTe col + Triton row beats fused dual (NCU 146µs vs 168µs) | `34b6899` |
| **Full breakdown script** | Subprocess-isolated memory+precision+nsys GPU projection | `8920c10` |

---

## 2. Kernel-Level GPU Time Breakdown (nsys, idle B200, 3564 µs/iter)

| # | Phase | Kernel | µs | % |
|---|-------|--------|:---:|:---:|
| 1 | FWD: x quant+gather | iso32(45) + pack(16) + gather(27) | **88** | 2.5% |
| 2 | FWD: GemmGated | fp8 zero-mat | **465** | 13.0% |
| 3 | FWD: z→z_fp8 | flat blockscaled | **123** | 3.4% |
| 4 | FWD: y1→y1_fp8 | pack ISA | **56** | 1.6% |
| 5 | FWD: DownProj | fp8 varlen | **232** | 6.5% |
| 6 | BWD: dout quant+gather | pack(16) + gather(27) | **43** | 1.2% |
| 7 | BWD: DGated | fp8 zero-mat + FP8CLoad | **414** | 11.6% |
| 8 | BWD: dw2 wgrad | bf16 GEMM | **369** | 10.3% |
| 9 | BWD: dz prequant | pack ISA | **109** | 3.0% |
| 10 | BWD: dw1 wgrad | bf16 GEMM | **756** | 21.1% |
| 11 | BWD: actgrad | fp8 varlen | **438** | 12.2% |
| 12 | Routing | various | **160** | 4.5% |
| 13 | Other | elementwise | **311** | 8.7% |

**FP8 quant overhead: 440 µs (12.3%). BF16 wgrad: 1125 µs (31.4%).**

---

## 3. FP8 Wgrad Pipeline Status (Session 43)

Wgrad is NOW integrated with CuTe DSL colwise quant. Pipeline breakdown (wall-clock):

| Component | Triton | CuTe DSL | Savings |
|-----------|:---:|:---:|:---:|
| colwise dz (65536×1536) | 138 µs | 101 µs | 37 µs |
| colwise x (65536×3072) | 267 µs | 189 µs | 78 µs |
| CUTLASS GEMM | 209 µs | 209 µs | — |
| **Total** | **614 µs** | **499 µs** | **115 µs (19%)** |

> Note: dz is usually pre-computed by `dual_quantize_varlen` (zero extra HBM read).
> Real savings = x colwise only: 78 µs per backward step.

### Dual Quant: Split Strategy (Session 43 cont'd)

Fused dual quant (single kernel, both row+col) was attempted but:
- CuTe fused: 300µs (288M instructions, 3.6× bloat from per-row warp shuffle + fp8 cast)
- Triton fused: 168µs (84% bank conflicts, 95% L1 saturated)
- **Split: CuTe col (84µs) + Triton row (62µs) = 146µs → 1.15× faster than Triton fused**

Split wins because: L2 cache is hot after colwise, rowwise rarely touches HBM again. Two specialized kernels > one kernel doing two different reductions.

Integrated in `functional/__init__.py` line 1552: dz pre-quant uses split strategy.

### Authoritative Full Breakdown (Session 43, `tools/fp8_frontier_breakdown.py`)

Subprocess-isolated measurement on busy B200 (8/8 GPUs active):

**Memory:**
| Mode | Fwd Peak | Bwd Peak | vs BF16 |
|------|:---:|:---:|:---:|
| BF16 | 1385.9 MiB | 1412.3 MiB | — |
| FP8 | 1486.8 MiB | 1490.9 MiB | +7.3% / +5.6% |
| FP8+stash | 1270.7 MiB | 1238.4 MiB | **-8.3% / -12.3%** |

**Performance (nsys GPU busy):** FP8 5691µs vs BF16 8530µs = 1.50× (busy node, inflated). Idle-node reference: 3564 vs 3993 = 1.12× (more trustworthy).

**Precision:**
| Tensor | RRMSE | Cosine | Stash RRMSE |
|--------|:---:|:---:|:---:|
| output | 6.604% | 1.000 | 6.604% (identical) |
| dx | 7.471% | 0.999 | 7.471% |
| dw1 | 6.408% | 1.012 | 6.408% |
| dw2 | 6.852% | 1.003 | 6.852% |

---

## 4. Critical Insights (for next agent)

### CuTe DSL Kernel Optimization Lessons (Session 43, NCU-verified)

1. **Row-major smem reads eliminate bank conflicts**: Column-wise quant reads `sSrc[tk, lane]` where 32 lanes access 32 consecutive bf16 = perfectly coalesced, zero bank conflicts. Triton's column-wise read pattern has 6.4-way bank conflicts.

2. **rcp.approx is bit-exact for E8M0**: `rcp.approx(amax)` has ≤1 ULP mantissa error, but E8M0 masks the mantissa entirely, so the exponent is always correct. Verified across all float32 ranges including near-subnormal (1e-38).

3. **abs.f32 PTX is 1 instruction vs fmax(x,-x) = 2**: Saves 5.9% total instructions.

4. **Scale layout (num_groups, dim) is coalesced; (dim, num_groups) is not**: 32 lanes writing different dim rows with same group = stride `num_groups` = 2048 bytes apart = 97% sector waste. Transposing the layout makes it coalesced.

5. **Smem-mediated fp8 vectorized store is WORSE**: Reduces instructions by 44% and regs from 30→25, but doubles L1 store traffic (write to smem + write to gmem). Direct gmem store wins.

6. **Runtime loops → low registers → high occupancy wins**: Two-pass smem read (30 regs, 89% occ, 91µs) beats single-pass unrolled (64 regs, 44% occ, 101µs). Occupancy compensates for extra smem reads.

7. **1D smem tensor view adds registers**: `cute.make_tensor(ptr+offset, layout)` creates tensor state in registers. The 2D indexing `sSrc[i, j]` is already optimized by the compiler for simple patterns.

8. **Double buffer not worth it for compute-bound kernels**: Pipeline overlap helps when memory-bound. Our kernel is SM 79% compute-bound; double buffer adds 16 regs for minimal benefit.

### Process Contamination
`SONIC_MOE_FP8_MODE` is cached at import time. **BF16 vs FP8 comparison MUST use separate subprocesses.**

### stash_bf16_to_cpu Cache Invalidation
`unstash_bf16()` changes `data_ptr` → old FP8 cache entries stale. **Must call `clear_all_fp8_weight_caches()` in unstash.**

### 32×32 Isotropic Weight Quant
Hardware only supports 1×32. Trick: compute amax over 32×32 tile, broadcast same scale to all 32 rows. Enables transpose without re-quantization. Precision = 1×32 for real weights.

---

## 5. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, stash, wgrad CuTe integration |
| `sonicmoe/moe.py` | MoE class, refresh_fp8, stash_bf16_to_cpu, unstash_bf16 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels (1×32, iso32, dual_varlen, colwise), weight caches, GEMM wrappers |
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | **CuTe DSL colwise quant** — 1.51× faster, ISA pack, gather support |
| `sonicmoe/quack_utils/cute_dual_quant.py` | **CuTe DSL dual quant** — fused (WIP) + split strategy entry point |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated CUTLASS DSL + BlockscaledScaleStore EpiOp |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated CUTLASS DSL + FP8PreActLoad EpiOp |
| `docs/FP8_ARCH_SPEC.md` | **Start here** — full architecture spec, data flow, memory lifecycle |
| `tests/fp8_large_project_contract_test.py` | 34-test contract suite |
| `tests/fp8_frontier_strict_test.py` | 6-test strict suite |
| `tests/test_cute_blockscaled.py` | CuTe DSL colwise quant correctness + perf tests |
| `tests/test_rcp_precision.py` | rcp.approx E8M0 precision verification |
| `tools/ncu_profile_colwise.py` | NCU profiling script for colwise quant comparison |

---

## 6. Validation Commands
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# CuTe DSL colwise quant test (quick, ~30s)
QUACK_CACHE_ENABLED=0 python tests/test_cute_blockscaled.py

# rcp.approx precision test
QUACK_CACHE_ENABLED=0 python tests/test_rcp_precision.py

# NCU comparison (needs idle GPU)
QUACK_CACHE_ENABLED=0 ncu --clock-control=none \
  --kernel-name "regex:ColwiseQuant|_colwise_quantize" \
  --launch-skip 3 --launch-count 2 \
  --metrics gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  python tools/ncu_profile_colwise.py

# Full test suite (40 tests, ~4 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py tests/fp8_frontier_strict_test.py -v --tb=short
```

---

## 7. Dead Ends (verified, do NOT retry)

| Approach | Why it fails | Session |
|----------|-------------|---------|
| Fused CuTe dual quant | 288M instructions (3.6× bloat). Per-row warp shuffle + rmem fp8 cast overhead. | 43 |
| [32][33] padded smem for dual quant | Eliminates bank conflict but cp.async can't align to stride-33. Element-wise load too slow. | 43 |
| Smem-mediated fp8 vectorized store | -44% instructions but +96% L1 traffic. Direct gmem store is faster. | 43 |
| 1D smem tensor view for address opt | Adds 18 regs (30→48) for tensor state. Net negative due to occupancy drop. | 43 |
| Double-buffer pipeline for colwise | Adds 16 regs, SM-bound kernel doesn't benefit from load/compute overlap. | 43 |
| Full loop unroll for fp8 store | 64 regs → 44% occupancy. Runtime loop (30 regs, 89% occ) is 10% faster. | 43 |
| Epilogue z-quant default ON | z.to(fp8) cast 170µs > standalone quant 122µs. Net +48µs. | 42 |
| FP8 wgrad at I=1536 default ON | col_fp8 transient +192 MiB > stash savings. | 42 |
| `torch.empty(0).as_strided()` | PyTorch bounds check rejects. | 41–42 |

---

## 8. Next Steps (prioritized)

### P0: CuTe DSL fused dual quant (est. −22 µs vs split)
Current split: CuTe col 84µs + Triton row 62µs = 146µs. Fused could eliminate 2nd HBM read. Key challenge: per-row warp shuffle + fp8 cast generates 3.6× instruction bloat. Need Paddle-style register-buffered approach with [128][129] or [32][33] padded smem. Reference: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/Paddle_B/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu` `quantize_1x128_kernel`.

### P1: CuTe DSL rowwise quant (est. −30 µs for activation quant)
`_quantize_flat_v2_kernel` at 123µs has 16.5/32 store coalescing. CuTe version could improve with the same techniques as colwise (30 regs, 89% occ, coalesced stores).

### P2: Reduce FP8 quant overhead further
GEMM dominates at 85%. Quant is 7.7% (440µs). Remaining quant targets:
- `_quantize_and_pack_kernel`: 197µs (ISA activation pack) — fuse with gather
- `_gather_isa_packed_scales_kernel`: 54µs — eliminate via pre-packed scales
- Epilogue quant fusion: blocked by SM100 fp8 D limitation

### P3: End-to-end idle-node validation
Run `tools/fp8_frontier_breakdown.py` on idle B200 (8/8 GPUs idle) for trustworthy absolute timing. Current busy-node nsys gives 1.50× but idle-node reference is 1.12×.

---

## 9. Information Sources

| Resource | Location | Use |
|----------|----------|-----|
| QuACK 0.3.7 CuTe DSL kernels | `/root/paddlejob/.../envs/xfer/lib/python3.13/site-packages/quack/` | Reference patterns: rmsnorm.py, softmax.py, reduce.py, reduction_base.py |
| CuTe DSL swizzle/copy patterns | Same path: `copy_utils.py`, `layout_utils.py`, `sm90_utils.py` | Swizzled smem, tiled_copy, position-independent partition |
| CUTLASS DSL | `.../nvidia_cutlass_dsl/python_packages/cutlass/` | Pipeline, TiledMma, TiledCopy, PipelineAsync |
| NCU reports | `reports/ncu_colwise.ncu-rep`, `/tmp/ncu_cute_final.ncu-rep` | Profile data for colwise quant optimization |
| FP8 architecture spec | `docs/FP8_ARCH_SPEC.md` | Full data flow, memory lifecycle, weight cache system |
