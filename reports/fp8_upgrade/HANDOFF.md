# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-13 (Session 52 — NCU-guided quant kernel 2.3x speedup, wgrad auto-tuning, dual-fused quant, CuTe-to-Triton migration)
> **Branch:** `native-fp8-exploration`
> **Status:** Zero-materialization FP8 forward+backward functional. Contract suite: **34/34 tests + 20 subtests PASS**. Quant kernels optimized 43%. Wgrad FP8 auto-tuned by shape.

---

## 0. Bottom Line (Session 52 — verified numbers)

### Performance (CUDA events, T=8192, E=8, top_k=8, TK=65536)

| Shape | BF16 total (ms) | FP8 total (ms) | Speedup | BF16 fwd | FP8 fwd | BF16 bwd | FP8 bwd |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I=1536** (Ernie) | 5.238 | 5.328 | **0.983x** | 2.062 | 2.457 | 3.176 | 2.871 |
| **I=2048** | 6.338 | 6.146 | **1.031x** | 2.499 | 2.899 | 3.839 | 3.247 |
| **I=3072** | 9.198 | 8.098 | **1.136x** | 3.582 | 3.786 | 5.616 | 4.312 |

**Methodology:** CUDA events, same-process, median of 15 trials, 5 warmup. B30Z GPU (275 GiB), shared cluster. Both modes experience identical contention.

**Key finding:** FP8 backward is consistently faster (9.6%-23.2% at I=1536-3072). FP8 forward is consistently slower (5.7%-19.2%) due to activation quantization overhead. Net effect crosses 1.0x between I=1536 and I=2048.

**Honest assessment:**
- **I=1536**: FP8 is slightly slower (~0.98x). Wgrad auto-tune correctly disables FP8 wgrad here.
- **I=2048**: Near break-even to slight win (~1.03x).
- **I=3072**: Clear win (~1.14x). GEMM savings dominate.
- FP8 advantage grows with I: GEMM savings ~ O(I^2), quant overhead ~ O(I).

### Memory (torch.cuda.max_memory_allocated, TK=65536)

| Shape | BF16 peak (MiB) | FP8 peak (MiB) | Delta |
|-------|:---:|:---:|:---:|
| I=1536 | 1627.8 | 1572.0 | **-3.4%** |
| I=2048 | 2116.3 | 2274.3 | +7.5% |
| I=3072 | 3094.1 | 3328.8 | +7.6% |

**Key insight:** At I=1536 (wgrad OFF by auto-tune), FP8 saves memory. At I>=2048 (wgrad ON), FP8 uses MORE memory due to wgrad quant temporaries. **FP8+Stash** saves 21-23% at all shapes (bf16 weights offloaded to CPU).

### Precision (unchanged, verified Session 52)

| Tensor | RRMSE | Correlation | Status |
|--------|:---:|:---:|:---:|
| output | 6.49% | 0.9979 | PASS |
| dx | 6.52% | 0.9979 | PASS |
| dw1 | 4.69% | 0.9989 | PASS |
| dw2 | 4.88% | 0.9988 | PASS |

All within guardrails: **RRMSE < 10%**, **correlation > 0.99**. 34/34 tests + 20 subtests PASS.

---

## 1. Session 52 Changes

| Change | Impact | Files |
|--------|--------|-------|
| **num_warps=1 on colwise quant** | 2.3x speedup on `_colwise_quantize_and_pack_kernel`. NCU-guided: fewer warps/block -> more blocks in-flight -> better SM utilization. Bitwise identical output. | `blockscaled_fp8_gemm.py` L1789 |
| **num_warps=1 on dual_quantize_varlen** | 2.0x speedup (314->157us at TK=65536). Same mechanism. | `blockscaled_fp8_gemm.py` L1969 |
| **CuTe-to-Triton nogather migration** | DownProj wgrad uses Triton nw=1 colwise for y1s (was CuTe). UpProj fallback also switched. CuTe nogather was 182us, Triton nw=1 is 137us. | `functional/__init__.py` L1755, L1183 |
| **Fused dual quant in DownProj** | `dual_quantize_varlen(dz)` replaces separate `colwise_quantize_cute(dz)` + `quantize_and_pack_activation(dz)`. 183us vs 311us = 41% faster, saves one HBM read. | `functional/__init__.py` L1760 |
| **Wgrad FP8 shape auto-tuning** | `_FP8Config.resolve_wgrad(I)`: ON for I>=2048, OFF for I<2048. Eliminates negative-ROI wgrad quant at small I. | `functional/__init__.py` L624-636 |
| **introspect.py: quant-bench mode** | Isolated CUDA-event kernel benchmark for all quant kernels with statistics. | `tools/introspect.py` |
| **introspect.py: wgrad-bench mode** | FP8 vs BF16 end-to-end with per-shape memory tracking. | `tools/introspect.py` |

### Total Quant Overhead Reduction (DownProj wgrad, TK=65536, dim=1536)

| Component | Before Session 52 | After Session 52 | Savings |
|-----------|:---:|:---:|:---:|
| y1s colwise nogather | CuTe 104us | Triton nw=1 82us | -22us |
| dz colwise nogather | CuTe 182us | (fused into dual) | -- |
| dz row quant | Triton 129us | (fused into dual) | -- |
| dz dual (row+col fused) | -- | Triton nw=1 183us | -128us |
| dout colwise gather | Triton nw=4 260us | Triton nw=1 122us | -138us |
| **Total** | **~676us** | **~388us** | **-288us (43%)** |

---

## 2. Quant Kernel Performance Data (Session 52 — CUDA events)

All at TK=65536, B30Z, median of 30 trials.

### Production Kernels (with num_warps=1 applied)

| Kernel | dim=3072 (us) | dim=1536 (us) | Notes |
|--------|:---:|:---:|-------|
| Triton colwise nogather | 137 | 82 | Production path for y1s |
| Triton colwise gather | 122 | 76 | Production path for dout, x |
| CuTe colwise nogather | 182 | 104 | **No longer in hot path** (kept as reference) |
| row_quant | 130 | 77 | Unchanged; at HBM ceiling |
| dual_quantize_varlen | 183 | 110 | Fuses row+col; replaces separate col(137)+row(130)=267us |

### NCU Root Cause (why num_warps=1 wins)

Triton default num_warps=4 uses 48 registers/thread -> 62.5% theoretical occupancy -> **63% cycles with no eligible warp** (latency-bound). NCU stall reason: `stall_barrier` dominated.

With num_warps=1 (32 threads/block):
- More blocks in-flight per SM (21x1 warp vs 10x4 warps)
- Reduced L1 contention from fewer concurrent random accesses per block
- Better cross-SM load balancing (more, smaller grid blocks)
- **Output is bitwise identical** across all num_warps values (verified at 6 shapes)

### What Cannot Be Further Optimized

| Kernel | Why | Evidence |
|--------|-----|----------|
| row_quant | 97% occupancy, 4613 GB/s effective BW, 56% DRAM utilization. At HBM ceiling. | NCU Session 48 |
| row_quant num_warps | nw=1/2/4 all give 108-109us. No sensitivity. | CUDA events Session 52 |

### Open Optimization Targets

| Target | Current | Theoretical floor | Gap | Approach |
|--------|:---:|:---:|:---:|----------|
| Triton colwise nw=1 | 137us (3072) | ~90us (match CuTe DRAM%) | 1.5x | Better L1 utilization; Triton achieves only 30% DRAM util vs CuTe's 48% |
| dual_quantize_varlen nw=1 | 183us | ~130us (row_quant floor) | 1.4x | Dual reads once but writes twice. Instruction count 288M is 3.6x row's 80M. |
| colwise gather nw=1 | 122us | ~80us | 1.5x | Gather scatter limits L1 reuse. Could try larger BLOCK_DIM. |

---

## 3. Design and Architecture

### Zero-Materialization FP8

SonicMoE avoids materializing TK-sized FP8 activations:
- `quantize_and_pack_activation(x)` produces T-sized FP8 + T-sized scales
- `_gather_isa_packed_scales_kernel` produces TK-sized ISA-packed scales
- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: T-FP8 + A_idx + TK-scales (no TK FP8 copy)

Zero-mat kernels subclass `GemmSm100` via MRO, override `__call__` with `@cute.jit`. Auto-selected in `gemm_gated.py`/`gemm_dgated.py` when `gather_A + blockscaled`.

### Epilogue Quant (default ON, Session 48)

GemmGated writes z directly as `float8_e4m3fn` in CUTLASS epilogue. No standalone quant kernel, no bf16 z allocation. BF16 placeholder with `as_strided((0,0))` wraps autograd graph node.

### Wgrad FP8 Auto-Tuning (Session 52)

`_FP8Config.resolve_wgrad(I)` applies shape-based heuristic:
- I >= 2048: FP8 wgrad ON (GEMM savings > quant overhead)
- I < 2048: FP8 wgrad OFF (quant overhead dominates)
- Explicitly settable via `SonicMoEConfig(fp8_wgrad=True/False)`

Benchmarked: I=1536 -> 0.913x (wgrad OFF), I=2048 -> 1.057x (ON), I=3072 -> 1.182x (ON).

### Weight Stash (FP8+Stash)

```python
optimizer.step()
moe.refresh_fp8_shadow_weights()  # bf16 -> FP8 caches
moe.stash_bf16_to_cpu()           # -216 MiB GPU
# ... forward + backward with FP8 ...
moe.unstash_bf16()                # restore for optimizer
```

Saves 21-23% peak memory at all shapes.

### Unaligned FP8

- **Forward**: `_padded_blockscaled_gated_forward()` pads expert segments to 128, runs zero-mat GEMM+SwiGLU, unpads. Works.
- **Forward down-proj**: `blockscaled_fp8_gemm_varlen(assume_aligned=False)` handles padding internally.
- **Backward**: Falls back to BF16 entirely (no padded FP8 dgated or wgrad implemented).

### Pythonic Config API

```python
cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True)
with cfg.activate():
    output, aux_loss = moe(x, use_fp8=True)
```

Priority: `SonicMoEConfig` (thread-local) > `enable_fp8()`/`enable_quack_gemm()` context managers > env vars.

---

## 4. Key Files

| File | Role |
|------|------|
| `sonicmoe/config.py` | `SonicMoEConfig` dataclass + thread-local context manager |
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, cache mgmt, wgrad auto-tune |
| `sonicmoe/moe.py` | MoE class, stash/unstash, refresh_fp8_shadow_weights |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels (colwise, row, dual, fused), weight caches, num_warps=1 |
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | CuTe DSL colwise quant (no longer in hot path; kept as reference) |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat CUTLASS kernel classes |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated wrapper (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated wrapper (auto ZeroMat selection) |
| `tools/introspect.py` | Profiling harness: quant-bench, wgrad-bench, nsys GPU-projection, precision |
| `tests/fp8_large_project_contract_test.py` | 34 precision tests + 20 subtests |

---

## 5. Validation Commands

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract suite (34 tests + 20 subtests, ~2.5 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Isolated quant kernel benchmark
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode quant-bench

# End-to-end FP8 vs BF16 benchmark with memory
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/introspect.py --mode wgrad-bench
```

---

## 6. Dead Ends (verified, do NOT retry)

| Approach | Why it fails | Evidence |
|----------|-------------|---------|
| FP8 wgrad at I=1536 | 0.913x net negative. Quant overhead > GEMM savings. Auto-tuned OFF. | Session 52 CUDA events |
| num_warps=1 on row_quant | nw=1/2/4 all give 108-109us. Already at HBM ceiling. | Session 52 CUDA events |
| num_warps=8 on any kernel | Always slower than nw=4. | Session 52 CUDA events |
| CuTe colwise for hot-path nogather | Triton nw=1 (137us) beats CuTe (182us) by 1.33x at dim=3072. | Session 52 |
| Pre-gather + CuTe colwise | index_select(24us) + CuTe(29us) = 53us > Triton fused(39us) | Session 48 NCU |
| Save x as fp8 between fwd/bwd | Dequant creates +24.8 MiB transient spike. Net WORSE. | Session 50 |
| torch.as_strided for fake TK shape | PyTorch storage bounds check rejects it | Session 41-42 |
| Fused CuTe dual quant | 288M instructions (3.6x bloat vs separate) | Session 43 |
| Early weight cache eviction at I=1536 | Only frees 37 MiB. Peak is at wgrad, not dgated. | Session 50 |
| Micro-optimizing row_quant | 97% occupancy, 56% DRAM. At ceiling. | Session 48 NCU |

---

## 7. Lessons (compact, high-value)

### Measurement
1. **CUDA events same-process** is most reliable under contention. nsys needs idle GPU.
2. **Env vars are process-global and cached at import.** `SONIC_MOE_FP8_MODE` leaks into BF16 baselines. Use subprocess isolation or the `_fp8_mode()` fix (Session 51).
3. **`_fp8_mode()` priority bug** (fixed Session 51): `enable_fp8(False)` did NOT work when env var was set. All pre-Session-51 "BF16" baselines with env var are invalid.
4. **NCU `--clock-control=none`** essential on contested nodes.

### Kernel Engineering
5. **num_warps=1 is dramatically better for quant kernels** -- counter-intuitive but verified: fewer warps/block -> more blocks/SM -> better utilization. The key diagnostic is NCU "% cycles with no eligible warp" (63% at nw=4, much lower at nw=1).
6. **Fused dual quant < separate row+col at nw=4** (314us vs 267us), **but dual WINS at nw=1** (183us vs 267us). The nw=1 fix made dual competitive.
7. **CuTe DSL scalar gather is unfixable** -- `mSrc[row, col]` compiles to individual SASS LDG. Triton `tl.load(ptr + offsets)` generates vectorized multi-address loads.
8. **QuACK JIT cache is source-fingerprint-based** -- editing CuTe kernels does NOT invalidate cache. Must manually `rm /tmp/root/quack_cache/<hash>/*.o` AND `_compile_colwise_quant.cache_clear()`.
9. **E8M0 scales encode BF16 magnitude, NOT FP8 magnitude** -- scales must be computed from original BF16 source.

### Architecture
10. **CUTLASS PreAct constraint** -- `assert PreAct.element_size() == 2` blocks feeding FP8 z to GemmDGated. Hard CUTLASS DSL limitation.
11. **Cross-stream memory reuse needs record_stream** -- single-stream is simpler.
12. **Weight stash (21-23% savings) is the dominant memory win.** FP8 alone saves only 3-5% (or costs more with wgrad ON).
13. **FP8 advantage scales with I**: O(I^2) GEMM savings vs O(I) quant overhead. Crossover ~I=1536.
14. **Per-element FP8 cast != blockscaled FP8** -- GEMM epilogue `.to(fp8)` is simple saturating cast. Blockscaled needs per-32-element amax -> E8M0 -> scale-aware cast.

---

## 8. Next Steps (Prioritized)

### P0: Compress forward quant overhead
FP8 forward is 19% slower at I=1536 (2.457 vs 2.062ms). This is the primary reason FP8 doesn't win at small I. Sources:
- `quantize_and_pack_activation(x)` for DownProj: ~130us at T=8192, H=3072
- FP8 GEMM dispatch overhead (CUTLASS scale parameter handling)
- Investigate whether x row-quant can be overlapped with router computation.

### P1: Further colwise quant optimization
Triton colwise nw=1 achieves only 30% DRAM utilization (vs CuTe's 48%). Gap suggests room for better vectorization or L1 utilization:
- Try larger BLOCK_DIM (256, 512) with nw=1
- Profile with NCU to verify L1 miss rate at nw=1

### P2: Unaligned FP8 backward
Currently falls back to BF16 entirely. Implementing padded FP8 backward would extend FP8 benefits to uneven routing:
- padded FP8 dgated (needs `_get_padding_plan` + GemmDGated with padded cu_seqlens)
- padded FP8 wgrad (needs padded CUTLASS varlen_k GEMM)

### P3: Larger shapes (I=4096+)
FP8 speedup should exceed 1.2x at I=4096 based on the O(I^2) vs O(I) scaling model. Validate.

---

## 9. Previous Session History

### Session 51 (fp8_mode fix + CUDA events benchmark)
- Fixed `_fp8_mode()` priority: context manager now properly overrides env var
- CUDA events 3-round benchmark established as primary methodology
- Kernel classifier fix (ZeroMat GEMM excluded from quant category)

### Sessions 48-50 (epilogue quant + memory deep-dive)
- Epilogue FP8 D output: GemmGated writes z as fp8 in epilogue (no standalone quant)
- BF16 placeholder via `as_strided((0,0))` for autograd graph
- Early weight cache eviction (37 MiB, ineffective at I=1536)
- Single-stream wgrad (better memory reuse vs cross-stream)
- FP8 backward peak +118 MiB over BF16 from wgrad quant temporaries
- CuTe gather coalesced loads: 154us -> 58us (2.7x) but still behind Triton 39us
- Save x as fp8: reverted (+24.8 MiB transient spike from dequant)

### Sessions 45-46 (Pythonic config + unaligned padding + CuTe DSL)
- SonicMoEConfig dataclass with thread-local context manager
- Unaligned FP8 padding for forward path
- CuTe DSL colwise quant (no-gather 29us, 93% occ -- was hot path before Session 52)
- wgrad FP8 default-ON

---

## 10. Environment

| Item | Value |
|------|-------|
| Python env | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate` |
| GPU | B30Z (275 GiB), shared cluster |
| PyTorch | 2.11.0+cu130, CUDA 13.0 |
| QuACK | 0.3.7 |
| JIT cache | `/tmp/root/quack_cache/` |
| Env docs | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` |
| BF16 baseline env | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16` |

### Data Sources

| File | Description |
|------|-------------|
| `reports/e2e_bench_session52.json` | Session 52 E2E benchmark (3 shapes x bf16/fp8) |
| `reports/quant_bench_final.json` | Session 52 quant kernel before/after comparison |
| `reports/final_benchmark.json` | Session 51 CUDA events 3-round benchmark |
| `reports/nsys_final/nsys_gpu_projection.json` | Session 50-51 nsys per-kernel category breakdown |
| `reports/fp8_upgrade/engineering_log.md` | Phase-by-phase development history |
