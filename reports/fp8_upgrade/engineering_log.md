# FP8 Engineering Log

> See [`HANDOFF.md`](HANDOFF.md) for current state. This log records milestones chronologically.

---

## Phase 1: Initial FP8 Integration (Sessions 1–21)

- Integrated blockscaled FP8 (1×32 UE8M0) into SonicMoE aligned training path
- Built quant/dequant Triton kernels for blockscaled FP8 on SM100a
- Implemented three FP8 weight caches (forward, dgated, actgrad)
- At this point: FP8 was **slower** than BF16 due to quant overhead > GEMM savings

## Phase 2: Selective FP8 (Sessions 22–23)

- Profiled every kernel with nsys to find where FP8 is net-positive
- Result: 7.0% faster GPU projection by using FP8 selectively

## Phase 3: Quant Kernel Optimization (Sessions 24–25)

- Rewrote quant kernel: 1D→2D grid, uint32 scale packing
- A_idx for backward dgated: quantize T rows (~8µs) instead of TK rows (~96µs)
- Forward A_idx attempted but failed (93.5% RRMSE — scales must be gathered with data)

## Phase 4: Multi-Shape + Adaptive FP8 (Sessions 25–26)

- Adaptive FP8 down-proj (net-positive at I≥2048 only)
- Wall-clock: 1.66× (I=1024), 2.15× (I=2048), 2.37× (I=4096)

## Phase 5: Zero-Materialization Kernels (Sessions 27–28)

- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat` (CUTLASS, SM100)
- Three-step gather pipeline (T-quant→fp8_gather→scale_gather)
- Z FP8 save (saves ~171 MiB at Ernie shape)

## Phase 6: Triton Weight Quant + Cleanup (Sessions 29–31)

- Triton weight quant: 8-op eager → single kernel (eliminated 3136µs/iter overhead)
- Z quant grid tuning: BR=32, GPB=12 → 44µs faster
- Weight cache retention (auto-invalidation via `w._version`)
- Dead code removal, 26/26 tests

## Phase 7: Fused Quant + Stream Parallel + Official Baseline (Session 32)

- **Fused z+y1 2D-grid quant kernel**: single launch, packed int32 scale writes
  - Grid (2048, 20) = 40960 blocks (vs 1D 2048 which was slower)
  - ncu-guided: uncoalesced excess 24%→6%, DRAM throughput 49%→67%
- **Stream parallelism**: z-dequant on side stream || dout-quant+s.float()+scale_gather
- **s.float() pre-cast**: 28µs elementwise hidden in stream overlap window
- **Official BF16 baseline**: fixed `moe(x)` API + `z.backward(dout)` for proper profiling
- **TMA investigation**: tested for both loads and stores — 2.3-2.4× slower (descriptor overhead)
- **CUTLASS constraint discovered**: GemmDGated requires bf16 PreAct (cannot feed FP8 z directly)
- 5 new precision tests → 31/31 total

## Phase 8: Authoritative Benchmarks (Session 33)

- **Fully idle node 0342** (8/8 GPUs idle, 0% utilization) — first clean measurement
- **Corrected performance**: BF16 3932µs/iter → FP8 3690µs/iter = **1.066× faster**
  - Previous 0344 data (6609 vs 6290µs) had GPU contention artifacts
  - GEMM savings: 764µs (21.1%), FP8 overhead: 532µs, net: 232µs
- **Corrected memory**: FP8 peak 1913.8 MiB > BF16 peak 1411.8 MiB (+502 MiB)
  - Weight caches (~650 MiB) outweigh Z FP8 savings (~186 MiB)
  - Previous claim "FP8 peak ≤ BF16" was wrong at production shape
- **Profiling runner**: `tools/_profiling_runner.sh` with `nsys_official_bf16`, `nsys_fp8_frontier`, `mem_fp8`, `mem_bf16`, `test` modes
- nsys install: `dpkg -i .../NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb` on remote nodes
- 31/31 tests pass (verified on node 0342)

## Phase 9: Native FP8 Exploration (Session 34)

- Initial `compute_scales_from_fp8_and_pack` was fundamentally wrong — removed
- After fixing, simulated native path was functionally identical to frontier

## Phase 10: True Native FP8 + A/B Comparison (Session 35)

- `NativeFP8Params` dataclass: 4 FP8 weight views via `_quantize_weight_3d_triton` — zero cache pollution
- `MoE.prepare_native_fp8()` + `forward(native_fp8_params=..., x_fp8_data=..., x_fp8_scales=...)`
- x-quant eliminated: `quantize_and_pack` 3→2 calls/iter
- fused_z_save_y1_quant restored (separate z+y1 was 15µs slower)
- **A/B on node 0267 GPU0**: BF16 3969µs → FP8 3669µs = **1.082×** (GEMM -22.4%, overhead +521µs, net -296µs)
- Per-variable precision: all RRMSE <8%, cosine >0.997. MaxRelErr only at near-zero values.
- Memory: FP8 2130 MiB vs BF16 1658 MiB (+472 MiB)
- **Direction A investigated**: z-dequant (129µs) is hard CUTLASS limitation — 5 constraints (view-f32 packing, TMA layout, epilogue recast, no dequant hook, need extra TMA for scales). Needs new kernel class.
- 31/31 frontier + 5/5 true native tests pass

## Phase 11: Comprehensive Benchmark Audit (Sessions 36–37)

- **Discovered process contamination**: `_IS_FP8_ACTIVE` cached at import time from env var (`utils.py:38`). Same-process FP8-vs-BF16 comparison produces fake bit-identical results. **All precision tests must use separate subprocesses.**
- **Bug 2 found & fixed**: `_DownProjection.backward` line 1310 `out=dw2_base` → `out=dw2.permute(2, 0, 1)` (matching BF16 path). c_proj.weight grad had wrong strides; expert 0 worked by luck.
- **Bug 1 found (OPEN)**: `GemmDGatedFP8CLoadSm100` writes dz only for expert 0. Experts 1-7 = denormalized garbage (~1e-36). y1s output is fine. Suspected C-output pointer offset bug in CUTLASS varlen path.
- **Workaround**: `SONIC_MOE_FP8_FUSED_GATED=0` disables fused dgated kernel, uses separate `blockscaled_fp8_gemm_varlen` + SwiGLU backward.
- **nsys profiling (idle node 00041, 8/8 GPUs idle)**:
  - BF16: 3930 µs/iter
  - FP8 fused (buggy): 3273 µs/iter → **1.20× faster** but numerically wrong
  - FP8 non-fused (correct): 5261 µs/iter → **0.75× (34% slower)** due to de-fused SwiGLU kernels adding 2203 µs/iter
- **Precision (subprocess-isolated, FUSED_GATED=0)**: all RRMSE 6.4-7.9%, cosine >0.997, uniform across 8 experts
- **Memory**: FP8 peak 3470 MiB vs Official BF16 1604 MiB (+1866 MiB, mostly QuACK 0.3.7 base overhead)
- **Conclusion**: Bug 1 fix is the critical path. Fused path has proven 1.20× speedup; non-fused workaround eliminates all gains.

---

## Lessons

1. **CUTLASS PreAct constraint is the bottleneck** — `assert PreAct.element_size() == 2` blocks the highest-value optimization (eliminating z-dequant 130µs). Requires CUTLASS DSL epilogue work.
2. **2D grids beat 1D grids for SM utilization** — the fused quant 1D grid (2048 blocks) was slower than separate 2D kernels. 2D grid (40960 blocks) matches separate kernel parallelism.
3. **Packed int32 scale writes fix coalescing** — accumulate 4 group bytes into uint32 before writing. Reduces excess sectors by 4×.
4. **TMA has overhead for small/fine-grained access** — descriptor creation costs dominate when data volume < ~100MB. Only beneficial for large structured GEMM tiles.
5. **nsys GPU projection is the only trustworthy metric** — wall-clock includes 40-60% CPU overhead on contested nodes.
6. **Official BF16 needs `z.backward(dout)`** not `z.sum().backward()` — the latter produces non-contiguous gradients that fail quack assertions.
7. **Stream parallelism has diminishing returns** — z-dequant (130µs) overlaps with dout-quant+gather (83µs), saving ~47µs. But adding more work to the overlap window only hides, doesn't save.
8. **GPU contention invalidates profiling data** — node 0344 (4/8 idle) showed 6609µs BF16; same on fully idle 0342 was 3932µs (1.68× inflation). Always use 8/8 idle nodes.
9. **FP8 weight caches dominate memory overhead** — 3 caches × 3 weights × ~72MB each ≈ 650 MiB, outweighing Z FP8 savings (186 MiB).
10. **nsys needs manual install on remote nodes** — `dpkg -i .../NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb` required before profiling.
11. **E8M0 scales encode BF16 magnitude, NOT FP8 magnitude** — You cannot reconstruct blockscaled scales from FP8 data. Scales must be computed from the original BF16 source and stored alongside FP8 data. `compute_scales_from_fp8_and_pack` was fundamentally wrong.
12. **PostAct FP8 from GemmGated has no ISA-packed scales** — CUTLASS clamp-casts without producing blockscaled metadata. Must still call `quantize_and_pack_activation(y1)` separately.
13. **Test isolation matters** — `enable_native_fp8()` modifies process-global state. Native FP8 tests must run in a separate pytest invocation from frontier tests.
14. **CUTLASS GemmDGated FP8 PreAct is a hard architectural limitation** — 5 constraints make it impossible to feed FP8 z directly: (1) view(f32) packing requires 2-byte elements, (2) recast_tensor assumes bf16 width, (3) TMA smem is dtype-specific, (4) no epilogue dequant hook, (5) scales need separate TMA. Requires a new kernel class, not a config change.
15. **MaxRelErr at near-zero values is FP8-normal** — when |gold|<1e-8, FP8 quantization noise dominates → relative error >1e6. But absolute error stays <5e-5 and all |gold|≥1e-4 elements have MaxRelErr < 0.35. Not an anomaly.
16. **z recompute increases peak memory in single-layer MoE** — GemmGated recompute allocates z(384)+y1(192) MiB during backward when other backward tensors are live. Only effective with `torch.utils.checkpoint` (framework-level lifetime management).
17. **Lazy bwd weights don't reduce peak** — bwd weights must exist during backward regardless of when they're allocated. Lazy allocation just shifts the allocation point, not the peak.
18. **Per-element FP8 cast ≠ blockscaled FP8** — GEMM epilogue's `.to(fp8)` is a simple saturating cast without per-group scaling. Blockscaled quantization (per-32-element amax → E8M0 → scale-aware cast) is required for precision. The two are NOT interchangeable.
19. **Epilogue blockscaled quant requires warp-level group reduction** — CUTLASS MMA register fragments are scattered across threads. Computing per-32-element amax needs `__shfl_xor_sync` within the epilogue, which is possible but requires knowledge of the SM100 MMA register-to-thread mapping.

## Phase 12: CuTe DSL Colwise Quant + Wgrad Integration (Session 43)

- **CuTe DSL `colwise_quantize_cute`**: 90µs vs Triton 136µs = 1.51× (NCU, clock-control=none, 65536×1536)
  - Bank conflicts: 11.1M → 110K (101× reduction) via row-major smem reads
  - Registers: 48 → 30/thread, occupancy: 60% → 89%
  - rcp.approx E8M0: bit-exact vs integer bitops, verified across all float32 ranges
  - abs.f32 PTX: 1 instruction vs fmax(x,-x) = 2, saves 5.9% total instructions
  - Coalesced (num_groups, dim) scale store: L1 store traffic -48%
  - ISA-packed scale output: 100% bit-exact vs Triton
  - gather_idx support: verified for x operand in wgrad path
- **Integrated into `functional/__init__.py`**: wgrad path uses CuTe DSL for both dz fallback and x colwise quant
- **Dead ends discovered**: smem fp8 store (+96% L1), 1D smem view (+18 regs), double-buffer (SM-bound), full unroll (64 regs)
- NCU profile of Triton `dual_quantize_varlen`: same bank-conflict pattern (84%, 11.2M), confirming CuTe DSL approach generalizes

### Lessons (Session 43)

20. **Row-major smem reads are the key to bank-conflict-free column-wise quant** — 32 lanes reading `sSrc[tk, lane]` = 32 consecutive bf16 bytes = coalesced, zero bank conflict. Column-wise reads cause 6.4-way conflicts.
21. **rcp.approx is bit-exact for E8M0** — mantissa is masked away in E8M0 encoding, so ≤1 ULP mantissa error in rcp.approx has zero effect on the scale byte.
22. **Scale layout matters for coalescing** — `(num_groups, dim)` is coalesced for warp writes; `(dim, num_groups)` causes 97% sector waste (stride = num_groups between lanes).
23. **Smem-mediated vectorized store can be WORSE** — writing fp8 to smem then gmem = double the L1 traffic. Only useful if store coalescing is terrible (not our case — fp8 stores are already 32-byte aligned).
24. **Runtime loops beat full unroll when occupancy matters** — 30 regs (89% occ) > 64 regs (44% occ) for this workload. The extra warps hide ALU latency better than ILP from unrolling.
25. **CuTe DSL `cute.make_tensor` creates register overhead** — each tensor object stores pointer + layout metadata in registers. Avoid creating temporary tensors in hot loops.
26. **NCU `--clock-control=none` is essential on contested nodes** — fixed clocks produce inconsistent results when GPU boost is affected by thermal throttling from other workloads.
27. **Split dual quant > fused dual quant [OUTDATED — reversed by nw=1 fix in Session 52]** — At nw=4: CuTe col (84µs) + Triton row (62µs) = 146µs beat Triton fused (168µs). But at nw=1: dual_quantize_varlen=183µs < separate col(137µs)+row(130µs)=267µs. The nw=1 fix reduces per-block overhead enough that fusion's single-HBM-read advantage dominates.
28. **Fused dual quant instruction bloat 3.6×** — per-TK-row warp butterfly shuffle (5 stages × fmax) + per-row E8M0 (10 ALU ops) + per-element rmem fp8 cast (4-element alloc+cast) = 288M instructions vs 80M for separate. The fp8 vector cast requiring 4-element rmem round-trip is the core overhead.
29. **[32][33] smem padding works for bank conflicts but breaks cp.async tiled_copy** — stride 33 is not multiple of 8, so 128-bit vector loads can't align. Element-wise load (no vectorization) is 10× slower.
30. **nsys GPU busy time ≠ wall-clock** — on busy nodes, GPU busy/iter = 5691µs but wall = 8144µs (30% gaps from CPU launch delay). Use nsys `cuda_gpu_trace` → merge intervals → compute actual GPU busy time.
31. **Paddle's `quantize_1x128_kernel`** — gold reference for fused dual quant. Key technique: load full 128×128 tile into registers, `ComputeRowScale` via warp shuffle + smem cross-warp reduce, `ComputeColumnScale` via smem tree reduce, transpose output via `shm[128][129]` (+1 padding). All data stays in registers — zero re-read from smem.

## Phase 13: NCU-Guided CuTe Gather Optimization + Epilogue FP8 D Output (Sessions 47-48)

- **23-kernel NCU report** (`/tmp/ncu_quant2.ncu-rep`): Full metrics for all quant kernels — time, regs, occupancy, DRAM%, LD efficiency, effective bandwidth
- **CuTe colwise+gather root cause**: Original kernel had 2.1 bytes/sector LD efficiency (93% waste) because each thread loaded one row with 32 sequential element loads — no coalescing across threads in a warp
- **Warp-cooperative coalesced gather**: Rewrote gather path so 32 lanes in a warp cooperatively load 32 consecutive columns of the same row (perfectly coalesced). 8 warps × 32 iterations = 256 rows = TILE_TK. Result: **154µs → 58µs** (2.7× improvement), LD efficiency 2.1→~16 bytes/sector
- **Pre-gather NOT viable**: `torch.index_select` (24µs) + CuTe colwise (29µs) = 53µs > Triton fused (39µs). The extra HBM write+read of the 48 MiB gathered buffer costs more than in-register gather. Verified with NCU.
- **Triton still wins gather** (39µs vs 58µs): Triton compiler generates efficient multi-address scatter-gather from `tl.load(ptr + offsets)`. CuTe element-wise `mSrc[row, col]` generates scalar loads.
- **Epilogue FP8 D output**: GemmGated writes z directly as `float8_e4m3fn` (eliminates standalone z quant ~141µs + z.to(fp8) cast ~288µs). BF16 placeholder with `as_strided((0,0))` for autograd graph.
- **Eager weight cache release**: After dgated GEMM, clear `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE`, `resize_(0)` consumed/unused weight caches (estimated -74 to -148 MiB peak)
- **Single-stream wgrad pipeline**: Removed cross-stream overlap. Single-stream enables caching allocator block reuse. +50µs latency but better memory reuse.
- **JIT cache trap discovered**: QuACK compiles CuTe kernels to `/tmp/root/quack_cache/<fingerprint>/*.o`. Fingerprint is based on quack package source, NOT user kernel source. Must manually clear `.o` files AND call `_compile_colwise_quant.cache_clear()` after editing CuTe kernels.
- **34/34 tests pass** (verified)

### Lessons (Sessions 47-48)

32. **CuTe element-wise gather generates scalar loads** — `mSrc[row, col]` compiles to individual SASS LDG instructions. Even with warp-cooperative coalescing (all lanes access same row, different columns), CuTe can't vectorize across gather indices. Triton's `tl.load(ptr + offsets)` generates vectorized multi-address loads.
33. **QuACK JIT disk cache is source-fingerprint-based** — the fingerprint covers `quack/` package source, not user kernel source. Editing CuTe kernels does NOT change the fingerprint. Must clear `/tmp/root/quack_cache/<hash>/*.o` manually.
34. **Pre-gather is mathematically correct but performance-worse** — bit-exact match vs fused approach, but the extra 48 MiB HBM roundtrip (write gathered bf16 + read it back) exceeds the cost of in-kernel scattered loads.
35. **Autograd + fp8 tensors = illegal memory access** — at large shapes, fp8 dtype tensors in the autograd graph cause segfaults in backward. Solution: store a bf16 placeholder with `as_strided((0,0))` (2-byte storage), actual fp8 data in a side dict.
36. **Cross-stream memory reuse needs record_stream** — PyTorch caching allocator won't reuse blocks across streams without explicit `record_stream` calls. Single-stream is simpler and lets freed blocks be immediately reusable.

## Phase 14: Pythonic Config + Unaligned Padding + nsys Engine (Sessions 44–46)

- **nsys GPU-projection engine** (Session 44): Integrated into `tools/introspect.py --mode nsys`. Launches separate BF16/FP8 subprocesses under nsys, parses `CUPTI_ACTIVITY_KIND_KERNEL` from sqlite, merges kernel intervals → per-iteration GPU busy time. Gold standard for latency measurement on idle GPUs.
- **Pythonic Config API** (`SonicMoEConfig`, Sessions 45–46): Dataclass with 10 fields + thread-local context manager. Priority: config > context managers > env vars. Replaced env-var-only control.
- **wgrad FP8 default-ON** (Session 45): `_use_fp8_wgrad()` returns True at all shapes. Trades ~19% slower wgrad GEMM for memory savings (early bf16 dz freeing).
- **Unaligned FP8 padding** (Session 45): `_padded_blockscaled_gated_forward()` pads expert segments to 128 for FP8 GEMM alignment. Backward stays BF16 (QuACK BF16 handles unaligned natively).
- **CuTe DSL colwise quant** (Session 43, continued): 29µs vs Triton 39µs = 1.3× faster without gather. 30 regs, 89–93% occupancy.
- **Wall-clock unreliable** (Session 44+): Established that shared-node contention swamps kernel-time signal in wall-clock measurements.

## Phase 15: Epilogue FP8 + CuTe Gather + NCU Profiling (Sessions 47–48)

- **Epilogue FP8 D output**: GemmGated writes z as `float8_e4m3fn` directly. Eliminates standalone z quant (~141µs) + z.to(fp8) cast (~288µs). No bf16 z allocation (saves 384 MiB transient).
- **BF16 autograd placeholder**: z stored as `as_strided((0,0))` bf16 (2 bytes). Actual fp8 z in `_PREQUANTIZED_SCALES["z_fp8"]`. Avoids fp8 tensors in autograd graph (segfaults at large shapes).
- **CuTe colwise+gather optimization**: Warp-cooperative coalesced loads — 154µs → 58µs (2.7×). Still behind Triton's 39µs for gather case.
- **23-kernel NCU report**: `/tmp/ncu_quant2.ncu-rep` with clock-control=none. Authoritative per-kernel metrics (time, regs, occ%, DRAM%, LD efficiency, effective BW).
- **Pre-gather proven suboptimal**: index_select(24µs) + CuTe colwise(29µs) = 53µs > Triton fused(39µs).
- **row_quant at ceiling**: 97% occupancy, 4613 GB/s. No further optimization possible.
- **Single-stream wgrad**: Removed cross-stream overlap. +50µs latency but better memory reuse via caching allocator.
- 34/34 tests pass.

## Phase 16: Memory Analysis + Env-Var Fix + `_fp8_mode()` Fix (Sessions 49–51)

- **Session 49**: First memory comparison attempted. **CONTAMINATED** by `SONIC_MOE_FP8_MODE=perf` env var leaking into "BF16" runs. All Session 49 data is invalid.
- **Session 50**: Fixed contamination via explicit `os.environ.pop('SONIC_MOE_FP8_MODE', None)` before BF16 measurement. Clean memory data obtained: FP8 saves 4–5% peak (fwd), +118 MiB bwd (wgrad quant temps). FP8+Stash saves 21–23% overall.
- **Session 50**: Backward memory deep-dive — peak is at wgrad (not dgated), so early cache eviction before dgated doesn't help at I=1536. Cache structure: only 37 MiB freeable (w2_varlen; w1T held by ctx reference).
- **Session 50**: "Save x as fp8" attempted and reverted — dequant creates +24.8 MiB transient spike.
- **Session 51**: **`_fp8_mode()` priority fix** — the actual root cause of Session 49 contamination. When `is_fp8_active()` returned False, the function fell through to env var check and returned "perf". Fix: return "off" immediately when `is_fp8_active()` is False.
- **Session 51**: CUDA events 3-round benchmark (20-trial median × 3 rounds, same-process). I=1536: 1.08× (BF16 high-variance), I=2048: 1.22× (consistent). CUDA events proved more reliable than nsys under 100% GPU contention.
- **Session 51**: Kernel classifier fix — ZeroMat GEMM kernels excluded from "Blockscaled Quant" category.

### Lessons (Sessions 44–51)

37. **env vars are process-global and cached at import** — `_IS_FP8_ACTIVE` is set once from env var. Same-process BF16/FP8 comparison with env var set will produce incorrect BF16 baselines.
38. **Context manager must override env var** — the `_fp8_mode()` priority chain must be: config > context manager > env var. A "False" context manager must short-circuit before reaching env var.
39. **CUDA events same-process is most reliable under contention** — both modes experience identical contention. nsys runs separate processes at different times with different contention levels.
40. **FP8 is more contention-resilient** — under 100% GPU util, BF16 showed 57% timing variance while FP8 showed 1.2% at I=1536. Compute-bound FP8 GEMMs are less affected by memory bandwidth contention.
41. **Weight stash is the dominant memory optimization** — 21–23% peak savings vs 4–5% from FP8 alone. The stash strategy works because FP8 caches serve backward, making bf16 master weights redundant during forward+backward.
42. **FP8 backward peak > BF16 backward peak without stash** — wgrad quant creates ~118 MiB temporaries. This is fundamental to blockscaled FP8 wgrad and cannot be eliminated without fusing quant into the GEMM epilogue.

## Phase 17: NCU-Guided Quant Optimization + Wgrad Auto-Tune (Session 52)

- **num_warps=1 discovery** (key insight): NCU showed Triton colwise at nw=4 spends 63% of cycles with no eligible warp (stall_barrier dominated). Reducing to nw=1 (32 threads/block) allows more blocks in-flight per SM → 2.3x speedup on colwise, 2.0x on dual_quantize_varlen. Bitwise identical output.
- **CuTe-to-Triton migration**: All hot-path nogather colwise calls now use Triton nw=1 (137us) instead of CuTe (182us). UpProj fallback also switched.
- **Fused dual quant in DownProj wgrad**: `dual_quantize_varlen(dz)` replaces separate `colwise_quantize_cute(dz)` + `quantize_and_pack_activation(dz)`. Saves one HBM read: 183us vs 311us.
- **Wgrad FP8 shape auto-tuning**: `_FP8Config.resolve_wgrad(I)` — ON for I>=2048, OFF for I<2048. Based on measured crossover: I=1536 wgrad is 0.913x (net negative).
- **Total quant overhead reduction in DownProj wgrad**: ~676us -> ~388us = 43% savings.
- **introspect.py: quant-bench + wgrad-bench modes** — isolated CUDA-event kernel benchmarks with statistics, JSON output.
- **End-to-end results** (CUDA events, TK=65536, B30Z): I=1536: 0.983x, I=2048: 1.031x, I=3072: 1.136x. FP8 forward consistently ~19% slower (quant overhead), FP8 backward consistently faster (9.6-23.2%).
- 34/34 tests + 20 subtests PASS.

### Lessons (Session 52)

43. **num_warps=1 dramatically better for bandwidth-bound Triton kernels** — counter-intuitive; fewer warps/block -> more blocks in-flight per SM -> better utilization. Key NCU diagnostic: "% cycles with no eligible warp". Does NOT help row_quant (nw=1/2/4 all 108us — already well-balanced).
44. **Fused dual quant viability depends on per-block overhead** — at nw=4, dual kernel's 288M instructions caused 3.6x bloat. At nw=1, overhead per block drops enough that single-HBM-read advantage makes dual competitive (183us vs 267us).
45. **CuTe nogather is no longer faster than Triton** — the nw=1 fix inverted the CuTe advantage: Triton nw=1 at 137us beats CuTe at 182us for nogather colwise at dim=3072. CuTe DSL retains its advantage only in theory (93% occ, 48% DRAM) but cannot match Triton's nw=1 block-level parallelism.
46. **FP8 forward overhead is the dominant bottleneck** — at I=1536, FP8 forward is 19% slower (2.457 vs 2.062ms). This single factor prevents FP8 from winning at small I. The forward quant overhead (quantize_and_pack_activation ~130us + GEMM dispatch overhead) cannot be hidden because it's on the critical path.
47. **FP8 wgrad auto-tuning is essential** — shape-dependent ROI: I=1536 wgrad is net negative (0.913x), I=2048 is break-even (1.057x), I=3072 is positive (1.182x). Without auto-tuning, small-I workloads pay an unnecessary penalty.

