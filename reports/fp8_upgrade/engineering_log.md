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

