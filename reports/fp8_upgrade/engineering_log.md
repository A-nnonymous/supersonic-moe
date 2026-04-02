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

- **Goal**: Full-chain FP8 params — x as FP8, weights as FP8, no quantization inside MoE
- **Implemented**: `enable_native_fp8()` context manager, `_native_fp8_gated_forward()`, forward routing, 12 precision tests, benchmark script
- **Initial bug**: `compute_scales_from_fp8_and_pack` kernel produced numerical garbage — E8M0 scales encode BF16 magnitude, not FP8 magnitude. Mean byte error ~12.8. **Removed.**
- **PostAct FP8 attempted**: GemmGated can output FP8 PostAct, but without blockscaled ISA scales (raw clamp-cast only). Scales must be stored alongside FP8 data, not reconstructed.
- **FP8 wgrad blocked**: `blockscaled_fp8_gemm_varlen` only supports `cu_seqlens_m`; wgrad needs `cu_seqlens_k + A_idx`
- **Result**: After fixing scale bug, native path is **functionally identical** to frontier (same kernels, same performance)
  - Frontier: 3935 µs/iter, Native: 3962 µs/iter (noise)
  - Memory: identical (2078.4 MiB both)
  - Precision: bit-identical (RRMSE=0.0000, correlation=1.000000)
- **Conclusion**: "Simulate native by quantizing inside MoE" approach provides zero value. True native requires pre-quantized x input + persistent FP8 weight buffers. Detailed plan written.
- 12/12 native tests pass, 31/31 frontier tests pass (run separately)

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
