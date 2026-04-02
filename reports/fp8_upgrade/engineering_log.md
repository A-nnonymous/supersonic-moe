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

---

## Lessons

1. **CUTLASS PreAct constraint is the bottleneck** — `assert PreAct.element_size() == 2` blocks the highest-value optimization (eliminating z-dequant 124µs). Requires CUTLASS DSL epilogue work.
2. **2D grids beat 1D grids for SM utilization** — the fused quant 1D grid (2048 blocks) was slower than separate 2D kernels. 2D grid (40960 blocks) matches separate kernel parallelism.
3. **Packed int32 scale writes fix coalescing** — accumulate 4 group bytes into uint32 before writing. Reduces excess sectors by 4×.
4. **TMA has overhead for small/fine-grained access** — descriptor creation costs dominate when data volume < ~100MB. Only beneficial for large structured GEMM tiles.
5. **nsys GPU projection is the only trustworthy metric** — wall-clock includes 40-60% CPU overhead on contested nodes.
6. **Official BF16 needs `z.backward(dout)`** not `z.sum().backward()` — the latter produces non-contiguous gradients that fail quack assertions.
7. **Stream parallelism has diminishing returns** — z-dequant (124µs) overlaps with dout-quant+gather (83µs), saving ~47µs. But adding more work to the overlap window (s.float() 28µs) only hides, doesn't save.
