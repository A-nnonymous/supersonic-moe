# FP8 Engineering Log

> Cleaned timeline of what actually happened, what was learned, and what survives.
> Dead-end details are in HANDOFF.md §3. Git history has the full journey.

---

## Phase 1: Initial FP8 integration (Sessions 1–21, pre-fork)

- Integrated blockscaled FP8 (1×32 UE8M0) into SonicMoE aligned training path
- Implemented fused GemmGated with SwiGLU epilogue (`SONIC_MOE_FP8_FUSED_GATED=1`)
- Built quant/dequant Triton kernels for blockscaled FP8 on SM100a
- Implemented three FP8 weight caches (forward, dgated, actgrad)
- Confirmed Triton blockscaled kernels work on SM100a Blackwell (earlier "illegal instruction" errors were CUDA context pollution, not SM100 incompatibility)
- Validated correctness at standard shape (T=4096, H=4096, I=1024, E=128, K=8)
- At this point: FP8 was **slower** than BF16 in GPU projection due to quant overhead (386µs) exceeding GEMM savings (259µs)

## Phase 2: Selective FP8 + A_idx optimization (Sessions 22–23) — commit `e6b78fd`

### Key insight: use FP8 only where net-positive
- Profiled every kernel individually with NSYS
- Found FP8 dgated saves 47µs in GEMM but costs 100µs for `gather_quant_dout` → net negative → keep BF16
- Found FP8 down-proj at I=1024 saves 9µs but costs 19µs for quant → net negative → keep BF16
- Result: **7.0% faster** GPU projection (2272µs vs 2442µs)

### Wall-clock vs GPU-projection divergence discovered
- Wall-clock: FP8 1.54× faster (3.14ms vs 4.83ms)
- GPU projection: FP8 only 7% faster
- Root cause: FP8 path has fewer kernel launches → less CPU dispatch overhead (~2ms)
- **Lesson:** For real-world throughput, wall-clock is the metric that matters

## Phase 3: Quant kernel optimization + A_idx (Sessions 24–25) — commits `2aaa278`, `0bcb474`

### 2D grid quant kernel
- Rewrote `_quantize_and_pack_kernel` from 1D to 2D grid
- ISA-packed scale writes: 1 byte → 4-byte uint32 packing (6.25% → 25% coalescing)
- quant_x: 58→8µs (with A_idx), quant_dz: 52→35µs

### A_idx for GemmGated and dgated
- **Core insight:** quantize at T=4096 pre-gather scale, not TK=32768 post-gather scale → 8× cheaper
- GemmGated CUTLASS kernel natively supports `A_idx` — quantize T rows, kernel gathers internally
- Applied same trick to dout quantization in backward dgated path
- Result: **14.9% faster** GPU projection at I=1024 (2511→2137µs)

## Phase 4: Adaptive FP8 + multi-shape validation (Sessions 25–26) — commits `9c95c14`, `60ee3c6`

### Adaptive FP8 down-proj
- At I≥2048, FP8 down-proj becomes net-positive (larger GEMM → more compute savings)
- Added configurable threshold (`SONIC_MOE_FP8_DOWNPROJ_THRESHOLD=2048`)

### Multi-shape benchmarking results
- GPU projection: 14.9% (I=1024), 42.5% (I=2048), 49.4% (I=4096) faster
- Wall-clock: 1.66× (I=1024), 2.15× (I=2048), 2.37× (I=4096) faster
- FP8 advantage scales dramatically with I because GemmGated has N=2×I

### Multi-seed validation
- 44/44 tests pass across seeds 42, 123, 777, 2024
- Includes `large_shape` test (T=4096, H=7168, I=2048, E=128, K=8)

## Phase 5: Memory investigation (Session 26) — commit `e5d3ca8`

### Eager cache eviction proven counterproductive
- Built `evict_fp8_weight_cache_entry()` utility
- Tested evicting caches after each forward/backward phase
- Result: Peak memory **increased** 10.26→12.87 GiB
- Root cause: `precompute_weight_fp8_*` does `w.permute(...).mT.contiguous()` creating BF16 temp copies (~2.15 GB for w1) larger than the cached FP8 tensor (~1.07 GB)
- Wall-clock regressed 17× (86ms vs 5ms) due to repeated re-quantization
- **Lesson:** Can't fix memory without changing the quantization architecture

---

## Lessons that matter

1. **Use official BF16 as the only baseline.** Exclude QuACK 0.3.7 `elementwise_kernel` from BF16 numbers.
2. **NSYS GPU projection ≠ wall-clock.** FP8 wins much bigger in wall-clock due to fewer kernel launches and less CPU overhead.
3. **Selective FP8 beats blanket FP8.** Only use FP8 where GEMM savings > quant overhead.
4. **A_idx is the key technique.** Quantize at pre-gather (T) scale instead of post-gather (TK) scale for massive quant cost reduction.
5. **FP8 advantage scales with GEMM size.** N=2×I for GemmGated means I=4096 is nearly 50% faster.
6. **FP8 wgrad is shape-dependent.** At K_per_expert=256 (T=4096, E=128) it's net-negative. At K_per_expert≥1024 (T≥16K) it may become viable.
7. **Memory overhead is architectural.** The 3 FP8 weight caches (~2.68 GiB at I=1024) can't be fixed by runtime eviction — requires strided quantization or on-the-fly quantization in CUTLASS.
8. **Don't claim memory or inference wins** — FP8 still uses more memory and inference is slower.
