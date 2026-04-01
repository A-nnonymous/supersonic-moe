# FP8 Engineering Log

> ⚠️ **Historical reference only.** See [`HANDOFF.md`](HANDOFF.md) for current implementation state.
>
> Cleaned timeline of what happened, what was learned, and the current state.
> Dead-end details are in HANDOFF.md §4. Git history has the full journey.

---

## Phase 1: Initial FP8 Integration (Sessions 1–21)

- Integrated blockscaled FP8 (1×32 UE8M0) into SonicMoE aligned training path
- Implemented fused GemmGated with SwiGLU epilogue
- Built quant/dequant Triton kernels for blockscaled FP8 on SM100a
- Implemented three FP8 weight caches (forward, dgated, actgrad)
- Confirmed Triton blockscaled kernels work on SM100a Blackwell
- At this point: FP8 was **slower** than BF16 due to quant overhead (386µs) > GEMM savings (259µs)

## Phase 2: Selective FP8 (Sessions 22–23) — commit `e6b78fd`

- Profiled every kernel with NSYS to find where FP8 is net-positive
- Result: 7.0% faster GPU projection by using FP8 selectively
- Discovered wall-clock vs GPU-projection divergence (FP8 1.54× wall-clock but only 7% GPU projection)

## Phase 3: Quant Kernel Optimization (Sessions 24–25) — commits `2aaa278`, `0bcb474`

- Rewrote quant kernel: 1D→2D grid, 1-byte→4-byte uint32 scale packing
- A_idx for backward dgated: quantize T rows (~8µs) instead of TK rows (~96µs)
- **A_idx for forward GemmGated was attempted but failed (93.5% RRMSE) — reverted.**
  Root cause: ISA-packed scale mismatch when GemmGated uses A_idx (see HANDOFF §8.1)
- Result at old shapes: 14.9% (I=1024), 42.5% (I=2048), 49.4% (I=4096) faster

## Phase 4: Multi-Shape + Adaptive FP8 (Sessions 25–26) — commits `9c95c14`, `60ee3c6`

- Adaptive FP8 down-proj (net-positive at I≥2048 only)
- Wall-clock: 1.66× (I=1024), 2.15× (I=2048), 2.37× (I=4096)

## Phase 5: Memory Investigation (Session 26) — commit `e5d3ca8`

- Eager FP8 weight cache eviction **proven counterproductive** (memory +2.6 GiB due to temp copies)
- Can't fix without architectural change to weight quantization

## Phase 6: Ernie Comparison + API Cleanup (Session 27) — uncommitted changes

### Three-way NSYS benchmark at Ernie production shape (T=8192, H=3072, I=1536, E=8, K=8):
| Config | NSYS GPU projection (µs/iter) | vs Sonic BF16 |
|--------|-------------------------------|---------------|
| Official Sonic BF16 | 3937.4 | baseline |
| Sonic FP8 frontier | 3786.6 | 1.04× faster |
| Ernie MOELayer BF16 | 15325.8 | 3.89× slower |

### FP8 overhead analysis at Ernie shape:
- `gather_quantize_and_pack` forward: ~96µs/iter (TK=65536 rows)
- `elementwise direct_copy`: ~156µs/iter × 10 instances = 1558µs/10 iters (unexplained)
- BF16 wgrad: ~1551µs combined (41% of total)

### Forward A_idx investigation (THE critical finding):
- Backward dgated successfully uses `quantize_and_pack_activation(dout)` + A_idx (T-sized, ~12µs)
- Forward attempted same: `quantize_and_pack_activation(x)` + A_idx → **93.5% RRMSE (garbage)**
- Root cause: `GemmWrapperBase.validate_and_prepare_tensors` sets M=A_idx.shape[0]=TK, but scale tensor has storage for T rows only → CUTLASS reads garbage scales beyond T
- User insight: "A和Scale需要同时被gather" — FP8 data and scales are row-correlated, both must be gathered
- Backward works because GemmDGated's scale indexing uses A_idx[j] for scale lookup; GemmGated apparently does not
- **This is the #1 optimization opportunity** — fixing it would save ~84µs/iter + eliminate TK-sized activation materialization

### API improvements:
- Added `MoE.forward(use_fp8=True)` parameter
- Added `enable_fp8()` context manager in `sonicmoe/functional/utils.py`
- Made `SONIC_MOE_FP8_FUSED_GATED` default to `"1"` (was `"0"`)
- Added `FP8AlignedContractTest` with 4 new tests (T=1024/8192, E=8, K=8) that actually exercise FP8 path
- 12/12 tests pass including production shape

---

## Lessons That Matter

1. **In FP8, data and scales are row-correlated.** A_idx must gather BOTH simultaneously. This is the key difference from BF16 where A_idx only gathers data.
2. **SonicMoE's core design is "no materialization of gathered activations."** The FP8 forward currently violates this by materializing TK-sized gathered FP8 data via `gather_quantize_and_pack_activation`. Fixing this aligns FP8 with the core design.
3. **Use official BF16 as the only baseline.** Exclude QuACK 0.3.7 `elementwise_kernel` from BF16 numbers.
4. **NSYS GPU projection ≠ wall-clock.** FP8 wins bigger in wall-clock (fewer kernel launches → less CPU overhead).
5. **FP8 advantage scales with GEMM size.** N=2×I for GemmGated → I=4096 is nearly 50% faster.
6. **FP8 wgrad viability depends on K_per_expert.** At E=128, K_per_expert=256 is too small. At E=8, K_per_expert=8192 — should be tested.
7. **Memory overhead is architectural.** 3 FP8 weight caches can't be fixed by eviction.
8. **The old shapes (T=4096, H=4096, E=128, K=8) are flattering.** At Ernie shape (T=8192, H=3072, I=1536, E=8, K=8), FP8 barely wins (1.04×). The gather overhead dominates at smaller I.
