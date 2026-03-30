# Blockscaled FP8 Varlen GEMM — Performance Optimization Log

> **Status: ABANDONED** — Triton `tl.dot_scaled` approach was abandoned because it is broken on SM100a + Triton 3.5.1.
> The production implementation uses CUTLASS `GemmDefaultSm100` via QuACK instead.
> See `reports/fp8_upgrade/HANDOFF.md` for current architecture.

## Historical Context

This file documents early attempts to use Triton's native `tl.dot_scaled` for blockscaled FP8 GEMM on Blackwell.
The approach was abandoned due to:

1. **`tl.dot_scaled` broken on SM100a** — Triton 3.5.1 generates incorrect code for SM100 blockscaled MMA
2. **Performance gap** — Even when working, Triton kernels could not match CUTLASS GEMM throughput for varlen batched workloads
3. **ISA packing complexity** — Scale factor packing for `tcgen05.mma` is complex (SF_TILE_M=128, SF_TILE_K=128, SF_VEC_SIZE=32) and poorly supported in Triton

## Baseline Measurements (historical, for reference)

| Version | Time (ms) | TFLOPS | vs BF16 | Notes |
|---------|-----------|--------|---------|-------|
| BF16 CUTLASS | 1.054 | 522 | 1.00x | `quack.gemm_interface.gemm` varlen |
| V1 original (E,N,K) | 3.182 | 173 | 0.33x | tl.dot_scaled, 128×128×128, 106 regs |
| V2 maxnreg=64 | 1.995 | 276 | 0.53x | Same tile, forced register cap |
| V3 (E,K,N) layout | 2.161 | 254 | 0.49x | B transposed for coalesced N loads |

**Lesson**: Native CUTLASS/QuACK GEMM is the only viable path for production blockscaled FP8 on SM100a until Triton fixes `tl.dot_scaled`.
