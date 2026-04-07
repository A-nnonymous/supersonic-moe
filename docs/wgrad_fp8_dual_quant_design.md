# Wgrad FP8 Optimization — Zero-Copy Dual Quantization Design

> Date: 2026-04-07
> Branch: `native-fp8-exploration`
> Status: Design approved, implementation pending

---

## 1. Problem

FP8 wgrad is 1.49x slower than BF16 (696µs vs 467µs) due to data preparation overhead.
The pure FP8 GEMM (277µs) is already 1.68x faster than BF16.

Measured breakdown:
```
fused_transpose_quant dout: 277µs (39.8%)  ← reads dout AGAIN
fused_transpose_quant y1s:  142µs (20.4%)
FP8 GEMM:                   277µs (39.8%)
Total:                       696µs
```

The root cause: `dout` is read from HBM twice (once for actgrad row-quant, once for wgrad col-quant).

---

## 2. Solution: Dual-Quantization Triton Kernel

### Concept

Read `dout` once from HBM, produce BOTH quantizations in one kernel:

```
dout (T, H) bf16  →  1 HBM read  →  {
  dout_fp8_row (T, H) fp8 + row_scales (ISA)    for actgrad (groups along H)
  dout_fp8_col (E, H, cap) fp8 + col_scales (ISA)  for wgrad (groups along cap/TK)
}
```

### Why the quantizations differ

Blockscaled FP8 (1×32) computes one E8M0 scale per 32 consecutive elements.
- Row-major (T, H): groups along H (contiguous dim) — correct for actgrad K=H
- Col-major (H, TK): groups along TK (after transpose) — correct for wgrad K=TK

The scale values are different because the group axes are different.
The fp8 values are also different (scaled by different factors).

### Kernel Design

```python
@triton.jit
def _dual_quantize_and_pack_kernel(
    # Inputs
    src_ptr,              # (T, H) bf16
    gather_idx_ptr,       # (TK,) int32 — T→TK scatter

    # Output 1: row-major for actgrad
    row_fp8_ptr,          # (T, H) fp8
    row_scales_ptr,       # (1, row_packed_size) ISA-packed

    # Output 2: col-major (transposed) for wgrad
    col_fp8_ptr,          # (E*H, capacity) fp8
    col_scales_ptr,       # (E, col_packed_size) ISA-packed

    # Grid: 2D — (M_blocks, K_groups)
    # Each block processes (BLOCK_ROWS, GROUP_K) tile
):
    # 1. Load (BLOCK_ROWS, GROUP_K) bf16 tile from src
    # 2. Compute row-major blockscaled quant (groups along K=H)
    #    → write to row_fp8_ptr + row_scales_ptr (ISA)
    # 3. Transpose tile in registers: (BLOCK_ROWS, GROUP_K) → (GROUP_K, BLOCK_ROWS)
    # 4. Compute col-major blockscaled quant (groups along M=TK)
    #    → write to col_fp8_ptr + col_scales_ptr (ISA) at transposed positions
```

### Bandwidth Analysis

```
Read:  T × H × 2 bytes = 403 MB  (one read, shared by both quants)
Write row: T × H × 1 + ISA_scales ≈ 207 MB
Write col: E × H × cap × 1 + ISA_scales ≈ 207 MB
Total I/O: 817 MB at 8 TB/s = ~102µs
```

vs current 117µs (row) + 277µs (col) = 394µs → **~3.9x reduction**

### Complexity: Medium

The kernel is essentially a merge of `_quantize_and_pack_kernel` (row path) and
`_fused_transpose_quantize_kernel` (col path) sharing the same input load.

Key challenges:
1. Different tile shapes (row: BLOCK_ROWS×GROUP_K, col: GROUP_SIZE×BLOCK_DIM after transpose)
2. Different ISA index computation (different M, K dimensions)
3. Gather index handling (only needed for col path)
4. Alignment: row path indexes by (T, H), col path by (E, H, capacity)

---

## 3. y1s Epilogue Quant (Future, Higher Complexity)

GemmDGated's epilogue already computes y1s values in registers. A transposed
blockscaled quant EpiOp could write (I, TK) fp8 + scales directly, eliminating
the 142µs `fused_transpose_quantize(y1s)`.

This requires:
- New EpiOp: `TransposedBlockscaledQuantStore`
- Transposed gmem write from epilogue (needs smem-based tile transpose)
- ISA scale packing inline (like `BlockscaledScaleStore` but transposed)

Estimated savings: 142µs → 0µs (compute-hidden, like Phase 3.1 dequant)

---

## 4. Expected Results

| Configuration | Time | vs BF16 467µs |
|--------------|------|---------------|
| Current FP8 wgrad | 696µs | 1.49x slower |
| + Dual-quant (A) | ~382µs | **1.22x faster** |
| + Epilogue y1s (A+B) | ~282µs | **1.66x faster** |

Phase A alone makes FP8 wgrad profitable.
Phase A+B approaches the theoretical minimum (pure GEMM 277µs).
