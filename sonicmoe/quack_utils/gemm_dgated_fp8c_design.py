# Phase 3.1 TMA-based FP8 C Load — Design Document
#
# Goal: Load fp8 z (PreAct) via TMA instead of LDG, eliminating 384MB z_bf16 temp.
#
# Architecture:
#   Standard path:  C = z_bf16.view(f32) → TMA(f32) → smem(f32) → reg(f32) → recast(bf16x2) → dSwiGLU
#   FP8 path:       C = z_fp8           → TMA(fp8) → smem(fp8) → reg(fp8) → cvt(f32) → dequant → dSwiGLU
#
# Key Changes:
#   1. c_dtype = Float8E4M3FN (not Float32)
#   2. C tensor shape = (TK, 2I) fp8 (not (TK, I) f32)
#   3. epi_c_tile_n = 2 * epi_tile_n (fp8 has 2x elements per byte-equivalent)
#   4. smem layout for fp8 C (half the bytes)
#   5. smem→register copy: fp8 load + fp8→f32 vectorized conversion
#   6. epi_visit_subtile: dequant (scale multiply) + existing dSwiGLU
#
# The C and D have DIFFERENT physical N dimensions:
#   D: (tile_M, tile_N) f32 = tile_M × tile_N × 4 bytes
#   C: (tile_M, 2*tile_N) fp8 = tile_M × 2*tile_N × 1 bytes = tile_M × tile_N × 2 bytes
#   C is HALF the bytes of D — TMA loads less data = better bandwidth utilization!
#
# Implementation Plan:
#   1. Override _setup_attributes: custom epi_c_smem_layout for fp8
#   2. Override __call__: create fp8 TMA atom with custom epi_tile for C
#   3. Override epilogue: custom smem→register copy (fp8 load + convert)
#   4. Override epi_visit_subtile: skip recast, apply dequant, existing dSwiGLU
#
# The epilogue loop iterates over epi_tiles. Each iteration:
#   - loads D accumulator subtile (f32, tile_M × epi_tile_N)
#   - loads C subtile via TMA (fp8, tile_M × 2*epi_tile_N) → smem → register
#   - epi_visit_subtile processes both
#
# TMA for C loads 2*epi_tile_N fp8 elements per row, which is epi_tile_N bf16 pairs.
# This exactly matches the N-dimension of D (epi_tile_N f32 = epi_tile_N bf16x2 pairs).
