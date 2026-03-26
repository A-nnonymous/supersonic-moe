# Blockscaled FP8 CTA-Tile Alignment: Why It Exists and Why It Doesn't Matter

## TL;DR

Blockscaled FP8 GEMM on Blackwell requires each expert's M-dimension segment to be
a multiple of **128**. This is a hardware ISA constraint from the `tcgen05.mma`
instruction's scale-factor atom layout — not a software design choice.

**This constraint is architecturally irrelevant in production** because SonicMoE's
**token rounding routing** (paper Section 5, Algorithm 4) already guarantees that
every expert receives a 128-aligned token count. The padding infrastructure in
`blockscaled_fp8_gemm_varlen` exists only as a safety net for non-rounded routing
(tests, debugging, research benchmarks).

Measured impact (production shape T=8192, H=4096, I=1024, E=128, K=8):

| GEMM Path | Token-rounded (no pad) | Vanilla top-K (padded) | Overhead |
|-----------|----------------------|----------------------|----------|
| Down-proj (TK,I)→(TK,H) | **2.9 ms** | 5.7 ms | +96% |
| Up-proj (TK,H)→(TK,2I) | **3.9 ms** | 7.7 ms | +97% |

With token rounding, blockscaled FP8 adds **zero padding overhead**.

---

## 1. The Hardware Constraint

### BF16: Dynamic Partial-Tile Handling

BF16 GEMM on Blackwell uses the `VarlenMTileScheduler` (QuACK `tile_scheduler.py`).
The scheduler dynamically assigns CTA tiles based on `cu_seqlens_m` expert boundaries.
When a segment doesn't fill the full CTA tile (e.g., 100 tokens with tile_m=128):

- The scheduler generates a **partial CTA tile** with only the actual row count.
- The CUTLASS mainloop processes only the valid rows.
- There is **no requirement** that expert segments align to any tile size.

This works because BF16 GEMM operates on **raw activation tensors** with no
pre-packed metadata structure that depends on the M-dimension tiling.

### Blockscaled FP8: Fixed Scale-Factor Atom Layout

Blockscaled FP8 on Blackwell uses the `tcgen05.mma` instruction with hardware
descaling via 1×32 UE8M0 scale factors. The scale factor layout is defined by the
Blackwell ISA (not CUTLASS):

```
Scale factor atom: 32 rows × 4 columns (from tcgen05 MMA spec)
Tile packing: 4 atoms stacked → 128 rows × 128 columns per tile
Storage: _SF_TILE_STORAGE = 128 × (128 / 32) = 512 bytes per tile
```

The `_scale_pack_index` function in `blockscaled_fp8_gemm.py` (line 169) encodes
this layout:

```python
row_tiles = row_ids // _SF_TILE_M        # Which 128-row tile
row_in_tile = row_ids % _SF_TILE_M       # Position within tile
tile_base = (row_tiles * k_tiles + k_tiles_idx) * _SF_TILE_STORAGE
row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
```

**This packing assumes complete 128-row tiles.** If an expert segment has, say,
100 rows, the scale factor tensor still occupies space for 128 rows, and the
activation tensor must be padded to match. There is no partial-tile mode for
blockscaled scale factors in the current hardware.

### Why BF16 Can But Blockscaled Cannot

| Aspect | BF16 | Blockscaled FP8 |
|--------|------|-----------------|
| Scale factors | None | Pre-packed 1×32 UE8M0 atoms |
| Tile scheduler | VarlenMTileScheduler (dynamic partial tiles) | Same scheduler, but scale layout requires full tiles |
| M-alignment | None required | Must be multiple of 128 (4 × 32-atom) |
| Root cause | Data is self-describing | Scale factors are pre-packed with fixed geometry |

The constraint originates from the **ISA-level scale factor atom shape** (32×4),
not from CUTLASS or QuACK software. It cannot be removed without hardware changes.

---

## 2. The Architectural Solution: Token Rounding

SonicMoE paper (Section 5) introduces **token rounding routing** — a drop-in
replacement for vanilla top-K that rounds per-expert token counts to multiples of
M_tile (128):

> "We propose to use token rounding to avoid launching such extra tiles, thereby
> leading to more efficient training."  — SonicMoE paper

**Algorithm 4** (paper): For each expert, the router either:
- **Pads** by adding low-score tokens from the expert-choice pool, OR
- **Drops** the lowest-score tokens

...to reach the nearest 128-multiple. The maximum deviation from vanilla top-K
is **one tile (128 tokens) per expert**.

When token rounding is active:
- `expert_frequency_offset` segments are all multiples of 128
- `_get_padding_plan()` returns `needs_pad=False`
- **All scatter/gather/caching overhead is eliminated**

---

## 3. Current Benchmark Discrepancy

The primary benchmark (`benchmarks/moe-cute.py`) uses `moe_TC_softmax_topk_layer`
with **vanilla top-K routing** (no token rounding). This causes every expert segment
to be misaligned, triggering padding on every GEMM call.

The token rounding benchmark exists at `benchmarks/moe-token-rounding.py` but is
not used for FP8 performance evaluation.

**Recommendation**: FP8 benchmarks should use token-rounded routing to reflect
production performance. The vanilla top-K numbers overstate FP8 overhead by ~2×.

---

## 4. What the Padding Safety Net Does (When Needed)

For non-rounded routing or edge cases, `blockscaled_fp8_gemm_varlen` (line 862)
handles alignment automatically:

1. **Detect**: Compute `seg_lens % 128` for each expert segment
2. **Pad**: Create zero-filled buffer at next 128-multiple, scatter real tokens
3. **Compute**: Recursive call with aligned `cu_seqlens_m`
4. **Extract**: Gather real outputs from padded result

This is cached per content of `cu_seqlens_m` to avoid recomputation.

The overhead is ~3 ms per GEMM call (dominated by scatter/gather of large
activation matrices, not the cache lookup). With 4 blockscaled GEMMs per E2E,
this adds ~12 ms — significant but **only when token rounding is not used**.

---

## 5. Remaining FP8 Overhead (With Token Rounding)

With alignment eliminated, the only overhead vs BF16 is **activation quantization**:

| Step | Time | Notes |
|------|------|-------|
| `quantize_activation_blockscaled_fast()` | ~1.8 ms | Triton kernel: bf16 → fp8 e4m3fn + 1×32 scales |
| `pack_blockscaled_1x32_scales()` | ~0.5 ms | Repack scales to ISA atom layout |
| Weight quantization | ~0 ms | Cached after first call |
| CUTLASS blockscaled kernel | ~0.15 ms | Actual GEMM execution |

Total overhead per GEMM call: **~2.5 ms** (vs BF16 ~2 ms per GEMM).
This is the irreducible cost of blockscaled FP8 quantization.

Potential further optimizations:
- Fuse quantization into GEMM prologue (requires CUTLASS kernel modification)
- Use TMA-aware quantization layout (avoids scale repacking)
- Pre-quantize recurring activations (e.g., in inference with static routing)
