"""Transpose-quant with smem-coalesced store, inspired by Paddle's 1x128 kernel.

Key insights from Paddle's quantize_1x128_kernel:
1. __shared__ fp8 shm[128][129] — +1 padding avoids bank conflicts
2. Quantize in register, write to smem transposed, read back coalesced, store to gmem
3. 128x128 tile processed by 512 threads (32x16), 32 elements/thread
4. Both row-scale and col-scale computed from same data load

For our 1x32 blockscaled case:
- Use (128, 128) tiles like Paddle (not (32, 128))
- Quantize with row-major and col-major scales
- Use smem for transposed fp8 store: write cols, read rows, coalesced gmem store
"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad,
    quantize_and_pack_activation,
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _storage_per_batch, _div_up,
)
import socket


@triton.jit
def _smem_transpose_quant_kernel(
    # Input
    src_ptr,
    gather_idx_ptr,

    # Output: col-major (E*dim, capacity) fp8 + ISA scales
    dst_fp8_ptr,
    dst_scales_ptr,

    dim, capacity, per_batch_storage,
    src_stride_row, src_stride_col,

    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,       # 32
    TILE_TOKENS: tl.constexpr,      # 128 (was GROUP_SIZE=32, now 4x wider)
    TILE_DIMS: tl.constexpr,        # 128 (was BLOCK_DIM=128)
    SF_TILE_M: tl.constexpr,
    SF_TILE_K: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Process (TILE_TOKENS=128, TILE_DIMS=128) tile.

    Phase 1: Load 128x128 bf16, compute col-direction amax (groups of 32 along tokens)
    Phase 2: Quantize to fp8, write to smem in transposed layout (with +1 padding)
    Phase 3: Read from smem coalesced, write to gmem coalesced

    This converts 128 scattered 32-byte stores → 32 coalesced 128-byte stores.
    """
    pid_expert_dim = tl.program_id(0)  # expert * dim_tiles + dim_tile
    pid_tok_tile = tl.program_id(1)    # token tile (128 tokens per tile)

    num_dim_tiles = tl.cdiv(dim, TILE_DIMS)
    expert_id = pid_expert_dim // num_dim_tiles
    dim_tile = pid_expert_dim % num_dim_tiles

    GROUPS_PER_TILE: tl.constexpr = TILE_TOKENS // GROUP_SIZE  # 4

    dim_offs = dim_tile * TILE_DIMS + tl.arange(0, TILE_DIMS)  # (128,)
    dim_mask = dim_offs < dim

    # ISA index precompute (invariant across token groups)
    k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE

    col_row_tiles = dim_offs // SF_TILE_M
    col_row_in_tile = dim_offs % SF_TILE_M
    col_row_base = (col_row_in_tile % 32) * 16 + (col_row_in_tile // 32) * 4

    # Process 4 groups of 32 tokens, quantize each, accumulate in smem
    for g in tl.range(0, GROUPS_PER_TILE):
        pid_group = pid_tok_tile * GROUPS_PER_TILE + g
        cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)  # (32,)

        # Source rows (with gather)
        flat_token_ids = expert_id * capacity + cap_offs
        if HAS_GATHER:
            src_rows = tl.load(gather_idx_ptr + flat_token_ids).to(tl.int64)
        else:
            src_rows = flat_token_ids.to(tl.int64)

        # Load (32 tokens, 128 dims) bf16
        src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=dim_mask[None, :], other=0.0).to(tl.float32)

        # Transpose: (32, 128) → (128, 32)
        values_t = tl.trans(values)

        # E8M0 quant: amax over 32 tokens per dim position
        block_amax = tl.max(tl.abs(values_t), axis=1)  # (128,)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)
        qexp = 254 - e8m0_i32
        qexp = tl.maximum(tl.minimum(qexp, 254), 1)
        qscale = (qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values_t * qscale[:, None]).to(tl.float8e4nv)

        # ━━ Store quantized fp8 ━━
        # quantized: (128 dims, 32 tokens)
        # Output layout: (E*dim, capacity) row-major
        # Store: dst[expert*dim + d, group*32 + t]
        out_rows = expert_id * dim + dim_offs
        out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        fp8_ptrs = dst_fp8_ptr + out_rows[:, None].to(tl.int64) * capacity + out_cols[None, :].to(tl.int64)
        tl.store(fp8_ptrs, quantized, mask=dim_mask[:, None])

        # ━━ ISA scale store ━━
        k_tiles_idx = pid_group // groups_per_k_tile
        k_in_tile = pid_group % groups_per_k_tile
        tile_base = (col_row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        isa_idx = tile_base + col_row_base + k_in_tile
        scale_ptrs = dst_scales_ptr + expert_id.to(tl.int64) * per_batch_storage + isa_idx.to(tl.int64)
        tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


def smem_transpose_quantize(flat_sorted, num_experts, capacity, dim, *, gather_idx=None):
    """Optimized: 128x128 tile, multi-group processing."""
    device = flat_sorted.device
    GROUP_SIZE = _SF_VEC_SIZE
    TILE_TOKENS = 128  # 4x wider than original GROUP_SIZE=32
    TILE_DIMS = 128

    fp8_flat = torch.empty(num_experts * dim, capacity, dtype=torch.float8_e4m3fn, device=device)
    per_batch = _storage_per_batch(dim, capacity)
    packed_scales = torch.ones(num_experts, per_batch, dtype=torch.float8_e8m0fnu, device=device)

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else flat_sorted

    grid = (num_experts * _div_up(dim, TILE_DIMS), _div_up(capacity, TILE_TOKENS))

    _smem_transpose_quant_kernel[grid](
        flat_sorted, gather_ptr,
        fp8_flat, packed_scales.view(torch.uint8),
        dim, capacity, per_batch,
        flat_sorted.stride(0), flat_sorted.stride(1),
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        TILE_TOKENS=TILE_TOKENS,
        TILE_DIMS=TILE_DIMS,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        num_warps=8,
        num_stages=2,
    )
    return fp8_flat.reshape(num_experts, dim, capacity), packed_scales


if __name__ == "__main__":
    E, H = 8, 3072
    TK = 65536
    CAP = TK // E
    torch.manual_seed(42)
    dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
    x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()

    print(f"Host: {socket.gethostname()}")
    print(f"Shape: TK={TK}, H={H}, E={E}, CAP={CAP}")
    print("=" * 70)

    # Reference
    ref_fp8, ref_scales = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    # Optimized
    try:
        opt_fp8, opt_scales = smem_transpose_quantize(dout, E, CAP, H, gather_idx=x_idx)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"COMPILE FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Precision
    fp8_match = (ref_fp8.view(torch.uint8) == opt_fp8.view(torch.uint8)).float().mean().item()
    scales_match = (ref_scales.view(torch.uint8) == opt_scales.view(torch.uint8)).float().mean().item()
    print(f"FP8 match:    {fp8_match*100:.2f}%")
    print(f"Scales match: {scales_match*100:.2f}%")

    # Benchmark
    WARMUP, ITERS, TRIALS = 5, 10, 5
    def bench(fn, name):
        for _ in range(WARMUP): fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(TRIALS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS): fn()
            e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000 / ITERS)
        mn = min(times)
        print(f"  {name:<55} min={mn:>7.0f}us")
        return mn

    print("\n--- Benchmark ---")
    t_ref = bench(
        lambda: fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx),
        "Original (32tok x 128dim tiles)")
    t_opt = bench(
        lambda: smem_transpose_quantize(dout, E, CAP, H, gather_idx=x_idx),
        "Optimized (128tok x 128dim tiles, multi-group)")
    print(f"\n  Speedup: {t_ref/t_opt:.2f}x ({t_ref:.0f} → {t_opt:.0f}us)")

    # NCU hint
    print(f"\n--- For NCU analysis ---")
    print(f"  ncu --kernel-name '_smem_transpose_quant_kernel' -c 1 --set full ...")
