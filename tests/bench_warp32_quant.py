"""32x32 single-warp transpose quant: minimal L1 pressure, no smem needed.

32x32 tile → only 32 output cache lines (vs 128 in original)
Single warp: no __syncthreads, no smem, pure register + warp shuffle
Each thread: 32 elements, amax is thread-local (no reduction needed)
Multiple groups per block to amortize launch overhead.
"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad,
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _storage_per_batch, _div_up,
)
import socket


@triton.jit
def _warp32x32_transpose_quant_kernel(
    src_ptr,
    gather_idx_ptr,
    dst_fp8_ptr,       # (E*dim, capacity) fp8
    dst_scales_ptr,    # (E, per_batch_storage) u8
    dim, capacity, per_batch_storage,
    src_stride_row, src_stride_col,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,       # 32
    BLOCK_DIM: tl.constexpr,        # 32 — matches GROUP_SIZE for square tile
    GROUPS_PER_BLOCK: tl.constexpr,  # process N groups per block (amortize overhead)
    SF_TILE_M: tl.constexpr,
    SF_TILE_K: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """32x32 tile, 1 warp, pure register transpose.

    Grid: (E * ceil(dim/32), ceil(capacity / (32 * GROUPS_PER_BLOCK)))
    Each block processes GROUPS_PER_BLOCK × (32 tokens × 32 dims) tiles.
    Only 32 cache lines per store (vs 128), 4x less L1 pressure.
    """
    pid_row = tl.program_id(0)
    pid_grp_block = tl.program_id(1)

    num_dim_blocks = tl.cdiv(dim, BLOCK_DIM)
    expert_id = pid_row // num_dim_blocks
    dim_block = pid_row % num_dim_blocks
    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)  # (32,)
    dim_mask = dim_offs < dim

    # ISA precompute (invariant across groups)
    k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    groups_per_k: tl.constexpr = SF_TILE_K // GROUP_SIZE
    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    out_rows = expert_id * dim + dim_offs

    for g in tl.range(0, GROUPS_PER_BLOCK):
        pid_group = pid_grp_block * GROUPS_PER_BLOCK + g
        cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)  # (32,)

        # Source rows
        flat_ids = expert_id * capacity + cap_offs
        if HAS_GATHER:
            src_rows = tl.load(gather_idx_ptr + flat_ids).to(tl.int64)
        else:
            src_rows = flat_ids.to(tl.int64)

        # Load (32, 32) bf16 tile — small, register-friendly
        src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=dim_mask[None, :], other=0.0).to(tl.float32)

        # Transpose (32, 32) → (32, 32) — square, efficient warp-level
        values_t = tl.trans(values)

        # E8M0 blockscaled quant: amax over 32 tokens per dim
        block_amax = tl.max(tl.abs(values_t), axis=1)  # (32,) — thread-local, no reduction!
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa > 0x600000, 1, 0)
        e8m0 = biased_exp - 8 + carry
        e8m0 = tl.where(biased_exp > 0, e8m0, 0)
        e8m0_byte = tl.maximum(e8m0, 0).to(tl.uint8)
        qexp = tl.maximum(tl.minimum(254 - e8m0, 254), 1)
        qscale = (qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values_t * qscale[:, None]).to(tl.float8e4nv)

        # Store fp8: 32 rows × 32 cols — 32 cache lines only
        out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        fp8_ptrs = dst_fp8_ptr + out_rows[:, None].to(tl.int64) * capacity + out_cols[None, :].to(tl.int64)
        tl.store(fp8_ptrs, quantized, mask=dim_mask[:, None])

        # ISA scale
        k_idx = pid_group // groups_per_k
        k_in = pid_group % groups_per_k
        tile_base = (row_tiles * k_tiles + k_idx) * SF_TILE_STORAGE
        isa_idx = tile_base + row_base + k_in
        scale_ptrs = dst_scales_ptr + expert_id.to(tl.int64) * per_batch_storage + isa_idx.to(tl.int64)
        tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


def warp32x32_transpose_quantize(flat_sorted, num_experts, capacity, dim, *, gather_idx=None):
    device = flat_sorted.device
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    BLOCK_DIM = 32
    GROUPS_PER_BLOCK = 8  # process 8 groups per block for amortization

    fp8_flat = torch.empty(num_experts * dim, capacity, dtype=torch.float8_e4m3fn, device=device)
    per_batch = _storage_per_batch(dim, capacity)
    packed_scales = torch.ones(num_experts, per_batch, dtype=torch.float8_e8m0fnu, device=device)

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else flat_sorted

    total_groups = capacity // GROUP_SIZE
    grid = (num_experts * _div_up(dim, BLOCK_DIM), _div_up(total_groups, GROUPS_PER_BLOCK))

    _warp32x32_transpose_quant_kernel[grid](
        flat_sorted, gather_ptr,
        fp8_flat, packed_scales.view(torch.uint8),
        dim, capacity, per_batch,
        flat_sorted.stride(0), flat_sorted.stride(1),
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        num_warps=1,  # Single warp!
    )
    return fp8_flat.reshape(num_experts, dim, capacity), packed_scales


if __name__ == "__main__":
    E, H = 8, 3072; TK = 65536; CAP = TK // E
    torch.manual_seed(42)
    dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
    x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()

    print(f"Host: {socket.gethostname()}")
    print(f"Shape: TK={TK}, H={H}, E={E}, CAP={CAP}")
    print("=" * 70)

    ref_fp8, ref_scales = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    try:
        opt_fp8, opt_scales = warp32x32_transpose_quantize(dout, E, CAP, H, gather_idx=x_idx)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    fp8_match = (ref_fp8.view(torch.uint8) == opt_fp8.view(torch.uint8)).float().mean().item()
    sc_match = (ref_scales.view(torch.uint8) == opt_scales.view(torch.uint8)).float().mean().item()
    print(f"FP8 match:    {fp8_match*100:.2f}%")
    print(f"Scales match: {sc_match*100:.2f}%")
    print(f"PRECISION: {'PASS' if fp8_match > 0.99 and sc_match > 0.99 else 'FAIL'}")

    WARMUP, ITERS, TRIALS = 5, 20, 5
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
    t_ref = bench(lambda: fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx),
                  "Original (32x128, 4 warps)")

    # Try different GROUPS_PER_BLOCK values
    for gpb in [4, 8, 16, 32]:
        def make_fn(g=gpb):
            def fn():
                device = dout.device
                fp8 = torch.empty(E * H, CAP, dtype=torch.float8_e4m3fn, device=device)
                pb = _storage_per_batch(H, CAP)
                sc = torch.ones(E, pb, dtype=torch.float8_e8m0fnu, device=device)
                total_groups = CAP // 32
                grid = (E * _div_up(H, 32), _div_up(total_groups, g))
                _warp32x32_transpose_quant_kernel[grid](
                    dout, x_idx, fp8, sc.view(torch.uint8),
                    H, CAP, pb, dout.stride(0), dout.stride(1),
                    HAS_GATHER=True, GROUP_SIZE=32, BLOCK_DIM=32,
                    GROUPS_PER_BLOCK=g,
                    SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K, SF_TILE_STORAGE=_SF_TILE_STORAGE,
                    num_warps=1)
            return fn
        bench(make_fn(gpb), f"32x32 1-warp GPB={gpb}")

    print(f"\n  Best vs Original: see above")
