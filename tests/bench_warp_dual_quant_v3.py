"""Warp dual-quant v3: register-optimized, no unroll, high occupancy target.

Changes vs v2:
- loop_unroll_factor=1 on GROUPS_PER_BLOCK loop (was auto-unrolled → 2x regs)
- Inline E8M0 (no function call overhead → compiler can reuse regs better)
- Try num_warps=1/2/4 with GPB=2/4/8 matrix to find optimal occupancy
- Target: regs < 48, IPC → 4, waves/SM > 20
"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad, quantize_and_pack_activation,
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _storage_per_batch, _div_up,
)
import socket


@triton.jit
def _warp_dual_quant_v3(
    src_ptr, gather_idx_ptr,
    row_fp8_ptr, row_scales_ptr,
    col_fp8_ptr, col_scales_ptr,
    T, H, capacity, col_per_batch,
    src_stride_row, src_stride_col,
    row_k_tiles,
    HAS_GATHER: tl.constexpr,
    GS: tl.constexpr,
    SF_TILE_M: tl.constexpr, SF_TILE_K: tl.constexpr, SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    pid_ed = tl.program_id(0)
    pid_gb = tl.program_id(1)
    num_db = tl.cdiv(H, GS)
    expert_id = pid_ed // num_db
    dim_block = pid_ed % num_db
    dim_offs = dim_block * GS + tl.arange(0, GS)
    dim_mask = dim_offs < H

    # Col ISA precompute (reused across loop — kept in regs)
    ck: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    cgk: tl.constexpr = SF_TILE_K // GS
    crt = dim_offs // SF_TILE_M
    cri = dim_offs % SF_TILE_M
    crb = (cri % 32) * 16 + (cri // 32) * 4
    cor = expert_id * H + dim_offs

    # Row ISA constants (dim_block is constant across loop)
    rki = dim_block // (SF_TILE_K // GS)
    rkn = dim_block % (SF_TILE_K // GS)

    # NO UNROLL: loop_unroll_factor=1 prevents register doubling from unroll
    for gb in tl.range(0, GROUPS_PER_BLOCK, loop_unroll_factor=1):
        pg = pid_gb * GROUPS_PER_BLOCK + gb
        co = pg * GS + tl.arange(0, GS)
        fids = expert_id * capacity + co
        if HAS_GATHER:
            sr = tl.load(gather_idx_ptr + fids).to(tl.int64)
        else:
            sr = fids.to(tl.int64)

        # Load (32, 32) — single HBM read per group
        ptrs = src_ptr + sr[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        v = tl.load(ptrs, mask=dim_mask[None, :], other=0.0).to(tl.float32)
        av = tl.abs(v)

        # ── Row quant: inline E8M0, reuse registers ──
        ra = tl.max(av, axis=1)  # (32,)
        ra_bits = ra.to(tl.int32, bitcast=True)
        ra_exp = (ra_bits >> 23) & 0xFF
        ra_carry = tl.where((ra_bits & 0x7FFFFF) > 0x600000, 1, 0)
        ra_e = ra_exp - 8 + ra_carry
        ra_e = tl.where(ra_exp > 0, ra_e, 0)
        ra_e = tl.maximum(ra_e, 0)
        ra_byte = ra_e.to(tl.uint8)
        ra_qe = tl.maximum(tl.minimum(254 - ra_e, 254), 1)
        ra_qs = (ra_qe.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        rq = (v * ra_qs[:, None]).to(tl.float8e4nv)
        tl.store(row_fp8_ptr + sr[:, None] * H + dim_offs[None, :], rq, mask=dim_mask[None, :])

        # Row ISA
        srt = sr // SF_TILE_M
        sri = sr % SF_TILE_M
        srb = (sri % 32) * 16 + (sri // 32) * 4
        rtb = (srt * row_k_tiles + rki) * SF_TILE_STORAGE
        tl.store(row_scales_ptr + rtb + srb + rkn, ra_byte)

        # ── Col quant: inline E8M0 ──
        ca = tl.max(av, axis=0)  # (32,)
        ca_bits = ca.to(tl.int32, bitcast=True)
        ca_exp = (ca_bits >> 23) & 0xFF
        ca_carry = tl.where((ca_bits & 0x7FFFFF) > 0x600000, 1, 0)
        ca_e = ca_exp - 8 + ca_carry
        ca_e = tl.where(ca_exp > 0, ca_e, 0)
        ca_e = tl.maximum(ca_e, 0)
        ca_byte = ca_e.to(tl.uint8)
        ca_qe = tl.maximum(tl.minimum(254 - ca_e, 254), 1)
        ca_qs = (ca_qe.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        cq = (v * ca_qs[None, :]).to(tl.float8e4nv)
        ct = tl.trans(cq)
        tl.store(col_fp8_ptr + cor[:, None].to(tl.int64) * capacity + co[None, :].to(tl.int64),
                 ct, mask=dim_mask[:, None])

        # Col ISA
        cki = pg // cgk
        ckn = pg % cgk
        ctb = (crt * ck + cki) * SF_TILE_STORAGE
        tl.store(col_scales_ptr + expert_id.to(tl.int64) * col_per_batch + (ctb + crb + ckn).to(tl.int64),
                 ca_byte, mask=dim_mask)


def warp_dual_quant_v3(src, num_experts, capacity, *, gather_idx=None, num_warps=1, gpb=4):
    src = src.contiguous()
    T, H = src.shape
    device = src.device
    GS = _SF_VEC_SIZE

    row_fp8 = torch.empty(T, H, dtype=torch.float8_e4m3fn, device=device)
    rpb = _storage_per_batch(T, H)
    rsc = torch.full((1, rpb), 127, dtype=torch.uint8, device=device)

    col_fp8 = torch.empty(num_experts * H, capacity, dtype=torch.float8_e4m3fn, device=device)
    cpb = _storage_per_batch(H, capacity)
    csc = torch.ones(num_experts, cpb, dtype=torch.uint8, device=device)

    hg = gather_idx is not None
    gp = gather_idx if hg else src
    grid = (num_experts * _div_up(H, GS), _div_up(capacity // GS, gpb))

    _warp_dual_quant_v3[grid](
        src, gp, row_fp8, rsc, col_fp8, csc,
        T, H, capacity, cpb,
        src.stride(0), src.stride(1), _div_up(H, _SF_TILE_K),
        HAS_GATHER=hg, GS=GS,
        SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_BLOCK=gpb, num_warps=num_warps,
    )
    return (row_fp8, rsc.view(torch.float8_e8m0fnu),
            col_fp8.reshape(num_experts, H, capacity), csc.view(torch.float8_e8m0fnu))


if __name__ == "__main__":
    E, H = 8, 3072; TK = 65536; CAP = TK // E
    torch.manual_seed(42)
    dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
    x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()

    print(f"Host: {socket.gethostname()}")
    print(f"TK={TK} H={H} E={E} CAP={CAP}")
    print("=" * 60)

    # Reference
    rr, rs = quantize_and_pack_activation(dout)
    rc, cs = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    # Verify v3
    r1, s1, c1, s2 = warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, num_warps=1, gpb=4)
    torch.cuda.synchronize()
    rm = (rr.view(torch.uint8) == r1.view(torch.uint8)).float().mean().item()
    cm = (rc.view(torch.uint8) == c1.view(torch.uint8)).float().mean().item()
    print(f"Precision: row {rm*100:.1f}%  col {cm*100:.1f}%")

    W, I, TR = 5, 10, 5
    def bench(fn, name):
        for _ in range(W): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(TR):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(I): fn()
            e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / I)
        print(f"  {name:<55} min={min(ts):>7.0f}us")
        return min(ts)

    print("\n--- Sweep num_warps × GPB ---")
    t_ref = bench(lambda: (quantize_and_pack_activation(dout),
                           fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)),
                  "Separate baseline (row+col)")

    best_t, best_cfg = 1e9, ""
    for nw in [1, 2, 4]:
        for gpb in [2, 4, 8]:
            name = f"v3 nw={nw} gpb={gpb}"
            t = bench(lambda nw=nw, gpb=gpb: warp_dual_quant_v3(
                dout, E, CAP, gather_idx=x_idx, num_warps=nw, gpb=gpb), name)
            if t < best_t:
                best_t, best_cfg = t, name

    print(f"\n  Best: {best_cfg} = {best_t:.0f}us (vs separate {t_ref:.0f}us, {t_ref/best_t:.2f}x)")
