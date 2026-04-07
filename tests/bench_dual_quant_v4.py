"""Dual-quant v4: bf16 data retention + register reuse + fused scale scatter.

Key optimizations vs v3:
1. Keep data as bf16 (16 regs/thread vs 32 for f32) — only convert on-the-fly
2. Two-phase: row quant first (free row regs), then col quant (reuse freed regs)
3. Fuse TK-space row ISA scale write (eliminates separate _gather_isa_packed_scales_kernel)
4. maxnreg=64 for optimal occupancy
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
def _warp_dual_quant_v4(
    src_ptr, gather_idx_ptr,
    row_fp8_ptr, row_scales_ptr,
    col_fp8_ptr, col_scales_ptr,
    # Optional: TK-space row scales (eliminates separate scatter kernel)
    row_tk_scales_ptr,
    T, H, capacity, col_per_batch, tk_row_per_batch,
    src_stride_row, src_stride_col,
    row_k_tiles, tk_k_tiles,
    HAS_GATHER: tl.constexpr,
    HAS_TK_SCALES: tl.constexpr,
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

    # Col ISA precompute (reusable, small)
    ck: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    cgk: tl.constexpr = SF_TILE_K // GS
    crt = dim_offs // SF_TILE_M
    cri = dim_offs % SF_TILE_M
    crb = (cri % 32) * 16 + (cri // 32) * 4
    cor = expert_id * H + dim_offs

    # Row ISA constants
    rki = dim_block // (SF_TILE_K // GS)
    rkn = dim_block % (SF_TILE_K // GS)

    for gb in tl.range(0, GROUPS_PER_BLOCK, loop_unroll_factor=1):
        pg = pid_gb * GROUPS_PER_BLOCK + gb
        co = pg * GS + tl.arange(0, GS)
        fids = expert_id * capacity + co
        if HAS_GATHER:
            sr = tl.load(gather_idx_ptr + fids).to(tl.int64)
        else:
            sr = fids.to(tl.int64)

        # ── Load (32, 32) as BF16 — half the register cost vs f32 ──
        ptrs = src_ptr + sr[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        v = tl.load(ptrs, mask=dim_mask[None, :], other=0.0)  # stays bf16!

        # ════════ Phase 1: Row quant (then free row-specific regs) ════════
        # Convert bf16→f32 ON THE FLY for amax only
        row_amax = tl.max(tl.abs(v.to(tl.float32)), axis=1)  # (32,) f32 temporary

        # Inline E8M0 — compute scale from amax
        rb = row_amax.to(tl.int32, bitcast=True)
        re = (rb >> 23) & 0xFF
        rc = tl.where((rb & 0x7FFFFF) > 0x600000, 1, 0)
        re = re - 8 + rc
        re = tl.maximum(tl.where(((rb >> 23) & 0xFF) > 0, re, 0), 0)
        r_byte = re.to(tl.uint8)
        rq_exp = tl.maximum(tl.minimum(254 - re, 254), 1)
        r_scale = (rq_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        # Quantize: bf16 → f32 → multiply → fp8 (temporary f32, not stored)
        rq = (v.to(tl.float32) * r_scale[:, None]).to(tl.float8e4nv)
        tl.store(row_fp8_ptr + sr[:, None] * H + dim_offs[None, :], rq, mask=dim_mask[None, :])

        # T-space row ISA scales
        srt = sr // SF_TILE_M
        sri = sr % SF_TILE_M
        srb = (sri % 32) * 16 + (sri // 32) * 4
        rtb = (srt * row_k_tiles + rki) * SF_TILE_STORAGE
        tl.store(row_scales_ptr + rtb + srb + rkn, r_byte)

        # TK-space row ISA scales (fused scatter — eliminates separate kernel!)
        if HAS_TK_SCALES:
            tk_rt = fids // SF_TILE_M
            tk_ri = fids % SF_TILE_M
            tk_rb = (tk_ri % 32) * 16 + (tk_ri // 32) * 4
            tk_tb = (tk_rt * tk_k_tiles + rki) * SF_TILE_STORAGE
            tl.store(row_tk_scales_ptr + tk_tb + tk_rb + rkn, r_byte)

        # Phase 1 done — r_scale, r_byte, rq can be freed by compiler

        # ════════ Phase 2: Col quant (reuses freed registers) ════════
        # Re-derive f32 abs values (cheaper than keeping them alive)
        col_amax = tl.max(tl.abs(v.to(tl.float32)), axis=0)  # (32,) f32

        # Inline E8M0
        cb = col_amax.to(tl.int32, bitcast=True)
        ce = (cb >> 23) & 0xFF
        cc = tl.where((cb & 0x7FFFFF) > 0x600000, 1, 0)
        ce = ce - 8 + cc
        ce = tl.maximum(tl.where(((cb >> 23) & 0xFF) > 0, ce, 0), 0)
        c_byte = ce.to(tl.uint8)
        cq_exp = tl.maximum(tl.minimum(254 - ce, 254), 1)
        c_scale = (cq_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        # Quantize + transpose
        cq = (v.to(tl.float32) * c_scale[None, :]).to(tl.float8e4nv)
        ct = tl.trans(cq)
        tl.store(col_fp8_ptr + cor[:, None].to(tl.int64) * capacity + co[None, :].to(tl.int64),
                 ct, mask=dim_mask[:, None])

        # Col ISA
        cki = pg // cgk
        ckn = pg % cgk
        ctb = (crt * ck + cki) * SF_TILE_STORAGE
        tl.store(col_scales_ptr + expert_id.to(tl.int64) * col_per_batch + (ctb + crb + ckn).to(tl.int64),
                 c_byte, mask=dim_mask)


def warp_dual_quant_v4(src, num_experts, capacity, *, gather_idx=None,
                        tk_row_scales=None, gpb=4, maxnreg=64):
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
    has_tk = tk_row_scales is not None
    tk_ptr = tk_row_scales if has_tk else rsc  # dummy

    TK = gather_idx.shape[0] if hg else T
    tk_rpb = _storage_per_batch(TK, H) if has_tk else 0

    grid = (num_experts * _div_up(H, GS), _div_up(capacity // GS, gpb))

    _warp_dual_quant_v4[grid](
        src, gp, row_fp8, rsc, col_fp8, csc, tk_ptr,
        T, H, capacity, cpb, tk_rpb,
        src.stride(0), src.stride(1),
        _div_up(H, _SF_TILE_K), _div_up(H, _SF_TILE_K),
        HAS_GATHER=hg, HAS_TK_SCALES=has_tk, GS=GS,
        SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_BLOCK=gpb, num_warps=1, maxnreg=maxnreg,
    )
    return (row_fp8, rsc.view(torch.float8_e8m0fnu),
            col_fp8.reshape(num_experts, H, capacity), csc.view(torch.float8_e8m0fnu),
            tk_row_scales)


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
    rc, rcs = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    # v4 without TK scales
    r1, s1, c1, cs1, _ = warp_dual_quant_v4(dout, E, CAP, gather_idx=x_idx)
    torch.cuda.synchronize()
    rm = (rr.view(torch.uint8) == r1.view(torch.uint8)).float().mean().item()
    cm = (rc.view(torch.uint8) == c1.view(torch.uint8)).float().mean().item()
    print(f"Precision: row {rm*100:.1f}%  col {cm*100:.1f}%")

    # v4 with TK scales (fused scatter)
    tk_rpb = _storage_per_batch(TK, H)
    tk_sc = torch.full((1, tk_rpb), 127, dtype=torch.uint8, device="cuda")
    r2, s2, c2, cs2, tks = warp_dual_quant_v4(dout, E, CAP, gather_idx=x_idx, tk_row_scales=tk_sc)
    torch.cuda.synchronize()
    print(f"TK scales: shape={tk_sc.shape}")

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

    print("\n--- Benchmark ---")
    t_sep = bench(lambda: (quantize_and_pack_activation(dout),
                           fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)),
                  "Separate (row + col, 2 kernels)")

    # Sweep maxnreg
    for mr in [48, 56, 64, 72, 80]:
        bench(lambda mr=mr: warp_dual_quant_v4(dout, E, CAP, gather_idx=x_idx, maxnreg=mr),
              f"v4 bf16-retain maxnreg={mr}")

    # Best with TK scales fused
    bench(lambda: warp_dual_quant_v4(dout, E, CAP, gather_idx=x_idx,
                                      tk_row_scales=tk_sc, maxnreg=64),
          "v4 bf16 + fused TK scales maxnreg=64")

    print(f"\n  (Fused TK scales saves ~20µs separate scatter kernel)")
