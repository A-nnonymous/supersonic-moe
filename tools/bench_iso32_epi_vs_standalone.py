#!/usr/bin/env python
"""Micro-bench: epi-fused iso32 vs (BF16 GEMM + standalone iso32 quant).

This isolates the FP8 GEMM + iso32 quant cost (the exact pattern used in
DGated bwd path) and measures kernel-level timing.  It does NOT measure
end-to-end — that requires wiring the epi into functional/__init__.py:2104,
which is a separate surgery on GemmDGatedMixin.

What this DOES prove: the standalone iso32 quant kernel cost is fully
absorbable into the GEMM epilogue (zero added kernels).
"""

from __future__ import annotations

import os
import sys

_VENV = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
_PY = f"{_VENV}/bin/python"
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    print(f"\033[33mSwitch venv: {_VENV}\033[0m")
    os.execv(_PY, [_PY, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bench(T=8192, E=8, K=4, H=2048, I=1408, n_warmup=10, n_iter=50):
    import torch
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        iso32_dual_quantize_varlen,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import (
        blockscaled_fp8_gemm_zeromat_iso32_quant,
        blockscaled_fp8_gemm_zeromat_bf16,
    )

    torch.manual_seed(0)
    device = "cuda"
    TK_real = T * K
    N2 = 2 * I

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.5
    w1 = torch.randn(N2, H, E, dtype=torch.bfloat16, device=device) * 0.05

    expert_assign = (torch.arange(TK_real, device=device, dtype=torch.int32) % E)
    sorted_assign, perm = torch.sort(expert_assign)
    counts = torch.bincount(sorted_assign, minlength=E).to(torch.int32)
    padded_counts = torch.where(counts > 0, ((counts + 127) // 128) * 128, counts)
    eFO = torch.zeros(E + 1, dtype=torch.int32, device=device)
    eFO[1:] = torch.cumsum(padded_counts, dim=0).to(torch.int32)
    TK = int(eFO[-1].item())

    base_token_real = perm // K
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    write = 0
    for e in range(E):
        real = int(counts[e].item())
        padded = int(padded_counts[e].item())
        if real:
            rows = base_token_real[sorted_assign == e]
            x_gather_idx[write:write + real] = rows.to(torch.int32)
        if padded > real:
            x_gather_idx[write + real:write + padded] = 0
        write += padded
    x_gather_idx = x_gather_idx.contiguous()

    x_fp8, x_scales_t = quantize_and_pack_activation(x)
    k_tiles_x = _div_up(H, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(TK, H)
    if TK % _SF_TILE_M == 0 and H % _SF_TILE_K == 0:
        x_scales_tk = torch.empty((1, per_batch_tk), dtype=torch.uint8, device=device)
    else:
        x_scales_tk = torch.full((1, per_batch_tk), 127, dtype=torch.uint8, device=device)
    BLOCK_ROWS = 128
    _gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK_ROWS), k_tiles_x)](
        x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
        src_k_tiles=k_tiles_x, dst_k_tiles=k_tiles_x,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_scales_tk_e8m0 = x_scales_tk.view(_E8M0)
    w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)

    print(f"[bench T={T} E={E} K={K} H={H} I={I}] TK={TK} N2={N2}")

    # JIT-warm both kernels.
    for _ in range(3):
        _ = blockscaled_fp8_gemm_zeromat_iso32_quant(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        z_bf16 = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        _ = iso32_dual_quantize_varlen(z_bf16, TK, N2)
    torch.cuda.synchronize()

    # Path A: BF16 GEMM + standalone iso32 quant (legacy pattern).
    for _ in range(n_warmup):
        z_bf16 = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        _ = iso32_dual_quantize_varlen(z_bf16, TK, N2)
    torch.cuda.synchronize()
    starts_a = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends_a   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts_a[i].record()
        z_bf16 = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        _ = iso32_dual_quantize_varlen(z_bf16, TK, N2)
        ends_a[i].record()
    torch.cuda.synchronize()
    times_a = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts_a, ends_a)])
    med_a = times_a[len(times_a) // 2]

    # Path B: epi-fused iso32 (single kernel).
    for _ in range(n_warmup):
        _ = blockscaled_fp8_gemm_zeromat_iso32_quant(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
    torch.cuda.synchronize()
    starts_b = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends_b   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts_b[i].record()
        _ = blockscaled_fp8_gemm_zeromat_iso32_quant(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        ends_b[i].record()
    torch.cuda.synchronize()
    times_b = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts_b, ends_b)])
    med_b = times_b[len(times_b) // 2]

    # Path C: BF16 GEMM alone (baseline for the quant cost split).
    for _ in range(n_warmup):
        _ = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
    torch.cuda.synchronize()
    starts_c = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends_c   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts_c[i].record()
        _ = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        ends_c[i].record()
    torch.cuda.synchronize()
    times_c = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts_c, ends_c)])
    med_c = times_c[len(times_c) // 2]

    quant_cost = med_a - med_c
    epi_overhead = med_b - med_c
    savings = med_a - med_b
    pct = 100.0 * savings / med_a

    print(f"  Path A (BF16 GEMM + iso32 quant kernel): {med_a:.1f} us")
    print(f"  Path B (FP8 GEMM with epi-fused iso32):  {med_b:.1f} us")
    print(f"  Path C (BF16 GEMM alone, no quant):      {med_c:.1f} us")
    print(f"  -> standalone iso32 quant kernel cost  : {quant_cost:.1f} us")
    print(f"  -> epi-fused quant overhead on GEMM    : {epi_overhead:.1f} us")
    print(f"  -> savings (A - B)                     : {savings:+.1f} us ({pct:+.1f}%)")


if __name__ == "__main__":
    # T=8192 E=8 K=4 H=2048 I=1408 is the production handoff target shape.
    _bench(T=8192, E=8, K=4, H=2048, I=1408)
