#!/usr/bin/env python
"""Session 1C test: colwise epi quant vs production colwise_quantize_and_pack.

Cross-checks `blockscaled_fp8_gemm_zeromat_colwise_quant` (epi-fused
colwise) against running `colwise_quantize_and_pack` on the BF16 D of
`blockscaled_fp8_gemm_zeromat_bf16` (the same accumulation, just no quant).

Validates that the epi colwise reproduces the production kernel within
the bf16-truncation precision envelope established by 1A-ext.

Run: CUDA_VISIBLE_DEVICES=0 python tests/ops/test_colwise_quant_epi.py
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


def _run(T=256, K=2, E=4, H=256, I=128, routing="uniform"):
    import torch
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        colwise_quantize_and_pack,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import (
        blockscaled_fp8_gemm_zeromat_colwise_quant,
        blockscaled_fp8_gemm_zeromat_bf16,
    )

    torch.manual_seed(0)
    device = "cuda"
    TK_real = T * K
    N2 = 2 * I

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.5
    w1 = torch.randn(N2, H, E, dtype=torch.bfloat16, device=device) * 0.05

    if routing == "uniform":
        expert_assign = (torch.arange(TK_real, device=device, dtype=torch.int32) % E)
    else:
        scores = torch.randn(T, E, device=device)
        expert_assign = scores.topk(K, dim=-1).indices.reshape(-1).to(torch.int32)

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

    print(f"[colwise T={T} K={K} E={E} H={H} I={I} routing={routing}] TK={TK} N2={N2}")

    # Path A: BF16 D + production colwise quant.
    z_bf16 = blockscaled_fp8_gemm_zeromat_bf16(
        x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()
    fp8_ref, col_sc_ref = colwise_quantize_and_pack(
        z_bf16, logical_rows=N2, logical_cols=TK,
    )
    torch.cuda.synchronize()

    # Path B: epi-fused colwise quant.
    fp8_new, col_sc_new = blockscaled_fp8_gemm_zeromat_colwise_quant(
        x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()

    # Compare.
    fp8_ref_u8 = fp8_ref.view(torch.uint8)
    fp8_new_u8 = fp8_new.view(torch.uint8)
    fp8_diff = (fp8_ref_u8 != fp8_new_u8).sum().item()
    fp8_pct = 100.0 * fp8_diff / fp8_ref.numel()
    print(f"  FP8 byte diff: {fp8_diff}/{fp8_ref.numel()} ({fp8_pct:.2f}%) "
          f"-- expected <5% from bf16-truncation in ref path")

    col_new_u8 = col_sc_new.view(torch.uint8).reshape(-1)
    col_ref_u8 = col_sc_ref.view(torch.uint8).reshape(-1)
    col_diff = "shape-mismatch" if col_new_u8.numel() != col_ref_u8.numel() else \
        (col_new_u8 != col_ref_u8).sum().item()
    col_total = col_ref_u8.numel()
    col_pct = (100.0 * col_diff / col_total) if isinstance(col_diff, int) else float('nan')
    print(f"  col-SF byte diff: {col_diff}/{col_total} ({col_pct:.2f}%) "
          f"-- expected ~0 (or <1% from bf16 amax dropout)")

    if isinstance(fp8_diff, int) and fp8_diff > 0:
        src_f32 = z_bf16.to(torch.float32)
        err_ref = (fp8_ref.to(torch.float32) - src_f32).abs()
        err_new = (fp8_new.to(torch.float32) - src_f32).abs()
        print(f"    rrmse new->bf16src: {(err_new**2).mean().sqrt().item():.6f}")
        print(f"    rrmse ref->bf16src: {(err_ref**2).mean().sqrt().item():.6f}")

    sf_ok = isinstance(col_diff, int) and col_diff <= col_total * 0.01
    fp8_ok = fp8_pct < 5.0
    return sf_ok and fp8_ok


if __name__ == "__main__":
    all_ok = True
    for c in [
        (256, 2, 4, 256, 128, "uniform"),
        (256, 2, 4, 256, 128, "random"),
        (512, 4, 8, 512, 256, "random"),
        (128, 1, 2, 128, 128, "uniform"),
    ]:
        ok = _run(T=c[0], K=c[1], E=c[2], H=c[3], I=c[4], routing=c[5])
        all_ok = all_ok and ok
        print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
