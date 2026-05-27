#!/usr/bin/env python
"""Session 1A-ext test: BlockscaledIso32QuantOnlyMixin warp_redux + dual ISA.

Validates the new iso32 epi-quant kernel
(`blockscaled_fp8_gemm_zeromat_iso32_quant`) against the 1A row-only kernel
(`blockscaled_fp8_gemm_zeromat_isa_quant`).  Three invariants:

  (a) **Block invariant**: iso32 row-SF must be CONSTANT within each
      (m_group=32 rows, n_group=32 cols) cell.  Since amax is reduced over
      the whole 32x32 block via warp_redux, every byte in the cell is the
      same e8m0.

  (b) **Block-max relation**: iso32 e8m0 for block (m_group, n_group)
      must equal MAX(1A per-row e8m0 over those 32 rows of the same
      n_group).  E8M0 is `ceil(log2(amax/448))` and max-of-32 amax >=
      each per-row amax, so block e8m0 >= each per-row e8m0; specifically
      it equals the max.

  (c) **iso32 invariant (row<->col)**: iso32 col-SF byte at
      (n_abs, m_group_abs) must equal iso32 row-SF byte at any
      m in m_group, n_group=n_abs//32.  Same e8m0 stored in both layouts.

Run: CUDA_VISIBLE_DEVICES=0 python tests/ops/test_iso32_quant_epi.py
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


def _isa_to_flat_row(isa, M, NG):
    """Decode (num_m_tiles, k_tiles, 512) row-ISA -> (M, NG) flat uint8."""
    import torch
    SF_TILE_M = 128
    GPK = 4
    flat = torch.zeros((M, NG), dtype=torch.uint8, device=isa.device)
    m_idx = torch.arange(M, device=isa.device)
    m_tile = (m_idx // SF_TILE_M).long()
    row_in_tile = (m_idx % SF_TILE_M).long()
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    n_idx = torch.arange(NG, device=isa.device)
    k_tile_idx = (n_idx // GPK).long()
    k_in_tile = (n_idx % GPK).long()
    inner_off = row_base[:, None] + k_in_tile[None, :]
    m_tile_b = m_tile[:, None].expand(M, NG)
    k_tile_b = k_tile_idx[None, :].expand(M, NG)
    flat[:, :] = isa[m_tile_b, k_tile_b, inner_off]
    return flat


def _isa_to_flat_col(isa, N, MG):
    """Decode (num_n_tiles, col_k_tiles, 512) col-ISA -> (N, MG) flat uint8.
    MG = number of m_groups = M // 32 (rounded up to multiple of 4 capacity)."""
    import torch
    SF_TILE_M = 128
    GPK = 4
    flat = torch.zeros((N, MG), dtype=torch.uint8, device=isa.device)
    n_idx = torch.arange(N, device=isa.device)
    n_tile = (n_idx // SF_TILE_M).long()
    col_row_in_tile = (n_idx % SF_TILE_M).long()
    col_row_base = (col_row_in_tile % 32) * 16 + (col_row_in_tile // 32) * 4
    mg_idx = torch.arange(MG, device=isa.device)
    col_k_tile_idx = (mg_idx // GPK).long()
    col_k_in_tile = (mg_idx % GPK).long()
    inner_off = col_row_base[:, None] + col_k_in_tile[None, :]
    n_tile_b = n_tile[:, None].expand(N, MG)
    k_tile_b = col_k_tile_idx[None, :].expand(N, MG)
    flat[:, :] = isa[n_tile_b, k_tile_b, inner_off]
    return flat


def _run(T=256, K=2, E=4, H=256, I=128, routing="uniform"):
    import torch
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import (
        blockscaled_fp8_gemm_zeromat_isa_quant,
        blockscaled_fp8_gemm_zeromat_iso32_quant,
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

    print(f"[iso32 T={T} K={K} E={E} H={H} I={I} routing={routing}] TK={TK} N2={N2}")
    print("  compiling 1A row-only ...")
    _, z_scale_1a = blockscaled_fp8_gemm_zeromat_isa_quant(
        x_fp8, w1_fp8.mT,
        cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()

    print("  compiling iso32 row+col ...")
    z_fp8_iso, z_scale_row_iso, z_scale_col_iso = blockscaled_fp8_gemm_zeromat_iso32_quant(
        x_fp8, w1_fp8.mT,
        cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()
    print(f"  shapes: row_iso={z_scale_row_iso.shape} col_iso={z_scale_col_iso.shape}")

    NG = N2 // 32
    MG = TK // 32

    flat_1a = _isa_to_flat_row(z_scale_1a, TK, NG)
    flat_iso = _isa_to_flat_row(z_scale_row_iso, TK, NG)
    flat_col = _isa_to_flat_col(z_scale_col_iso, N2, MG)

    ok = True

    # (a) Block invariant on iso32 row-SF: bytes constant within each 32-row m_group.
    iso_reshaped = flat_iso.view(MG, 32, NG)
    block_const = (iso_reshaped == iso_reshaped[:, 0:1, :]).all().item()
    print(f"  (a) iso32 row-SF block-constant within m_group: {block_const}")
    ok = ok and block_const

    # (b) iso32 block e8m0 == max(1A per-row e8m0) over each 32-row m_group.
    flat_1a_reshaped = flat_1a.view(MG, 32, NG)
    per_block_max_1a = flat_1a_reshaped.max(dim=1).values  # (MG, NG)
    iso_per_block = iso_reshaped[:, 0, :]  # (MG, NG)
    rel_diff = (iso_per_block != per_block_max_1a).sum().item()
    print(f"  (b) iso32 e8m0 == max(1A e8m0 over 32 rows): mismatches {rel_diff}/{MG*NG}")
    if rel_diff != 0:
        bad = (iso_per_block != per_block_max_1a).nonzero(as_tuple=False)[:5]
        for mg, ng in bad.tolist():
            print(f"    mg={mg} ng={ng}: 1A_max={int(per_block_max_1a[mg,ng])} iso={int(iso_per_block[mg,ng])}")
        ok = False

    # (c) iso32 invariant: col-SF (n_abs, m_group_abs) == row-SF (m in m_group, n_abs//32).
    # flat_col shape (N2, MG); flat_iso shape (TK, NG).  For each (n, mg):
    #   col_byte = flat_col[n, mg]
    #   row_byte = flat_iso[mg*32, n//32]
    flat_iso_per_block = iso_reshaped[:, 0, :]  # (MG, NG)
    # Build expected col layout from row layout:  expected[n, mg] = iso[mg, n//32]
    expected_col = flat_iso_per_block.T  # (NG, MG)
    expected_col = expected_col.repeat_interleave(32, dim=0)  # (N2, MG)
    col_diff = (flat_col != expected_col).sum().item()
    print(f"  (c) iso32 col-SF == row-SF (transposed view): mismatches {col_diff}/{N2*MG}")
    if col_diff != 0:
        bad = (flat_col != expected_col).nonzero(as_tuple=False)[:5]
        for n, mg in bad.tolist():
            print(f"    n={n} mg={mg}: row_view={int(expected_col[n,mg])} col={int(flat_col[n,mg])}")
        ok = False

    # FP8 sanity: not all zeros.
    nz = (z_fp8_iso.view(torch.uint8) != 0).sum().item()
    print(f"  fp8 nonzero bytes: {nz}/{z_fp8_iso.numel()}")

    return ok


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
