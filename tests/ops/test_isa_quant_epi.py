#!/usr/bin/env python
"""Session 1A foundation test: BlockscaledIsaQuantOnlyMixin byte-exact oracle.

Validates that the new ISA-pack epi quant kernel
(`blockscaled_fp8_gemm_zeromat_isa_quant`) produces UE8M0 scale bytes
byte-equivalent to the existing flat-layout kernel
(`blockscaled_fp8_gemm_zeromat_quant`) after a Python-level flat->ISA
pack conversion.

Both kernels share the same GEMM accumulation path and same E8M0 arithmetic
in the epi; only the final scale store layout differs.  Differences would
indicate offset-math bugs in the new EpiOp / Mixin.

Run: CUDA_VISIBLE_DEVICES=0 python tests/ops/test_isa_quant_epi.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def _flat_to_isa(flat_scales: "torch.Tensor") -> "torch.Tensor":
    """Convert (M, N//32) uint8 flat scales -> (num_m_tiles, k_tiles, 512) ISA pack.

    Mirrors the offset math in `_quantize_and_pack_kernel`:
        m_tile          = m // 128
        row_in_tile     = m %  128
        k_tile_idx      = n_group // 4
        k_in_tile       = n_group %  4
        row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
        isa[m_tile, k_tile_idx, row_base_offset + k_in_tile] = flat[m, n_group]
    """
    import torch
    M, NG = flat_scales.shape
    SF_TILE_M = 128
    GROUPS_PER_K_TILE = 4
    num_m_tiles = (M + SF_TILE_M - 1) // SF_TILE_M
    k_tiles = (NG + GROUPS_PER_K_TILE - 1) // GROUPS_PER_K_TILE
    isa = torch.zeros((num_m_tiles, k_tiles, 512), dtype=torch.uint8, device=flat_scales.device)

    m_idx = torch.arange(M, device=flat_scales.device)
    m_tile = (m_idx // SF_TILE_M).long()
    row_in_tile = (m_idx % SF_TILE_M).long()
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4  # (M,)

    n_idx = torch.arange(NG, device=flat_scales.device)
    k_tile_idx = (n_idx // GROUPS_PER_K_TILE).long()
    k_in_tile = (n_idx % GROUPS_PER_K_TILE).long()
    inner_off = row_base[:, None] + k_in_tile[None, :]  # (M, NG)
    m_tile_b = m_tile[:, None].expand(M, NG)
    k_tile_b = k_tile_idx[None, :].expand(M, NG)
    isa[m_tile_b, k_tile_b, inner_off] = flat_scales
    return isa


def _run(T: int = 256, K: int = 2, E: int = 4, H: int = 256, I: int = 128, routing: str = "uniform"):
    import torch
    import numpy as np

    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import (
        blockscaled_fp8_gemm_zeromat_quant,
        blockscaled_fp8_gemm_zeromat_isa_quant,
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

    # Quantize x at T-size + gather scales to padded TK.
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

    print(f"[isa-epi T={T} K={K} E={E} H={H} I={I} routing={routing}] TK_padded={TK} N2={N2}")
    print(f"  compiling FLAT kernel ...")
    z_fp8_flat, z_scale_flat = blockscaled_fp8_gemm_zeromat_quant(
        x_fp8, w1_fp8.mT,
        cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()
    print(f"  FLAT done: z_fp8={z_fp8_flat.shape} z_scale={z_scale_flat.shape}")

    print(f"  compiling ISA kernel ...")
    z_fp8_isa, z_scale_isa = blockscaled_fp8_gemm_zeromat_isa_quant(
        x_fp8, w1_fp8.mT,
        cu_seqlens_m=eFO, A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0, b_scales=w1_scales,
    )
    torch.cuda.synchronize()
    print(f"  ISA done: z_fp8={z_fp8_isa.shape} z_scale_isa={z_scale_isa.shape}")

    # z_fp8 itself must match byte-exact (same GEMM path).
    z_fp8_diff = (z_fp8_flat.view(torch.uint8) != z_fp8_isa.view(torch.uint8)).sum().item()
    print(f"  z_fp8 mismatched bytes: {z_fp8_diff}/{z_fp8_flat.numel()}")

    # Convert flat scales -> ISA layout in Python, compare to direct ISA store.
    isa_from_flat = _flat_to_isa(z_scale_flat)
    s_diff = (isa_from_flat != z_scale_isa).sum().item()
    print(f"  z_scale_isa mismatched bytes vs flat->ISA: {s_diff}/{z_scale_isa.numel()}")

    if z_fp8_diff != 0 or s_diff != 0:
        idx = (isa_from_flat != z_scale_isa).nonzero(as_tuple=False)[:10].cpu().numpy()
        print("  first ISA scale mismatches (m_tile, k_tile, off, ref, got):")
        for i in idx:
            mt, kt, of = int(i[0]), int(i[1]), int(i[2])
            print(f"    ({mt},{kt},{of}): ref={int(isa_from_flat[mt,kt,of].item())} got={int(z_scale_isa[mt,kt,of].item())}")
        return False
    return True


if __name__ == "__main__":
    all_ok = True
    cases = [
        # (T, K, E, H, I, routing)
        (256, 2, 4, 256, 128, "uniform"),
        (256, 2, 4, 256, 128, "random"),
        (512, 4, 8, 512, 256, "random"),
        (1024, 2, 8, 1024, 512, "random"),
        (128, 1, 2, 128, 128, "uniform"),  # smallest aligned
    ]
    for c in cases:
        ok = _run(T=c[0], K=c[1], E=c[2], H=c[3], I=c[4], routing=c[5])
        all_ok = all_ok and ok
        print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
