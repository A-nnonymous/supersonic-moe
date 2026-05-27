#!/usr/bin/env python
"""Microbench: row-only ISA quant epi (1A) vs standalone quantize_and_pack_activation.

At production shape T=8192 E=8 K=4 H=2048 I=1408 — comparing:
- Path A: BF16-D GEMM + standalone quantize_and_pack_activation (current prod-equiv)
- Path B: FP8-D GEMM with epi-fused row-only ISA quant
- Path C: BF16-D GEMM alone (baseline)

This is the proof-of-value for Phase A. Row-quant has no inter-lane dependency
(each lane owns one full row in its fragment) so it should add minimal epi cost.
"""

from __future__ import annotations
import os, sys, time

_VENV = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    os.execv(f"{_VENV}/bin/python", [f"{_VENV}/bin/python", *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

for _p in (
    "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack",
    "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def bench(fn, n_warmup=10, n_iter=50):
    import torch
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6


def main():
    import torch
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation, precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import (
        blockscaled_fp8_gemm_zeromat_isa_quant,
        blockscaled_fp8_gemm_zeromat_bf16,
    )

    torch.manual_seed(0)
    device = "cuda"
    T, K, E, H, I = 8192, 4, 8, 2048, 1408
    TK_real = T * K
    N2 = 2 * I

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.5
    w1 = torch.randn(N2, H, E, dtype=torch.bfloat16, device=device) * 0.05
    scores = torch.randn(T, E, device=device)
    expert_assign = scores.topk(K, dim=-1).indices.reshape(-1).to(torch.int32)
    sorted_assign, perm = torch.sort(expert_assign)
    counts = torch.bincount(sorted_assign, minlength=E).to(torch.int32)
    padded_counts = torch.where(counts > 0, ((counts + 127) // 128) * 128, counts)
    eFO = torch.zeros(E + 1, dtype=torch.int32, device=device)
    eFO[1:] = torch.cumsum(padded_counts, dim=0).to(torch.int32)
    TK = int(eFO[-1].item())

    base_token = perm // K
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    write = 0
    for e in range(E):
        real, padded = int(counts[e].item()), int(padded_counts[e].item())
        if real:
            x_gather_idx[write:write + real] = base_token[sorted_assign == e].to(torch.int32)
        if padded > real:
            x_gather_idx[write + real:write + padded] = 0
        write += padded

    x_fp8, x_scales_t = quantize_and_pack_activation(x)
    k_tiles = _div_up(H, _SF_TILE_K)
    per_batch = _storage_per_batch(TK, H)
    x_scales_tk = torch.empty((1, per_batch), dtype=torch.uint8, device=device) if (TK % _SF_TILE_M == 0 and H % _SF_TILE_K == 0) else torch.full((1, per_batch), 127, dtype=torch.uint8, device=device)
    BLOCK = 128
    _gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK), k_tiles)](
        x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_scales_tk_e8m0 = x_scales_tk.view(_E8M0)
    w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)
    print(f"Shape: T={T} K={K} E={E} H={H} I={I} | TK={TK} N2={N2}")

    def path_a():
        z = blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)
        return quantize_and_pack_activation(z)

    def path_b():
        return blockscaled_fp8_gemm_zeromat_isa_quant(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)

    def path_c():
        return blockscaled_fp8_gemm_zeromat_bf16(
            x_fp8, w1_fp8.mT, cu_seqlens_m=eFO, A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0, b_scales=w1_scales)

    path_a(); path_b(); path_c()
    torch.cuda.synchronize()

    t_a = bench(path_a); t_b = bench(path_b); t_c = bench(path_c)
    print(f"  Path A (BF16 GEMM + standalone row quant):     {t_a:7.1f} µs")
    print(f"  Path B (FP8 GEMM + epi-fused row-ISA quant):   {t_b:7.1f} µs")
    print(f"  Path C (BF16 GEMM alone):                       {t_c:7.1f} µs")
    print(f"  Standalone row quant cost (A−C):                {t_a - t_c:7.1f} µs")
    print(f"  Epi overhead on GEMM   (B−C):                   {t_b - t_c:7.1f} µs")
    print(f"  REALIZABLE SAVINGS (A−B):                       {t_a - t_b:7.1f} µs ({100*(t_a-t_b)/t_a:.1f}%)")


if __name__ == "__main__":
    main()
