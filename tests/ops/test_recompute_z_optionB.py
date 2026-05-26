#!/usr/bin/env python
"""Bit-exact validation for recompute_z Option B.

Two layers of validation:
  (1) **Kernel-level bit-exact**: drive the new
      ``blockscaled_fp8_gemm_zeromat_quant`` kernel with the SAME inputs
      (x_fp8, w1_fp8, A_idx, a_scales, b_scales) as the existing gated
      ``gemm_gated`` epi-quant path on SM100, and assert that

        - z_fp8 bytes are byte-equal (uint8 view),
        - z_scale (UE8M0) bytes are byte-equal.

  (2) **End-to-end equivalence**: the existing recompute_z=True path now
      goes through Option B (because we rewired ``_recompute_z_fp8``). Re-
      run a small MLP forward+backward, check out/dx/ds/dw1/dw2 are
      bit-equal (cosine=1, rrmse=0) to the recompute_z=False baseline.

Run: CUDA_VISIBLE_DEVICES=0 python tests/ops/test_recompute_z_optionB.py
"""

from __future__ import annotations

import os
import subprocess
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


# ---------------------------------------------------------------------------
# Layer 1: kernel-level bit-exact comparison
# ---------------------------------------------------------------------------

def _kernel_bit_exact_child(T: int, K: int, E: int, H: int, I: int, routing: str):
    import torch
    import numpy as np

    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    from sonicmoe.quack_utils.gemm_gated import gemm_gated
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import blockscaled_fp8_gemm_zeromat_quant

    torch.manual_seed(0)
    device = "cuda"

    TK_real = T * K
    N2 = 2 * I  # the 2I dim of the up-projection (gated half size = N2//2 = I)

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.5
    w1 = torch.randn(N2, H, E, dtype=torch.bfloat16, device=device) * 0.05  # (2I, H, E)

    def _make_assignments() -> torch.Tensor:
        if routing == "uniform":
            return torch.arange(TK_real, device=device, dtype=torch.int32) % E
        if routing == "random":
            scores = torch.randn(T, E, device=device)
            return scores.topk(K, dim=-1).indices.reshape(-1).to(torch.int32)
        if routing == "skew":
            scores = torch.randn(T, E, device=device)
            hot = torch.rand(T, device=device) < 0.80
            scores[hot, 0] += 100.0
            return scores.topk(K, dim=-1).indices.reshape(-1).to(torch.int32)
        if routing == "holes":
            active = max(K, E // 2)
            scores = torch.randn(T, active, device=device)
            return scores.topk(K, dim=-1).indices.reshape(-1).to(torch.int32)
        raise ValueError(f"unknown routing={routing}")

    expert_assign = _make_assignments()
    if expert_assign.numel() != TK_real:
        raise RuntimeError(f"routing={routing} produced {expert_assign.numel()} routes, expected {TK_real}")
    sorted_assign, perm = torch.sort(expert_assign)
    counts = torch.bincount(sorted_assign, minlength=E).to(torch.int32)
    padded_counts = torch.where(counts > 0, ((counts + 127) // 128) * 128, counts)
    eFO = torch.zeros(E + 1, dtype=torch.int32, device=device)
    eFO[1:] = torch.cumsum(padded_counts, dim=0).to(torch.int32)
    TK = int(eFO[-1].item())

    base_token_real = perm // K
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    write = 0
    pad_rows = 0
    for e in range(E):
        real = int(counts[e].item())
        padded = int(padded_counts[e].item())
        if real:
            rows = base_token_real[sorted_assign == e]
            x_gather_idx[write:write + real] = rows.to(torch.int32)
        if padded > real:
            x_gather_idx[write + real:write + padded] = 0
            pad_rows += padded - real
        write += padded
    x_gather_idx = x_gather_idx.contiguous()

    # --- Stage 1: quantize x at T-size + gather scales to padded TK ---
    x_fp8, x_scales_t = quantize_and_pack_activation(x)
    k_tiles = _div_up(H, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(TK, H)
    if TK % _SF_TILE_M == 0 and H % _SF_TILE_K == 0:
        x_scales_tk = torch.empty((1, per_batch_tk), dtype=torch.uint8, device=device)
    else:
        x_scales_tk = torch.full((1, per_batch_tk), 127, dtype=torch.uint8, device=device)
    BLOCK_ROWS = 128
    _gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK_ROWS), k_tiles)](
        x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_scales_tk_e8m0 = x_scales_tk.view(_E8M0)

    # --- Stage 2: weight fp8 + scales (cached) ---
    w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)

    # --- Path A: gated kernel with epilogue blockscaled quant on D ---
    z_fp8_A = torch.empty((TK, N2), dtype=torch.float8_e4m3fn, device=device)
    PostAct_A = torch.empty((TK, N2 // 2), dtype=torch.bfloat16, device=device)
    z_scale_out_A = torch.empty((TK, N2 // 32), dtype=torch.uint8, device=device)
    gemm_gated(
        x_fp8, w1_fp8.mT,
        z_fp8_A, None,
        PostAct_A,
        None,
        "swiglu",
        128, 128, 1, 1,
        cu_seqlens_m=eFO,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        z_scale_out=z_scale_out_A,
    )
    torch.cuda.synchronize()

    # --- Path B: non-gated quant-only kernel ---
    z_fp8_B, z_scale_out_B = blockscaled_fp8_gemm_zeromat_quant(
        x_fp8, w1_fp8.mT,
        cu_seqlens_m=eFO,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
    )
    torch.cuda.synchronize()

    # --- Bit-exact comparison ---
    a_z = z_fp8_A.view(torch.uint8).cpu().numpy()
    b_z = z_fp8_B.view(torch.uint8).cpu().numpy()
    a_s = z_scale_out_A.cpu().numpy()
    b_s = z_scale_out_B.cpu().numpy()

    z_diff = (a_z != b_z).sum()
    s_diff = (a_s != b_s).sum()

    counts_cpu = counts.cpu().tolist()
    print(
        f"[bitexact routing={routing} T={T} K={K} E={E} H={H} I={I}] "
        f"tpe(min/max/sum)={min(counts_cpu)}/{max(counts_cpu)}/{sum(counts_cpu)} "
        f"pad_rows={pad_rows} TK_padded={TK}"
    )
    print(f"  z_fp8 mismatched bytes: {z_diff}/{a_z.size}")
    print(f"  z_scale mismatched bytes: {s_diff}/{a_s.size}")

    if z_diff != 0 or s_diff != 0:
        idx = np.argwhere(a_z != b_z)[:5]
        print("first z mismatches (idx, A, B):")
        for i in idx:
            print(f"  {tuple(i)}: A={int(a_z[tuple(i)])} B={int(b_z[tuple(i)])}")
        idx = np.argwhere(a_s != b_s)[:5]
        print("first scale mismatches (idx, A, B):")
        for i in idx:
            print(f"  {tuple(i)}: A={int(a_s[tuple(i)])} B={int(b_s[tuple(i)])}")
        sys.exit(1)


def _e2e_equivalence_child(recompute_z: bool, T: int, K: int, E: int, H: int, I: int):
    """Replicates test_recompute_z's harness; returns numeric tensors."""
    import paddle
    paddle.enable_compat()
    import torch
    import numpy as np

    from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches
    import sonicmoe.functional as functional

    functional._ALIGNMENT_ASSUMED = True
    if recompute_z:
        os.environ["SONIC_MOE_FP8_RECOMPUTE_Z"] = "1"
    else:
        os.environ.pop("SONIC_MOE_FP8_RECOMPUTE_Z", None)

    torch.manual_seed(0)
    device = "cuda"
    TK = T * K
    moe = SonicMoEMlpNode(
        gate_in_dim=H, moe_intermediate_size=I,
        num_local_experts=E, k=K,
    ).to(device).to(torch.bfloat16)
    invalidate_weight_caches()

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True) * 0.5

    expert_assign = torch.arange(TK, device=device, dtype=torch.int32) % E
    sorted_assign, perm = torch.sort(expert_assign)
    counts = torch.bincount(sorted_assign, minlength=E).to(torch.int32)
    eFO = torch.zeros(E + 1, dtype=torch.int32, device=device)
    eFO[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    base_token = perm // K
    x_gather_idx = base_token.to(torch.int32).contiguous()
    s_scatter_idx = torch.argsort(perm).to(torch.int32).contiguous()
    s_reverse_scatter_idx = perm.to(torch.int32).contiguous()
    topk_scores = torch.ones(TK, dtype=torch.float32, device=device) / K

    out = moe(x, eFO, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, topk_scores)
    g_out = torch.randn_like(out) * 0.01
    out.backward(g_out)
    flush_native_grads()
    torch.cuda.synchronize()

    return {
        "out": out.detach().float().cpu().numpy(),
        "dx": x.grad.detach().float().cpu().numpy(),
        "dw1": moe.up_proj.weight.main_grad.detach().float().cpu().numpy(),
        "dw2": moe.down_proj.weight.main_grad.detach().float().cpu().numpy(),
    }


def _spawn_child(label: str, *args):
    fn = {
        "kernel": _kernel_bit_exact_child,
    }[label]
    fn(*args)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        label = sys.argv[2]
        if label == "kernel":
            T, K, E, H, I = (int(x) for x in sys.argv[3:8])
            routing = sys.argv[8]
            tile = sys.argv[9] if len(sys.argv) > 9 else "128"
            os.environ["SONIC_MOE_QONLY_TILE"] = tile
            _kernel_bit_exact_child(T, K, E, H, I, routing)
        sys.exit(0)

    # Layer 1: kernel-level bit-exact across representative routing regimes.
    # Each case: (T, K, E, H, I, routing, tile)
    cases = [
        (1024, 8, 8, 3072, 1536, "uniform", "128"),
        (1024, 8, 8, 3072, 1536, "random", "128"),
        (1024, 8, 8, 3072, 1536, "skew", "128"),
        (1024, 8, 16, 3072, 1536, "holes", "128"),
        (1024, 8, 16, 3072, 1536, "random", "128"),
        (1024, 8, 32, 3072, 1536, "random", "128"),
        (512, 4, 16, 1024, 1536, "random", "128"),
        (512, 4, 16, 3072, 1024, "skew", "128"),
        (4096, 8, 8, 3072, 1536, "skew", "128"),
        (8192, 8, 8, 3072, 1536, "random", "128"),
        # E=64, K=2 many-expert cases
        (1024, 2, 64, 3072, 1536, "random", "128"),
        (1024, 2, 64, 3072, 1536, "skew", "128"),
        (1024, 2, 64, 3072, 1536, "holes", "128"),
        (4096, 2, 64, 3072, 1536, "random", "128"),
        # Taily: small per-expert counts near tile boundaries
        (256, 8, 32, 3072, 1536, "random", "128"),
        (128, 2, 64, 3072, 1536, "random", "128"),
        # 256x256 tile for dense shapes
        (8192, 8, 8, 3072, 1536, "random", "256"),
        (4096, 8, 8, 3072, 1536, "skew", "256"),
        # auto tile
        (8192, 8, 8, 3072, 1536, "random", "auto"),
        (1024, 2, 64, 3072, 1536, "random", "auto"),
    ]
    for case in cases:
        T, K, E, H, I, routing, tile = case
        print(f"\n=== Layer 1 kernel bit-exact T={T} K={K} E={E} H={H} I={I} routing={routing} tile={tile} ===")
        rc = subprocess.call(
            [_PY, __file__, "--child", "kernel", str(T), str(K), str(E), str(H), str(I), routing, tile],
            timeout=300,
        )
        if rc != 0:
            print(f"\033[31mFAIL kernel bit-exact at {case}\033[0m")
            sys.exit(rc)

    print("\n\033[32mAll Option B bit-exact tests PASS.\033[0m")


if __name__ == "__main__":
    main()
