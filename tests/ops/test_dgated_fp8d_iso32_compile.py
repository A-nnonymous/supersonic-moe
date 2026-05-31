"""Phase 1 (DGated FP8-D iso32) — JIT compile + smoke run.

Goal: exercise the *new* GemmDGatedFP8CLoadIso32QuantSm100ZeroMat kernel
class through the production `gemm_dgated` wrapper, in fp8_preact + gather_A
+ blockscaled + varlen mode with the iso32 side-channel outputs allocated.

This test does NOT verify numerical correctness; it only ensures the JIT
compile succeeds, the kernel runs without faulting, and the iso32 outputs
are written (non-zero).  Byte-exact correctness is tracked separately by
`test_dgated_fp8d_iso32_correctness.py`.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import paddle  # noqa: E402
paddle.enable_compat()
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _quantize_and_pack(bf16: torch.Tensor):
    """Rowwise blockscaled FP8 + ISA-packed scales (matches GEMM A/B inputs)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation
    return quantize_and_pack_activation(bf16)


def _quantize_raw(bf16: torch.Tensor):
    """Rowwise blockscaled FP8 + raw scales (matches DGated preact input)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
    return quantize_activation_blockscaled_fast(bf16)


def _build_inputs(TK: int, K: int, N: int, E: int, device="cuda"):
    """Build tensors that mimic ctx-saved state in DownProjection.backward."""
    torch.manual_seed(0)
    # dout (TK, K) bf16  ->  FP8 + scales
    dout = torch.randn(TK, K, dtype=torch.bfloat16, device=device) * 0.1
    dout_fp8, dout_scales = _quantize_and_pack(dout)

    # z (TK, 2N) bf16 -> FP8 + raw scales (the DGated preact input)
    z = torch.randn(TK, 2 * N, dtype=torch.bfloat16, device=device) * 0.5
    z_fp8, z_scales = _quantize_raw(z)
    z_scales_u8 = z_scales.view(torch.uint8)

    # w (H, I, E)  -> per-expert FP8 weights packed for the GEMM, shape (E, N=I, K=H)
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_direct_fused_dgated,
    )
    w2 = torch.randn(K, N, E, dtype=torch.bfloat16, device=device) * 0.1
    w2_fp8, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)

    # cu_seqlens_m: split TK into E equal chunks
    chunk = TK // E
    cu = torch.tensor(
        [chunk * i for i in range(E + 1)], dtype=torch.int32, device=device
    )
    cu[-1] = TK  # ensure last == TK

    # x_gather_idx (TK,) — identity gather for the test
    x_gather_idx = torch.arange(TK, dtype=torch.int32, device=device)

    # gathered dout scales (ISA-packed) — production gathers per-token; use
    # a fresh empty buffer of the right size (identity gather, but the
    # kernel still expects ISA packing layout).
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _gather_isa_packed_scales_kernel, _div_up, _SF_TILE_K, _SF_TILE_M,
        _SF_TILE_STORAGE, _SF_VEC_SIZE, _storage_per_batch,
    )
    k_tiles = _div_up(K, _SF_TILE_K)
    per = _storage_per_batch(TK, K)
    aligned = (TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0)
    dout_scales_tk = (
        torch.empty((1, per), dtype=torch.uint8, device=device)
        if aligned
        else torch.full((1, per), 127, dtype=torch.uint8, device=device)
    )
    _gather_isa_packed_scales_kernel[(_div_up(TK, 128), k_tiles)](
        dout_scales.view(torch.uint8), x_gather_idx, dout_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=128, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    _E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
    dout_scales_gathered = dout_scales_tk.view(_E8M0_DTYPE)

    # iso32 outputs
    dz_fp8 = torch.zeros(TK, 2 * N, dtype=torch.float8_e4m3fn, device=device)
    row_per = _storage_per_batch(TK, 2 * N)
    col_per = _storage_per_batch(2 * N, TK)
    dz_row_scales = torch.zeros((1, row_per), dtype=torch.uint8, device=device)
    dz_col_scales = torch.zeros((1, col_per), dtype=torch.uint8, device=device)

    # 3D view for the mixin's ISA-pack stores: (num_m_tiles, k_tiles_n, 512)
    n_k_tiles = _div_up(2 * N, _SF_TILE_K)
    num_m_tiles = _div_up(TK, _SF_TILE_M)
    dz_row_scales_3d = dz_row_scales.view(num_m_tiles, n_k_tiles, _SF_TILE_STORAGE)

    col_k_tiles = _div_up(TK, _SF_TILE_K)
    num_n_tiles = _div_up(2 * N, _SF_TILE_M)
    dz_col_scales_3d = dz_col_scales.view(num_n_tiles, col_k_tiles, _SF_TILE_STORAGE)

    return dict(
        dout_fp8=dout_fp8,
        w2_fp8=w2_fp8,
        z_fp8=z_fp8,
        z_scales_u8=z_scales_u8,
        cu_seqlens=cu,
        x_gather_idx=x_gather_idx,
        dout_scales=dout_scales_gathered,
        w2_scales=w2_scales,
        dz_fp8=dz_fp8,
        dz_row_scales=dz_row_scales,
        dz_col_scales=dz_col_scales,
        dz_row_scales_3d=dz_row_scales_3d,
        dz_col_scales_3d=dz_col_scales_3d,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)
def test_direct_dgated_uses_transposed_iso32_scales():
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _ISO32_WEIGHT_CACHE,
        _cache_iso32_w2,
        _simple_weight_key,
        precompute_weight_fp8_for_direct_fused_dgated,
    )

    H, I, E = 256, 384, 2
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.1
    _ISO32_WEIGHT_CACHE.clear()
    _cache_iso32_w2(w2)
    cached_fp8, row_scales, col_scales = _ISO32_WEIGHT_CACHE[_simple_weight_key(w2)]

    w2_fp8, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)

    assert w2_fp8.data_ptr() == cached_fp8.data_ptr()
    assert tuple(w2_fp8.shape) == (E, I, H)
    assert w2_scales.data_ptr() == col_scales.data_ptr()
    assert w2_scales.data_ptr() != row_scales.data_ptr()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)
def test_dgated_iso32_jit_smoke():
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel

    # Production-ish shape (small)
    TK, K, N, E = 1024, 1024, 512, 4
    state = _build_inputs(TK, K, N, E)

    # Output buffers
    device = state["dout_fp8"].device
    dz = torch.zeros(TK, 2 * N, dtype=torch.bfloat16, device=device)
    y1s = torch.zeros(TK, N, dtype=torch.bfloat16, device=device)

    # default config for this shape — borrow GemmDGatedFP8CLoadSm100ZeroMat's autotuner
    tile_m, tile_n = 128, 128
    cluster_m, cluster_n = 1, 1

    s_float = torch.ones(TK, dtype=torch.float32, device=device)
    colvec_reduce = torch.zeros(
        (TK, (N + tile_n - 1) // tile_n), dtype=torch.float32, device=device
    )

    gemm_dgated_kernel(
        state["dout_fp8"],
        state["w2_fp8"],
        dz,
        dz,  # PreAct is unused in fp8_preact_mode (overwritten by preact_fp8)
        y1s,
        None,
        "swiglu",
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        True,
        persistent=True,
        max_swizzle_size=8,
        colvec_scale=s_float,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=state["cu_seqlens"],
        A_idx=state["x_gather_idx"],
        a_scales=state["dout_scales"],
        b_scales=state["w2_scales"],
        preact_fp8=state["z_fp8"],
        preact_scales=state["z_scales_u8"],
        iso32_dz_fp8=state["dz_fp8"],
        iso32_dz_row_scales=state["dz_row_scales_3d"],
        iso32_dz_col_scales=state["dz_col_scales_3d"],
    )
    torch.cuda.synchronize()

    # Smoke: dz_fp8 + scales must be touched (non-all-zero) on this random input
    assert state["dz_fp8"].view(torch.uint8).to(torch.int32).any(), "dz_fp8 not written"
    assert state["dz_row_scales"].to(torch.int32).any(), "dz_row_scales not written"
    assert state["dz_col_scales"].to(torch.int32).any(), "dz_col_scales not written"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v", "-s"]))
