"""Test: can CUTLASS GemmGatedSm100ZeroMatBlockscaledQuant output fp8 D?

If yes, we can eliminate the standalone z quant kernel entirely (~141µs savings).
"""
import os
os.environ["USE_QUACK_GEMM"] = "1"
import sys
import torch

def test_fp8_d_output():
    """Test if CUTLASS accepts fp8 preact_out (D tensor)."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        _gather_isa_packed_scales_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
        precompute_weight_fp8_for_fused_gated,
    )
    from sonicmoe.quack_utils import gemm_gated

    torch.manual_seed(42)
    device = "cuda:0"

    # Ernie shape
    T, H, I, E, K = 8192, 3072, 1536, 8, 8
    N = 2 * I  # gated
    TK = T * K  # assume all experts active

    x = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(N, H, E, device=device, dtype=torch.bfloat16) * 0.01
    expert_freq = torch.tensor(
        [i * (TK // E) for i in range(E + 1)], dtype=torch.int32, device=device
    )
    x_gather_idx = torch.arange(TK, device=device, dtype=torch.int32) % T

    # FP8 weight cache
    w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)

    # Quantize activation
    x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Gather scales T→TK
    k_tiles = _div_up(H, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(TK, H)
    x_scales_tk = torch.empty((1, per_batch_tk), dtype=torch.uint8, device=device)
    BLOCK_ROWS = 128
    _gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK_ROWS), k_tiles)](
        x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    x_scales_tk_e8m0 = x_scales_tk.view(torch.float8_e8m0fnu)

    # z_scale_out for epilogue quant
    z_scale_out = torch.empty(TK, N // 32, dtype=torch.uint8, device=device)

    # --- Test 1: bf16 D (current, should work) ---
    print("Test 1: bf16 D output with epilogue quant...")
    z_bf16, y1_bf16 = gemm_gated(
        x_fp8, w1_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=expert_freq,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        dynamic_scheduler=False,
        tuned=False,
        z_scale_out=z_scale_out,
    )
    print(f"  z shape={z_bf16.shape}, dtype={z_bf16.dtype}")
    print(f"  y1 shape={y1_bf16.shape}, dtype={y1_bf16.dtype}")
    z_bf16_fp8 = z_bf16.to(torch.float8_e4m3fn)
    print(f"  z cast to fp8: OK, dtype={z_bf16_fp8.dtype}")

    # --- Test 2: fp8 D output (new, key optimization) ---
    print("\nTest 2: fp8 D output with epilogue quant...")
    z_scale_out2 = torch.empty(TK, N // 32, dtype=torch.uint8, device=device)
    try:
        z_fp8, y1_fp8test = gemm_gated(
            x_fp8, w1_fp8,
            activation="swiglu",
            out_dtype=torch.float8_e4m3fn,
            postact_dtype=torch.bfloat16,
            cu_seqlens_m=expert_freq,
            A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0,
            b_scales=w1_scales,
            dynamic_scheduler=False,
            tuned=False,
            z_scale_out=z_scale_out2,
        )
        print(f"  z shape={z_fp8.shape}, dtype={z_fp8.dtype}")
        print(f"  y1 shape={y1_fp8test.shape}, dtype={y1_fp8test.dtype}")
        print("  SUCCESS: CUTLASS supports fp8 D output!")

        # Compare: z_bf16→fp8 vs direct fp8 D
        # Both should produce the same fp8 values (both use fp32→fp8 saturating cast
        # after the same epilogue quant scaling)
        match = (z_bf16_fp8.view(torch.uint8) == z_fp8.view(torch.uint8)).float().mean()
        print(f"  Byte-exact match rate: {match:.4f}")

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  -> fp8 D output not supported, will use alternative optimization")

    # --- Test 3: No D output (store_preact=False) ---
    print("\nTest 3: No D output (skip z store entirely)...")
    z_scale_out3 = torch.empty(TK, N // 32, dtype=torch.uint8, device=device)
    try:
        z_none, y1_only = gemm_gated(
            x_fp8, w1_fp8,
            activation="swiglu",
            out_dtype=torch.bfloat16,
            postact_dtype=torch.bfloat16,
            cu_seqlens_m=expert_freq,
            A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0,
            b_scales=w1_scales,
            dynamic_scheduler=False,
            tuned=False,
            z_scale_out=z_scale_out3,
            store_preact=False,
        )
        print(f"  z={z_none}, y1 shape={y1_only.shape}")
        print("  SUCCESS: No-preact mode works!")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_fp8_d_output()
