"""Minimal compilation test for GemmDGated FP8 C Load via TMA.

Tests that the TMA atom creation succeeds (the prior blocker was
smem layout / CTA V-map shape mismatch with integer epi_tile).
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated

E, H, I = 8, 3072, 1536
TK = 1024  # Small for quick compilation test

torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2 * I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc.view(torch.uint8)

cu = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    torch.full((E,), TK // E, dtype=torch.int32, device="cuda").cumsum(0),
]).int()

df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)

dx = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
pa = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")

print("=" * 60)
print(f"FP8 C TMA Compilation Test (TK={TK}, E={E}, H={H}, I={I})")
print("=" * 60)

print("\n1. Testing standard BF16 C path (baseline)...")
gemm_dgated(
    df, wf, dx, z, pa,
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    "swiglu", 128, 128, 1, 1,
    cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
)
print("   BF16 C path: OK")

print("\n2. Testing FP8 C TMA path...")
gemm_dgated.compile_cache.clear()
try:
    gemm_dgated(
        df, wf, dx, z, pa,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
        preact_fp8=z_fp8, preact_scales=z_sc_u8,
    )
    print("   FP8 C TMA path: COMPILED AND RAN SUCCESSFULLY!")
except Exception as e:
    print(f"   FP8 C TMA path: FAILED — {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Quick precision check...")
try:
    # BF16 reference
    from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
    dx_ref = torch.empty_like(dx)
    pa_ref = torch.empty_like(pa)
    gemm_dgated.compile_cache.clear()
    gemm_dgated(
        df, wf, dx_ref, z_bf16, pa_ref,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
    )
    # FP8 result (already in dx, pa)
    gemm_dgated.compile_cache.clear()
    gemm_dgated(
        df, wf, dx, z, pa,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
        preact_fp8=z_fp8, preact_scales=z_sc_u8,
    )
    dx_rrmse = ((dx.float() - dx_ref.float()).pow(2).mean().sqrt() /
                dx_ref.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    pa_rrmse = ((pa.float() - pa_ref.float()).pow(2).mean().sqrt() /
                pa_ref.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    print(f"   dx RRMSE: {dx_rrmse:.6f}")
    print(f"   pa RRMSE: {pa_rrmse:.6f}")
    if dx_rrmse < 0.05 and pa_rrmse < 0.05:
        print("   PRECISION: PASS")
    else:
        print("   PRECISION: NEEDS INVESTIGATION (high RRMSE)")
except Exception as e:
    print(f"   Precision check skipped — {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
