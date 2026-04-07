"""Direct test: epilogue quant scale write correctness.

Calls gemm_gated directly (not through MoE) to isolate the epilogue quant
and compare scale output with standalone quantize_activation_blockscaled_fast.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_EPILOGUE_QUANT"] = "1"

from sonicmoe.quack_utils.gemm_interface import gemm_gated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
    _gather_isa_packed_scales_kernel,
    _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
    _storage_per_batch,
)

torch.manual_seed(42)

# Production-representative shape (128-aligned segments)
E, K_topk, H, I = 8, 8, 3072, 1536
T = 2048
TK = T * K_topk  # 16384

# Create inputs
x = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
w1 = 0.02 * torch.randn(2 * I, H, E, device="cuda", dtype=torch.bfloat16)

# Create fake routing metadata (int32 cu_seqlens matching production format)
expert_freq = torch.full((E,), TK // E, dtype=torch.int32, device="cuda")
expert_frequency_offset = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    expert_freq.cumsum(0)
]).to(torch.int32)
x_gather_idx = (torch.arange(TK, dtype=torch.int32, device="cuda") % T).to(torch.int32)

# Prepare FP8 inputs (same as _fused_blockscaled_gated_forward)
w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)
x_fp8, x_scales_t = quantize_and_pack_activation(x)

k_tiles = _div_up(H, _SF_TILE_K)
per_batch_tk = _storage_per_batch(TK, H)
x_scales_tk = torch.full((1, per_batch_tk), 127, dtype=torch.uint8, device=x.device)
BLOCK_ROWS = 32
_gather_isa_packed_scales_kernel[(_div_up(TK, BLOCK_ROWS), k_tiles)](
    x_scales_t.view(torch.uint8), x_gather_idx, x_scales_tk, TK,
    src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
    SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
    BLOCK_ROWS=BLOCK_ROWS, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
)
x_scales_tk_e8m0 = x_scales_tk.view(torch.float8_e8m0fnu)

N = 2 * I  # preactivation width

# ── Step 1: Run without epilogue quant → get reference z ──
print("Step 1: Reference (no epilogue quant)...")
z_ref, y1_ref = gemm_gated(
    x_fp8, w1_fp8,
    activation="swiglu",
    out_dtype=torch.bfloat16,
    postact_dtype=torch.bfloat16,
    cu_seqlens_m=expert_frequency_offset,
    A_idx=x_gather_idx,
    a_scales=x_scales_tk_e8m0,
    b_scales=w1_scales,
    dynamic_scheduler=False,
    tuned=False,
)
torch.cuda.synchronize()
print(f"  z_ref: {z_ref.shape} {z_ref.dtype}")
print(f"  y1_ref: {y1_ref.shape} {y1_ref.dtype}")

# Standalone quant of reference z
z_fp8_ref, z_scales_ref = quantize_activation_blockscaled_fast(z_ref)
z_scales_ref_u8 = z_scales_ref.view(torch.uint8)
print(f"  z_scales_ref: {z_scales_ref_u8.shape}, range=[{z_scales_ref_u8.min().item()}, {z_scales_ref_u8.max().item()}]")

# ── Step 2: Run WITH epilogue quant → get z + scales ──
print("\nStep 2: Epilogue quant...")
from sonicmoe.quack_utils.gemm_gated import gemm_gated as _gg
_gg.compile_cache.clear()

z_scale_out = torch.zeros(TK, N // 32, dtype=torch.uint8, device="cuda")
z_epi, y1_epi = gemm_gated(
    x_fp8, w1_fp8,
    activation="swiglu",
    out_dtype=torch.bfloat16,
    postact_dtype=torch.bfloat16,
    cu_seqlens_m=expert_frequency_offset,
    A_idx=x_gather_idx,
    a_scales=x_scales_tk_e8m0,
    b_scales=w1_scales,
    dynamic_scheduler=False,
    tuned=False,
    z_scale_out=z_scale_out,
)
torch.cuda.synchronize()
print(f"  z_epi: {z_epi.shape} {z_epi.dtype}")
print(f"  z_scale_out: {z_scale_out.shape}, range=[{z_scale_out.min().item()}, {z_scale_out.max().item()}]")
print(f"  z_scale_out nonzero: {(z_scale_out > 0).sum().item()} / {z_scale_out.numel()}")
print(f"  z_scale_out all-zero rows: {(z_scale_out.sum(dim=1) == 0).sum().item()} / {z_scale_out.shape[0]}")

# ── Step 3: Compare ──
print("\n--- Comparison ---")

# y1 should be identical (epilogue quant doesn't change SwiGLU output)
y1_match = torch.allclose(y1_ref, y1_epi, atol=0, rtol=0)
print(f"y1 exact match: {y1_match}")
if not y1_match:
    diff = (y1_ref.float() - y1_epi.float()).abs()
    print(f"  y1 max diff: {diff.max().item():.6e}")

# Scale comparison
if z_scales_ref_u8.shape == z_scale_out.shape:
    match_rate = (z_scales_ref_u8 == z_scale_out).float().mean().item()
    mismatch = (z_scales_ref_u8 != z_scale_out).sum().item()
    print(f"Scale match rate: {match_rate*100:.2f}% ({z_scale_out.numel() - mismatch}/{z_scale_out.numel()})")
    if mismatch > 0 and mismatch < 20:
        rows, cols = torch.where(z_scales_ref_u8 != z_scale_out)
        for i in range(min(10, len(rows))):
            r, c = rows[i].item(), cols[i].item()
            print(f"  [{r},{c}]: ref={z_scales_ref_u8[r,c].item()}, epi={z_scale_out[r,c].item()}")
    elif mismatch > 0:
        # Analyze mismatch pattern
        diff = z_scale_out.int() - z_scales_ref_u8.int()
        print(f"  Mismatch diff distribution:")
        for d in [-2, -1, 0, 1, 2]:
            count = (diff == d).sum().item()
            if count > 0:
                print(f"    diff={d:+d}: {count}")
        # Check if mismatches are always ±1 (rounding boundary)
        abs_diff = diff.abs()
        max_diff = abs_diff.max().item()
        print(f"  Max absolute diff: {max_diff}")
        if max_diff <= 1:
            print(f"  ALL mismatches are ±1 → rounding boundary differences (acceptable)")

# Step 4: Verify backward dequant works with epilogue scales
print("\nStep 4: Backward dequant test...")
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
z_fp8_epi = z_epi.to(torch.float8_e4m3fn)
try:
    z_bf16_from_epi = dequantize_blockscaled_fp8(z_fp8_epi, z_scale_out)
    print(f"  dequant success: {z_bf16_from_epi.shape} {z_bf16_from_epi.dtype}")
    # Compare with reference dequant
    z_bf16_from_ref = dequantize_blockscaled_fp8(z_fp8_ref, z_scales_ref_u8)
    rrmse = ((z_bf16_from_epi.float() - z_bf16_from_ref.float()).norm()
             / z_bf16_from_ref.float().norm().clamp(min=1e-8)).item()
    print(f"  dequant RRMSE vs ref: {rrmse:.6f}")
except Exception as e:
    print(f"  dequant FAILED: {e}")
else:
    print(f"Shape mismatch: ref={z_scales_ref_u8.shape} vs epi={z_scale_out.shape}")

print("\nDONE")
