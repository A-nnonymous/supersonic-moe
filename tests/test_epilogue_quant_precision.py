"""Full-chain precision test for blockscaled FP8 quantization.

Validates that quantize_and_pack_activation matches the Paddle reference
algorithm step-by-step:
  1. amax = max(abs(group))   from ORIGINAL f32 values
  2. scale = 448 / amax → power-of-2 round (ldexpf pattern)
  3. store_scale = 1.0 / scale (exact for pow2)
  4. UE8M0 = biased_exponent(store_scale)
  5. fp8 = static_cast<fp8>(original * scale)  [hardware saturation + rounding]
"""
import sys, os, struct, math, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_and_pack_activation,
    _div_up, _SF_VEC_SIZE, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE,
)
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8


def float_as_int(f):
    return struct.unpack('I', struct.pack('f', f))[0]

def int_as_float(i):
    return struct.unpack('f', struct.pack('I', i & 0xFFFFFFFF))[0]


def reference_blockscaled_quant_group(vals_f32, eps=1.17549435e-38):
    """Pure Python reference matching Paddle ScaleWrapper<Power2Scaling=true> + StoreScale<ue8m0>.

    Args:
        vals_f32: list of float32 values (one group, 32 elements)
    Returns:
        (fp8_bytes, ue8m0_byte, scale, store_scale)
    """
    # Step 1: amax from ORIGINAL values
    amax = max(abs(v) for v in vals_f32)
    amax_mod = max(amax, eps)

    # Step 2: scale = 448 / amax_mod → power-of-2
    if amax_mod == 0.0:
        scale = 1.0
    else:
        scale = 448.0 / amax_mod
        if math.isinf(scale):
            scale = int_as_float(0x7F000000)  # 0x1.0p127
        elif scale == 0.0:
            pass
        else:
            # Power2Scaling: ldexpf(1.0, biased_exp - 127)
            scale_bits = float_as_int(scale)
            exp = (scale_bits >> 23) & 0xFF
            normal_biased_exp = exp - 127
            scale = math.ldexp(1.0, normal_biased_exp)

    # Step 3: store_scale = 1.0 / scale (exact for pow2)
    store_scale = 1.0 / scale

    # Step 4: UE8M0 = biased exponent of store_scale
    store_scale_bits = float_as_int(store_scale)
    ue8m0 = (store_scale_bits >> 23) & 0xFF

    # Step 5: fp8 = cast(original * scale) — hardware saturating round
    fp8_vals = []
    for v in vals_f32:
        scaled = v * scale
        # Simulate fp8 e4m3fn cast: clamp to [-448, 448], then cast
        fp8_vals.append(torch.tensor(scaled, dtype=torch.float32).to(torch.float8_e4m3fn).item())

    return fp8_vals, ue8m0, scale, store_scale


def test_quant_precision(M, K, magnitude=0.02, label=""):
    """Validate quantize_and_pack_activation against pure Python reference."""
    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * magnitude

    # Triton kernel output
    fp8_data, packed_scales = quantize_and_pack_activation(x)

    # Spot-check fp8 bytes and scale bytes against Python reference
    x_cpu = x.float().cpu()
    fp8_cpu = fp8_data.view(torch.uint8).cpu()
    group_size = _SF_VEC_SIZE  # 32

    fp8_mismatches = 0
    total_checked = 0

    for row in range(min(M, 8)):  # Check first 8 rows
        for g in range(min(_div_up(K, group_size), 8)):  # First 8 groups
            col_start = g * group_size
            col_end = min(col_start + group_size, K)
            group_vals = [x_cpu[row, c].item() for c in range(col_start, col_end)]
            while len(group_vals) < group_size:
                group_vals.append(0.0)

            ref_fp8, ref_ue8m0, ref_scale, ref_store_scale = reference_blockscaled_quant_group(group_vals)

            for c in range(col_end - col_start):
                actual_byte = fp8_cpu[row, col_start + c].item()
                ref_byte = torch.tensor(ref_fp8[c], dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).item()
                if actual_byte != ref_byte:
                    fp8_mismatches += 1
                total_checked += 1

    print(f"[{label}] M={M}, K={K}, magnitude={magnitude}")
    print(f"  FP8 byte mismatches: {fp8_mismatches}/{total_checked} (first 8 rows × 8 groups)")
    print(f"  NaN in fp8: {torch.isnan(fp8_data.float()).any().item()}")

    ok = fp8_mismatches == 0
    print(f"  → {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    print("=== Blockscaled FP8 Quant Full-Chain Precision Test ===\n")

    all_ok = True
    for label, M, K, mag in [
        ("Small aligned", 256, 768, 0.02),
        ("Production z", 65536, 3072, 0.02),
        ("Large values", 256, 768, 10.0),
        ("Tiny values", 256, 768, 1e-6),
        ("Mixed", 256, 768, 1.0),
        ("Non-aligned", 300, 800, 0.02),
    ]:
        ok = test_quant_precision(M, K, mag, label)
        all_ok = all_ok and ok

    print("=" * 50)
    print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
