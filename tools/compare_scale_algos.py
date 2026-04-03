"""Compare the two scale computation algorithms for UE8M0 consistency."""
import struct, random

def triton_method(amax_f32):
    """From _quantize_and_pack_kernel: integer bit manipulation."""
    bits = struct.unpack("I", struct.pack("f", amax_f32))[0]
    biased_exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    carry = 1 if mantissa > 0x600000 else 0
    e8m0 = biased_exp - 8 + carry
    if biased_exp == 0:
        e8m0 = 0
    e8m0 = max(e8m0, 0)
    return e8m0

def epilogue_method(amax_f32):
    """From epilogue override: fdiv 448/amax → pow2 → inv_scale → exp bits."""
    if amax_f32 == 0:
        return 127
    scale = 448.0 / amax_f32
    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]
    scale_exp = (scale_bits >> 23) & 0xFF
    scale_pow2 = struct.unpack("f", struct.pack("I", scale_exp << 23))[0]
    amax_scaled = amax_f32 * scale_pow2
    inv_scale = amax_scaled / 448.0
    inv_bits = struct.unpack("I", struct.pack("f", inv_scale))[0]
    ue8m0 = (inv_bits >> 23) & 0xFF
    return ue8m0

def paddle_reference(amax_f32):
    """From Paddle ScaleWrapper<Power2Scaling=true> + StoreScale<ue8m0>."""
    if amax_f32 == 0:
        return 127
    scale = 448.0 / max(amax_f32, 1.17549435e-38)
    if scale == float('inf'):
        return 0  # scale=2^127, store_scale=2^-127, exp=0
    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]
    exp = (scale_bits >> 23) & 0xFF
    normal_biased_exp = exp - 127
    import math
    scale_pow2 = math.ldexp(1.0, normal_biased_exp)
    store_scale = 1.0 / scale_pow2
    store_bits = struct.unpack("I", struct.pack("f", store_scale))[0]
    ue8m0 = (store_bits >> 23) & 0xFF
    return ue8m0

random.seed(42)
mismatches_te = 0
mismatches_tp = 0
mismatches_ep = 0
total = 100000

for _ in range(total):
    amax = random.uniform(1e-10, 1000.0)
    t = triton_method(amax)
    e = epilogue_method(amax)
    p = paddle_reference(amax)
    if t != e:
        mismatches_te += 1
    if t != p:
        mismatches_tp += 1
    if e != p:
        mismatches_ep += 1

print(f"Scale algorithm comparison ({total} random amax values):")
print(f"  Triton vs Epilogue:  {mismatches_te} mismatches")
print(f"  Triton vs Paddle:    {mismatches_tp} mismatches")
print(f"  Epilogue vs Paddle:  {mismatches_ep} mismatches")

if mismatches_te == 0 and mismatches_ep == 0:
    print("ALL THREE ALGORITHMS PRODUCE IDENTICAL UE8M0!")
elif mismatches_ep == 0:
    print("Epilogue matches Paddle reference exactly!")
else:
    print("WARNING: algorithms differ — need investigation")

    # Show first few mismatches
    random.seed(42)
    shown = 0
    for _ in range(total):
        amax = random.uniform(1e-10, 1000.0)
        t, e, p = triton_method(amax), epilogue_method(amax), paddle_reference(amax)
        if e != p and shown < 5:
            print(f"  amax={amax:.8e}: triton={t} epilogue={e} paddle={p}")
            shown += 1
