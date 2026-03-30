"""Benchmark decomposed backward dSwiGLU path vs fused path at production shape."""
import torch, time

TK, two_I, I = 32768, 2048, 1024

z = torch.randn(TK, two_I, device='cuda', dtype=torch.bfloat16)
dy1 = torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
s = torch.randn(TK, device='cuda', dtype=torch.float32).abs()

from sonicmoe.quack_utils.swiglu_triton import (
    swiglu_backward_quant_pack_triton,
    dequantize_blockscaled_fp8,
    swiglu_backward_from_fp8_quant_pack_triton,
    swiglu_forward_quant_pack_zsave_triton,
)

def bench(fn, warmup=5, iters=20, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1-t0)/iters*1000
    print(f"{label}: {ms:.3f} ms")
    return ms

# 1. bwd_quant_pack without dz_bf16
t_no = bench(lambda: swiglu_backward_quant_pack_triton(dy1, z, s, return_dz_bf16=False),
             label="bwd_quant_pack (no dz_bf16) ")

# 2. bwd_quant_pack WITH dz_bf16
t_with = bench(lambda: swiglu_backward_quant_pack_triton(dy1, z, s, return_dz_bf16=True),
               label="bwd_quant_pack (with dz_bf16)")
print(f"  dz_bf16 overhead: {t_with-t_no:.3f} ms ({(t_with/t_no - 1)*100:.1f}%)")

# 3. Create fp8 z with raw scales for backward
_, _, z_fp8, z_scales_e8m0 = swiglu_forward_quant_pack_zsave_triton(z)
z_raw_u8 = z_scales_e8m0.view(torch.uint8)

# 4. Decomposed: dequant(z_fp8) + bwd_quant_pack(with dz_bf16)
def decomposed_with_dz():
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_u8)
    return swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=True)
t_decomposed = bench(decomposed_with_dz, label="Decomposed (dequant+bwd+dz) ")

# 5. Decomposed: dequant(z_fp8) + bwd_quant_pack(no dz_bf16)
def decomposed_no_dz():
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_u8)
    return swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=False)
t_decomposed_no_dz = bench(decomposed_no_dz, label="Decomposed (dequant+bwd)     ")

# 6. Old fused path
def fused_old():
    return swiglu_backward_from_fp8_quant_pack_triton(dy1, z_fp8, z_raw_u8, s)
t_fused = bench(fused_old, label="Fused (old path)              ")

# 7. Standalone dequant
t_dequant = bench(lambda: dequantize_blockscaled_fp8(z_fp8, z_raw_u8),
                  label="dequant only                  ")

print(f"\nSummary at TK={TK}, I={I}, two_I={two_I}:")
print(f"  Fused (all-in-one):          {t_fused:.3f} ms")
print(f"  Decomposed with dz_bf16:     {t_decomposed:.3f} ms ({t_fused/t_decomposed:.2f}x vs fused)")
print(f"  Decomposed no dz_bf16:       {t_decomposed_no_dz:.3f} ms ({t_fused/t_decomposed_no_dz:.2f}x vs fused)")
print(f"  bwd kernel only (no dz):     {t_no:.3f} ms")
print(f"  bwd kernel only (with dz):   {t_with:.3f} ms")
print(f"  dequant kernel:              {t_dequant:.3f} ms")
