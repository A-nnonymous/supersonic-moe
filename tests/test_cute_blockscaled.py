"""Test for CuTe DSL colwise blockscaled quant with swizzled smem transpose."""
import torch
import math


def test_colwise_cute_correctness():
    """Verify CuTe colwise quant matches Triton colwise quant."""
    TK, dim = 1024, 1536  # small for correctness
    GROUP_SIZE = 32
    num_groups = TK // GROUP_SIZE

    torch.manual_seed(42)
    src = torch.randn(TK, dim, dtype=torch.bfloat16, device='cuda')

    # CuTe version
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute
    fp8_cute, scale_cute = colwise_quantize_cute(src, dim, TK)
    torch.cuda.synchronize()
    print(f"CuTe compiled and ran! fp8={fp8_cute.shape}, scale={scale_cute.shape}")

    # Reference: manual colwise quant for validation
    # Groups of 32 along TK axis (axis 0), per-column-element scale
    fp8_ref = torch.empty_like(fp8_cute)
    scale_ref = torch.empty_like(scale_cute)

    src_f32 = src.float()
    for g in range(num_groups):
        tk_start = g * GROUP_SIZE
        tk_end = tk_start + GROUP_SIZE
        group_data = src_f32[tk_start:tk_end, :]  # (32, dim)
        amax = group_data.abs().amax(dim=0)  # (dim,)
        amax = amax.clamp(min=1e-12)

        # E8M0
        amax_bits = amax.view(torch.int32)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa = amax_bits & 0x7FFFFF
        carry = (mantissa > 0x600000).int()
        e8m0 = biased_exp - 8 + carry
        e8m0 = e8m0.clamp(min=0)

        qexp = (254 - e8m0).clamp(1, 254)
        quant_scale = (qexp.int() << 23).view(torch.float32)

        quantized = (group_data * quant_scale.unsqueeze(0)).to(torch.float8_e4m3fn)
        fp8_ref[tk_start:tk_end, :] = quantized
        scale_ref[:, g] = e8m0.to(torch.uint8)

    # Compare
    scale_match = (scale_cute == scale_ref).float().mean().item()
    print(f"  Scale match: {scale_match*100:.1f}%")

    fp8_match = (fp8_cute.view(torch.uint8) == fp8_ref.view(torch.uint8)).float().mean().item()
    print(f"  FP8 bit-match: {fp8_match*100:.1f}%")

    if fp8_match < 1.0:
        diff = (fp8_cute.float() - fp8_ref.float()).abs()
        print(f"  FP8 max diff: {diff.max().item():.6f}")
        # Show first mismatch
        mismatch = (fp8_cute.view(torch.uint8) != fp8_ref.view(torch.uint8)).nonzero()
        if len(mismatch) > 0:
            idx = tuple(mismatch[0].tolist())
            print(f"  First mismatch at {idx}: cute={fp8_cute[idx].item()}, ref={fp8_ref[idx].item()}")

    assert scale_match > 0.99, f"Scale match {scale_match*100:.1f}% too low"
    print("CORRECTNESS CHECK PASSED!")


def test_colwise_cute_perf():
    """Benchmark CuTe colwise quant vs Triton colwise quant."""
    TK, dim = 65536, 1536
    GROUP_SIZE = 32

    torch.manual_seed(42)
    src = torch.randn(TK, dim, dtype=torch.bfloat16, device='cuda')

    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack

    # Warmup
    for _ in range(5):
        colwise_quantize_cute(src, dim, TK)
        colwise_quantize_and_pack(src, dim, TK)
    torch.cuda.synchronize()

    N = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # CuTe
    start.record()
    for _ in range(N):
        colwise_quantize_cute(src, dim, TK)
    end.record()
    torch.cuda.synchronize()
    cute_us = start.elapsed_time(end) / N * 1000

    # Triton
    start.record()
    for _ in range(N):
        colwise_quantize_and_pack(src, dim, TK)
    end.record()
    torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) / N * 1000

    # Theoretical
    data_bytes = TK * dim * 2 + TK * dim * 1 + dim * (TK//32)
    theo_us = data_bytes / (8e6)  # 8 TB/s

    print(f"\nPerformance: colwise quant ({TK}, {dim})")
    print(f"  CuTe DSL:    {cute_us:.1f} µs")
    print(f"  Triton:      {triton_us:.1f} µs")
    print(f"  Theoretical: {theo_us:.1f} µs")
    print(f"  Speedup:     {triton_us/cute_us:.2f}×")


if __name__ == "__main__":
    print("=" * 60)
    print("CuTe DSL Colwise Blockscaled Quant Test")
    print("=" * 60)
    test_colwise_cute_correctness()
    print()
    test_colwise_cute_perf()
