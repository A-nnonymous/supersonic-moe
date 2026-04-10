"""Rigorous precision test: rcp.approx E8M0 vs integer bitops E8M0."""
import torch
import struct
import math


def test_rcp_e8m0_precision():
    """Compare rcp.approx E8M0 with integer bitops E8M0 across all relevant amax ranges."""
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute

    torch.manual_seed(42)
    TK, dim = 1024, 1536

    # Test 1: Random data
    src = torch.randn(TK, dim, dtype=torch.bfloat16, device='cuda')
    fp8_rcp, scale_rcp = colwise_quantize_cute(src, dim, TK, use_rcp=True)
    fp8_int, scale_int = colwise_quantize_cute(src, dim, TK, use_rcp=False)

    scale_match = (scale_rcp == scale_int).float().mean().item()
    fp8_match = (fp8_rcp.view(torch.uint8) == fp8_int.view(torch.uint8)).float().mean().item()
    print(f"Random data: scale match={scale_match*100:.2f}%, fp8 match={fp8_match*100:.2f}%")

    if scale_match < 1.0:
        mismatch = (scale_rcp != scale_int).nonzero()
        for idx in mismatch[:5]:
            d, g = idx[0].item(), idx[1].item()
            group_data = src[g*32:(g+1)*32, d].float()
            amax = group_data.abs().max().item()
            print(f"  dim={d}, group={g}: amax={amax:.6e}, rcp_e8m0={scale_rcp[d,g].item()}, int_e8m0={scale_int[d,g].item()}")

    # Test 2: Stress test with extreme values
    for val_range, name in [
        ((1e-6, 1e-3), "tiny"),
        ((0.1, 10.0), "normal"),
        ((100, 10000), "large"),
        ((1e-38, 1e-30), "near-subnormal"),
    ]:
        src_stress = torch.empty(TK, dim, dtype=torch.bfloat16, device='cuda').uniform_(*val_range)
        fp8_r, sc_r = colwise_quantize_cute(src_stress, dim, TK, use_rcp=True)
        fp8_i, sc_i = colwise_quantize_cute(src_stress, dim, TK, use_rcp=False)
        sm = (sc_r == sc_i).float().mean().item()
        fm = (fp8_r.view(torch.uint8) == fp8_i.view(torch.uint8)).float().mean().item()
        print(f"  {name:>15s} [{val_range[0]:.0e},{val_range[1]:.0e}]: scale={sm*100:.2f}%, fp8={fm*100:.2f}%")

    # Test 3: Edge case — values near power-of-2 boundaries (where carry matters)
    # amax near 1.5, 3.0, 6.0 etc. (mantissa = 0.5 = 0x400000, threshold = 0x600000)
    edge_vals = [1.0, 1.5, 1.75, 1.875, 2.0, 3.0, 3.5, 4.0, 0.5, 0.75, 0.875]
    print("\n  Edge cases (amax near power-of-2 boundaries):")
    for amax_target in edge_vals:
        src_edge = torch.full((32, 1), amax_target, dtype=torch.bfloat16, device='cuda')
        fp8_r, sc_r = colwise_quantize_cute(src_edge, 1, 32, use_rcp=True)
        fp8_i, sc_i = colwise_quantize_cute(src_edge, 1, 32, use_rcp=False)
        match = "✓" if (sc_r == sc_i).all() else "✗"
        print(f"    amax={amax_target:>8.4f}: rcp_e8m0={sc_r[0,0].item():>3d}, int_e8m0={sc_i[0,0].item():>3d} {match}")


def test_correctness_vs_reference():
    """Verify both rcp and int versions against manual reference."""
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute

    torch.manual_seed(123)
    TK, dim = 1024, 512

    src = torch.randn(TK, dim, dtype=torch.bfloat16, device='cuda')
    fp8_rcp, scale_rcp = colwise_quantize_cute(src, dim, TK, use_rcp=True)
    fp8_int, scale_int = colwise_quantize_cute(src, dim, TK, use_rcp=False)

    # Manual reference using integer bitops
    src_f32 = src.float()
    for g in range(TK // 32):
        group_data = src_f32[g*32:(g+1)*32, :]
        amax = group_data.abs().amax(dim=0).clamp(min=1e-12)
        ab = amax.view(torch.int32)
        be = (ab >> 23) & 0xFF
        m = ab & 0x7FFFFF
        c = (m > 0x600000).int()
        e = (be - 8 + c).clamp(min=0)
        ref_e8m0 = e.to(torch.uint8)

        int_ok = (scale_int[:, g] == ref_e8m0).all()
        rcp_vs_ref = (scale_rcp[:, g] == ref_e8m0).float().mean().item()

        if not int_ok:
            print(f"  WARNING: int_e8m0 mismatch at group {g}!")
        if rcp_vs_ref < 1.0 and g < 5:
            # Show mismatches
            mismatch_mask = scale_rcp[:, g] != ref_e8m0
            n_mismatch = mismatch_mask.sum().item()
            print(f"  Group {g}: rcp vs ref match={rcp_vs_ref*100:.2f}% ({n_mismatch} mismatches)")
            for idx in mismatch_mask.nonzero()[:3]:
                d = idx[0].item()
                print(f"    dim={d}: amax={amax[d].item():.6e}, rcp={scale_rcp[d,g].item()}, ref={ref_e8m0[d].item()}")

    total_match = (scale_rcp == scale_int).float().mean().item()
    print(f"\nOverall rcp vs int: {total_match*100:.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("rcp.approx E8M0 Precision Verification")
    print("=" * 60)
    test_rcp_e8m0_precision()
    print()
    test_correctness_vs_reference()
