"""Phase 1 precision validation: epilogue quant vs. standalone quant.

Compares the full MoE forward+backward with:
  A) SONIC_MOE_FP8_EPILOGUE_QUANT=0 (standalone fused_z_save_y1_quant)
  B) SONIC_MOE_FP8_EPILOGUE_QUANT=1 (epilogue blockscaled quant)

Reports per-tensor RRMSE, max abs error, and cosine similarity.
Both paths should produce identical results for y1/y2/gradients since
epilogue quant only changes HOW z is quantized, not the z→y1 path.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

# Shape: production-representative
E, K, H, I = 8, 8, 3072, 1536
T = 4096
SEED = 42


def rrmse(a, b):
    diff = (a.float() - b.float()).norm()
    ref = b.float().norm().clamp(min=1e-8)
    return (diff / ref).item()


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def run_mode(epilogue_quant: bool):
    """Run full fwd+bwd and return (output, dx, dw1, dw2)."""
    if epilogue_quant:
        os.environ["SONIC_MOE_FP8_EPILOGUE_QUANT"] = "1"
    else:
        os.environ.pop("SONIC_MOE_FP8_EPILOGUE_QUANT", None)

    # Force fresh compile (different kernel variant)
    from sonicmoe.quack_utils.gemm_gated import gemm_gated as _gg
    _gg.compile_cache.clear()

    torch.manual_seed(SEED)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()

    torch.manual_seed(SEED + 1)
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["USE_QUACK_GEMM"] = "1"

    with enable_quack_gemm(True):
        out, _ = moe(x)
    torch.cuda.synchronize()

    # Check z_fp8 saved in ctx by examining _PREQUANTIZED_SCALES state
    from sonicmoe.functional import _PREQUANTIZED_SCALES
    print(f"  [{mode}] _PREQUANTIZED_SCALES keys after fwd: {list(_PREQUANTIZED_SCALES.keys())}")

    # Test dequant directly before backward
    if epilogue_quant:
        # The z_fp8 should have been consumed by _DownProjection.forward
        # and saved to ctx. Let's verify the backward doesn't crash.
        print(f"  [{mode}] out shape: {out.shape}, requires_grad: {out.requires_grad}")
        print(f"  [{mode}] Starting backward...")

    out.backward(dout)
    torch.cuda.synchronize()

    # Collect grads
    dx = x.grad.clone()
    dw1 = moe.c_fc.weight.grad.clone()
    dw2 = moe.c_proj.weight.grad.clone()

    # Cleanup for next run
    x.grad = None
    moe.zero_grad(set_to_none=True)
    del moe, x

    return out.detach(), dx, dw1, dw2


def main():
    print("=" * 70)
    print("Phase 1 Precision Validation: Epilogue Quant vs Standalone Quant")
    print("=" * 70)
    print(f"Shape: T={T}, E={E}, K={K}, H={H}, I={I}")
    print()

    print("[1/2] Running standalone quant (epilogue OFF)...")
    out_off, dx_off, dw1_off, dw2_off = run_mode(epilogue_quant=False)
    torch.cuda.synchronize()

    print("[2/2] Running epilogue quant (epilogue ON)...")
    out_on, dx_on, dw1_on, dw2_on = run_mode(epilogue_quant=True)
    torch.cuda.synchronize()

    print()
    print("-" * 70)
    print(f"{'Tensor':<12} {'RRMSE':>12} {'MaxAbsErr':>14} {'Cosine':>10} {'Status':>8}")
    print("-" * 70)

    pairs = [
        ("output", out_off, out_on),
        ("dx", dx_off, dx_on),
        ("dw1", dw1_off, dw1_on),
        ("dw2", dw2_off, dw2_on),
    ]

    all_pass = True
    for name, ref, test in pairs:
        r = rrmse(test, ref)
        m = max_abs_err(test, ref)
        c = cosine(test, ref)
        # Thresholds: forward should be near-identical (z quant only affects backward z),
        # backward may differ due to different z quantization paths
        threshold = 0.05  # 5% RRMSE tolerance
        ok = r < threshold and c > 0.99
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"{name:<12} {r:>12.6f} {m:>14.6f} {c:>10.6f} {status:>8}")

    print("-" * 70)
    if all_pass:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: SOME FAILED — investigate before committing")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
