"""Minimal forward+backward test with epilogue_quant + fp8 D output."""
import os
os.environ["USE_QUACK_GEMM"] = "1"
import torch

def test_fwd_bwd():
    from sonicmoe.moe import MoE
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.config import SonicMoEConfig

    device = "cuda:0"
    torch.manual_seed(42)

    T, H, I, E, K = 1024, 3072, 1536, 8, 8
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function="swiglu",
        add_bias=True, std=0.01,
    ).to(device).to(torch.bfloat16)

    x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Test with epilogue_quant=True (default now)
    cfg = SonicMoEConfig(epilogue_quant=True)
    print(f"Config: epilogue_quant={cfg.epilogue_quant}")
    with cfg.activate(), enable_fp8():
        print("Forward...")
        y, aux_loss = moe(x, use_fp8=True)
        print(f"  y shape={y.shape}, dtype={y.dtype}")
        print("Backward...")
        try:
            y.sum().backward()
            print("  SUCCESS!")
            print(f"  x.grad norm: {x.grad.norm().item():.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_fwd_bwd()
