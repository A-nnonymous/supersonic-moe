"""E2E: FP8 TMA vs FP8 frontier (standalone dequant) — should be 0 RRMSE."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import socket

E, K, H, I = 8, 4, 768, 256
T = 128

def make_moe():
    return MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()

def make_sample():
    torch.manual_seed(42)
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)
    return x, dout

def rrmse(a, b):
    return ((a.float() - b.float()).pow(2).mean().sqrt() /
            b.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()

host = socket.gethostname()
print(f"Host: {host}")
print("=" * 60)

# Run 1: FP8 with standalone dequant (disable SAVE_Z_FP8 to force bf16 z save)
print("\n1. FP8 frontier (z saved as bf16, standalone dequant)...")
os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "0"
model1 = make_moe()
x1, dout1 = make_sample()
with enable_fp8(), enable_quack_gemm():
    y1, _ = model1(x1)
    y1.backward(dout1)
dx1 = x1.grad.clone()
dw1 = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

# Run 2: FP8 with TMA preact (z saved as fp8, TMA-based dequant)
print("2. FP8 TMA (z saved as fp8, in-kernel TMA dequant)...")
os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
model2 = make_moe()
model2.load_state_dict(model1.state_dict())
x2, dout2 = make_sample()
with enable_fp8(), enable_quack_gemm():
    y2, _ = model2(x2)
    y2.backward(dout2)
dx2 = x2.grad.clone()
dw2 = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

print(f"\n--- FP8 TMA vs FP8 Frontier ---")
print(f"y  RRMSE: {rrmse(y2, y1):.6f}")
print(f"dx RRMSE: {rrmse(dx2, dx1):.6f}")
for n in dw1:
    if n in dw2:
        print(f"dw[{n}] RRMSE: {rrmse(dw2[n], dw1[n]):.6f}")

all_zero = (rrmse(y2, y1) < 1e-5 and rrmse(dx2, dx1) < 1e-5
            and all(rrmse(dw2[n], dw1[n]) < 1e-5 for n in dw1 if n in dw2))
print(f"\nBIT-EXACT: {'YES' if all_zero else 'NO (check above)'}")
