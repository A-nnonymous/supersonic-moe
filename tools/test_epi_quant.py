"""Test epilogue blockscaled quant — directly compiled into ZeroMat kernel."""
import sys, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192
torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

# Run FP8 forward (now uses BlockscaledQuantMixin in ZeroMat kernel)
moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True), enable_fp8():
    out, _ = moe(x, use_fp8=True)

print(f"out: nan={torch.isnan(out).any().item()} norm={out.float().norm().item():.6f}")

# Backward too
out.backward(dout)
print(f"dx:  nan={torch.isnan(x.grad).any().item()} norm={x.grad.float().norm().item():.6f}")

# Compare with BF16 gold
x2 = x.detach().clone().requires_grad_()
moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out2, _ = moe(x2)
out2.backward(dout)

rrmse_fwd = ((out.float()-out2.float()).pow(2).mean().sqrt() / out2.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
rrmse_bwd = ((x.grad.float()-x2.grad.float()).pow(2).mean().sqrt() / x2.grad.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
print(f"vs BF16: fwd RRMSE={rrmse_fwd:.4f}, bwd RRMSE={rrmse_bwd:.4f}")
if rrmse_fwd < 0.15:
    print("EPILOGUE BLOCKSCALED QUANT: COMPILED, RAN, PRECISION OK!")
else:
    print(f"PRECISION ISSUE: fwd RRMSE={rrmse_fwd:.4f}")
