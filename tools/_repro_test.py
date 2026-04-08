"""Test if two identical runs produce same dx (no offload)."""
import sys, os, gc, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
torch.manual_seed(42)
T, H, I, E, K = 8192, 3072, 1536, 8, 8
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02
).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
for _ in range(2):
    with enable_quack_gemm(True):
        o = moe(x, use_fp8=True)[0]
    o.backward(dout)
    x.grad = None
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None

# Run 1
moe.refresh_fp8_shadow_weights()
x1 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o1 = moe(x1, use_fp8=True)[0]
o1.backward(dout)
r1_dx = x1.grad.float().cpu()
r1_dw1 = moe.c_fc.weight.grad.float().cpu()
x1.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None

# Run 2 (identical, no offload)
moe.refresh_fp8_shadow_weights()
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
o2.backward(dout)
r2_dx = x2.grad.float().cpu()
r2_dw1 = moe.c_fc.weight.grad.float().cpu()

dx_diff = (r1_dx - r2_dx).abs().max().item()
dw1_diff = (r1_dw1 - r2_dw1).abs().max().item()
print(f"Run1 vs Run2 (NO offload): dx diff={dx_diff:.2e}, dw1 diff={dw1_diff:.2e}")
