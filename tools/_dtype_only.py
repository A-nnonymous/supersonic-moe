"""Test: change dtype to fp8 WITHOUT touching caches."""
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

# Reference: shadow only, keep bf16
moe.refresh_fp8_shadow_weights()
x1 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o1 = moe(x1, use_fp8=True)[0]
o1.backward(dout)
ref_dx = x1.grad.float().cpu()
ref_dw1 = moe.c_fc.weight.grad.float().cpu()
x1.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None

# Test: same caches, but CHANGE dtype to fp8 (no cache re-pop)
moe.refresh_fp8_shadow_weights()
# Just change dtype, DON'T touch caches
moe.c_fc.weight.data = moe.c_fc.weight.data.to(torch.float8_e4m3fn)
moe.c_proj.weight.data = moe.c_proj.weight.data.to(torch.float8_e4m3fn)
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
o2.backward(dout)
test_dx = x2.grad.float().cpu() if x2.grad is not None else None
test_dw1 = moe.c_fc.weight.grad.float().cpu() if moe.c_fc.weight.grad is not None else None

print("=== DTYPE-ONLY CHANGE (no cache re-pop) ===")
if test_dx is not None:
    print(f"dx diff:  {(ref_dx - test_dx).abs().max().item():.2e}")
else:
    print("dx: None!")
if test_dw1 is not None:
    print(f"dw1 diff: {(ref_dw1 - test_dw1).abs().max().item():.2e}")
else:
    print("dw1: None!")
