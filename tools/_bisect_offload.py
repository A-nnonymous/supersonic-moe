"""Bisect: which part of offload causes the diff? dtype change or cache re-population?"""
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

# Reference
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

# Test A: offload (full pipeline)
moe.refresh_fp8_shadow_weights()
moe.offload_bf16_weights()
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
o2.backward(dout)
off_dx = x2.grad.float().cpu() if x2.grad is not None else None
off_dw1 = moe.c_fc.weight.grad.float().cpu() if moe.c_fc.weight.grad is not None else None

# Restore for next test
moe.restore_bf16_weights()
x2.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None

# Test B: refresh + clear_all + re-populate ALL caches (like offload does) but DON'T change dtype
moe.refresh_fp8_shadow_weights()
from sonicmoe.functional import clear_all_fp8_weight_caches
from sonicmoe.quack_utils import precompute_weight_fp8, precompute_weight_fp8_for_fused_gated, precompute_weight_fp8_for_direct_fused_dgated
clear_all_fp8_weight_caches()
w1_perm = moe.c_fc.weight.permute(1, 2, 0)
w2_perm = moe.c_proj.weight.permute(1, 2, 0)
precompute_weight_fp8_for_fused_gated(w1_perm)
precompute_weight_fp8(w2_perm)
precompute_weight_fp8(w1_perm.permute(1, 0, 2))
precompute_weight_fp8_for_direct_fused_dgated(w2_perm)
# Keep dtype as bf16!

x3 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o3 = moe(x3, use_fp8=True)[0]
o3.backward(dout)
repop_dx = x3.grad.float().cpu()
repop_dw1 = moe.c_fc.weight.grad.float().cpu()

print("=== BISECT OFFLOAD DIFF ===")
if off_dx is not None:
    print(f"offload dx diff:    {(ref_dx - off_dx).abs().max().item():.2e}")
else:
    print("offload dx: None!")
if off_dw1 is not None:
    print(f"offload dw1 diff:   {(ref_dw1 - off_dw1).abs().max().item():.2e}")
else:
    print("offload dw1: None!")
print(f"repopulate dx diff: {(ref_dx - repop_dx).abs().max().item():.2e}")
print(f"repopulate dw1 diff:{(ref_dw1 - repop_dw1).abs().max().item():.2e}")
print(f"\nIf repopulate=0 but offload!=0: dtype change causes the diff")
print(f"If repopulate!=0:  cache re-population itself causes the diff")
