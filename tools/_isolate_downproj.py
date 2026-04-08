"""Isolate: does _DownProjection.backward produce different dz when w2 is fp8?"""
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

# Hook to capture dz at _UpProjection.backward entry
_captured_dz = [None, None]

import sonicmoe.functional.__init__ as fi
_orig_up_bwd = fi._UpProjection.backward

@staticmethod
def _hooked_bwd(ctx, _unused, dz):
    idx = 0 if _captured_dz[0] is None else 1
    _captured_dz[idx] = dz.detach().float().cpu() if dz is not None else None
    return _orig_up_bwd(ctx, _unused, dz)

fi._UpProjection.backward = _hooked_bwd

# Run 1: bf16 weights
moe.refresh_fp8_shadow_weights()
_captured_dz[0] = None
x1 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o1 = moe(x1, use_fp8=True)[0]
o1.backward(dout)
x1.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None

# Run 2: fp8 weights
moe.refresh_fp8_shadow_weights()
moe.c_fc.weight.data = moe.c_fc.weight.data.to(torch.float8_e4m3fn)
moe.c_proj.weight.data = moe.c_proj.weight.data.to(torch.float8_e4m3fn)
moe.c_fc.weight.grad_dtype = None
moe.c_proj.weight.grad_dtype = None
_captured_dz[1] = None
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
o2.backward(dout)

dz1, dz2 = _captured_dz
if dz1 is not None and dz2 is not None:
    diff = (dz1 - dz2).abs().max().item()
    rrmse = torch.sqrt(torch.mean((dz1 - dz2)**2)).item() / torch.sqrt(torch.mean(dz1**2)).item()
    tag = "BIT-IDENTICAL" if diff == 0 else f"max_diff={diff:.2e} rrmse={rrmse*100:.4f}%"
    print(f"dz entering _UpProjection.backward: {tag}")
else:
    print(f"dz capture: {dz1 is not None}, {dz2 is not None}")
