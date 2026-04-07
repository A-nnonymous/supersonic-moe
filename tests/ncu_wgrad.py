"""ncu target: profile ONLY the wgrad up-proj gemm kernel."""
import os, torch
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192
torch.manual_seed(42)

moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

# Warmup
for _ in range(3):
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        out, _ = moe(x)
    out.backward(dout)
torch.cuda.synchronize()

# Profiled run — ncu will capture all kernels
x.grad = None; moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()
out.backward(dout)
torch.cuda.synchronize()
print("DONE")
