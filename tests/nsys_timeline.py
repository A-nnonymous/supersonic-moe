"""nsys timeline: capture full backward with stream annotations."""
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

# Warmup (including JIT compilation)
for _ in range(5):
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        out, _ = moe(x)
    out.backward(dout)
torch.cuda.synchronize()

# Profiled iteration — capture only backward
x.grad = None; moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()

# Use NVTX ranges for backward phases
torch.cuda.nvtx.range_push("backward_total")
out.backward(dout)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("DONE")
