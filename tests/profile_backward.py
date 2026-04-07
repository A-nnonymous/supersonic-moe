"""Profile backward kernel breakdown with ncu-style analysis."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

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

# Profile backward only
x.grad = None; moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()

# Time individual backward stages
import time

def time_cuda(fn, name, iters=10):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) * 1000 / iters  # µs
    print(f"  {name:<40} {t:>8.0f} µs")
    return t

print("=" * 60)
print("Backward Stage Breakdown")
print("=" * 60)

# Full backward
x.grad = None; moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()

t_total = time_cuda(lambda: (out.backward(dout, retain_graph=True), torch.cuda.synchronize()), "Total backward", iters=5)

# Memory analysis
torch.cuda.reset_peak_memory_stats()
x.grad = None; moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()
mem_before = torch.cuda.memory_allocated()
out.backward(dout)
torch.cuda.synchronize()
mem_peak = torch.cuda.max_memory_allocated()
mem_after = torch.cuda.memory_allocated()

print()
print(f"Memory before backward: {mem_before / 1024**2:.1f} MiB")
print(f"Memory peak during backward: {mem_peak / 1024**2:.1f} MiB")
print(f"Memory after backward: {mem_after / 1024**2:.1f} MiB")
print(f"Backward temp memory: {(mem_peak - mem_before) / 1024**2:.1f} MiB")
