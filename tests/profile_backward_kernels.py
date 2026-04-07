"""ncu-compatible: trace backward kernel names + durations via torch profiler."""
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

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    out.backward(dout)
    torch.cuda.synchronize()

# Print CUDA kernels sorted by duration
print("=" * 80)
print("Backward CUDA Kernel Breakdown (sorted by duration)")
print("=" * 80)
events = []
for evt in prof.events():
    if evt.device_type == torch.autograd.DeviceType.CUDA and evt.duration > 0:
        events.append((evt.duration, evt.name))

events.sort(reverse=True)
total = sum(d for d, _ in events)
for dur, name in events[:20]:
    pct = dur / total * 100
    print(f"  {dur:>8.0f} µs ({pct:>5.1f}%)  {name[:70]}")
print(f"  {'─'*8}──────────")
print(f"  {total:>8.0f} µs (100.0%)  TOTAL")
