"""Precise memory tracing: Phase 3.1 vs baseline around GemmDGated."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192

def measure_backward_memory():
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

    # Measure: forward, then track backward memory
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        out, _ = moe(x)
    torch.cuda.synchronize()

    mem_after_fwd = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    out.backward(dout)
    torch.cuda.synchronize()

    mem_peak_bwd = torch.cuda.max_memory_allocated()
    mem_after_bwd = torch.cuda.memory_allocated()

    bwd_peak_delta = (mem_peak_bwd - mem_after_fwd) / 1024**2
    print(f"  After fwd:     {mem_after_fwd / 1024**2:.1f} MiB")
    print(f"  Peak during bwd: {mem_peak_bwd / 1024**2:.1f} MiB")
    print(f"  Bwd temp peak: {bwd_peak_delta:.1f} MiB")
    print(f"  After bwd:     {mem_after_bwd / 1024**2:.1f} MiB")
    return bwd_peak_delta

print("=" * 60)
print("Phase 3.1 Memory Analysis")
print("=" * 60)
result = measure_backward_memory()
print(f"\nBackward temp memory: {result:.1f} MiB")
print(f"z_bf16 size: {T * K * 2 * I * 2 / 1024**2:.1f} MiB (if allocated)")
