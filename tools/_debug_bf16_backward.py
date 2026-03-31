"""Debug: pinpoint the 4578µs elementwise_kernel in BF16 backward."""
import os, sys
os.environ["USE_QUACK_GEMM"] = "1"
for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT", "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ.pop("SONIC_MOE_FP8_MODE", None)  # Force BF16

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
import sonicmoe.functional as F_mod

T, H, I, E, K = 4096, 4096, 1024, 128, 8

def build_uniform_routing(device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices

torch.manual_seed(42)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
          intermediate_size=I, activation_function=ActivationType.SWIGLU,
          add_bias=False, std=0.02).to("cuda", torch.bfloat16)
enable_quack_gemm()

x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
_scores, _indices = build_uniform_routing(x_base.device)

class _UniformRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits, E_arg, K_arg):
        ctx.save_for_backward(_scores, _indices)
        ctx.E = E_arg; ctx.dtype = router_logits.dtype
        return _scores.clone(), _indices.clone()
    @staticmethod
    def backward(ctx, grad_scores, _grad_indices):
        scores, _ = ctx.saved_tensors
        return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None

F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter

# Warmup
for _ in range(3):
    for p in moe.parameters(): p.grad = None
    x_ = x_base.clone().requires_grad_(True)
    out, _ = moe(x_)
    out.sum().backward()
torch.cuda.synchronize()

# Profile with torch profiler
for p in moe.parameters(): p.grad = None
x_ = x_base.clone().requires_grad_(True)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True
) as prof:
    torch.cuda.synchronize()
    out, _ = moe(x_)
    torch.cuda.synchronize()
    out.sum().backward()
    torch.cuda.synchronize()

# Show CUDA events sorted by duration, focusing on elementwise
print("\n=== TOP 30 CUDA EVENTS (by GPU time) ===")
events = prof.key_averages()
events_sorted = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)
for e in events_sorted[:30]:
    print(f"  {e.cuda_time_total:10.0f}µs  {e.cpu_time_total:10.0f}µs cpu  "
          f"count={e.count:3d}  {e.key}")

# Show events with stack traces for the largest elementwise operations
print("\n=== EVENTS WITH STACK (top 10 by GPU time) ===")
for e in events_sorted[:10]:
    print(f"\n--- {e.key} ({e.cuda_time_total:.0f}µs GPU) ---")
    if hasattr(e, 'stack') and e.stack:
        for frame in e.stack[:10]:
            print(f"  {frame}")
