"""Stage-by-stage memory breakdown with torch.cuda hooks."""
import sys, os, gc, json, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = os.environ.get("FP8_MODE", "perf")
os.environ["SONIC_MOE_STAGEWISE_MEMORY"] = "1"
torch.manual_seed(42)
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

# Warmup
for _ in range(2):
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); x.grad = None
    for p in moe.parameters():
        if p.grad is not None: p.grad = None

moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

print(f"=== STAGEWISE MEMORY ({os.environ['SONIC_MOE_FP8_MODE']} mode) ===")
print(f"Base: {torch.cuda.memory_allocated()/MiB:.1f} MiB")

torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True):
    o = moe(x, use_fp8=True)[0]
torch.cuda.synchronize()
print(f"Fwd peak: {torch.cuda.max_memory_allocated()/MiB:.1f} MiB")

torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize()
print(f"Bwd peak: {torch.cuda.max_memory_allocated()/MiB:.1f} MiB")
print(f"Bwd alloc: {torch.cuda.memory_allocated()/MiB:.1f} MiB")
