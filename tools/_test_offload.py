"""Correctness + memory test for bf16 weight offload."""
import sys, os, gc, json, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
torch.manual_seed(42)
MiB = 1024**2
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

# === 1. Reference (shadow only, no offload) ===
moe.refresh_fp8_shadow_weights()
x1 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o1 = moe(x1, use_fp8=True)[0]
o1.backward(dout)
ref_out = o1.detach().float().cpu()
ref_dx = x1.grad.float().cpu()
ref_dw1 = moe.c_fc.weight.grad.float().cpu()
ref_dw2 = moe.c_proj.weight.grad.float().cpu()
x1.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None

# === 2. Test (shadow + offload) ===
moe.refresh_fp8_shadow_weights()
moe.offload_bf16_weights()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

base = torch.cuda.memory_allocated() / MiB
x2 = x.detach().clone().requires_grad_()
with enable_quack_gemm(True):
    o2 = moe(x2, use_fp8=True)[0]
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / MiB

torch.cuda.reset_peak_memory_stats()
o2.backward(dout)
torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / MiB

test_out = o2.detach().float().cpu()
test_dx = x2.grad.float().cpu() if x2.grad is not None else None
test_dw1 = moe.c_fc.weight.grad.float().cpu() if moe.c_fc.weight.grad is not None else None
test_dw2 = moe.c_proj.weight.grad.float().cpu() if moe.c_proj.weight.grad is not None else None

# === 3. Compare ===
print("=== CORRECTNESS (offload vs no-offload) ===")
for name, ref, test in [
    ("output", ref_out, test_out),
    ("dx", ref_dx, test_dx),
    ("dw1", ref_dw1, test_dw1),
    ("dw2", ref_dw2, test_dw2),
]:
    if test is None:
        print(f"  {name:10s}  NONE (not computed)")
    else:
        diff = (ref - test).abs().max().item()
        tag = "BIT-IDENTICAL" if diff == 0 else f"max_diff={diff:.2e}"
        print(f"  {name:10s}  {tag}")

print(f"\n=== MEMORY (with offload) ===")
print(f"  base:     {base:.1f} MiB")
print(f"  fwd_peak: {fwd_peak:.1f} MiB")
print(f"  bwd_peak: {bwd_peak:.1f} MiB")
