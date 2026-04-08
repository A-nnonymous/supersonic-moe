"""Exhaustive memory snapshot at backward peak — account for EVERY byte."""
import sys, os, gc, json, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = os.environ.get("FP8_MODE", "perf")
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

# Enable memory history recording
torch.cuda.memory._record_memory_history(max_entries=100000)

# Measured iter
torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
o.backward(dout)
torch.cuda.synchronize()

# Capture snapshot
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)

# Analyze: find all live allocations at the peak moment
peak_bytes = torch.cuda.max_memory_allocated()
current_bytes = torch.cuda.memory_allocated()

print(f"=== EXHAUSTIVE MEMORY AUDIT ({os.environ['SONIC_MOE_FP8_MODE']} mode) ===")
print(f"Peak allocated: {peak_bytes/MiB:.1f} MiB")
print(f"Current allocated: {current_bytes/MiB:.1f} MiB")
print()

# Method 2: enumerate ALL current live tensors via gc
print("=== ALL LIVE CUDA TENSORS (sorted by size) ===")
gc.collect()
tensors = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda and obj.storage().size() > 0:
            size_bytes = obj.storage().size() * obj.storage().element_size()
            tensors.append((size_bytes, obj.shape, obj.dtype, obj.device))
    except:
        pass

tensors.sort(key=lambda x: -x[0])
total_accounted = 0
print(f"{'Size (MiB)':>10s}  {'Shape':>30s}  {'Dtype':>15s}")
print("-" * 60)
for size_b, shape, dtype, dev in tensors[:40]:
    size_m = size_b / MiB
    total_accounted += size_b
    print(f"{size_m:>10.2f}  {str(tuple(shape)):>30s}  {str(dtype):>15s}")

print(f"\n{'TOTAL accounted':>10s}  {total_accounted/MiB:>28.1f} MiB")
print(f"{'torch reported':>10s}  {current_bytes/MiB:>28.1f} MiB")
print(f"{'Unaccounted':>10s}  {(current_bytes - total_accounted)/MiB:>28.1f} MiB")
print(f"{'Peak':>10s}  {peak_bytes/MiB:>28.1f} MiB")
