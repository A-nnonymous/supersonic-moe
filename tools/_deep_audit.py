"""Deep audit: instrument backward to print tensor sizes at peak moments."""
import sys, os, gc, torch
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

for _ in range(2):
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); x.grad = None
    for p in moe.parameters():
        if p.grad is not None: p.grad = None

moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Use memory_stats for breakdown
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
base = torch.cuda.memory_allocated() / MiB

with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / MiB
fwd_alloc = torch.cuda.memory_allocated() / MiB

torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / MiB
bwd_alloc = torch.cuda.memory_allocated() / MiB

# Now enumerate every live allocation via memory_stats
stats = torch.cuda.memory_stats()

mode = os.environ["SONIC_MOE_FP8_MODE"]
print(f"=== DEEP MEMORY AUDIT ({mode}) ===")
print(f"Base: {base:.1f} MiB")
print(f"Fwd peak: {fwd_peak:.1f} MiB, Fwd alloc: {fwd_alloc:.1f} MiB")
print(f"Bwd peak: {bwd_peak:.1f} MiB, Bwd alloc: {bwd_alloc:.1f} MiB")
print()

# Breakdown base: known parameters + caches
w1 = moe.c_fc.weight
w2 = moe.c_proj.weight
rw = moe.router.weight
print("=== KNOWN ALLOCATIONS AT BASE ===")
known = []
for name, p in [("c_fc.weight", w1), ("c_proj.weight", w2), ("router.weight", rw)]:
    s = p.nelement() * p.element_size() / MiB
    known.append((name, s, p.shape, p.dtype))
    print(f"  {name:<25s} {s:>8.2f} MiB  {tuple(p.shape)}  {p.dtype}")

# Gradients
for name, p in [("c_fc.weight.grad", w1), ("c_proj.weight.grad", w2), ("router.weight.grad", rw)]:
    if p.grad is not None:
        s = p.grad.nelement() * p.grad.element_size() / MiB
        known.append((name, s, p.grad.shape, p.grad.dtype))
        print(f"  {name:<25s} {s:>8.2f} MiB  {tuple(p.grad.shape)}  {p.grad.dtype}")

# Input tensors
for name, t in [("x", x), ("dout", dout)]:
    s = t.nelement() * t.element_size() / MiB
    known.append((name, s, t.shape, t.dtype))
    print(f"  {name:<25s} {s:>8.2f} MiB  {tuple(t.shape)}  {t.dtype}")

if x.grad is not None:
    s = x.grad.nelement() * x.grad.element_size() / MiB
    known.append(("x.grad", s, x.grad.shape, x.grad.dtype))
    print(f"  {'x.grad':<25s} {s:>8.2f} MiB  {tuple(x.grad.shape)}  {x.grad.dtype}")

# Weight caches
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _VARLEN_WEIGHT_CACHE, _FUSED_WEIGHT_CACHE
for cache_name, cache in [("_VARLEN_WEIGHT_CACHE", _VARLEN_WEIGHT_CACHE), ("_FUSED_WEIGHT_CACHE", _FUSED_WEIGHT_CACHE)]:
    for key, val in cache.items():
        for i, t in enumerate(val):
            if torch.is_tensor(t) and t.is_cuda and t.untyped_storage().size() > 0:
                s = t.untyped_storage().size() / MiB
                known.append((f"{cache_name}[{i}]", s, t.shape, t.dtype))
                print(f"  {cache_name}[{i}]{'':10s} {s:>8.2f} MiB  {tuple(t.shape)}  {t.dtype}")

total_known = sum(x[1] for x in known)
print(f"\n  KNOWN TOTAL:             {total_known:>8.1f} MiB")
print(f"  torch reported:          {bwd_alloc:>8.1f} MiB")
print(f"  UNKNOWN (dark memory):   {bwd_alloc - total_known:>8.1f} MiB ({(bwd_alloc - total_known)/bwd_alloc*100:.0f}%)")
print(f"  Peak:                    {bwd_peak:>8.1f} MiB")
print(f"  Peak UNKNOWN:            {bwd_peak - total_known:>8.1f} MiB")
