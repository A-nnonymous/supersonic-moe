"""Full benchmark: memory + per-tensor precision + nsys timing prep."""
import sys, os, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe import MoE, enable_fp8, enable_native_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import quantize_and_pack_activation

E, K, H, I = 8, 8, 3072, 1536
T = 8192

def mb(x): return x / 1024**2

def rrmse(a, b):
    return ((a.float()-b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-12)).item()

def cosine(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    return (af @ bf / (af.norm() * bf.norm()).clamp(min=1e-12)).item()

# ═══ 1. Per-tensor precision ═══
print("=" * 70)
print("1. PER-TENSOR PRECISION (production shape E=8 K=8 T=8192 H=3072 I=1536)")
print("=" * 70)

torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

# BF16 gold
x0 = x.detach().clone().requires_grad_()
moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True):
    o0, _ = moe(x0)
o0.backward(dout)
gold = {"out": o0.detach(), "dx": x0.grad.detach()}
for n, p in moe.named_parameters():
    if p.grad is not None:
        gold[f"d_{n}"] = p.grad.detach().clone()

# Frontier FP8
x1 = x.detach().clone().requires_grad_()
moe.zero_grad(set_to_none=True)
with enable_quack_gemm(True), enable_fp8():
    o1, _ = moe(x1, use_fp8=True)
o1.backward(dout)
frontier = {"out": o1.detach(), "dx": x1.grad.detach()}
for n, p in moe.named_parameters():
    if p.grad is not None:
        frontier[f"d_{n}"] = p.grad.detach().clone()

# Native FP8 with epilogue quant
nfp = moe.prepare_native_fp8()
xf, xs = quantize_and_pack_activation(x.detach())
x2 = x.detach().clone().requires_grad_()
moe.zero_grad(set_to_none=True)
with enable_native_fp8():
    o2, _ = moe(x2, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
o2.backward(dout)
native = {"out": o2.detach(), "dx": x2.grad.detach()}
for n, p in moe.named_parameters():
    if p.grad is not None:
        native[f"d_{n}"] = p.grad.detach().clone()

print(f"{'Variable':<25} {'Frontier RRMSE':>14} {'Native RRMSE':>14} {'Frontier cos':>14} {'Native cos':>14}")
print("-" * 85)
for key in ["out", "dx", "d_c_fc.weight", "d_c_proj.weight", "d_router.weight"]:
    if key in gold:
        g = gold[key]
        f_rr = rrmse(frontier.get(key, g), g)
        n_rr = rrmse(native.get(key, g), g)
        f_cos = cosine(frontier.get(key, g), g)
        n_cos = cosine(native.get(key, g), g)
        print(f"{key:<25} {f_rr:>14.6f} {n_rr:>14.6f} {f_cos:>14.8f} {n_cos:>14.8f}")

# ═══ 2. Memory ═══
print("\n" + "=" * 70)
print("2. MEMORY (isolated processes would be more accurate, but inline is OK)")
print("=" * 70)

del moe, x, x0, x1, x2, dout, nfp, xf, xs, gold, frontier, native, o0, o1, o2
torch.cuda.empty_cache()

for mode_name in ["BF16", "Frontier FP8", "Native FP8 + Epilogue Quant"]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)

    if "Native" in mode_name:
        nfp = moe.prepare_native_fp8()
        xf, xs = quantize_and_pack_activation(x.detach())

    torch.cuda.reset_peak_memory_stats()
    moe.zero_grad(set_to_none=True); x.grad = None
    if mode_name == "BF16":
        with enable_quack_gemm(True):
            out, _ = moe(x)
    elif mode_name == "Frontier FP8":
        with enable_quack_gemm(True), enable_fp8():
            out, _ = moe(x, use_fp8=True)
    else:
        with enable_native_fp8():
            out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
    out.backward(dout)
    peak = torch.cuda.max_memory_allocated()
    print(f"  {mode_name:<35} peak={mb(peak):>8.1f} MiB")
    del moe, x, dout, out
    if "Native" in mode_name:
        del nfp, xf, xs

# ═══ 3. Timing (wall-clock, GPU projection needs nsys) ═══
print("\n" + "=" * 70)
print("3. TIMING (CUDA event, 20 warmup + 3 trials × 30 iters)")
print("=" * 70)

torch.cuda.empty_cache()
torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)
nfp = moe.prepare_native_fp8()
xf, xs = quantize_and_pack_activation(x.detach())

WARMUP, ITERS = 20, 30

def bench(label, run_fn):
    for _ in range(WARMUP):
        run_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            run_fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)
    avg = sum(times)/len(times)
    mn = min(times)
    print(f"  {label:<35} avg={avg:>6.0f} min={mn:>6.0f} µs/iter")
    return mn

def run_bf16():
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        z, _ = moe(x)
    z.backward(dout)

def run_frontier():
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True), enable_fp8():
        z, _ = moe(x, use_fp8=True)
    z.backward(dout)

def run_native():
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_native_fp8():
        z, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
    z.backward(dout)

t_bf16 = bench("BF16 QuACK", run_bf16)
t_fp8 = bench("Frontier FP8", run_frontier)
t_nat = bench("Native FP8 + Epilogue Quant", run_native)

print(f"\n  FP8 Frontier vs BF16:    {t_bf16/t_fp8:.3f}×")
print(f"  Native+EpiQuant vs BF16: {t_bf16/t_nat:.3f}×")
print(f"  Native+EpiQuant vs Frontier: {t_fp8/t_nat:.3f}×")
