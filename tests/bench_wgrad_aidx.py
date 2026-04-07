"""Test: pre-gather x + blockscaled_fp8_weight_grad_gemm vs current A_idx wgrad."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import blockscaled_fp8_weight_grad_gemm
from quack.gemm_interface import gemm

E, K, H, I = 8, 8, 3072, 1536
T = 8192
TK = T * K  # 65536

torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
dz = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)

# Create routing metadata
expert_freq = torch.full((E,), TK // E, dtype=torch.int32, device="cuda")
expert_frequency_offset = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    expert_freq.cumsum(0)
]).to(torch.int32)
x_gather_idx = (torch.arange(TK, dtype=torch.int32, device="cuda") % T)

# --- Benchmark A: current A_idx path ---
def bench_aidx(warmup=10, iters=20, trials=5):
    dw1_base = torch.empty((E, 2*I, H), dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        gemm(x.T, dz, out=dw1_base.permute(0, 2, 1),
             cu_seqlens_k=expert_frequency_offset, A_idx=x_gather_idx,
             dynamic_scheduler=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            gemm(x.T, dz, out=dw1_base.permute(0, 2, 1),
                 cu_seqlens_k=expert_frequency_offset, A_idx=x_gather_idx,
                 dynamic_scheduler=False)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return dw1_base.clone(), times

# --- Benchmark B: pre-gather + FP8 wgrad ---
def bench_fp8_wgrad(warmup=10, iters=20, trials=5):
    for _ in range(warmup):
        x_gathered = x[x_gather_idx]
        blockscaled_fp8_weight_grad_gemm(x_gathered, dz, cu_seqlens_m=expert_frequency_offset)
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            x_gathered = x[x_gather_idx]
            out = blockscaled_fp8_weight_grad_gemm(
                x_gathered, dz, cu_seqlens_m=expert_frequency_offset)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return out, times

# --- Benchmark C: pre-gather + BF16 regular gemm (no A_idx) ---
def bench_gathered_bf16(warmup=10, iters=20, trials=5):
    dw1_base = torch.empty((E, 2*I, H), dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        x_gathered = x[x_gather_idx]
        gemm(x_gathered.T, dz, out=dw1_base.permute(0, 2, 1),
             cu_seqlens_k=expert_frequency_offset, dynamic_scheduler=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            x_gathered = x[x_gather_idx]
            gemm(x_gathered.T, dz, out=dw1_base.permute(0, 2, 1),
                 cu_seqlens_k=expert_frequency_offset, dynamic_scheduler=False)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return dw1_base.clone(), times

print("=" * 70)
print(f"wgrad up-proj Benchmark: T={T}, E={E}, K={K}, H={H}, I={I}, TK={TK}")
print("=" * 70)

print("\n[A] Current: gemm(x.T, dz, A_idx=gather_idx)")
dw1_a, times_a = bench_aidx()
print(f"  min={min(times_a):.0f}µs avg={sum(times_a)/len(times_a):.0f}µs all={[f'{t:.0f}' for t in times_a]}")

print("\n[B] FP8 wgrad: gather(x) + blockscaled_fp8_weight_grad_gemm")
dw1_b, times_b = bench_fp8_wgrad()
print(f"  min={min(times_b):.0f}µs avg={sum(times_b)/len(times_b):.0f}µs all={[f'{t:.0f}' for t in times_b]}")

print("\n[C] BF16 no-A_idx: gather(x) + gemm(x_gathered.T, dz)")
dw1_c, times_c = bench_gathered_bf16()
print(f"  min={min(times_c):.0f}µs avg={sum(times_c)/len(times_c):.0f}µs all={[f'{t:.0f}' for t in times_c]}")

# Precision comparison
def rrmse(a, b):
    return ((a.float()-b.float()).norm() / b.float().norm().clamp(min=1e-8)).item()

print(f"\n--- Precision ---")
print(f"[A] vs [C] (A_idx vs no-A_idx, same BF16): RRMSE={rrmse(dw1_a, dw1_c):.6f}")
if dw1_b.shape == dw1_a.shape:
    print(f"[B] vs [A] (FP8 vs BF16): RRMSE={rrmse(dw1_b, dw1_a):.6f}")
else:
    print(f"[B] shape={dw1_b.shape} vs [A] shape={dw1_a.shape}")

print(f"\n--- Speedup ---")
min_a = min(times_a)
min_b = min(times_b)
min_c = min(times_c)
print(f"[B] vs [A]: {min_a/min_b:.2f}x ({min_a:.0f} → {min_b:.0f}µs, Δ={min_a-min_b:+.0f}µs)")
print(f"[C] vs [A]: {min_a/min_c:.2f}x ({min_a:.0f} → {min_c:.0f}µs, Δ={min_a-min_c:+.0f}µs)")
