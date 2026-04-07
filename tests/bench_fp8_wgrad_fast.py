"""Benchmark: fp8_weight_grad_gemm_fast for wgrad."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_weight_grad_gemm_fast
from quack.gemm_interface import gemm

E, K, H, I = 8, 8, 3072, 1536
T = 8192
TK = T * K

torch.manual_seed(42)
x = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
dz = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
y1 = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)

expert_freq = torch.full((E,), TK // E, dtype=torch.int32, device="cuda")
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), expert_freq.cumsum(0)]).int()
x_gather_idx = (torch.arange(TK, dtype=torch.int32, device="cuda") % T)

def time_fn(fn, name, warmup=10, iters=20, trials=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    mn = min(times)
    print(f"  {name:<50} min={mn:.0f}µs all={[f'{t:.0f}' for t in times]}")
    return mn

print("=" * 70)
print(f"FP8 Wgrad Benchmark (T={T}, E={E}, H={H}, I={I})")
print("=" * 70)

# --- Down-proj wgrad: gemm(dout.T, y1) → (E, H, I) ---
# dout is (TK, H), y1 is (TK, I), output is (E, H, I)
# gemm(A=dout.T(H,TK), B=y1(TK,I)) → (H, I) per expert
# out shape from gemm with cu_seqlens_k: (E, H, I)
dw2_base = torch.empty((E, H, I), dtype=torch.bfloat16, device="cuda")
t_down_bf16 = time_fn(
    lambda: gemm(dout.T, y1, out=dw2_base,
                 cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False),
    "down-proj wgrad BF16 (A_idx)")

try:
    t_down_fp8 = time_fn(
        lambda: blockscaled_fp8_weight_grad_gemm_fast(
            dout, y1, cu, a_gather_idx=x_gather_idx, b_gather_idx=x_gather_idx),
        "down-proj wgrad FP8 fast (gather)")
except Exception as e:
    print(f"  down-proj FP8 fast FAILED: {e}")
    t_down_fp8 = None

# --- Up-proj wgrad: gemm(x.T, dz) ---
dw1_base = torch.empty((E, 2*I, H), dtype=torch.bfloat16, device="cuda")
t_up_bf16 = time_fn(
    lambda: gemm(x.T, dz, out=dw1_base.permute(0, 2, 1),
                 cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False),
    "up-proj wgrad BF16 (A_idx)")

try:
    t_up_fp8 = time_fn(
        lambda: blockscaled_fp8_weight_grad_gemm_fast(
            x, dz, cu, a_gather_idx=x_gather_idx, b_gather_idx=None),
        "up-proj wgrad FP8 fast (a_gather)")
except Exception as e:
    print(f"  up-proj FP8 fast FAILED: {e}")
    t_up_fp8 = None

print("\n--- Summary ---")
if t_down_fp8:
    print(f"Down-proj: BF16 {t_down_bf16:.0f}µs → FP8 {t_down_fp8:.0f}µs ({t_down_bf16/t_down_fp8:.2f}x)")
if t_up_fp8:
    print(f"Up-proj:   BF16 {t_up_bf16:.0f}µs → FP8 {t_up_fp8:.0f}µs ({t_up_bf16/t_up_fp8:.2f}x)")
