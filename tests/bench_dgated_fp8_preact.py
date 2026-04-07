"""Isolated benchmark: GemmDGated FP8 PreAct vs BF16 PreAct."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

E, H, I = 8, 3072, 1536
TK = 65536  # production size

torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)

cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.full((E,), TK//E, dtype=torch.int32, device="cuda").cumsum(0)]).int()

df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)

dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device="cuda")
pa = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")

WARMUP, ITERS, TRIALS = 5, 10, 5

def bench(fn, name):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS): fn()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)
    mn = min(times)
    print(f"  {name:<45} min={mn:>7.0f}us  all={[f'{t:.0f}' for t in times]}")
    return mn

print("=" * 70)
print(f"GemmDGated Isolated Benchmark (TK={TK})")
print("=" * 70)

# BF16 PreAct (baseline): includes dequant
def run_baseline():
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc.view(torch.uint8))
    gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)

t_bf16 = bench(run_baseline, "BF16 PreAct (dequant + GemmDGated)")

# FP8 PreAct (Phase 3.1): no dequant
gemm_dgated.compile_cache.clear()
def run_fp8():
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc.view(torch.uint8))

t_fp8 = bench(run_fp8, "FP8 PreAct (no dequant)")

print(f"\n--- Comparison ---")
print(f"BF16 (dequant+kernel): {t_bf16:.0f}us")
print(f"FP8 (kernel only):     {t_fp8:.0f}us")
print(f"Speedup:               {t_bf16/t_fp8:.2f}x")
print(f"Delta:                 {t_fp8-t_bf16:+.0f}us")
