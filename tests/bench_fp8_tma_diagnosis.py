"""Diagnose FP8 TMA performance: isolate dequant cost vs TMA+convert cost."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast, quantize_and_pack_activation,
)
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

E, H, I = 8, 3072, 1536
TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2 * I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc.view(torch.uint8)

cu = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    torch.full((E,), TK // E, dtype=torch.int32, device="cuda").cumsum(0),
]).int()
df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
dx = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
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
    print(f"  {name:<55} min={mn:>7.0f}us  all={[f'{t:.0f}' for t in times]}")
    return mn

print("=" * 70)
print(f"FP8 TMA Performance Diagnosis (TK={TK})")
print("=" * 70)

# 1. BF16 baseline: dequant + GemmDGated
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
def run_baseline():
    gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)
t_base = bench(run_baseline, "BF16 GemmDGated (dequant already done)")

# 2. BF16 dequant only
def run_dequant_only():
    dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
t_dequant = bench(run_dequant_only, "Triton dequant only")

# 3. BF16 baseline total = dequant + gemm
def run_baseline_total():
    z_tmp = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
    gemm_dgated(df, wf, dx, z_tmp, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)
t_total = bench(run_baseline_total, "BF16 total (dequant + GemmDGated)")

# 4. FP8 TMA with dequant
gemm_dgated.compile_cache.clear()
def run_fp8_tma():
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_u8)
t_fp8 = bench(run_fp8_tma, "FP8 TMA (Int16 + in-kernel dequant)")

# 5. FP8 TMA without dequant (pass zero scales to measure TMA+convert cost)
# Create all-zero scales (effectively scale=1.0 since 2^(0<<23) = 2^0 = 1.0)
gemm_dgated.compile_cache.clear()
z_sc_zeros = torch.zeros_like(z_sc_u8)
def run_fp8_tma_nodequant():
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device="cuda"),
                "swiglu", 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_zeros)
t_fp8_noscale = bench(run_fp8_tma_nodequant, "FP8 TMA (Int16 + zero scales = no real dequant)")

print(f"\n--- Analysis ---")
print(f"BF16 GEMM only:       {t_base:.0f}us")
print(f"Triton dequant:       {t_dequant:.0f}us")
print(f"BF16 total:           {t_total:.0f}us")
print(f"FP8 TMA + dequant:    {t_fp8:.0f}us")
print(f"FP8 TMA + zero scale: {t_fp8_noscale:.0f}us")
print(f"  → TMA overhead vs BF16 GEMM: {t_fp8_noscale - t_base:+.0f}us (Int16 TMA + fp8→f32 conv)")
print(f"  → Dequant overhead in FP8:    {t_fp8 - t_fp8_noscale:+.0f}us (scale LDG + multiply)")
print(f"  → Net FP8 vs BF16 total:      {t_fp8 - t_total:+.0f}us")
