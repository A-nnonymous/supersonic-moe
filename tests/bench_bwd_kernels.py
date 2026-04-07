"""Precise backward kernel-by-kernel timing via individual benchmarks."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import (
    blockscaled_fp8_gemm_varlen, precompute_weight_fp8,
    quantize_and_pack_activation,
)
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
from quack.gemm_interface import gemm
import quack.activation

E, K, H, I = 8, 8, 3072, 1536
T = 8192
TK = T * K

torch.manual_seed(42)

# Setup tensors
x = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
w1 = 0.02 * torch.randn(2*I, H, E, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
z_bf16 = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_scales = quantize_activation_blockscaled_fast(z_bf16)
dz = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)
y1 = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)

expert_freq = torch.full((E,), TK // E, dtype=torch.int32, device="cuda")
cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), expert_freq.cumsum(0)]).int()
x_gather_idx = (torch.arange(TK, dtype=torch.int32, device="cuda") % T)

# Precompute FP8 weights
w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
w2_enk_fp8 = precompute_weight_fp8(w2)

def time_kernel(fn, name, warmup=10, iters=30, trials=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    mn = min(times)
    avg = sum(times)/len(times)
    print(f"  {name:<45} min={mn:>7.0f}µs  avg={avg:>7.0f}µs")
    return mn

print("=" * 70)
print(f"Backward Kernel-by-Kernel Benchmark (T={T}, TK={TK})")
print("=" * 70)

# 1. z dequant
t_dequant = time_kernel(
    lambda: dequantize_blockscaled_fp8(z_fp8, z_scales.view(torch.uint8)),
    "z dequant (fp8→bf16)")

# 2. dout quant
t_dout_quant = time_kernel(
    lambda: quantize_and_pack_activation(dout),
    "dout quant+pack")

# 3. GemmDGated (dout × w2^T + dSwiGLU)
dout_fp8, dout_scales = quantize_and_pack_activation(dout)
w2_fp8, w2_scales = precompute_weight_fp8(w2)
dx_out = torch.empty(TK, 2*I, dtype=torch.bfloat16, device="cuda")
postact_out = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")
t_dgated = time_kernel(
    lambda: gemm_dgated_kernel(
        dout_fp8, w2_fp8.permute(2, 1, 0).mT.contiguous().unsqueeze(0),
        z_bf16.view(torch.float32), z_bf16.view(torch.float32),  # D=C=PreAct
        dx_out.view(torch.float32), postact_out,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        persistent=True, a_scales=dout_scales, b_scales=w2_scales,
    ) if False else None,  # Skip — complex setup
    "GemmDGated (skipped — complex setup)")

# 4. wgrad up-proj: gemm(x.T, dz, A_idx)
dw1_base = torch.empty((E, 2*I, H), dtype=torch.bfloat16, device="cuda")
t_wgrad_up = time_kernel(
    lambda: gemm(x.T, dz, out=dw1_base.permute(0, 2, 1),
                 cu_seqlens_k=cu_seqlens, A_idx=x_gather_idx,
                 dynamic_scheduler=False),
    "wgrad up-proj gemm(x.T, dz, A_idx)")

# 5. wgrad down-proj: gemm(dout.T, y1)
dw2_base = torch.empty((E, H, I), dtype=torch.bfloat16, device="cuda")
t_wgrad_down = time_kernel(
    lambda: gemm(dout.T, y1, out=dw2_base.permute(0, 2, 1),
                 cu_seqlens_k=cu_seqlens, A_idx=x_gather_idx,
                 dynamic_scheduler=False),
    "wgrad down-proj gemm(dout.T, y1, A_idx)")

# 6. actgrad: FP8 blockscaled GEMM
t_actgrad = time_kernel(
    lambda: blockscaled_fp8_gemm_varlen(
        dout_fp8, w1.permute(1, 0, 2), cu_seqlens,
        a_scales=dout_scales, w_fp8=w1T_fp8, w_scales=w1T_scales,
        out_dtype=torch.bfloat16, assume_aligned=True),
    "actgrad fp8_gemm(dout, w1^T)")

# 7. token gather/scatter
s_reverse = torch.arange(TK, dtype=torch.int32, device="cuda")
dx_reduced = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
from sonicmoe.functional.backward import _token_broadcast_backward
naept_offset = torch.arange(T+1, dtype=torch.int32, device="cuda")
t_scatter = time_kernel(
    lambda: _token_broadcast_backward(
        dx_reduced, dout, s_reverse, naept_offset, K, H, False),
    "token scatter/reduce")

print()
print("--- Summary ---")
total = t_dequant + t_dout_quant + t_wgrad_up + t_wgrad_down + t_actgrad + t_scatter
print(f"  Sum of kernels: {total:.0f}µs")
print(f"  Note: wgrad + actgrad run in PARALLEL → actual wall time ≈ max(wgrad, actgrad) + rest")
wall = max(t_wgrad_up, t_actgrad) + t_dequant + t_dout_quant + t_wgrad_down + t_scatter
print(f"  Estimated wall time (with overlap): {wall:.0f}µs")
