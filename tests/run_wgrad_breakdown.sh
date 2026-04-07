#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_weight_grad_gemm_fast,
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)
from sonicmoe.quack_utils.gemm_interface import gemm_gated_tuned
import socket

E, H, I = 8, 3072, 1536; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()

print(f'Host: {socket.gethostname()}')
print(f'Shape: TK={TK}, H={H}, I={I}, E={E}')
print('='*70)

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
    print(f'  {name:<50} min={mn:>7.0f}us  all={[f\"{t:.0f}\" for t in times]}')
    return mn

# 1. BF16 wgrad with A_idx (down-proj shape: dout.T @ y1s → H×I×E)
print('--- Down-proj wgrad: dout(TK,H).T @ y1s(TK,I) → dw2(H,I,E) ---')
# The actual BF16 wgrad uses gemm(dout.T, y1s_wgrad, out=dw2, cu_seqlens_k=cu, A_idx=x_idx)
# But gemm_gated_tuned doesn't exist in the right format for wgrad.
# Let me use the raw gemm_gated or the direct quack gemm call.
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated

# For wgrad, we need: A=dout.T (H, TK), B=y1s (TK, I, E), out=(H, I, E)
# This is a different GEMM than GemmDGated. Let's use a direct approach.

# Actually BF16 wgrad uses quack's gemm() which is a standard GEMM with A_idx.
# Let me just directly time both end-to-end.

def run_fp8_wgrad():
    blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx)

t_fp8 = bench(run_fp8_wgrad, 'FP8 wgrad fast (full pipeline)')

# Sub-operations of FP8 wgrad:
# Step 1: Quantize both inputs
def run_quant_dout():
    quantize_and_pack_activation(dout)
t_q1 = bench(run_quant_dout, 'Quantize dout (TK×H)')

def run_quant_y1s():
    quantize_activation_blockscaled_fast(y1s)
t_q2 = bench(run_quant_y1s, 'Quantize y1s (TK×I)')

# Can't easily isolate the pack/transpose/GEMM steps inside the function.
# But we can compute:
t_overhead = t_fp8 - 200  # Estimate: FP8 GEMM ~200µs (2x faster than BF16 ~400µs)
print(f'\\n--- Wgrad FP8 Breakdown ---')
print(f'Total FP8 wgrad:     {t_fp8:.0f}us')
print(f'  Quantize dout:     {t_q1:.0f}us')
print(f'  Quantize y1s:      {t_q2:.0f}us')
print(f'  Sum quant:         {t_q1 + t_q2:.0f}us')
print(f'  Rest (pack+GEMM):  {t_fp8 - t_q1 - t_q2:.0f}us')
print(f'')
print(f'Theoretical analysis:')
tflops = 2 * TK * H * I / E / 1e12  # per expert
bf16_peak = 106e12 / 1e12  # B200 BF16 peak TFLOPS
fp8_peak = 212e12 / 1e12   # B200 FP8 peak TFLOPS (TBC)
print(f'  FLOPS per expert:  {tflops*1e3:.1f} GFLOPS')
print(f'  BF16 theory min:   {tflops / bf16_peak * 1e6:.0f}us (at {bf16_peak:.0f} TFLOPS peak)')
print(f'  FP8 theory min:    {tflops / fp8_peak * 1e6:.0f}us (at {fp8_peak:.0f} TFLOPS peak)')
print(f'  8 experts total:   {tflops * 8 / bf16_peak * 1e6:.0f}us BF16 / {tflops * 8 / fp8_peak * 1e6:.0f}us FP8')
"
