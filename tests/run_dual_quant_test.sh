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
    dual_quantize_and_pack,
    quantize_and_pack_activation,
    fused_transpose_quantize_for_wgrad,
    _auto_capacity,
)
import socket

E, H, I = 8, 3072, 1536; TK = 65536
CAP = TK // E  # 8192 with uniform routing
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
torch.cuda.synchronize()

print(f'Host: {socket.gethostname()}')
print(f'Shape: TK={TK}, H={H}, E={E}, CAP={CAP}')
print('=' * 70)

# Reference: separate row + col quantization
print('1. Computing references...')
ref_row_fp8, ref_row_scales = quantize_and_pack_activation(dout)
ref_col_fp8, ref_col_scales = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
torch.cuda.synchronize()

# Dual quant
print('2. Running dual_quantize_and_pack...')
try:
    row_fp8, row_scales, col_fp8, col_scales = dual_quantize_and_pack(
        dout, E, CAP, gather_idx=x_idx
    )
    torch.cuda.synchronize()
    print('   Compiled OK')
except Exception as e:
    print(f'   FAILED: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

# Precision check
print('3. Precision check...')
row_fp8_match = (ref_row_fp8.view(torch.uint8) == row_fp8.view(torch.uint8)).float().mean().item()
row_scales_match = (ref_row_scales.view(torch.uint8) == row_scales.view(torch.uint8)).float().mean().item()
col_fp8_match = (ref_col_fp8.view(torch.uint8) == col_fp8.view(torch.uint8)).float().mean().item()
col_scales_match = (ref_col_scales.view(torch.uint8) == col_scales.view(torch.uint8)).float().mean().item()
print(f'   Row fp8 match:    {row_fp8_match*100:.2f}%')
print(f'   Row scales match: {row_scales_match*100:.2f}%')
print(f'   Col fp8 match:    {col_fp8_match*100:.2f}%')
print(f'   Col scales match: {col_scales_match*100:.2f}%')
all_match = row_fp8_match > 0.99 and col_fp8_match > 0.99 and row_scales_match > 0.99 and col_scales_match > 0.99
print(f'   PRECISION: {\"PASS\" if all_match else \"FAIL\"} ({\"bit-exact\" if all(x == 1.0 for x in [row_fp8_match, row_scales_match, col_fp8_match, col_scales_match]) else \"near-match\"})')

# Benchmark
print('4. Benchmark...')
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
    print(f'   {name:<55} min={mn:>7.0f}us')
    return mn

t_row = bench(lambda: quantize_and_pack_activation(dout), 'Separate: row quant')
t_col = bench(lambda: fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx), 'Separate: col quant')
t_dual = bench(lambda: dual_quantize_and_pack(dout, E, CAP, gather_idx=x_idx), 'Dual quant (1 read, 2 writes)')

print(f'')
print(f'--- Results ---')
print(f'Separate: {t_row + t_col:.0f}us ({t_row:.0f} + {t_col:.0f})')
print(f'Dual:     {t_dual:.0f}us')
print(f'Saving:   {t_row + t_col - t_dual:.0f}us ({(1 - t_dual/(t_row+t_col))*100:.1f}%)')
"
