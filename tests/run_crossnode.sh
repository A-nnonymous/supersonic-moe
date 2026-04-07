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

from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
import socket

E, H, I = 8, 3072, 1536
TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device='cuda', dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2*I, device='cuda', dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc.view(torch.uint8)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')

WARMUP, ITERS, TRIALS = 10, 20, 5

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

host = socket.gethostname()
gpu = torch.cuda.get_device_name(0)
print(f'Host: {host}  GPU: {gpu}')
print('=' * 70)

z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)

def run_bf16_total():
    z_tmp = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
    gemm_dgated(df, wf, dx, z_tmp, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)

def run_bf16_gemm():
    gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)

t_bf16_total = bench(run_bf16_total, 'BF16 total (dequant+GEMM)')
t_bf16_gemm = bench(run_bf16_gemm, 'BF16 GEMM only')

gemm_dgated.compile_cache.clear()
def run_fp8():
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_u8)
t_fp8 = bench(run_fp8, 'FP8 TMA (Int16+dequant)')

print(f'\\nBF16 total: {t_bf16_total:.0f}us  |  BF16 GEMM: {t_bf16_gemm:.0f}us  |  FP8 TMA: {t_fp8:.0f}us')
print(f'FP8 vs BF16 total: {t_fp8 - t_bf16_total:+.0f}us ({(t_fp8/t_bf16_total - 1)*100:+.1f}%)')
print(f'FP8 vs BF16 GEMM:  {t_fp8 - t_bf16_gemm:+.0f}us ({(t_fp8/t_bf16_gemm - 1)*100:+.1f}%)')
"
