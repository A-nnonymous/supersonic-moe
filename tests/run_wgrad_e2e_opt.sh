#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c "
import os, sys, torch; sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM']='1'; os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_weight_grad_gemm_fast
import socket

E, H, I = 8, 3072, 1536; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
torch.cuda.synchronize()

print(f'Host: {socket.gethostname()}')
print('='*60)

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
    print(f'  {name:<50} min={mn:>7.0f}us')
    return mn

t_fp8 = bench(lambda: blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx),
              'FP8 wgrad (with 32x32 kernel)')
print(f'\\nPrevious: 696us  |  Now: {t_fp8:.0f}us  |  BF16 baseline: 467us')
print(f'Speedup vs previous: {696/t_fp8:.2f}x')
if t_fp8 < 467:
    print(f'FP8 is now {467/t_fp8:.2f}x FASTER than BF16!')
else:
    print(f'FP8 is still {t_fp8/467:.2f}x slower than BF16')
"
