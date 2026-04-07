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
    fused_transpose_quantize_for_wgrad,
    _auto_capacity,
    _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE, _SF_VEC_SIZE,
    _storage_per_batch, _div_up,
)
import socket

E, H, I = 8, 3072, 1536; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
capacity = int(((TK // E + 127) // 128) * 128)
torch.cuda.synchronize()

print(f'Host: {socket.gethostname()}')
print(f'Shape: TK={TK}, H={H}, I={I}, E={E}, capacity={capacity}')
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
    print(f'  {name:<55} min={mn:>7.0f}us  all={[f\"{t:.0f}\" for t in times]}')
    return mn

# Individual sub-operations
print('--- Sub-operation timing ---')

# 1. quantize_and_pack_activation (row-major fp8 + ISA scales)
def run_qpa_dout():
    quantize_and_pack_activation(dout)
t_qpa_dout = bench(run_qpa_dout, 'quantize_and_pack dout (T×H, row)')

def run_qpa_y1s():
    quantize_and_pack_activation(y1s)
t_qpa_y1s = bench(run_qpa_y1s, 'quantize_and_pack y1s (TK×I, row)')

# 2. fused_transpose_quantize (transpose + fp8 + ISA scales)
def run_ftq_dout():
    fused_transpose_quantize_for_wgrad(dout, E, capacity, H, gather_idx=x_idx)
t_ftq_dout = bench(run_ftq_dout, 'fused_transpose_quantize dout (→H×TK, col)')

def run_ftq_y1s():
    fused_transpose_quantize_for_wgrad(y1s, E, capacity, I)
t_ftq_y1s = bench(run_ftq_y1s, 'fused_transpose_quantize y1s (→I×TK, col)')

# 3. Full FP8 wgrad pipeline
def run_fp8_wgrad():
    blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx)
t_fp8_full = bench(run_fp8_wgrad, 'FP8 wgrad full pipeline')

# 4. Isolated GEMM (pre-quantize, then measure just GEMM)
a_fp8, a_scales = fused_transpose_quantize_for_wgrad(dout, E, capacity, H, gather_idx=x_idx)
b_fp8, b_scales = fused_transpose_quantize_for_wgrad(y1s, E, capacity, I)
torch.cuda.synchronize()

# Can't easily isolate the GEMM without calling the full function.
# Compute by subtraction.
t_gemm_est = t_fp8_full - t_ftq_dout - t_ftq_y1s

# 5. Bandwidth analysis
dout_bytes = TK * H * 2  # bf16
y1s_bytes = TK * I * 2
dout_fp8_bytes = E * H * capacity  # fp8 transposed
y1s_fp8_bytes = E * I * capacity
hbm_bw = 8e12  # B200 ~8 TB/s

print(f'\\n--- Breakdown ---')
print(f'FP8 wgrad total:              {t_fp8_full:.0f}us')
print(f'  fused_transpose_quant dout: {t_ftq_dout:.0f}us')
print(f'  fused_transpose_quant y1s:  {t_ftq_y1s:.0f}us')
print(f'  GEMM (estimated):           {t_gemm_est:.0f}us')
print(f'')
print(f'--- Bandwidth analysis ---')
print(f'dout read (bf16):    {dout_bytes/1e6:.0f} MB → {dout_bytes/hbm_bw*1e6:.0f}us at {hbm_bw/1e12:.0f} TB/s')
print(f'dout_fp8 write:      {dout_fp8_bytes/1e6:.0f} MB → {dout_fp8_bytes/hbm_bw*1e6:.0f}us')
print(f'y1s read (bf16):     {y1s_bytes/1e6:.0f} MB → {y1s_bytes/hbm_bw*1e6:.0f}us')
print(f'y1s_fp8 write:       {y1s_fp8_bytes/1e6:.0f} MB → {y1s_fp8_bytes/hbm_bw*1e6:.0f}us')

# If we fused dout quant (row+col) in one kernel
dout_fused_io = dout_bytes + dout_bytes // 2 + dout_fp8_bytes + dout_fp8_bytes // 32
print(f'\\n--- Potential savings ---')
print(f'Current: separate row-quant ({t_qpa_dout:.0f}us) + col-quant ({t_ftq_dout:.0f}us)')
print(f'If fused dual-quant dout (1 read, 2 writes): ~{max(dout_bytes + dout_bytes//2 + dout_fp8_bytes, dout_bytes)/ hbm_bw*1e6:.0f}us bandwidth-limited')
print(f'Saving from dout dual-quant: ~{t_qpa_dout + t_ftq_dout - 120:.0f}us')
print(f'')
print(f'If y1s quant fused into GemmDGated epilogue: ~0us (compute-hidden)')
print(f'Saving from y1s epilogue-quant: ~{t_ftq_y1s:.0f}us')
print(f'')
print(f'Total potential FP8 wgrad with all fusions:')
print(f'  ~{t_gemm_est:.0f}us (pure GEMM) + ~5us overhead')
print(f'  vs BF16 wgrad: 467us')
if t_gemm_est < 467:
    print(f'  → FP8 would be {467/t_gemm_est:.2f}x FASTER')
else:
    print(f'  → FP8 would still be {t_gemm_est/467:.2f}x slower (GEMM itself is bottleneck)')
"
