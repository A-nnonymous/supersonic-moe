#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'; os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_weight_grad_gemm_fast,
    dual_quantize_and_pack,
    quantize_and_pack_activation,
    fused_transpose_quantize_for_wgrad,
    _auto_capacity, _gather_isa_packed_scales_kernel,
    _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE, _storage_per_batch,
)
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
import socket

E, H, I = 8, 3072, 1536; TK = 65536; CAP = TK // E
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2*I, device='cuda', dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z) if False else (None, None)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
w2 = 0.02 * torch.randn(H, I, E, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()

from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)

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
    print(f'  {name:<60} min={mn:>7.0f}us')
    return mn

# 1. BF16 wgrad (baseline)
dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc.view(torch.uint8))
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')
# GemmDGated to get y1s, then wgrad
print('--- Current backward pipeline ---')

# Simulate the full fused backward for down-proj wgrad
def run_current_bf16_wgrad():
    # Step 1: quant dout + scale gather
    dout_fp8_t, dout_scales_t = quantize_and_pack_activation(dout)
    K_bwd = H
    k_tiles_bwd = _div_up(K_bwd, _SF_TILE_K)
    per_batch_bwd = _storage_per_batch(TK, K_bwd)
    dout_scales_tk = torch.empty((1, per_batch_bwd), dtype=torch.uint8, device='cuda')
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles_bwd)](
        dout_scales_t.view(torch.uint8), x_idx, dout_scales_tk, TK,
        src_k_tiles=k_tiles_bwd, dst_k_tiles=k_tiles_bwd,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    # Step 2: GemmDGated (BF16 C)
    gemm_dgated(dout_fp8_t, wf, dx, z_bf16, pa, None, 'swiglu', 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_scales_tk.view(torch.float8_e8m0fnu), b_scales=ws)
    # Step 3: wgrad (BF16 A_idx)
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx)

t_current = bench(run_current_bf16_wgrad, 'Current: row_quant + scatter + GemmDGated(bf16) + wgrad')

# 2. With dual-quant
def run_dual_quant_wgrad():
    # Step 1: dual quant (1 read, row + col outputs)
    row_fp8, row_scales, col_fp8_a, col_scales_a = dual_quantize_and_pack(
        dout, E, CAP, gather_idx=x_idx)
    # Step 2: scale gather (still needed for GemmDGated's TK-space scales)
    K_bwd = H
    k_tiles_bwd = _div_up(K_bwd, _SF_TILE_K)
    per_batch_bwd = _storage_per_batch(TK, K_bwd)
    dout_scales_tk = torch.empty((1, per_batch_bwd), dtype=torch.uint8, device='cuda')
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles_bwd)](
        row_scales.view(torch.uint8), x_idx, dout_scales_tk, TK,
        src_k_tiles=k_tiles_bwd, dst_k_tiles=k_tiles_bwd,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    # Step 2: GemmDGated (BF16 C)
    gemm_dgated(row_fp8, wf, dx, z_bf16, pa, None, 'swiglu', 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_scales_tk.view(torch.float8_e8m0fnu), b_scales=ws)
    # Step 3: wgrad (FP8, col_fp8_a already prepared!)
    # Only need to quant y1s (= pa from GemmDGated)
    col_fp8_b, col_scales_b = fused_transpose_quantize_for_wgrad(pa, E, CAP, I)
    # Direct CUTLASS GEMM with pre-quantized A
    # TODO: need to call CUTLASS GEMM directly, skipping re-quantization of A

t_dual = bench(run_dual_quant_wgrad, 'Dual-quant: dual_quant + scatter + GemmDGated + wgrad(partial)')

print(f'\\n--- Summary ---')
print(f'Current pipeline:  {t_current:.0f}us')
print(f'Dual-quant pipeline: {t_dual:.0f}us')
print(f'Saving: {t_current - t_dual:.0f}us ({(1 - t_dual/t_current)*100:.1f}%)')
"
