#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c "
import torch, os, sys
sys.path.insert(0, '.')
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

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

# BF16 reference
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
dx_ref = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa_ref = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')
gemm_dgated(df, wf, dx_ref, z_bf16, pa_ref, torch.zeros(1, dtype=torch.int32, device='cuda'),
            'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)

# FP8 PreAct via TMA
gemm_dgated.compile_cache.clear()
dx_fp8 = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa_fp8 = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')
gemm_dgated(df, wf, dx_fp8, z, pa_fp8, torch.zeros(1, dtype=torch.int32, device='cuda'),
            'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
            preact_fp8=z_fp8, preact_scales=z_sc_u8)

# Precision
def rrmse(a, b):
    return ((a.float()-b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
def cosine(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm()).clamp(min=1e-12)).item()

print(f'dx RRMSE: {rrmse(dx_fp8, dx_ref):.6f}  cosine: {cosine(dx_fp8, dx_ref):.6f}')
print(f'pa RRMSE: {rrmse(pa_fp8, pa_ref):.6f}  cosine: {cosine(pa_fp8, pa_ref):.6f}')
print()

# Memory check
import gc; gc.collect(); torch.cuda.empty_cache()
print(f'z_fp8:  {z_fp8.numel() * z_fp8.element_size() / 1024**2:.0f} MiB')
print(f'z_bf16: {z.numel() * z.element_size() / 1024**2:.0f} MiB')
print(f'Saving: {(z.numel() * z.element_size() - z_fp8.numel() * z_fp8.element_size()) / 1024**2:.0f} MiB')
"
