#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
export NNODES=1 PADDLE_TRAINERS_NUM=1

OUTDIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/wgrad_ncu
mkdir -p $OUTDIR

echo "=== Wgrad NCU + nsys Analysis ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# 1. NCU profile: BF16 wgrad (A_idx) — capture 1 kernel
echo "--- NCU: BF16 wgrad down-proj ---"
ncu --target-processes all -c 1 --set full \
    -o $OUTDIR/wgrad_bf16_downproj \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
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
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc.view(torch.uint8))

# Warmup
for _ in range(3):
    gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)
torch.cuda.synchronize()

# Profile this one
gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
            'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)
torch.cuda.synchronize()
print('NCU BF16 done')
" 2>&1 | tail -5

echo ""
echo "--- NCU: FP8 TMA GemmDGated ---"
ncu --target-processes all -c 1 --set full \
    -o $OUTDIR/wgrad_fp8_tma \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated

E, H, I = 8, 3072, 1536
TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device='cuda', dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2*I, device='cuda', dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device='cuda')
pa = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')

# Warmup
for _ in range(3):
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc.view(torch.uint8))
torch.cuda.synchronize()

# Profile this one
gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
            'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
            preact_fp8=z_fp8, preact_scales=z_sc.view(torch.uint8))
torch.cuda.synchronize()
print('NCU FP8 TMA done')
" 2>&1 | tail -5

echo ""
echo "--- NCU: BF16 wgrad up-proj (the heaviest kernel) ---"
ncu --target-processes all -c 1 --set full \
    -o $OUTDIR/wgrad_bf16_upproj \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_interface import gemm

E, H, I = 8, 3072, 1536
TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_gather_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')

# Warmup
for _ in range(3):
    gemm(dout.T, y1s, out=dw2.permute(2,0,1).contiguous().permute(1,2,0),
         cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False)
torch.cuda.synchronize()

gemm(dout.T, y1s, out=dw2.permute(2,0,1).contiguous().permute(1,2,0),
     cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False)
torch.cuda.synchronize()
print('NCU wgrad up-proj done')
" 2>&1 | tail -5

echo ""
echo "--- nsys: full backward timeline ---"
nsys profile --stats=true -t cuda,nvtx --force-overwrite=true \
    -o $OUTDIR/backward_timeline \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
from sonicmoe.quack_utils.gemm_interface import gemm

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
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
x_gather_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')

# Warmup
for _ in range(3):
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_u8)
    gemm(dout.T, y1s, out=dw2.permute(2,0,1).contiguous().permute(1,2,0),
         cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False)
torch.cuda.synchronize()

# Profiled iteration
torch.cuda.nvtx.range_push('backward_fp8_tma')
gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
            'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
            preact_fp8=z_fp8, preact_scales=z_sc_u8)
gemm(dout.T, y1s, out=dw2.permute(2,0,1).contiguous().permute(1,2,0),
     cu_seqlens_k=cu, A_idx=x_gather_idx, dynamic_scheduler=False)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print('nsys done')
" 2>&1 | grep -E 'cuda_gpu_kern|Total|nsys done' | head -20

echo ""
echo "=== Analysis complete. Output in $OUTDIR ==="
ls -la $OUTDIR/
