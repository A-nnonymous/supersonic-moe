#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

OUTDIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/wgrad_ncu_v2
mkdir -p $OUTDIR

echo "=== Wgrad NCU Analysis v2 (with kernel-name filter) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Use --kernel-name to filter only the CUTLASS kernel, and --launch-skip to skip warmup
# The CUTLASS kernel name contains "kernel" from quack

echo ""
echo "--- NCU: BF16 GemmDGated ---"
ncu --target-processes all --launch-skip 3 -c 1 --set full \
    -o $OUTDIR/gemmdgated_bf16 \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

E, H, I = 8, 3072, 1536; TK = 65536
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
torch.cuda.synchronize()

# 3 warmup (skipped by --launch-skip 3) then 1 profiled
for i in range(4):
    gemm_dgated(df, wf, dx, z_bf16, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws)
torch.cuda.synchronize()
print('NCU BF16 GemmDGated done')
" 2>&1 | tail -5

echo ""
echo "--- NCU: FP8 TMA GemmDGated ---"
ncu --target-processes all --launch-skip 3 -c 1 --set full \
    -o $OUTDIR/gemmdgated_fp8_tma \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast, quantize_and_pack_activation
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated

E, H, I = 8, 3072, 1536; TK = 65536
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
torch.cuda.synchronize()

for i in range(4):
    gemm_dgated(df, wf, dx, z, pa, torch.zeros(1, dtype=torch.int32, device='cuda'),
                'swiglu', 128, 128, 1, 1, cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc.view(torch.uint8))
torch.cuda.synchronize()
print('NCU FP8 TMA done')
" 2>&1 | tail -5

echo ""
echo "--- NCU: BF16 wgrad (gemm with A_idx) ---"
ncu --target-processes all --launch-skip 3 -c 1 --set full \
    -o $OUTDIR/wgrad_bf16 \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.gemm_gated import gemm_gated

E, H, I = 8, 3072, 1536; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')
# Use gemm_gated wrapper for BF16 wgrad (PostAct = y1s)
pa = torch.empty(TK, I, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()

# This uses the BF16 A_idx varlen GEMM
from sonicmoe.quack_utils.gemm_interface import gemm_gated_tuned
for i in range(4):
    gemm_gated_tuned(dout, y1s, out=dw2, cu_seqlens_m=cu, A_idx=x_idx)
torch.cuda.synchronize()
print('NCU wgrad BF16 done')
" 2>&1 | tail -5

echo ""
echo "--- NCU: FP8 wgrad (blockscaled_fp8_weight_grad_gemm_fast) ---"
ncu --target-processes all --launch-skip 3 -c 1 --set full \
    -o $OUTDIR/wgrad_fp8 \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_weight_grad_gemm_fast,
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)

E, H, I = 8, 3072, 1536; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device='cuda', dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                torch.full((E,), TK//E, dtype=torch.int32, device='cuda').cumsum(0)]).int()
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()

for i in range(4):
    blockscaled_fp8_weight_grad_gemm_fast(
        dout, y1s, cu, dw2, x_gather_idx=x_idx
    )
torch.cuda.synchronize()
print('NCU FP8 wgrad done')
" 2>&1 | tail -5

echo ""
echo "=== Output files ==="
ls -lh $OUTDIR/
