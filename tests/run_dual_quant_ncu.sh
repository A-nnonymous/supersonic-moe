#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

OUTDIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/dual_quant_ncu
mkdir -p $OUTDIR

echo "=== NCU: Dual-quant vs Separate kernels ==="
echo "Host: $(hostname)"

# 1. NCU profile of dual_quantize_and_pack
echo "--- NCU: dual_quantize_and_pack ---"
ncu --target-processes all --kernel-name '_dual_quantize_kernel' -c 1 --set full \
    -o $OUTDIR/dual_quant \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'; os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import dual_quantize_and_pack
E, H = 8, 3072; TK = 65536; CAP = TK // E
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
torch.cuda.synchronize()
for _ in range(3): dual_quantize_and_pack(dout, E, CAP, gather_idx=x_idx)
torch.cuda.synchronize()
dual_quantize_and_pack(dout, E, CAP, gather_idx=x_idx)
torch.cuda.synchronize()
print('done')
" 2>&1 | tail -5

# 2. NCU profile of separate row quant
echo "--- NCU: quantize_and_pack_activation (row) ---"
ncu --target-processes all --kernel-name '_quantize_and_pack_kernel' -c 1 --set full \
    -o $OUTDIR/row_quant \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'; os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation
E, H = 8, 3072; TK = 65536
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
for _ in range(3): quantize_and_pack_activation(dout)
torch.cuda.synchronize()
quantize_and_pack_activation(dout)
torch.cuda.synchronize()
print('done')
" 2>&1 | tail -5

# 3. NCU profile of fused_transpose_quantize (col)
echo "--- NCU: fused_transpose_quantize (col) ---"
ncu --target-processes all --kernel-name '_fused_transpose_quantize_kernel' -c 1 --set full \
    -o $OUTDIR/col_quant \
    python -c "
import os, sys, torch
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'; os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import fused_transpose_quantize_for_wgrad
E, H = 8, 3072; TK = 65536; CAP = TK // E
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device='cuda', dtype=torch.bfloat16)
x_idx = torch.arange(TK, device='cuda', dtype=torch.int32)
torch.cuda.synchronize()
for _ in range(3): fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
torch.cuda.synchronize()
fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
torch.cuda.synchronize()
print('done')
" 2>&1 | tail -5

echo ""
echo "=== Reading NCU metrics ==="
for f in dual_quant row_quant col_quant; do
    echo ""
    echo "--- $f ---"
    ncu --import $OUTDIR/${f}.ncu-rep --print-summary per-kernel 2>&1 | grep -E 'Duration|Compute.*Throughput|Memory Throughput|DRAM Throughput|SM Busy|Executed Ipc|L1|L2' | head -12
done

echo ""
echo "=== Output ==="
ls -lh $OUTDIR/
