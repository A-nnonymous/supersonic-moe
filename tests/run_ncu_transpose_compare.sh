#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

OUTDIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/transpose_quant_ncu
mkdir -p $OUTDIR

echo "=== NCU GPU-Projection: Original vs 32x32 ==="

# 1. Original kernel
echo "--- Original (32x128, 4 warps) ---"
ncu --target-processes all --kernel-name '_fused_transpose_quantize_kernel' -c 1 --set full \
    -o $OUTDIR/original \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
# Temporarily use original kernel
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _fused_transpose_quantize_kernel, _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE, _storage_per_batch, _div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,128),CAP//32)
for _ in range(3):
    _fused_transpose_quantize_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=128,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE)
torch.cuda.synchronize()
_fused_transpose_quantize_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=128,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE)
torch.cuda.synchronize();print('done')
" 2>&1 | tail -3

# 2. 32x32 kernel
echo "--- 32x32 (1 warp, GPB=16) ---"
ncu --target-processes all --kernel-name '_warp32x32' -c 1 --set full \
    -o $OUTDIR/warp32 \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _warp32x32_transpose_quant_kernel, _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE, _storage_per_batch, _div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,32),_div_up(CAP//32,16))
for _ in range(3):
    _warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=16,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize()
_warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=16,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize();print('done')
" 2>&1 | tail -3

echo ""
echo "=== Comparison ==="
for f in original warp32; do
    echo "--- $f ---"
    ncu --import $OUTDIR/${f}.ncu-rep --print-summary per-kernel 2>&1 | grep -E 'Duration|Compute.*Throughput|Memory Throughput|DRAM Throughput|SM Busy|Executed Ipc|L1|L2|Occupancy|Registers|Waves' | head -15
    echo ""
done
