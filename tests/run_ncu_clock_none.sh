#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

OUTDIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/transpose_quant_ncu_v2
mkdir -p $OUTDIR

echo "=== NCU --clock-control=none: fair comparison ==="
echo "Host: $(hostname)"

# Helper: NCU a specific kernel call
run_ncu() {
    local NAME=$1; shift
    echo "--- $NAME ---"
    ncu --target-processes all --kernel-name "$1" -c 1 --clock-control=none \
        --set full -o $OUTDIR/$NAME "${@:2}" 2>&1 | grep -E 'Profiling|Disconnected' | tail -2
    echo "  Metrics:"
    ncu --import $OUTDIR/${NAME}.ncu-rep --print-summary per-kernel 2>&1 | \
        grep -E 'Duration|Memory Throughput|DRAM Throughput|L1.*Throughput|L2.*Throughput|Compute.*Throughput|SM Busy|Executed Ipc|Registers|Waves|Occupancy' | head -15
    echo ""
}

# 1. Original kernel (32x128)
run_ncu "original" "_fused_transpose_quantize_kernel" \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _fused_transpose_quantize_kernel,_SF_TILE_M,_SF_TILE_K,_SF_TILE_STORAGE,_storage_per_batch,_div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,128),CAP//32)
for _ in range(3): _fused_transpose_quantize_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=128,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE)
torch.cuda.synchronize()
_fused_transpose_quantize_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=128,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE)
torch.cuda.synchronize()
"

# 2. 32x32 kernel GPB=8
run_ncu "warp32_gpb8" "_warp32x32_transpose_quant_kernel" \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _warp32x32_transpose_quant_kernel,_SF_TILE_M,_SF_TILE_K,_SF_TILE_STORAGE,_storage_per_batch,_div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,32),_div_up(CAP//32,8))
for _ in range(3): _warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=8,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize()
_warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=8,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize()
"

# 3. 32x32 kernel GPB=4 (less reg pressure)
run_ncu "warp32_gpb4" "_warp32x32_transpose_quant_kernel" \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _warp32x32_transpose_quant_kernel,_SF_TILE_M,_SF_TILE_K,_SF_TILE_STORAGE,_storage_per_batch,_div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,32),_div_up(CAP//32,4))
for _ in range(3): _warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=4,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize()
_warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=4,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=1)
torch.cuda.synchronize()
"

# 4. 32x32 kernel GPB=4, 2 warps (more parallelism)
run_ncu "warp32_gpb4_2w" "_warp32x32_transpose_quant_kernel" \
    python -c "
import os,sys,torch;sys.path.insert(0,'.')
os.environ['USE_QUACK_GEMM']='1';os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _warp32x32_transpose_quant_kernel,_SF_TILE_M,_SF_TILE_K,_SF_TILE_STORAGE,_storage_per_batch,_div_up
E,H=8,3072;TK=65536;CAP=TK//E
torch.manual_seed(42);dout=0.02*torch.randn(TK,H,device='cuda',dtype=torch.bfloat16)
x_idx=torch.arange(TK,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
fp8=torch.empty(E*H,CAP,dtype=torch.float8_e4m3fn,device='cuda')
pb=_storage_per_batch(H,CAP);sc=torch.ones(E,pb,dtype=torch.float8_e8m0fnu,device='cuda')
grid=(E*_div_up(H,32),_div_up(CAP//32,4))
for _ in range(3): _warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=4,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=2)
torch.cuda.synchronize()
_warp32x32_transpose_quant_kernel[grid](dout,x_idx,fp8,sc.view(torch.uint8),H,CAP,pb,dout.stride(0),dout.stride(1),HAS_GATHER=True,GROUP_SIZE=32,BLOCK_DIM=32,GROUPS_PER_BLOCK=4,SF_TILE_M=_SF_TILE_M,SF_TILE_K=_SF_TILE_K,SF_TILE_STORAGE=_SF_TILE_STORAGE,num_warps=2)
torch.cuda.synchronize()
"

echo "=== Done ==="
