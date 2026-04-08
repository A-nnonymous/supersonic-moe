#!/bin/bash
export VSCODE_SHELL_INTEGRATION=0
FP8_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python
BF16_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/python

echo "=== BF16 Cold Start ==="
CUDA_VISIBLE_DEVICES=1 $BF16_PY << 'PY'
import torch, os, sys
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe")
os.environ["USE_QUACK_GEMM"]="1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
torch.cuda.reset_peak_memory_stats()
moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072, intermediate_size=1536, activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
moe.train()
torch.manual_seed(42)
x = (0.02 * torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_(True)
dout = 0.02 * torch.randn_like(x)
with enable_quack_gemm(True):
    out, loss = moe(x)
    out.backward(dout)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1024**2
alloc = torch.cuda.memory_allocated() / 1024**2
print(f"BF16 cold-start peak: {peak:.2f} MiB, after-bwd alloc: {alloc:.2f} MiB")
PY

echo ""
echo "=== FP8 Cold Start ==="
CUDA_VISIBLE_DEVICES=6 $FP8_PY << 'PY'
import torch, os, sys
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"]="1"
os.environ["SONIC_MOE_FP8_MODE"]="perf"
os.environ["SONIC_MOE_FP8_DOUBLE_QUANT"]="1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
torch.cuda.reset_peak_memory_stats()
moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072, intermediate_size=1536, activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
moe.train()
torch.manual_seed(42)
x = (0.02 * torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_(True)
dout = 0.02 * torch.randn_like(x)
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
    out.backward(dout)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1024**2
alloc = torch.cuda.memory_allocated() / 1024**2
print(f"FP8 cold-start peak: {peak:.2f} MiB, after-bwd alloc: {alloc:.2f} MiB")
PY
