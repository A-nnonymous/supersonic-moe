#!/bin/bash
# Self-contained nsys benchmark script for remote execution on idle node.
# Usage: ssh <host> bash /path/to/nsys_remote_bench.sh
set -e

export VSCODE_SHELL_INTEGRATION=0
WORK=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
OUT=$WORK/benchmarks/nsys_clean
mkdir -p $OUT

FP8_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python
BF16_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/python

# Install nsys if needed
which nsys >/dev/null 2>&1 || dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb 2>/dev/null

# Verify GPU is idle
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ==================== BF16 Script ====================
cat > /tmp/_nsys_bf16.py << 'PYEOF'
import torch, os, sys
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

T, H, I, E, K = 8192, 3072, 1536, 8, 8
device = torch.device("cuda")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
moe.train()

# Warmup
for i in range(5):
    torch.manual_seed(42)
    x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
    dout = 0.02 * torch.randn_like(x)
    with enable_quack_gemm(True):
        out, loss = moe(x)
        out.backward(dout)
    moe.zero_grad()
torch.cuda.synchronize()

# Profiled iterations
for i in range(10):
    torch.manual_seed(42 + i)
    x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
    dout = 0.02 * torch.randn_like(x)
    
    torch.cuda.nvtx.range_push("forward")
    with enable_quack_gemm(True):
        out, loss = moe(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("backward")
    out.backward(dout)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    moe.zero_grad()
torch.cuda.synchronize()
print("BF16 profiling done")
PYEOF

# ==================== FP8 Script ====================
cat > /tmp/_nsys_fp8.py << 'PYEOF'
import torch, os, sys
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_DOUBLE_QUANT"] = "1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

T, H, I, E, K = 8192, 3072, 1536, 8, 8
device = torch.device("cuda")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
moe.train()

# Warmup
for i in range(5):
    torch.manual_seed(42)
    x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
    dout = 0.02 * torch.randn_like(x)
    with enable_quack_gemm(True):
        out, loss = moe(x, use_fp8=True)
        out.backward(dout)
    moe.zero_grad()
torch.cuda.synchronize()

# Profiled iterations
for i in range(10):
    torch.manual_seed(42 + i)
    x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
    dout = 0.02 * torch.randn_like(x)
    
    torch.cuda.nvtx.range_push("forward")
    with enable_quack_gemm(True):
        out, loss = moe(x, use_fp8=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("backward")
    out.backward(dout)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    moe.zero_grad()
torch.cuda.synchronize()
print("FP8 profiling done")
PYEOF

echo ""
echo "=== Running BF16 nsys profile (GPU 1) ==="
CUDA_VISIBLE_DEVICES=1 nsys profile \
  --output=$OUT/bf16_clean \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --capture-range=none \
  --sample=none \
  $BF16_PY /tmp/_nsys_bf16.py 2>&1

echo ""
echo "=== Running FP8 nsys profile (GPU 2) ==="
CUDA_VISIBLE_DEVICES=6 nsys profile \
  --output=$OUT/fp8_clean \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --capture-range=none \
  --sample=none \
  $FP8_PY /tmp/_nsys_fp8.py 2>&1

echo ""
echo "=== Exporting to SQLite ==="
nsys export --type=sqlite --output=$OUT/bf16_clean.sqlite $OUT/bf16_clean.nsys-rep 2>&1
nsys export --type=sqlite --output=$OUT/fp8_clean.sqlite $OUT/fp8_clean.nsys-rep 2>&1

echo ""
echo "=== Analysis ==="
$FP8_PY $WORK/tools/nsys_full_breakdown.py $OUT/bf16_clean.sqlite $OUT/fp8_clean.sqlite --labels bf16 fp8 2>&1

echo ""
echo "=== DONE ==="
