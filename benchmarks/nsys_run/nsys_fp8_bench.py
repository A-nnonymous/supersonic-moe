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
