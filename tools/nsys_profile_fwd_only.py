"""Forward-only nsys profiling for BF16 baseline comparison."""
import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_TC_softmax_topk_layer
T,H,I,E,K = 8192,4096,1024,128,8
torch.manual_seed(42)
moe = MoE(E,K,H,I,ActivationType.SWIGLU,False,0.02).to(torch.bfloat16).cuda()
x = 0.2*torch.randn(T,H,device="cuda",dtype=torch.bfloat16)
w1,w2,rw = moe.c_fc.weight,moe.c_proj.weight,moe.router.weight
mode = os.environ.get("SONIC_MOE_FP8_MODE", "off")
label = f"FP8_MODE={mode}"
print(f"Profiling forward-only: {label}")
def fwd():
    return moe_TC_softmax_topk_layer(x,rw,w1.permute(1,2,0),None,w2.permute(1,2,0),None,K,moe.stream_id,ActivationType.SWIGLU,True,None)
for _ in range(5): fwd()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for i in range(10):
    torch.cuda.nvtx.range_push(f"{label}_fwd_{i}")
    fwd()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("Done")
