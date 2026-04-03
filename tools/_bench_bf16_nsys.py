import torch, os
os.environ.setdefault("USE_QUACK_GEMM", "1")
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import enable_quack_gemm

T,H,I,E,K=8192,3072,1536,8,8
torch.manual_seed(42)
moe=MoE(E,K,H,I,ActivationType.SWIGLU,add_bias=False,std=0.02).cuda().bfloat16()
x=(0.02*torch.randn(T,H,device="cuda",dtype=torch.bfloat16)).detach().requires_grad_()
dout=0.02*torch.randn_like(x)

for _ in range(10):
    x.grad=None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        out,_=moe(x)
    out.backward(dout)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    x.grad=None; moe.zero_grad(set_to_none=True)
    with enable_quack_gemm(True):
        out,_=moe(x)
    out.backward(dout)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("bf16 nsys done")
