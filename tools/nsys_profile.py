"""NSYS profiling with NVTX markers for FP8 vs BF16 comparison."""
import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs, count_cumsum
import torch.nn.functional as F

T,H,I,E,K = 8192,4096,1024,128,8
torch.manual_seed(42)
moe = MoE(E,K,H,I,ActivationType.SWIGLU,False,0.02).to(torch.bfloat16).cuda()
x = 0.2*torch.randn(T,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
dout = 0.2*torch.randn_like(x)
w1,w2,rw = moe.c_fc.weight,moe.c_proj.weight,moe.router.weight

# Token rounding routing (fixed)
router_logits = F.linear(x, rw)
scores = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)
topk_values, topk_indices = scores.topk(K, dim=-1)
expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
expert_freq_rounded = torch.round(expert_freq / 128).to(torch.int32) * 128
topk_values /= topk_values.sum(dim=-1, keepdim=True)
sc = scores.detach().clone() - 1; sc.scatter_(1, topk_indices, topk_values)
idx2 = sc.argsort(dim=0, descending=True).int()
mask = torch.arange(T, device="cuda", dtype=torch.int32)[:,None].expand(-1,E) < expert_freq_rounded[None,:]
ti = idx2[mask]; ei = torch.arange(E, device="cuda", dtype=torch.int32)[None,:].expand(T,-1)[mask]
order = ti.argsort().int(); ti = ti[order]; ei = ei[order]
rs = scores[ti, ei].contiguous()

mode = os.environ.get("SONIC_MOE_FP8_MODE", "off")
fused = os.environ.get("SONIC_MOE_FP8_FUSED_GATED", "0")
label = f"FP8_MODE={mode}_FUSED={fused}"
print(f"Profiling: {label}, T={T}, H={H}, I={I}, E={E}, K={K}")

def fwd_bwd():
    o, _ = moe_general_routing_inputs(x,rs,ti,ei,w1.permute(1,2,0),None,w2.permute(1,2,0),None,E,moe.stream_id,ActivationType.SWIGLU,False)
    torch.autograd.grad(o, [x, rs, w1, w2], dout, retain_graph=True)
    rs.grad = x.grad = w1.grad = w2.grad = None

# Warmup (5 iterations, includes all JIT compilation)
for i in range(5):
    fwd_bwd()
torch.cuda.synchronize()

# Profile 10 steady-state iterations with NVTX markers
torch.cuda.cudart().cudaProfilerStart()
for i in range(10):
    torch.cuda.nvtx.range_push(f"{label}_iter_{i}")
    torch.cuda.nvtx.range_push("forward")
    o, _ = moe_general_routing_inputs(x,rs,ti,ei,w1.permute(1,2,0),None,w2.permute(1,2,0),None,E,moe.stream_id,ActivationType.SWIGLU,False)
    torch.cuda.nvtx.range_pop()  # forward
    torch.cuda.nvtx.range_push("backward")
    torch.autograd.grad(o, [x, rs, w1, w2], dout, retain_graph=True)
    rs.grad = x.grad = w1.grad = w2.grad = None
    torch.cuda.nvtx.range_pop()  # backward
    torch.cuda.nvtx.range_pop()  # iter
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"Profiling complete: {label}")
