"""Detailed Nsight profiling with per-kernel NVTX markers."""
import os, sys, torch, time
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
label = f"FP8={mode}_FUSED={fused}"
print(f"Profiling: {label}, T={T}, H={H}, I={I}, E={E}, K={K}, TK={ti.shape[0]}")

def fwd_bwd():
    x_local = x.detach().clone().requires_grad_(True)
    o, _ = moe_general_routing_inputs(x_local,rs,ti,ei,w1.permute(1,2,0),None,w2.permute(1,2,0),None,E,moe.stream_id,ActivationType.SWIGLU,False)
    torch.autograd.grad(o, [x_local, rs, w1, w2], dout)

# Warmup (includes JIT compilation)
for i in range(5):
    fwd_bwd()
torch.cuda.synchronize()

# Measure wall-clock time
start = time.perf_counter()
for i in range(20):
    fwd_bwd()
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 20
print(f"Avg E2E: {elapsed*1000:.3f} ms")

# Profile 3 iterations for nsys/ncu
torch.cuda.cudart().cudaProfilerStart()
for i in range(3):
    torch.cuda.nvtx.range_push(f"{label}_iter_{i}")
    fwd_bwd()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
print(f"Profiling done: {label}")
