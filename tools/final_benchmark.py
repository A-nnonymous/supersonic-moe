"""Realistic training benchmark: weights NOT re-initialized between trials."""
import os, sys, torch, time
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = os.environ.get("SONIC_MOE_FP8_MODE", "perf")
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs, count_cumsum
from triton.testing import do_bench
import torch.nn.functional as F

T,H,I,E,K = 8192,4096,1024,128,8
torch.manual_seed(42)
moe = MoE(E,K,H,I,ActivationType.SWIGLU,False,0.02).to(torch.bfloat16).cuda()
x = 0.2*torch.randn(T,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
dout = 0.2*torch.randn_like(x)
w1,w2,rw = moe.c_fc.weight,moe.c_proj.weight,moe.router.weight

# Token rounding routing (fixed, not re-generated each trial)
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
fused = os.environ.get("SONIC_MOE_FP8_FUSED_GATED", "1")  # match default in _use_fused_blockscaled_gated()
print(f"Config: FP8_MODE={mode}, FUSED_GATED={fused}, T={T}, H={H}, I={I}, E={E}, K={K}")

def fwd():
    return moe_general_routing_inputs(x,rs,ti,ei,w1.permute(1,2,0),None,w2.permute(1,2,0),None,E,moe.stream_id,ActivationType.SWIGLU,False)

def fwd_bwd():
    o, _ = fwd()
    torch.autograd.grad(o, [x, rs, w1, w2], dout, retain_graph=True)
    rs.grad = x.grad = w1.grad = w2.grad = None

# Warmup (includes CUTLASS compile + weight quantization)
for _ in range(3): fwd_bwd()

# Timing
fwd_time = do_bench(fwd, warmup=10, rep=200)
e2e_time = do_bench(fwd_bwd, warmup=10, rep=200, grad_to_none=[x, w1, w2, rw, dout])
bwd_time = e2e_time - fwd_time

TK = rs.shape[0]
fwd_tflops = 6*TK*I*H / (fwd_time * 1e9)
e2e_tflops = 18*TK*I*H / (e2e_time * 1e9)

# Memory
torch.cuda.reset_peak_memory_stats()
fwd_bwd()
torch.cuda.synchronize()
peak_mib = torch.cuda.max_memory_allocated() / 1024**2

print(f"Fwd:     {fwd_time:.3f} ms ({fwd_tflops:.1f} TFLOPS)")
print(f"E2E:     {e2e_time:.3f} ms ({e2e_tflops:.1f} TFLOPS)")
print(f"Bwd:     {bwd_time:.3f} ms")
print(f"Peak:    {peak_mib:.0f} MiB")
print(f"TK:      {TK}")
