import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = os.environ.get("SONIC_MOE_FP8_MODE", "perf")
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

# Token rounding routing
router_logits = F.linear(x, rw)
scores = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)
topk_values, topk_indices = scores.topk(K, dim=-1)
expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
Mtile = 128
expert_freq_rounded = torch.round(expert_freq / Mtile).to(torch.int32) * Mtile
topk_values /= topk_values.sum(dim=-1, keepdim=True)
scores_clone = scores.detach().clone() - 1
scores_clone.scatter_(1, topk_indices, topk_values)
topk_idx2 = scores_clone.argsort(dim=0, descending=True).int()
mask = torch.arange(T, device="cuda", dtype=torch.int32)[:,None].expand(-1,E) < expert_freq_rounded[None,:]
token_indices = topk_idx2[mask]
expert_indices = torch.arange(E, device="cuda", dtype=torch.int32)[None,:].expand(T,-1)[mask]
order = token_indices.argsort().int()
token_indices = token_indices[order]
expert_indices = expert_indices[order]
router_scores = scores[token_indices, expert_indices].contiguous()

fp8_mode = os.environ.get("SONIC_MOE_FP8_MODE", "off")
fused = os.environ.get("SONIC_MOE_FP8_FUSED_GATED", "1")
print(f"Config: FP8_MODE={fp8_mode}, FUSED_GATED={fused}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated() / 1024**2

# Forward
o, _ = moe_general_routing_inputs(x,router_scores,token_indices,expert_indices,
    w1.permute(1,2,0),None,w2.permute(1,2,0),None,E,moe.stream_id,ActivationType.SWIGLU,False)
torch.cuda.synchronize()
after_fwd = torch.cuda.memory_allocated() / 1024**2
peak_fwd = torch.cuda.max_memory_allocated() / 1024**2

# Backward
torch.cuda.reset_peak_memory_stats()
torch.autograd.grad(o, [x, router_scores, w1, w2], dout, retain_graph=False)
torch.cuda.synchronize()
after_bwd = torch.cuda.memory_allocated() / 1024**2
peak_bwd = torch.cuda.max_memory_allocated() / 1024**2

print(f"Before:      {before:.1f} MiB")
print(f"After fwd:   {after_fwd:.1f} MiB (peak: {peak_fwd:.1f} MiB)")
print(f"After bwd:   {after_bwd:.1f} MiB (peak during bwd: {peak_bwd:.1f} MiB)")
print(f"Fwd delta:   +{after_fwd - before:.1f} MiB")
print(f"Peak fwd:    {peak_fwd:.1f} MiB")
print(f"Peak bwd:    {peak_bwd:.1f} MiB")
