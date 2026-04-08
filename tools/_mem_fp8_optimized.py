import sys
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
import os, gc, json, torch
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
# NOTE: FUSED_ZY1_QUANT is OFF (default) — split quant saves 96 MiB forward peak
torch.manual_seed(42)
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm
ck = {}
def c(n): torch.cuda.synchronize(); ck[n] = round(torch.cuda.memory_allocated()/MiB, 2)
def p(n): torch.cuda.synchronize(); ck[n] = round(torch.cuda.max_memory_allocated()/MiB, 2)

c("00_empty")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
c("01_model")
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
c("02_input")

for _ in range(3):
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); x.grad = None
    for pp in moe.parameters():
        if pp.grad is not None: pp.grad = None

# Forward-only shadow refresh (only w1_fused + w2_varlen = 111 MiB, NOT backward caches)
moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
c("03_post_warmup")

with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
torch.cuda.synchronize(); p("04_fwd_peak"); c("04_fwd_alloc")
torch.cuda.reset_peak_memory_stats()
o.backward(dout); torch.cuda.synchronize(); p("05_bwd_peak"); c("05_bwd_alloc")
x.grad = None
for pp in moe.parameters():
    if pp.grad is not None: pp.grad = None
del o; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
c("06_cleanup")

# Timing
moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
times = []
for _ in range(10):
    for pp in moe.parameters():
        if pp.grad is not None: pp.grad = None
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e)); del o
ck["timing_avg_ms"] = round(sum(times)/len(times), 3)
ck["timing_std_ms"] = round((sum((t-sum(times)/len(times))**2 for t in times)/len(times))**0.5, 3)
print(json.dumps(ck))
