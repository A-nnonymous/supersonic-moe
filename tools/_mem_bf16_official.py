import sys, os, gc, json, torch
sys.path.insert(0, '/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe')
os.environ['USE_QUACK_GEMM'] = '1'
torch.manual_seed(42)
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.utils import enable_quack_gemm
from sonicmoe import MoE

ck = {}
def c(n):
    torch.cuda.synchronize()
    ck[n] = round(torch.cuda.memory_allocated() / MiB, 2)
def p(n):
    torch.cuda.synchronize()
    ck[n] = round(torch.cuda.max_memory_allocated() / MiB, 2)

c("00_empty")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02
).to(device="cuda", dtype=torch.bfloat16)
c("01_model")
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
c("02_input")

def run_fwd():
    with enable_quack_gemm(True):
        return moe_TC_softmax_topk_layer(
            x, moe.router.weight, moe.c_fc.weight.permute(1,2,0), None,
            moe.c_proj.weight.permute(1,2,0), None, K, 0, ActivationType.SWIGLU, False)

for _ in range(2):
    o, _, _ = run_fwd()
    o.backward(dout)
    x.grad = None
    for pp in moe.parameters():
        if pp.grad is not None: pp.grad = None

gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
c("03_post_warmup")
o, _, _ = run_fwd()
torch.cuda.synchronize(); p("04_fwd_peak"); c("05_post_fwd")
torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize(); p("06_bwd_peak"); c("07_post_bwd")
x.grad = None
for pp in moe.parameters():
    if pp.grad is not None: pp.grad = None
del o; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
c("08_cleanup")
print(json.dumps(ck))
