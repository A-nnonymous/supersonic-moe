"""Capture memory snapshot at the EXACT backward peak moment."""
import sys, os, gc, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = os.environ.get("FP8_MODE", "perf")
os.environ["SONIC_MOE_STAGEWISE_MEMORY"] = "1"
torch.manual_seed(42)
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

for _ in range(2):
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); x.grad = None
    for p in moe.parameters():
        if p.grad is not None: p.grad = None

moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Patch _log_stage_memory to capture tensor inventory at each stage
import sonicmoe.functional.__init__ as func_module
_original_log = func_module._log_stage_memory
_snapshots = {}

def _patched_log(stage):
    _original_log(stage)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / MiB
    peak = torch.cuda.max_memory_allocated() / MiB
    # Enumerate live tensors
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                storage_size = obj.untyped_storage().size()
                if storage_size > 0:
                    elem_size = obj.element_size()
                    nbytes = obj.nelement() * elem_size
                    tensors.append((nbytes, tuple(obj.shape), str(obj.dtype), storage_size))
        except: pass
    tensors.sort(key=lambda x: -x[0])
    _snapshots[stage] = (alloc, peak, tensors)

func_module._log_stage_memory = _patched_log

# Run measured iter
torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
o.backward(dout)
torch.cuda.synchronize()

# Print snapshots
mode = os.environ["SONIC_MOE_FP8_MODE"]
print(f"=== TENSOR INVENTORY AT EACH STAGE ({mode} mode) ===\n")
for stage in sorted(_snapshots.keys()):
    alloc, peak, tensors = _snapshots[stage]
    total_tensor_bytes = sum(t[0] for t in tensors)
    print(f"[{stage}] alloc={alloc:.1f}M peak={peak:.1f}M tensors_sum={total_tensor_bytes/MiB:.1f}M")
    for nbytes, shape, dtype, storage in tensors[:12]:
        print(f"    {nbytes/MiB:>8.2f}M  {str(shape):>30s}  {dtype:>25s}  storage={storage}")
    if len(tensors) > 12:
        rest = sum(t[0] for t in tensors[12:])
        print(f"    ... +{len(tensors)-12} more tensors ({rest/MiB:.1f}M)")
    print()
