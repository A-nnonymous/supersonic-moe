"""Trace cache hits/misses during forward+backward with offload."""
import sys, os, gc, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
torch.manual_seed(42)
T, H, I, E, K = 8192, 3072, 1536, 8, 8
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

# Monkey-patch cache functions to trace hits/misses
import importlib; bfp8 = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")
_orig_precompute = bfp8.precompute_weight_fp8
_orig_fused_gated = bfp8.precompute_weight_fp8_for_fused_gated
_orig_fused_dgated = bfp8.precompute_weight_fp8_for_direct_fused_dgated

def traced_precompute(w):
    key = (w._version, tuple(w.shape), tuple(w.stride()))
    hit = key in bfp8._VARLEN_WEIGHT_CACHE
    print(f"  precompute_weight_fp8: key={key[:2]}... {'HIT' if hit else 'MISS'} dtype={w.dtype}")
    return _orig_precompute(w)

def traced_fused_gated(w):
    key = (w._version, tuple(w.shape), tuple(w.stride()))
    hit = key in bfp8._FUSED_WEIGHT_CACHE
    print(f"  precompute_fp8_fused_gated: key={key[:2]}... {'HIT' if hit else 'MISS'} dtype={w.dtype}")
    return _orig_fused_gated(w)

def traced_fused_dgated(w):
    key = (w._version, tuple(w.shape), tuple(w.stride()))
    hit = key in bfp8._FUSED_WEIGHT_CACHE
    print(f"  precompute_fp8_fused_dgated: key={key[:2]}... {'HIT' if hit else 'MISS'} dtype={w.dtype}")
    return _orig_fused_dgated(w)

bfp8.precompute_weight_fp8 = traced_precompute
bfp8.precompute_weight_fp8_for_fused_gated = traced_fused_gated
bfp8.precompute_weight_fp8_for_direct_fused_dgated = traced_fused_dgated
# Also patch the imported names in __init__.py
import sonicmoe.functional.__init__ as fi
fi.precompute_weight_fp8 = traced_precompute
fi.precompute_weight_fp8_for_fused_gated = traced_fused_gated

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02
).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
for _ in range(2):
    with enable_quack_gemm(True):
        o = moe(x, use_fp8=True)[0]
    o.backward(dout)
    x.grad = None
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None

print("=== OFFLOAD + FORWARD + BACKWARD ===")
moe.refresh_fp8_shadow_weights()
moe.offload_bf16_weights()
print("Cache filled, weights offloaded to fp8.")

x_test = x.detach().clone().requires_grad_()
print("\n--- FORWARD ---")
with enable_quack_gemm(True):
    o = moe(x_test, use_fp8=True)[0]
print("\n--- BACKWARD ---")
o.backward(dout)
print("\nDone.")
