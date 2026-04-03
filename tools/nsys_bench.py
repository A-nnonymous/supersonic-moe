"""Nsys profiling script for native FP8 + epilogue quant."""
import sys, os, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe import MoE, enable_native_fp8
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import quantize_and_pack_activation

mode = sys.argv[1] if len(sys.argv) > 1 else "native"

E, K, H, I = 8, 8, 3072, 1536
T = 8192
torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

if mode == "native":
    nfp = moe.prepare_native_fp8()
    xf, xs = quantize_and_pack_activation(x.detach())

for _ in range(10):
    x.grad = None; moe.zero_grad(set_to_none=True)
    if mode == "native":
        with enable_native_fp8():
            out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
    else:
        from sonicmoe import enable_fp8, enable_quack_gemm
        with enable_quack_gemm(True), enable_fp8():
            out, _ = moe(x, use_fp8=True)
    out.backward(dout)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    x.grad = None; moe.zero_grad(set_to_none=True)
    if mode == "native":
        with enable_native_fp8():
            out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
    else:
        from sonicmoe import enable_fp8, enable_quack_gemm
        with enable_quack_gemm(True), enable_fp8():
            out, _ = moe(x, use_fp8=True)
    out.backward(dout)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"nsys {mode} done")
