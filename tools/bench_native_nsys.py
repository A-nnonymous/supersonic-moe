import sys, os, torch
FORK_DIR = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
sys.path.insert(0, FORK_DIR)
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe import MoE, enable_native_fp8
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import quantize_and_pack_activation

T, H, I, E, K = 8192, 3072, 1536, 8, 8
torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

nfp = moe.prepare_native_fp8()
x_fp8, x_scales = quantize_and_pack_activation(x.detach())

for _ in range(10):
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_native_fp8():
        out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=x_fp8, x_fp8_scales=x_scales)
    out.backward(dout)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()

for _ in range(5):
    x.grad = None; moe.zero_grad(set_to_none=True)
    with enable_native_fp8():
        out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=x_fp8, x_fp8_scales=x_scales)
    out.backward(dout)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("nsys done")
