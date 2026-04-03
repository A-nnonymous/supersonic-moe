"""Isolated memory + timing comparison (separate process per mode)."""
import sys, os, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

mode = sys.argv[1]  # "bf16", "frontier", "native"

from sonicmoe import MoE, enable_fp8, enable_native_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils import quantize_and_pack_activation

E, K, H, I = 8, 8, 3072, 1536
T = 8192

def mb(x): return x / 1024**2

torch.manual_seed(42)
moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

if mode == "native":
    nfp = moe.prepare_native_fp8()
    xf, xs = quantize_and_pack_activation(x.detach())

# Memory
torch.cuda.reset_peak_memory_stats()
moe.zero_grad(set_to_none=True); x.grad = None
if mode == "bf16":
    with enable_quack_gemm(True):
        out, _ = moe(x)
elif mode == "frontier":
    with enable_quack_gemm(True), enable_fp8():
        out, _ = moe(x, use_fp8=True)
else:
    with enable_native_fp8():
        out, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
out.backward(dout)
peak = torch.cuda.max_memory_allocated()

# Timing
WARMUP, ITERS = 20, 30
def run():
    x.grad = None; moe.zero_grad(set_to_none=True)
    if mode == "bf16":
        with enable_quack_gemm(True):
            z, _ = moe(x)
    elif mode == "frontier":
        with enable_quack_gemm(True), enable_fp8():
            z, _ = moe(x, use_fp8=True)
    else:
        with enable_native_fp8():
            z, _ = moe(x, use_fp8=True, native_fp8_params=nfp, x_fp8_data=xf, x_fp8_scales=xs)
    z.backward(dout)

for _ in range(WARMUP):
    run()
torch.cuda.synchronize()
times = []
for _ in range(5):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        run()
    e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e) * 1000 / ITERS)

mn = min(times)
avg = sum(times)/len(times)
print(f"{mode}: peak={mb(peak):.1f}MiB avg={avg:.0f}µs min={mn:.0f}µs")
