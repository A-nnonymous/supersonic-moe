#!/usr/bin/env python3
"""Profile BF16 official or FP8 fork — run with appropriate env and codebase."""
import argparse, os, sys, torch

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["bf16", "fp8"], required=True)
args = parser.parse_args()

T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP, ITERS = 10, 5

if args.mode == "bf16":
    sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe")
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ.pop("SONIC_MOE_FP8_MODE", None)
    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(42)

    def run_fwd():
        with enable_quack_gemm(True):
            return moe(x)[0]

    def run_bwd(z):
        z.backward(dout)
        x.grad = None; moe.zero_grad(set_to_none=True)

elif args.mode == "fp8":
    sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    from sonicmoe import MoE, enable_fp8, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(42)

    def run_fwd():
        with enable_quack_gemm(True), enable_fp8():
            return moe(x, use_fp8=True)[0]

    def run_bwd(z):
        z.backward(dout)
        x.grad = None; moe.zero_grad(set_to_none=True)

# Warmup
for i in range(WARMUP):
    z = run_fwd()
    run_bwd(z)
    if i == 0:
        torch.cuda.synchronize()
        print(f"[{args.mode}] warmup iter 0 done (JIT)")
torch.cuda.synchronize()
print(f"[{args.mode}] {WARMUP} warmup done")

# Profiled iterations with NVTX + sync
torch.cuda.cudart().cudaProfilerStart()
for i in range(ITERS):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"{args.mode}_iter{i}")

    torch.cuda.nvtx.range_push(f"{args.mode}_forward")
    z = run_fwd()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{args.mode}_backward")
    run_bwd(z)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
print(f"[{args.mode}] {ITERS} profiled iters done")
