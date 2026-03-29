"""
E2E nsys profiling script for SonicMoE FP8 vs BF16.
Captures CUDA kernel timelines, launch overhead, memory ops.

Usage:
    # Profile FP8
    CUDA_VISIBLE_DEVICES=7 nsys profile -t cuda,nvtx,osrt \
        --cuda-memory-usage=true -o /tmp/sonicmoe_fp8 --force-overwrite=true \
        python tools/nsys_profile_e2e.py --mode fp8

    # Profile BF16
    CUDA_VISIBLE_DEVICES=7 nsys profile -t cuda,nvtx,osrt \
        --cuda-memory-usage=true -o /tmp/sonicmoe_bf16 --force-overwrite=true \
        python tools/nsys_profile_e2e.py --mode bf16

    # Export to SQLite for analysis
    nsys export --type=sqlite -o /tmp/sonicmoe_fp8.sqlite /tmp/sonicmoe_fp8.nsys-rep
"""
import argparse
import os
import sys

import torch
import torch.cuda.nvtx as nvtx


def setup_env(mode: str):
    os.environ["USE_QUACK_GEMM"] = "1"
    if mode == "fp8":
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"


def create_moe_and_inputs(hidden_size=4096, intermediate_size=2048, num_experts=8,
                          num_experts_per_tok=8, seq_len=512):
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType

    moe = MoE(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=torch.bfloat16)

    x = (0.02 * torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)
    return moe, x, dout


def run_fwd_bwd(moe, x, dout, protocol, label):
    from sonicmoe import KernelBackendMoE, enable_quack_gemm

    with enable_quack_gemm(True):
        nvtx.range_push(f"{label}_forward")
        output, _aux = moe(
            x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol,
        )
        nvtx.range_pop()

        nvtx.range_push(f"{label}_backward")
        output.backward(dout)
        nvtx.range_pop()

    torch.cuda.synchronize()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp8", "bf16"], required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    setup_env(args.mode)

    from sonicmoe import get_default_fp8_protocol

    moe, x, dout = create_moe_and_inputs(
        args.hidden, args.intermediate, args.experts, args.topk, args.seq
    )
    protocol = get_default_fp8_protocol() if args.mode == "fp8" else None

    # Warmup (JIT compile + cache fill)
    for i in range(args.warmup):
        x_w = x.detach().clone().requires_grad_()
        nvtx.range_push(f"warmup_{i}")
        run_fwd_bwd(moe, x_w, dout, protocol, f"warmup_{i}")
        nvtx.range_pop()
        moe.zero_grad()

    torch.cuda.synchronize()

    # Profiled iterations
    for i in range(args.iters):
        x_p = x.detach().clone().requires_grad_()
        nvtx.range_push(f"iter_{i}")
        run_fwd_bwd(moe, x_p, dout, protocol, f"iter_{i}")
        nvtx.range_pop()
        moe.zero_grad()

    torch.cuda.synchronize()
    print(f"Profiling done: mode={args.mode}, {args.iters} iterations captured")


if __name__ == "__main__":
    main()
