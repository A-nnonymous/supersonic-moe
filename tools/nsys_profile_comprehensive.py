"""Comprehensive nsys profiling: BF16 vs FP8 with NVTX markers.

Captures all GPU activity: kernels, malloc, memset, memcpy, NVTX ranges.
Uses the same uniform routing as bench_aligned_e2e.py for production shapes.

Usage (on remote node):
    nsys profile -t cuda,nvtx --cuda-memory-usage=true \
        -o /tmp/sonic_bf16 --force-overwrite=true \
        python tools/nsys_profile_comprehensive.py --mode bf16

    nsys profile -t cuda,nvtx --cuda-memory-usage=true \
        -o /tmp/sonic_fp8 --force-overwrite=true \
        python tools/nsys_profile_comprehensive.py --mode fp8

    nsys profile -t cuda,nvtx --cuda-memory-usage=true \
        -o /tmp/sonic_fp8wg --force-overwrite=true \
        python tools/nsys_profile_comprehensive.py --mode fp8_wgrad
"""
import argparse
import os
import sys

# Must set env before any sonicmoe import
os.environ["USE_QUACK_GEMM"] = "1"
for _k in [
    "PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

import torch
import torch.cuda.nvtx as nvtx

# ====================== Shape ======================
T, H, I, E, K = 4096, 4096, 1024, 128, 8
TK = T * K
tpe = TK // E
assert tpe % 128 == 0

WARMUP = 5
PROFILE_ITERS = 3


def setup_mode(mode: str):
    if mode in ("fp8", "fp8_wgrad"):
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
        if mode == "fp8_wgrad":
            os.environ["SONIC_MOE_FP8_WGRAD"] = "1"
        else:
            os.environ["SONIC_MOE_FP8_WGRAD"] = "0"
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"
        os.environ.pop("SONIC_MOE_FP8_ASSUME_ALIGNED", None)
        os.environ.pop("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", None)
        os.environ.pop("SONIC_MOE_FP8_SAVE_Z_FP8", None)
        os.environ.pop("SONIC_MOE_FP8_WGRAD", None)


def build_uniform_routing(device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "fp8", "fp8_wgrad"], required=True)
    args = parser.parse_args()
    setup_mode(args.mode)

    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import clear_all_fp8_weight_caches
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache,
    )
    import sonicmoe.functional as F_mod

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # Uniform routing patch
    _scores, _indices = build_uniform_routing(x_base.device)

    class _UniformRouter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, router_logits, E_arg, K_arg):
            ctx.save_for_backward(_scores, _indices)
            ctx.E = E_arg; ctx.dtype = router_logits.dtype
            return _scores.clone(), _indices.clone()
        @staticmethod
        def backward(ctx, grad_scores, _grad_indices):
            scores, _ = ctx.saved_tensors
            return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None

    orig_router = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter

    if args.mode != "bf16":
        F_mod._ALIGNMENT_ASSUMED = True

    def zero_grads():
        for p in moe.parameters():
            p.grad = None

    # Warmup (JIT compile everything)
    print(f"Warming up {args.mode} mode ({WARMUP} iters)...")
    for _ in range(WARMUP):
        zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    torch.cuda.synchronize()
    print("Warmup done.")

    # Profiled iterations with fine-grained NVTX
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(PROFILE_ITERS):
        nvtx.range_push(f"{args.mode}_iter_{i}")

        nvtx.range_push("zero_grad")
        zero_grads()
        nvtx.range_pop()

        nvtx.range_push("clone_input")
        x_ = x_base.clone().requires_grad_(True)
        nvtx.range_pop()

        # Sync before forward to ensure clean GPU state
        torch.cuda.synchronize()
        nvtx.range_push("forward")
        out, _ = moe(x_)
        torch.cuda.synchronize()
        nvtx.range_pop()

        # Sync before backward to ensure clean GPU state
        nvtx.range_push("backward")
        out.sum().backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_pop()  # iter
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    F_mod.TC_Softmax_Topk_Router_Function = orig_router
    print(f"Profiling complete: mode={args.mode}, {PROFILE_ITERS} iters, shape T={T} H={H} I={I} E={E} K={K}")


if __name__ == "__main__":
    main()
