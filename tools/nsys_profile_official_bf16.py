"""NSYS profiling script for official SonicMoE BF16 baseline.

Requires the official_bf16 virtualenv with quack-kernels 0.2.x (PyTorch main branch).

Run via nsys:
    source .../envs/official_bf16/bin/activate
    cd .../lab/official/sonic-moe
    CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda,nvtx --gpu-metrics-devices=0 \
        --cuda-memory-usage=false --cuda-event-trace=false -f true \
        -o /path/to/output --export=sqlite \
        python /path/to/nsys_profile_official_bf16.py

Shape: T=4096, H=4096, I=1024, E=128, K=8 (uniform 128-aligned routing)
"""
import os
import sys

# Must set env before imports
os.environ["USE_QUACK_GEMM"] = "1"
for _k in [
    "PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"
# Ensure no FP8 flags leak
for _k in ["SONIC_MOE_FP8_MODE", "SONIC_MOE_FP8_FUSED_GATED",
            "SONIC_MOE_FP8_WGRAD", "SONIC_MOE_FP8_ASSUME_ALIGNED",
            "SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "SONIC_MOE_FP8_SAVE_Z_FP8"]:
    os.environ.pop(_k, None)

# Use official sonic-moe (must be pip installed or on sys.path)
OFFICIAL_PATH = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
if OFFICIAL_PATH not in sys.path:
    sys.path.insert(0, OFFICIAL_PATH)

import torch
import torch.cuda.nvtx as nvtx

T, H, I, E, K = 4096, 4096, 1024, 128, 8
TK = T * K
WARMUP = 5
PROFILE_ITERS = 3


def build_uniform_routing(device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def main():
    from sonicmoe import MoE, enable_quack_gemm, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F_mod

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)

    _scores, _indices = build_uniform_routing(torch.device("cuda"))

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

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    def zero_grads():
        for p in moe.parameters():
            p.grad = None

    # Pre-allocate grad output (avoid stride-0 from sum().backward())
    grad_out = torch.ones(T, H, device="cuda", dtype=torch.bfloat16)

    # Warmup with QuACK enabled
    print(f"Warming up official BF16 ({WARMUP} iters)...")
    for _ in range(WARMUP):
        zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        with enable_quack_gemm(True):
            out, _ = moe(x_, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        out.backward(grad_out)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Profile
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(PROFILE_ITERS):
        nvtx.range_push(f"official_bf16_iter_{i}")

        nvtx.range_push("zero_grad")
        zero_grads()
        nvtx.range_pop()

        nvtx.range_push("clone_input")
        x_ = x_base.clone().requires_grad_(True)
        nvtx.range_pop()

        torch.cuda.synchronize()
        nvtx.range_push("forward")
        with enable_quack_gemm(True):
            out, _ = moe(x_, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("backward")
        out.backward(grad_out)
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_pop()  # iter
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    F_mod.TC_Softmax_Topk_Router_Function = orig_router
    print(f"Official BF16 profiling complete: {PROFILE_ITERS} iters, T={T} H={H} I={I} E={E} K={K}")


if __name__ == "__main__":
    main()
