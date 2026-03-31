"""Profile official vs fork baselines with nsys NVTX GPU projection.

Supports: official BF16, fork BF16, fork FP8.
Uses sys.path isolation so both codebases run under the xfer env (same quack).

Usage:
  # On GPU node with xfer env activated:
  nsys profile -t cuda,nvtx -o /tmp/sonic_official_bf16 --force-overwrite true \
    --capture-range=cudaProfilerApi \
    python tools/profile_official_vs_fork.py --impl official

  nsys profile -t cuda,nvtx -o /tmp/sonic_fork_bf16 --force-overwrite true \
    --capture-range=cudaProfilerApi \
    python tools/profile_official_vs_fork.py --impl fork

  nsys profile -t cuda,nvtx -o /tmp/sonic_fork_fp8 --force-overwrite true \
    --capture-range=cudaProfilerApi \
    python tools/profile_official_vs_fork.py --impl fork --fp8

Then analyze with: python tools/nsys_full_breakdown.py /tmp/sonic_*.sqlite
"""
import argparse
import os
import sys

# --- sys.path isolation: MUST happen before any sonicmoe import ---
OFFICIAL_PATH = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
FORK_PATH = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"


def _isolate_imports(impl: str):
    """Ensure only the target impl's sonicmoe is importable."""
    # Remove any sonicmoe-related paths from sys.path
    sys.path = [p for p in sys.path if "sonic-moe" not in p and "sonic_moe" not in p]
    # Insert the target first
    target = OFFICIAL_PATH if impl == "official" else FORK_PATH
    sys.path.insert(0, target)
    # Purge any cached imports
    for mod_name in list(sys.modules.keys()):
        if "sonicmoe" in mod_name:
            del sys.modules[mod_name]


# --- Environment setup ---
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

# Production shape
T, H, I, E, K = 4096, 4096, 1024, 128, 8
WARMUP, PROFILE_ITERS = 5, 3


def build_uniform_routing(device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["official", "fork"], required=True)
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 mode (fork only)")
    args = parser.parse_args()

    if args.fp8 and args.impl == "official":
        print("ERROR: --fp8 is only supported with --impl fork")
        sys.exit(1)

    # Set FP8 env vars before import
    if args.fp8:
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
        os.environ["SONIC_MOE_FP8_WGRAD"] = "0"
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"

    # For official code: inject ArgumentsBase shim into quack (removed in 0.3.7)
    if args.impl == "official":
        import quack.cute_dsl_utils as _qutils
        if not hasattr(_qutils, "ArgumentsBase"):
            from dataclasses import fields as _fields
            from cutlass.base_dsl.typing import JitArgument as _JitArgument

            class ArgumentsBase(_JitArgument):
                def __c_pointers__(self):
                    all_flds = [getattr(self, f.name) for f in _fields(self)]
                    non_ce = [f for f in all_flds
                              if not isinstance(f, _qutils.StaticTypes)]
                    ptrs = []
                    for obj in non_ce:
                        if hasattr(obj, "__c_pointers__"):
                            ptrs.extend(obj.__c_pointers__())
                    return ptrs

            _qutils.ArgumentsBase = ArgumentsBase

    # Isolate imports
    _isolate_imports(args.impl)

    from sonicmoe.moe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe import enable_quack_gemm
    import sonicmoe.functional as F_mod

    label = f"{args.impl}_{'fp8' if args.fp8 else 'bf16'}"
    print(f"Loaded sonicmoe from: {F_mod.__file__}")

    # Official code bug: backward passes non-contiguous dout to gemm_dgated which
    # asserts k-major. Our fork fixed this (dout.contiguous()). Monkey-patch here
    # so we can profile official without modifying official repo files.
    if args.impl == "official":
        _DP = F_mod._DownProjection
        _orig_bwd = _DP.backward

        @staticmethod
        def _patched_bwd(ctx, dout):
            return _orig_bwd(ctx, dout.contiguous())

        _DP.backward = _patched_bwd
        print("Patched official _DownProjection.backward with dout.contiguous()")

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
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
            return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype,
                               device=scores.device), None, None

    # Patch router (same attr name in both official and fork)
    if hasattr(F_mod, "TC_Softmax_Topk_Router_Function"):
        F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter

    # For FP8 path, ensure alignment flag is set
    if args.fp8 and hasattr(F_mod, "_ALIGNMENT_ASSUMED"):
        F_mod._ALIGNMENT_ASSUMED = True

    def zero_grads():
        for p in moe.parameters():
            p.grad = None

    # Warmup (JIT compile)
    print(f"Warming up {label} ({WARMUP} iters)...")
    for _ in range(WARMUP):
        zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    torch.cuda.synchronize()
    print("Warmup done.")

    # Profiled iterations with NVTX + sync barriers for accurate GPU projection
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(PROFILE_ITERS):
        nvtx.range_push(f"{label}_iter_{i}")

        nvtx.range_push("zero_grad")
        zero_grads()
        nvtx.range_pop()

        nvtx.range_push("clone_input")
        x_ = x_base.clone().requires_grad_(True)
        nvtx.range_pop()

        torch.cuda.synchronize()
        nvtx.range_push("forward")
        out, _ = moe(x_)
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("backward")
        out.sum().backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_pop()  # iter
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Done: {label}, {PROFILE_ITERS} iters, shape T={T} H={H} I={I} E={E} K={K}")
    print(f"Peak GPU memory: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
