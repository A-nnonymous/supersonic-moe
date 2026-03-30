"""End-to-end aligned MoE benchmark: BF16 vs FP8 with 128-aligned routing.

Production MoE systems use token-rounding that guarantees every expert segment
is 128-aligned.  Random routing (as in bench_fwd_bwd_e2e.py) produces
non-aligned segments, which either forces a padding fallback or crashes
the FP8 kernel with ILLEGAL_INSTRUCTION when ASSUME_ALIGNED=1.

This script monkey-patches TC_Softmax_Topk_Router_Function to emit perfectly
uniform routing -- each of the E=128 experts receives exactly tpe = T*K/E = 256
tokens (256 = 128*2, so 128-aligned).  The rest of the MoE forward+backward
runs through the *normal* code path (including TC_topk_router_metadata_triton
for scatter/gather index computation), giving a realistic production benchmark.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py
"""

import os
import sys

# ---------- environment (must precede any sonicmoe import) ----------
os.environ["USE_QUACK_GEMM"] = "1"
for _k in [
    "PADDLE_ELASTIC_JOB_ID",
    "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS",
    "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

import torch
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches
import sonicmoe.functional as F_mod
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _COMPILE_CACHE,
    clear_blockscaled_fp8_weight_cache,
)

# ====================== Shape Configuration ======================
T, H, I, E, K = 4096, 4096, 1024, 128, 8
TK = T * K  # 32768
tpe = TK // E  # 256 tokens per expert
assert tpe % 128 == 0, f"tpe={tpe} is not 128-aligned; adjust T, K, E"

WARMUP = 8
ITERS = 20


# ====================== Uniform Router Patch ======================
#
# Token i is assigned to experts  (i*K + j) % E  for j in 0..K-1.
# Because K=8 divides E=128, every expert receives exactly tpe = T*K/E = 256
# tokens -- each a multiple of 128.

_uniform_indices: torch.Tensor | None = None
_uniform_scores: torch.Tensor | None = None


def _build_uniform_routing(device: torch.device) -> None:
    """Build and verify the deterministic uniform routing table (once)."""
    global _uniform_indices, _uniform_scores
    if _uniform_indices is not None:
        return
    tok = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    off = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
    _uniform_indices = ((tok * K + off) % E).to(torch.int32)  # (T, K)
    _uniform_scores = torch.full(
        (T, K), 1.0 / K, dtype=torch.float32, device=device
    )
    counts = torch.bincount(_uniform_indices.flatten(), minlength=E)
    assert counts.min().item() == counts.max().item() == tpe, (
        f"Routing not uniform: min={counts.min()}, max={counts.max()}, want={tpe}"
    )


class _UniformSoftmaxTopk(torch.autograd.Function):
    """Drop-in replacement for TC_Softmax_Topk_Router_Function.

    Ignores actual router logits and returns pre-computed uniform routing.
    The backward returns zeros for the router-logit gradient (negligible
    contribution to wall-clock timing).
    """

    @staticmethod
    def forward(ctx, router_logits, E_arg, K_arg):
        _build_uniform_routing(router_logits.device)
        ctx.save_for_backward(_uniform_scores, _uniform_indices)
        ctx.E = E_arg
        ctx.dtype = router_logits.dtype
        return _uniform_scores.clone(), _uniform_indices.clone()

    @staticmethod
    def backward(ctx, grad_scores, _grad_indices):
        scores, _ = ctx.saved_tensors
        T_local = scores.size(0)
        return (
            torch.zeros(T_local, ctx.E, dtype=ctx.dtype, device=scores.device),
            None,
            None,
        )


# ====================== Helpers ======================

# Module-level references set in main(), used by bench()
moe: MoE
x_base: torch.Tensor


def _reset_fp8_state() -> None:
    """Clear all FP8 caches and alignment tracking between modes."""
    clear_all_fp8_weight_caches()
    clear_blockscaled_fp8_weight_cache()
    _COMPILE_CACHE.clear()
    F_mod._ALIGNMENT_STREAK = 0


def bench(mode: str, warmup: int = WARMUP, iters: int = ITERS):
    """Benchmark one mode ('bf16' or 'fp8').

    Returns (fwd_ms, bwd_ms, total_ms).
    """
    if mode == "fp8":
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
        F_mod._ALIGNMENT_ASSUMED = True
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"
        os.environ.pop("SONIC_MOE_FP8_ASSUME_ALIGNED", None)
        os.environ.pop("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", None)
        os.environ.pop("SONIC_MOE_FP8_SAVE_Z_FP8", None)
        F_mod._ALIGNMENT_ASSUMED = False
    _reset_fp8_state()

    # Warmup (compiles kernels, populates caches)
    for _ in range(warmup):
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    # Forward only
    ev_s.record()
    for _ in range(iters):
        with torch.no_grad():
            out, _ = moe(x_base)
    ev_e.record()
    torch.cuda.synchronize()
    fwd_ms = ev_s.elapsed_time(ev_e) / iters

    # Forward + Backward
    ev_s.record()
    for _ in range(iters):
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    ev_e.record()
    torch.cuda.synchronize()
    total_ms = ev_s.elapsed_time(ev_e) / iters
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


# ====================== Main ======================


def main() -> None:
    global moe, x_base

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # Install uniform routing -- replaces the real softmax-topk router
    # so every expert gets exactly tpe tokens (128-aligned).
    orig_router_fn = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformSoftmaxTopk

    try:
        print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K} (aligned: tpe={tpe})")

        bf16_fwd, bf16_bwd, bf16_total = bench("bf16")
        print(
            f"BF16 baseline:     fwd={bf16_fwd:.3f}ms  "
            f"bwd={bf16_bwd:.3f}ms  total={bf16_total:.3f}ms"
        )

        fp8_fwd, fp8_bwd, fp8_total = bench("fp8")
        print(
            f"FP8 (aligned):     fwd={fp8_fwd:.3f}ms  "
            f"bwd={fp8_bwd:.3f}ms  total={fp8_total:.3f}ms"
        )

        print(
            f"Speedup: fwd={bf16_fwd / fp8_fwd:.2f}x  "
            f"bwd={bf16_bwd / fp8_bwd:.2f}x  "
            f"total={bf16_total / fp8_total:.2f}x"
        )
    finally:
        F_mod.TC_Softmax_Topk_Router_Function = orig_router_fn


if __name__ == "__main__":
    main()
