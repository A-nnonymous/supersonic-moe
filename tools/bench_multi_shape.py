"""Multi-shape MoE benchmark: BF16 vs FP8 with 128-aligned routing.

Runs three shapes:
  1. T=4096, H=4096, I=1024, E=128, K=8  (original small)
  2. T=4096, H=4096, I=2048, E=128, K=8  (original large)
  3. T=8192, H=3072, I=1536, E=8,   K=8  (Ernie)

Based on bench_aligned_e2e.py — uses uniform routing to ensure 128-alignment.
"""

import os
import sys
import gc

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

WARMUP = 8
ITERS = 20

SHAPES = [
    # (T, H, I, E, K)
    (4096, 4096, 1024, 128, 8),
    (4096, 4096, 2048, 128, 8),
    (8192, 3072, 1536, 8, 8),
]


# ====================== Uniform Router Patch ======================

_uniform_indices = None
_uniform_scores = None
_current_T = None
_current_E = None
_current_K = None


def _build_uniform_routing(device, T, E, K):
    global _uniform_indices, _uniform_scores, _current_T, _current_E, _current_K
    if _uniform_indices is not None and _current_T == T and _current_E == E and _current_K == K:
        return
    _current_T, _current_E, _current_K = T, E, K
    TK = T * K
    tpe = TK // E
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    _uniform_indices = ((tok * K + off) % E).to(torch.int32)
    _uniform_scores = torch.full(
        (T, K), 1.0 / K, dtype=torch.float32, device=device
    )
    counts = torch.bincount(_uniform_indices.flatten(), minlength=E)
    assert counts.min().item() == counts.max().item() == tpe, (
        f"Routing not uniform: min={counts.min()}, max={counts.max()}, want={tpe}"
    )


class _UniformSoftmaxTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits, E_arg, K_arg):
        T = router_logits.size(0)
        _build_uniform_routing(router_logits.device, T, E_arg, K_arg)
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

def _reset_fp8_state():
    clear_all_fp8_weight_caches()
    clear_blockscaled_fp8_weight_cache()
    _COMPILE_CACHE.clear()
    F_mod._ALIGNMENT_STREAK = 0


def bench(moe_model, x_base, mode, warmup=WARMUP, iters=ITERS):
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

    def _zero_grads():
        for p in moe_model.parameters():
            p.grad = None

    # Warmup
    for _ in range(warmup):
        _zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe_model(x_)
        out.sum().backward()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    # Forward only
    ev_s.record()
    for _ in range(iters):
        with torch.no_grad():
            out, _ = moe_model(x_base)
    ev_e.record()
    torch.cuda.synchronize()
    fwd_ms = ev_s.elapsed_time(ev_e) / iters

    # Forward + Backward
    ev_s.record()
    for _ in range(iters):
        _zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe_model(x_)
        out.sum().backward()
    ev_e.record()
    torch.cuda.synchronize()
    total_ms = ev_s.elapsed_time(ev_e) / iters
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


def main():
    enable_quack_gemm()
    orig_router_fn = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformSoftmaxTopk

    results = []

    try:
        for T, H, I, E, K in SHAPES:
            TK = T * K
            tpe = TK // E
            assert tpe % 128 == 0, f"tpe={tpe} not 128-aligned for T={T} E={E} K={K}"

            print(f"\n{'='*70}")
            print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K} (tpe={tpe})")
            print(f"{'='*70}")

            # Reset global routing state
            global _uniform_indices, _uniform_scores, _current_T
            _uniform_indices = None
            _uniform_scores = None
            _current_T = None

            torch.manual_seed(42)
            moe_model = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to("cuda", torch.bfloat16)

            x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

            bf16_fwd, bf16_bwd, bf16_total = bench(moe_model, x_base, "bf16")
            print(
                f"  BF16:  fwd={bf16_fwd:.3f}ms  bwd={bf16_bwd:.3f}ms  total={bf16_total:.3f}ms"
            )

            fp8_fwd, fp8_bwd, fp8_total = bench(moe_model, x_base, "fp8")
            print(
                f"  FP8:   fwd={fp8_fwd:.3f}ms  bwd={fp8_bwd:.3f}ms  total={fp8_total:.3f}ms"
            )

            print(
                f"  Speedup: fwd={bf16_fwd / fp8_fwd:.2f}x  "
                f"bwd={bf16_bwd / fp8_bwd:.2f}x  "
                f"total={bf16_total / fp8_total:.2f}x"
            )

            results.append({
                "shape": (T, H, I, E, K),
                "bf16": (bf16_fwd, bf16_bwd, bf16_total),
                "fp8": (fp8_fwd, fp8_bwd, fp8_total),
            })

            # Free memory before next shape
            del moe_model, x_base
            _reset_fp8_state()
            gc.collect()
            torch.cuda.empty_cache()

        # Summary table
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Shape':<35} {'BF16 total':>10} {'FP8 total':>10} {'Speedup':>8}")
        print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*8}")
        for r in results:
            T, H, I, E, K = r["shape"]
            bf16_t = r["bf16"][2]
            fp8_t = r["fp8"][2]
            shape_str = f"T={T} H={H} I={I} E={E} K={K}"
            print(f"{shape_str:<35} {bf16_t:>9.3f}ms {fp8_t:>9.3f}ms {bf16_t/fp8_t:>7.2f}x")

    finally:
        F_mod.TC_Softmax_Topk_Router_Function = orig_router_fn


if __name__ == "__main__":
    main()
