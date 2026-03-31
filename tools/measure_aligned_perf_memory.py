"""Measure current aligned BF16/FP8 train+infer timing and peak memory.

This is a local event-based helper for iteration speed / memory trend checks.
Do not use it as the authoritative cross-branch baseline; for that, use
NSYS NVTX GPU projection against `reports/sonic_official_bf16.sqlite`.
"""

import os

# Must be set before sonicmoe imports.
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

import sonicmoe.functional as F_mod
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _COMPILE_CACHE,
    _COMPILE_CACHE_VK,
    clear_blockscaled_fp8_weight_cache,
)

T, H, I, E, K = 4096, 4096, 1024, 128, 8
TRAIN_WARMUP = 3
TRAIN_ITERS = 6
INFER_WARMUP = 3
INFER_ITERS = 20


class _UniformRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits, E_arg, K_arg):
        ctx.save_for_backward(_SCORES, _INDICES)
        ctx.E = E_arg
        ctx.dtype = router_logits.dtype
        return _SCORES.clone(), _INDICES.clone()

    @staticmethod
    def backward(ctx, grad_scores, _grad_indices):
        scores, _ = ctx.saved_tensors
        return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None


def _build_uniform(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    idx = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, idx


def _reset_state() -> None:
    clear_all_fp8_weight_caches()
    clear_blockscaled_fp8_weight_cache()
    _COMPILE_CACHE.clear()
    _COMPILE_CACHE_VK.clear()
    F_mod._ALIGNMENT_STREAK = 0


def _setup_mode(mode: str, *, fused_gated: bool = False, fp8_wgrad: bool = False) -> None:
    if mode == "bf16":
        os.environ["SONIC_MOE_FP8_MODE"] = "off"
        for key in [
            "SONIC_MOE_FP8_ASSUME_ALIGNED",
            "SONIC_MOE_FP8_FUSED_SWIGLU_QUANT",
            "SONIC_MOE_FP8_SAVE_Z_FP8",
            "SONIC_MOE_FP8_FUSED_GATED",
            "SONIC_MOE_FP8_WGRAD",
        ]:
            os.environ.pop(key, None)
        F_mod._ALIGNMENT_ASSUMED = False
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
        os.environ["SONIC_MOE_FP8_FUSED_GATED"] = "1" if fused_gated else "0"
        os.environ["SONIC_MOE_FP8_WGRAD"] = "1" if fp8_wgrad else "0"
        F_mod._ALIGNMENT_ASSUMED = True
    _reset_state()


def _measure_train(moe: MoE, x_base: torch.Tensor, grad_out: torch.Tensor, *, mode: str, fused_gated: bool, fp8_wgrad: bool) -> tuple[float, float]:
    _setup_mode(mode, fused_gated=fused_gated, fp8_wgrad=fp8_wgrad)
    moe.train()
    for _ in range(TRAIN_WARMUP):
        for p in moe.parameters():
            p.grad = None
        x = x_base.clone().requires_grad_(True)
        out, _ = moe(x)
        out.backward(grad_out)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TRAIN_ITERS):
        for p in moe.parameters():
            p.grad = None
        x = x_base.clone().requires_grad_(True)
        out, _ = moe(x)
        out.backward(grad_out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TRAIN_ITERS, torch.cuda.max_memory_allocated() / (1024 ** 3)


def _measure_infer(moe: MoE, x_base: torch.Tensor, *, mode: str, fused_gated: bool) -> tuple[float, float]:
    _setup_mode(mode, fused_gated=fused_gated, fp8_wgrad=False)
    moe.eval()
    for _ in range(INFER_WARMUP):
        with torch.inference_mode():
            moe(x_base, is_inference_mode=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(INFER_ITERS):
        with torch.inference_mode():
            moe(x_base, is_inference_mode=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / INFER_ITERS, torch.cuda.max_memory_allocated() / (1024 ** 3)


def main() -> None:
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
    grad_out = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    orig_router = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter
    try:
        train_bf16_ms, train_bf16_gb = _measure_train(moe, x_base, grad_out, mode="bf16", fused_gated=False, fp8_wgrad=False)
        train_fp8_ms, train_fp8_gb = _measure_train(moe, x_base, grad_out, mode="fp8", fused_gated=True, fp8_wgrad=False)
        train_fp8wg_ms, train_fp8wg_gb = _measure_train(moe, x_base, grad_out, mode="fp8", fused_gated=True, fp8_wgrad=True)
        infer_bf16_ms, infer_bf16_gb = _measure_infer(moe, x_base, mode="bf16", fused_gated=False)
        infer_fp8_ms, infer_fp8_gb = _measure_infer(moe, x_base, mode="fp8", fused_gated=True)
    finally:
        F_mod.TC_Softmax_Topk_Router_Function = orig_router

    print(f"TRAIN BF16 total:                {train_bf16_ms:.3f} ms, peak {train_bf16_gb:.3f} GiB")
    print(f"TRAIN FP8 + BF16 wgrad total:    {train_fp8_ms:.3f} ms, peak {train_fp8_gb:.3f} GiB")
    print(f"TRAIN FP8 + FP8  wgrad total:    {train_fp8wg_ms:.3f} ms, peak {train_fp8wg_gb:.3f} GiB")
    print(f"INFER BF16 total:                {infer_bf16_ms:.3f} ms, peak {infer_bf16_gb:.3f} GiB")
    print(f"INFER FP8 total:                 {infer_fp8_ms:.3f} ms, peak {infer_fp8_gb:.3f} GiB")


if __name__ == "__main__":
    _SCORES, _INDICES = _build_uniform(torch.device("cuda"))
    main()
