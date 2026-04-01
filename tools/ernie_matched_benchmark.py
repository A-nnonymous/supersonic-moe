"""Comprehensive benchmark: SonicMoE BF16/FP8 at Ernie-compatible shapes.

Shapes derived from Ernie DeepEPMOELayer test configs:
  - Shape A: T=1024, H=1536, I=1536, E=16, K=2  (Ernie distributed test)
  - Shape B: T=4096, H=4096, I=4096, E=64, K=2   (medium production)
  - Shape C: T=4096, H=4096, I=11008, E=64, K=2  (Ernie default I)
  - Shape D: T=8192, H=4096, I=4096, E=64, K=2   (large token count)

All shapes enforce 128-alignment: (T*K) % E == 0 AND (T*K/E) % 128 == 0.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_matched_benchmark.py
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_matched_benchmark.py --shapes A B
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_matched_benchmark.py --nsys  # for nsys profiling mode
"""
import argparse
import gc
import os
import sys
import time

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

SHAPES = {
    "A": dict(T=1024, H=1536, I=1536, E=16, K=2, desc="Ernie-distrib"),
    "B": dict(T=4096, H=4096, I=4096, E=64, K=2, desc="medium"),
    "C": dict(T=4096, H=4096, I=11008, E=64, K=2, desc="Ernie-default-I"),
    "D": dict(T=8192, H=4096, I=4096, E=64, K=2, desc="large-T"),
}

WARMUP = 5
BENCH_ITERS = 10
PROFILE_ITERS = 3


def verify_alignment(T, K, E):
    TK = T * K
    tpe = TK // E
    assert TK % E == 0, f"TK={TK} not divisible by E={E}"
    assert tpe % 128 == 0, f"tpe={tpe} not 128-aligned (T={T}, K={K}, E={E})"
    return tpe


def build_uniform_routing(T, K, E, device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def setup_mode(mode: str):
    import sonicmoe.functional as F_mod
    from sonicmoe.functional import clear_all_fp8_weight_caches
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _COMPILE_CACHE, _COMPILE_CACHE_VK, clear_blockscaled_fp8_weight_cache,
    )

    clear_all_fp8_weight_caches()
    clear_blockscaled_fp8_weight_cache()
    _COMPILE_CACHE.clear()
    _COMPILE_CACHE_VK.clear()
    F_mod._ALIGNMENT_STREAK = 0

    if mode == "bf16":
        os.environ["SONIC_MOE_FP8_MODE"] = "off"
        for key in [
            "SONIC_MOE_FP8_ASSUME_ALIGNED", "SONIC_MOE_FP8_FUSED_SWIGLU_QUANT",
            "SONIC_MOE_FP8_SAVE_Z_FP8", "SONIC_MOE_FP8_FUSED_GATED", "SONIC_MOE_FP8_WGRAD",
        ]:
            os.environ.pop(key, None)
        F_mod._ALIGNMENT_ASSUMED = False
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
        os.environ["SONIC_MOE_FP8_FUSED_GATED"] = "1"
        os.environ["SONIC_MOE_FP8_WGRAD"] = "0"
        F_mod._ALIGNMENT_ASSUMED = True


def measure_shape(shape_name, shape_cfg, modes, nsys_mode=False):
    """Measure wall-clock, memory, and optionally NSYS-profile a shape."""
    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F_mod

    T, H, I, E, K = shape_cfg["T"], shape_cfg["H"], shape_cfg["I"], shape_cfg["E"], shape_cfg["K"]
    tpe = verify_alignment(T, K, E)
    print(f"\n{'='*70}")
    print(f"Shape {shape_name}: T={T}, H={H}, I={I}, E={E}, K={K} (tpe={tpe}) [{shape_cfg['desc']}]")
    print(f"{'='*70}")

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    _scores, _indices = build_uniform_routing(T, K, E, torch.device("cuda"))

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

    results = {}

    for mode in modes:
        setup_mode(mode)
        moe.train()
        label = f"{shape_name}_{mode}"

        # Warmup
        for _ in range(WARMUP):
            for p in moe.parameters():
                p.grad = None
            x = x_base.clone().requires_grad_(True)
            out, _ = moe(x)
            out.backward(grad_out)
        torch.cuda.synchronize()

        if nsys_mode:
            # NSYS profiling mode: NVTX markers, no timing
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(PROFILE_ITERS):
                nvtx.range_push(f"{label}_iter_{i}")
                for p in moe.parameters():
                    p.grad = None
                x = x_base.clone().requires_grad_(True)
                torch.cuda.synchronize()
                nvtx.range_push("forward")
                out, _ = moe(x)
                torch.cuda.synchronize()
                nvtx.range_pop()
                nvtx.range_push("backward")
                out.backward(grad_out)
                torch.cuda.synchronize()
                nvtx.range_pop()
                nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"  {mode}: NSYS profiled {PROFILE_ITERS} iters")
        else:
            # Wall-clock + memory mode
            torch.cuda.reset_peak_memory_stats()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(BENCH_ITERS):
                for p in moe.parameters():
                    p.grad = None
                x = x_base.clone().requires_grad_(True)
                out, _ = moe(x)
                out.backward(grad_out)
            end_evt.record()
            torch.cuda.synchronize()

            ms = start_evt.elapsed_time(end_evt) / BENCH_ITERS
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            results[mode] = {"ms": ms, "peak_gb": peak_gb}
            print(f"  {mode:6s}: {ms:8.2f} ms/iter, peak {peak_gb:.2f} GiB")

        # Reset for next mode
        torch.cuda.empty_cache()
        gc.collect()

    F_mod.TC_Softmax_Topk_Router_Function = orig_router

    # Cleanup model
    del moe, x_base, grad_out, _scores, _indices
    torch.cuda.empty_cache()
    gc.collect()

    return results


def precision_check(shape_name, shape_cfg, seeds=(42, 123, 777)):
    """Quick precision check: BF16 vs FP8 RelRMSE and correlation."""
    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F_mod

    T, H, I, E, K = shape_cfg["T"], shape_cfg["H"], shape_cfg["I"], shape_cfg["E"], shape_cfg["K"]

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        moe = MoE(
            num_experts=E, num_experts_per_tok=K, hidden_size=H,
            intermediate_size=I, activation_function=ActivationType.SWIGLU,
            add_bias=False, std=0.02,
        ).to("cuda", torch.bfloat16)
        enable_quack_gemm()

        x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

        _scores, _indices = build_uniform_routing(T, K, E, torch.device("cuda"))

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

        # Run BF16
        setup_mode("bf16")
        moe.train()
        for p in moe.parameters():
            p.grad = None
        x_bf16 = x_base.clone().requires_grad_(True)
        out_bf16, _ = moe(x_bf16)
        out_bf16.backward(grad_out)
        torch.cuda.synchronize()
        fwd_bf16 = out_bf16.detach().float()
        grad_bf16 = x_bf16.grad.detach().float()

        # Run FP8
        setup_mode("fp8")
        F_mod._ALIGNMENT_ASSUMED = True
        for p in moe.parameters():
            p.grad = None
        x_fp8 = x_base.clone().requires_grad_(True)
        out_fp8, _ = moe(x_fp8)
        out_fp8.backward(grad_out)
        torch.cuda.synchronize()
        fwd_fp8 = out_fp8.detach().float()
        grad_fp8 = x_fp8.grad.detach().float()

        F_mod.TC_Softmax_Topk_Router_Function = orig_router

        # RelRMSE and correlation
        def relrmse(a, b):
            return (a - b).norm() / b.norm() * 100  # percent

        def corr(a, b):
            a_flat, b_flat = a.flatten(), b.flatten()
            a_flat = a_flat - a_flat.mean()
            b_flat = b_flat - b_flat.mean()
            return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())

        fwd_rrmse = relrmse(fwd_fp8, fwd_bf16).item()
        grad_rrmse = relrmse(grad_fp8, grad_bf16).item()
        fwd_corr = corr(fwd_fp8, fwd_bf16).item()
        grad_corr = corr(grad_fp8, grad_bf16).item()

        results.append({
            "seed": seed,
            "fwd_relrmse": fwd_rrmse,
            "grad_relrmse": grad_rrmse,
            "fwd_corr": fwd_corr,
            "grad_corr": grad_corr,
        })

        del moe, x_base, grad_out, _scores, _indices
        torch.cuda.empty_cache()
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Ernie-matched multi-shape benchmark")
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()),
                        choices=list(SHAPES.keys()), help="Shapes to benchmark")
    parser.add_argument("--nsys", action="store_true", help="NSYS profiling mode (NVTX only)")
    parser.add_argument("--precision", action="store_true", help="Run precision checks")
    parser.add_argument("--modes", nargs="+", default=["bf16", "fp8"],
                        help="Modes to test (bf16, fp8)")
    args = parser.parse_args()

    print("=" * 70)
    print("SonicMoE Ernie-Matched Benchmark")
    print(f"Shapes: {args.shapes}, Modes: {args.modes}")
    print(f"NSYS={args.nsys}, Precision={args.precision}")
    print("=" * 70)

    all_results = {}

    for shape_name in args.shapes:
        shape_cfg = SHAPES[shape_name]
        perf_results = measure_shape(shape_name, shape_cfg, args.modes, nsys_mode=args.nsys)
        all_results[shape_name] = {"perf": perf_results, "shape": shape_cfg}

        if args.precision and not args.nsys:
            prec_results = precision_check(shape_name, shape_cfg)
            all_results[shape_name]["precision"] = prec_results

    if not args.nsys:
        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Shape':<8} {'Desc':<16} {'BF16 (ms)':>10} {'FP8 (ms)':>10} {'Speedup':>8} {'BF16 Mem':>10} {'FP8 Mem':>10}")
        print("-" * 80)
        for sn in args.shapes:
            res = all_results[sn]
            cfg = res["shape"]
            perf = res["perf"]
            bf16_ms = perf.get("bf16", {}).get("ms", float("nan"))
            fp8_ms = perf.get("fp8", {}).get("ms", float("nan"))
            bf16_gb = perf.get("bf16", {}).get("peak_gb", float("nan"))
            fp8_gb = perf.get("fp8", {}).get("peak_gb", float("nan"))
            speedup = bf16_ms / fp8_ms if fp8_ms > 0 else float("nan")
            print(f"{sn:<8} {cfg['desc']:<16} {bf16_ms:10.2f} {fp8_ms:10.2f} {speedup:7.2f}x {bf16_gb:9.2f}G {fp8_gb:9.2f}G")

        if args.precision:
            print(f"\n{'='*70}")
            print("PRECISION")
            print(f"{'='*70}")
            print(f"{'Shape':<8} {'Seed':>6} {'Fwd RelRMSE':>12} {'Grad RelRMSE':>13} {'Fwd Corr':>10} {'Grad Corr':>10} {'Pass':>6}")
            print("-" * 70)
            for sn in args.shapes:
                for p in all_results[sn].get("precision", []):
                    fwd_pass = p["fwd_relrmse"] < 10 and p["fwd_corr"] > 0.99
                    grad_pass = p["grad_relrmse"] < 10 and p["grad_corr"] > 0.99
                    status = "✓" if (fwd_pass and grad_pass) else "✗"
                    print(f"{sn:<8} {p['seed']:>6} {p['fwd_relrmse']:11.4f}% {p['grad_relrmse']:12.4f}% {p['fwd_corr']:10.6f} {p['grad_corr']:10.6f} {status:>6}")


if __name__ == "__main__":
    main()
