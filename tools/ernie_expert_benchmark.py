"""Benchmark Ernie ErnieMoeMLP (PaddlePaddle) for FP8 expert-level comparison.

Runs ErnieMoeMLP in BF16 and FP8 modes on single GPU for direct comparison
with SonicMoE expert computation. DeepEPMOELayer requires EP>1 so cannot
be benchmarked on single GPU; ErnieMoeMLP is the expert-level equivalent.

Shapes match the Ernie-compatible parameter space from test configs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_expert_benchmark.py
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_expert_benchmark.py --shapes A B
    CUDA_VISIBLE_DEVICES=0 python tools/ernie_expert_benchmark.py --fp8  # FP8 only
"""
import argparse
import gc
import os
import sys
import time

os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
for _k in [
    "PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np

# Add ernie-core to path
ernie_src = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/baidu/ernie/ernie-core/src"
sys.path.insert(0, ernie_src)

# Ernie-compatible shapes: T (tokens dispatched to expert), H, I
# In full MoE: tokens_per_expert = T*K/E (uniform routing)
# For expert-level comparison: we run the expert MLP on T_expert tokens
SHAPES = {
    # tpe = T*K/E tokens per expert in full MoE context
    # Shape A: Full MoE T=1024,K=2,E=16 → tpe=128 per expert, 16 experts
    # Expert sees 128 tokens, total compute = 128*16 = 2048 token-expert pairs
    "A": dict(T_expert=128, H=1536, I=1536, E=16, desc="Ernie-distrib (per-expert)"),
    # Shape B: Full MoE T=4096,K=2,E=64 → tpe=128 per expert, 64 experts
    "B": dict(T_expert=128, H=4096, I=4096, E=64, desc="medium (per-expert)"),
    # Shape C: Full MoE T=4096,K=2,E=64 → tpe=128, with Ernie default I
    "C": dict(T_expert=128, H=4096, I=11008, E=64, desc="Ernie-default-I (per-expert)"),
    # Shape D: Full MoE T=8192,K=2,E=64 → tpe=256 per expert
    "D": dict(T_expert=256, H=4096, I=4096, E=64, desc="large-T (per-expert)"),
    # Aggregate shapes: total tokens = T_expert * E (what the full MoE processes)
    "A_agg": dict(T_expert=2048, H=1536, I=1536, E=1, desc="Ernie-distrib (aggregate)"),
    "B_agg": dict(T_expert=8192, H=4096, I=4096, E=1, desc="medium (aggregate)"),
    "C_agg": dict(T_expert=8192, H=4096, I=11008, E=1, desc="Ernie-default-I (aggregate)"),
    "D_agg": dict(T_expert=16384, H=4096, I=4096, E=1, desc="large-T (aggregate)"),
}

WARMUP = 5
BENCH_ITERS = 10


def create_ernie_config(H, I, fp8=False):
    """Create a minimal ErnieMoE config for expert MLP."""
    from ernie_core.models.ernie5_moe.configuration import ErniemmMoEConfig

    config = ErniemmMoEConfig(hidden_size=H)
    config.intermediate_size = I
    config.tensor_model_parallel_size = 1
    config.fuse_ffn = True
    config.fuse_attn_ffn = True
    config.hidden_act = "silu"
    config.use_bias = False
    if fp8:
        config.fp8 = "e4m3"
        config.fp8_wgrad = True
        config.use_fuse_node = True
    else:
        config.fp8 = None
    return config


def benchmark_ernie_expert(shape_name, shape_cfg, fp8=False):
    """Benchmark a single ErnieMoeMLP expert."""
    from ernie_core.models.ernie_moe.modeling import ErnieMoeMLP

    T_exp = shape_cfg["T_expert"]
    H = shape_cfg["H"]
    I = shape_cfg["I"]
    mode_str = "FP8" if fp8 else "BF16"

    print(f"\n  Ernie {mode_str}: T_expert={T_exp}, H={H}, I={I}")

    paddle.seed(42)
    paddle.set_default_dtype("bfloat16")

    config = create_ernie_config(H, I, fp8=fp8)
    expert = ErnieMoeMLP(config, is_shared_expert=False)

    x_base = paddle.randn([T_exp, H], dtype='bfloat16')

    # Warmup
    for _ in range(WARMUP):
        expert.clear_gradients()
        x = x_base.clone()
        x.stop_gradient = False
        out = expert(x)
        out.sum().backward()
    paddle.device.cuda.synchronize()

    # Benchmark
    start_event = paddle.device.cuda.Event(enable_timing=True)
    end_event = paddle.device.cuda.Event(enable_timing=True)

    paddle.device.cuda.synchronize()
    # Get peak memory before benchmark
    # PaddlePaddle doesn't have reset_peak_memory_stats, use manual tracking
    start_event.record()
    for _ in range(BENCH_ITERS):
        expert.clear_gradients()
        x = x_base.clone()
        x.stop_gradient = False
        out = expert(x)
        out.sum().backward()
    end_event.record()
    paddle.device.cuda.synchronize()

    ms = start_event.elapsed_time(end_event) / BENCH_ITERS

    # Memory: use paddle's memory stats
    try:
        peak_gb = paddle.device.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        peak_gb = float("nan")

    print(f"    {ms:.3f} ms/iter, peak ~{peak_gb:.2f} GiB")

    del expert, x_base
    paddle.device.cuda.empty_cache()
    gc.collect()

    return {"ms": ms, "peak_gb": peak_gb}


def benchmark_ernie_moe_aggregate(shape_name, shape_cfg, fp8=False):
    """Benchmark aggregate expert computation (T_expert * E tokens through 1 expert).

    This simulates the total compute of the MoE layer by running all tokens
    through a single expert, matching the total FLOP count.
    """
    from ernie_core.models.ernie_moe.modeling import ErnieMoeMLP

    T_total = shape_cfg["T_expert"]  # Already aggregate for _agg shapes
    H = shape_cfg["H"]
    I = shape_cfg["I"]
    mode_str = "FP8" if fp8 else "BF16"

    print(f"\n  Ernie {mode_str} aggregate: T_total={T_total}, H={H}, I={I}")

    paddle.seed(42)
    paddle.set_default_dtype("bfloat16")

    config = create_ernie_config(H, I, fp8=fp8)
    expert = ErnieMoeMLP(config, is_shared_expert=False)

    x_base = paddle.randn([T_total, H], dtype='bfloat16')

    # Warmup
    for _ in range(WARMUP):
        expert.clear_gradients()
        x = x_base.clone()
        x.stop_gradient = False
        out = expert(x)
        out.sum().backward()
    paddle.device.cuda.synchronize()

    # Benchmark
    start_event = paddle.device.cuda.Event(enable_timing=True)
    end_event = paddle.device.cuda.Event(enable_timing=True)

    paddle.device.cuda.synchronize()
    start_event.record()
    for _ in range(BENCH_ITERS):
        expert.clear_gradients()
        x = x_base.clone()
        x.stop_gradient = False
        out = expert(x)
        out.sum().backward()
    end_event.record()
    paddle.device.cuda.synchronize()

    ms = start_event.elapsed_time(end_event) / BENCH_ITERS

    try:
        peak_gb = paddle.device.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        peak_gb = float("nan")

    print(f"    {ms:.3f} ms/iter, peak ~{peak_gb:.2f} GiB")

    del expert, x_base
    paddle.device.cuda.empty_cache()
    gc.collect()

    return {"ms": ms, "peak_gb": peak_gb}


def main():
    parser = argparse.ArgumentParser(description="Ernie ErnieMoeMLP expert benchmark")
    parser.add_argument("--shapes", nargs="+",
                        default=["A_agg", "B_agg", "C_agg", "D_agg"],
                        help="Shapes to benchmark")
    parser.add_argument("--fp8", action="store_true", help="FP8 only")
    parser.add_argument("--bf16", action="store_true", help="BF16 only")
    args = parser.parse_args()

    modes = []
    if not args.fp8:
        modes.append(False)  # BF16
    if not args.bf16:
        modes.append(True)   # FP8

    print("=" * 70)
    print("Ernie ErnieMoeMLP Expert Benchmark")
    print(f"Shapes: {args.shapes}")
    print(f"Modes: {['FP8' if m else 'BF16' for m in modes]}")
    print("=" * 70)

    results = {}
    for shape_name in args.shapes:
        shape_cfg = SHAPES[shape_name]
        print(f"\n{'='*50}")
        print(f"Shape {shape_name}: {shape_cfg['desc']}")
        print(f"{'='*50}")

        results[shape_name] = {}
        for fp8 in modes:
            mode_key = "fp8" if fp8 else "bf16"
            try:
                if "_agg" in shape_name:
                    r = benchmark_ernie_moe_aggregate(shape_name, shape_cfg, fp8=fp8)
                else:
                    r = benchmark_ernie_expert(shape_name, shape_cfg, fp8=fp8)
                results[shape_name][mode_key] = r
            except Exception as e:
                print(f"    ERROR: {e}")
                results[shape_name][mode_key] = {"ms": float("nan"), "peak_gb": float("nan"), "error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("ERNIE EXPERT BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Shape':<10} {'Desc':<28} {'BF16 (ms)':>10} {'FP8 (ms)':>10} {'Speedup':>8}")
    print("-" * 70)
    for sn in args.shapes:
        cfg = SHAPES[sn]
        bf16_ms = results[sn].get("bf16", {}).get("ms", float("nan"))
        fp8_ms = results[sn].get("fp8", {}).get("ms", float("nan"))
        speedup = bf16_ms / fp8_ms if fp8_ms > 0 and not np.isnan(fp8_ms) else float("nan")
        print(f"{sn:<10} {cfg['desc']:<28} {bf16_ms:10.3f} {fp8_ms:10.3f} {speedup:7.2f}x")


if __name__ == "__main__":
    main()
