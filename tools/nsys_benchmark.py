#!/usr/bin/env python3
"""nsys-compatible benchmark: BF16 vs FP8 with NVTX markers.

Usage (run under nsys profile):
  CUDA_VISIBLE_DEVICES=X nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    -o /tmp/sonic_{mode} python tools/nsys_benchmark.py --mode {bf16|fp8} --gpu X

Or run directly for timing (CUDA events):
  CUDA_VISIBLE_DEVICES=X python tools/nsys_benchmark.py --mode fp8 --gpu X --no-nsys
"""
import argparse
import gc
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "fp8"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-nsys", action="store_true", help="Skip cudaProfiler markers")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["USE_QUACK_GEMM"] = "1"
    if args.mode == "fp8":
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"

    torch.manual_seed(42)
    device = "cuda"
    MiB = 1024**2

    T, H, I, E, K = 8192, 3072, 1536, 8, 8

    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_quack_gemm

    moe = MoE(
        num_experts=E, num_experts_per_tok=K,
        hidden_size=H, intermediate_size=I,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to(device=device, dtype=torch.bfloat16)

    x_base = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)

    # Warmup (JIT compile)
    for i in range(args.warmup):
        x = x_base.detach().clone().requires_grad_()
        with enable_quack_gemm(True):
            o = moe(x)[0]
        o.backward(dout)
        del o, x
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / MiB

    # Profiled iterations
    if not args.no_nsys:
        torch.cuda.cudart().cudaProfilerStart()

    fwd_times = []
    bwd_times = []
    fwd_peaks = []
    bwd_peaks = []

    for i in range(args.iters):
        x = x_base.detach().clone().requires_grad_()
        for p in moe.parameters():
            if p.grad is not None:
                p.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Forward
        if not args.no_nsys:
            torch.cuda.nvtx.range_push(f"forward_{i}")
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()

        with enable_quack_gemm(True):
            o = moe(x)[0]

        fwd_end.record()
        if not args.no_nsys:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        fwd_peak = torch.cuda.max_memory_allocated() / MiB
        fwd_times.append(fwd_start.elapsed_time(fwd_end))
        fwd_peaks.append(fwd_peak)

        torch.cuda.reset_peak_memory_stats()

        # Backward
        if not args.no_nsys:
            torch.cuda.nvtx.range_push(f"backward_{i}")
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start.record()

        o.backward(dout)

        bwd_end.record()
        if not args.no_nsys:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        bwd_peak = torch.cuda.max_memory_allocated() / MiB
        bwd_times.append(bwd_start.elapsed_time(bwd_end))
        bwd_peaks.append(bwd_peak)

        del o, x

    if not args.no_nsys:
        torch.cuda.cudart().cudaProfilerStop()

    # Cleanup and measure residual
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    residual = torch.cuda.memory_allocated() / MiB

    # Report
    import statistics
    fwd_avg = statistics.mean(fwd_times)
    bwd_avg = statistics.mean(bwd_times)
    total_avg = fwd_avg + bwd_avg
    fwd_peak_avg = statistics.mean(fwd_peaks)
    bwd_peak_avg = statistics.mean(bwd_peaks)

    print(f"=== {args.mode.upper()} Benchmark Results (GPU {args.gpu}, {args.iters} iters) ===")
    print(f"Forward:  {fwd_avg:.1f} ms (± {statistics.stdev(fwd_times):.2f})")
    print(f"Backward: {bwd_avg:.1f} ms (± {statistics.stdev(bwd_times):.2f})")
    print(f"Total:    {total_avg:.1f} ms")
    print(f"Fwd peak: {fwd_peak_avg:.1f} MiB")
    print(f"Bwd peak: {bwd_peak_avg:.1f} MiB")
    print(f"Base mem: {base_mem:.1f} MiB")
    print(f"Residual: {residual:.1f} MiB")

    # Machine-readable output
    print(f"\n[DATA] mode={args.mode} fwd_ms={fwd_avg:.3f} bwd_ms={bwd_avg:.3f} total_ms={total_avg:.3f} fwd_peak_mib={fwd_peak_avg:.1f} bwd_peak_mib={bwd_peak_avg:.1f} base_mib={base_mem:.1f} residual_mib={residual:.1f}")


if __name__ == "__main__":
    main()
