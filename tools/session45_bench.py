#!/usr/bin/env python3
"""Subprocess-isolated BF16/FP8 benchmark with CUDA events."""
import json
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r'''
import os, torch, gc, json
os.environ['USE_QUACK_GEMM'] = '1'
{extra_env}
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
device = 'cuda'
moe = MoE(
    num_experts=E, num_experts_per_tok=K, hidden_size=H,
    intermediate_size=I, activation_function=ActivationType.SWIGLU,
    add_bias=False, std=0.01,
).to(device).to(torch.bfloat16)
x = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.02

# Warmup (5 iters)
for _ in range(5):
    with enable_quack_gemm(True){fp8_ctx}:
        out, _ = moe(x{use_fp8})
        out.sum().backward()
        torch.cuda.synchronize()
del out; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# Measure (7 iters, drop min/max)
times = []
for _ in range(7):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with enable_quack_gemm(True){fp8_ctx}:
        out, _ = moe(x{use_fp8})
        out.sum().backward()
    e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
    del out; gc.collect(); torch.cuda.empty_cache()

peak = torch.cuda.max_memory_allocated() / 1024**2
times_trimmed = sorted(times)[1:-1]  # drop min/max
print(json.dumps(dict(
    times=times, mean=sum(times_trimmed)/len(times_trimmed),
    peak_mib=peak, trimmed_mean=sum(times_trimmed)/len(times_trimmed),
)))
''')


def run_bench(T, H, I, E, K, mode, gpu=7):
    if mode == 'bf16':
        code = SCRIPT.format(T=T, H=H, I=I, E=E, K=K,
                             extra_env='', fp8_ctx='', use_fp8='')
    else:
        code = SCRIPT.format(T=T, H=H, I=I, E=E, K=K,
                             extra_env="os.environ['SONIC_MOE_FP8_MODE'] = 'perf'",
                             fp8_ctx=', enable_fp8()', use_fp8=', use_fp8=True')
    env = {k: v for k, v in os.environ.items()}
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # Don't inherit FP8 mode from parent
    env.pop('SONIC_MOE_FP8_MODE', None)
    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if proc.returncode != 0:
        print(f"  FAILED (exit={proc.returncode})")
        stderr = proc.stderr[-1000:] if proc.stderr else ''
        print(stderr)
        return None
    return json.loads(proc.stdout.strip().split('\n')[-1])


if __name__ == '__main__':
    shapes = [
        ('Ernie', 8192, 3072, 1536, 8, 8),
        ('I2048', 8192, 3072, 2048, 8, 8),
    ]
    all_results = {}
    for name, T, H, I, E, K in shapes:
        print(f'\n=== {name} (T={T}, H={H}, I={I}, E={E}, K={K}) ===')
        for mode in ['bf16', 'fp8']:
            print(f'  [{mode.upper()}] running...', end=' ', flush=True)
            data = run_bench(T, H, I, E, K, mode)
            if data:
                print(f'{data["trimmed_mean"]:.2f}ms  peak={data["peak_mib"]:.0f}MiB  raw={[f"{t:.2f}" for t in data["times"]]}')
                all_results[f'{name}_{mode}'] = data
            else:
                print('FAILED')
        bf16 = all_results.get(f'{name}_bf16')
        fp8 = all_results.get(f'{name}_fp8')
        if bf16 and fp8:
            speedup = bf16['trimmed_mean'] / fp8['trimmed_mean']
            mem_delta = fp8['peak_mib'] - bf16['peak_mib']
            print(f'  => Speedup: {speedup:.3f}x  Memory: {mem_delta:+.0f} MiB ({mem_delta/bf16["peak_mib"]*100:+.1f}%)')

    with open('session45_benchmark.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to session45_benchmark.json')
