"""Benchmark: epilogue quant ON vs OFF latency + memory."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192
WARMUP, ITERS = 10, 20


def bench(mode):
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    if mode == "epilogue":
        os.environ["SONIC_MOE_FP8_EPILOGUE_QUANT"] = "1"
    else:
        os.environ.pop("SONIC_MOE_FP8_EPILOGUE_QUANT", None)

    from sonicmoe.quack_utils.gemm_gated import gemm_gated as _gg
    _gg.compile_cache.clear()

    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)

    def run():
        x.grad = None
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True):
            out, _ = moe(x)
        out.backward(dout)

    # Warmup
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    # Memory
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    # Timing
    times = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)  # µs per iter

    avg = sum(times) / len(times)
    mn = min(times)
    return peak / 1024**2, avg, mn


def main():
    print("=" * 60)
    print("Epilogue Quant Benchmark")
    print(f"Shape: T={T}, E={E}, K={K}, H={H}, I={I}")
    print("=" * 60)

    print("\n[1/2] Standalone quant (OFF)...")
    peak_off, avg_off, min_off = bench("standalone")
    print(f"  peak={peak_off:.1f}MiB avg={avg_off:.0f}µs min={min_off:.0f}µs")

    print("\n[2/2] Epilogue quant (ON)...")
    peak_on, avg_on, min_on = bench("epilogue")
    print(f"  peak={peak_on:.1f}MiB avg={avg_on:.0f}µs min={min_on:.0f}µs")

    print("\n--- Comparison ---")
    print(f"Memory: {peak_off:.1f} → {peak_on:.1f} MiB (delta={peak_on-peak_off:+.1f})")
    print(f"Latency avg: {avg_off:.0f} → {avg_on:.0f} µs (delta={avg_on-avg_off:+.0f})")
    print(f"Latency min: {min_off:.0f} → {min_on:.0f} µs (delta={min_on-min_off:+.0f})")


if __name__ == "__main__":
    main()
