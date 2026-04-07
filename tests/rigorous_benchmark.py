"""Rigorous benchmark with GPU isolation verification + theoretical analysis."""
import os, sys, torch, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192
TK = T * K  # 65536

def check_gpu_idle():
    """Verify GPU utilization is near 0 before benchmarking."""
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}",
         "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    line = result.stdout.strip().split("\n")[0]
    parts = [p.strip() for p in line.split(",")]
    util, mem_used, mem_total = int(parts[0]), int(parts[1]), int(parts[2])
    print(f"GPU {gpu_id}: util={util}%, mem={mem_used}/{mem_total} MiB")
    if util > 5:
        print(f"WARNING: GPU utilization {util}% > 5% — results may be noisy!")
        return False
    return True

def compute_theoretical():
    """Compute theoretical FLOP counts and expected times."""
    # B200 specs: FP8 ~4500 TFLOPS, BF16 ~2250 TFLOPS (tensor core)
    # Actual sustained is ~70-80% of peak
    FP8_TFLOPS = 4500 * 0.75  # sustained
    BF16_TFLOPS = 2250 * 0.75
    HBM_BW = 8000  # GB/s (B200 HBM3e)

    print("\n=== Theoretical Analysis ===")

    # wgrad up-proj: gemm(x.T, dz) — (H, T) × (T, 2I) → (H, 2I) per expert, varlen
    # Total FLOPs = 2 * TK * H * 2I (GEMM FLOP = 2*M*N*K)
    flops_wgrad_up = 2 * TK * H * 2 * I
    t_wgrad_up_bf16 = flops_wgrad_up / (BF16_TFLOPS * 1e12) * 1e6  # µs
    t_wgrad_up_fp8 = flops_wgrad_up / (FP8_TFLOPS * 1e12) * 1e6
    print(f"wgrad up-proj: {flops_wgrad_up/1e12:.2f} TFLOP")
    print(f"  BF16 theoretical: {t_wgrad_up_bf16:.0f} µs")
    print(f"  FP8 theoretical:  {t_wgrad_up_fp8:.0f} µs")

    # wgrad down-proj: gemm(dout.T, y1) — (H, TK) × (TK, I) → (H, I) per expert
    flops_wgrad_down = 2 * TK * H * I
    t_wgrad_down_bf16 = flops_wgrad_down / (BF16_TFLOPS * 1e12) * 1e6
    t_wgrad_down_fp8 = flops_wgrad_down / (FP8_TFLOPS * 1e12) * 1e6
    print(f"\nwgrad down-proj: {flops_wgrad_down/1e12:.2f} TFLOP")
    print(f"  BF16 theoretical: {t_wgrad_down_bf16:.0f} µs")
    print(f"  FP8 theoretical:  {t_wgrad_down_fp8:.0f} µs")

    # GemmDGated: dout × w2^T — (TK, H) × (H, I) → (TK, I)
    flops_dgated = 2 * TK * H * I
    t_dgated_fp8 = flops_dgated / (FP8_TFLOPS * 1e12) * 1e6
    print(f"\nGemmDGated: {flops_dgated/1e12:.2f} TFLOP")
    print(f"  FP8 theoretical: {t_dgated_fp8:.0f} µs")

    # actgrad: dz × w1^T — (TK, 2I) × (2I, H) → (TK, H)
    flops_actgrad = 2 * TK * 2 * I * H
    t_actgrad_fp8 = flops_actgrad / (FP8_TFLOPS * 1e12) * 1e6
    print(f"\nactgrad: {flops_actgrad/1e12:.2f} TFLOP")
    print(f"  FP8 theoretical: {t_actgrad_fp8:.0f} µs")

    # z dequant: (TK, 2I) fp8 read + bf16 write
    bytes_dequant = TK * 2 * I * (1 + 2)  # read fp8 + write bf16
    t_dequant = bytes_dequant / (HBM_BW * 1e9) * 1e6
    print(f"\nz dequant: {bytes_dequant/1e6:.0f} MB HBM traffic")
    print(f"  Bandwidth-limited: {t_dequant:.0f} µs")

    return {
        "wgrad_up_bf16": t_wgrad_up_bf16,
        "wgrad_up_fp8": t_wgrad_up_fp8,
        "wgrad_down_bf16": t_wgrad_down_bf16,
        "wgrad_down_fp8": t_wgrad_down_fp8,
    }

def benchmark_backward(warmup=20, iters=30, trials=5):
    """Benchmark with statistical rigor."""
    torch.manual_seed(42)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)

    def run():
        x.grad = None; moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True):
            out, _ = moe(x)
        out.backward(dout)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Memory
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    # Timing with multiple trials
    times = []
    for trial in range(trials):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            run()
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e) * 1000 / iters  # µs per iter
        times.append(t)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    cv = std / avg * 100  # coefficient of variation

    print(f"\n=== Benchmark Results (warmup={warmup}, iters={iters}, trials={trials}) ===")
    print(f"  Peak memory: {peak / 1024**2:.1f} MiB")
    print(f"  Latency: avg={avg:.0f}µs, min={mn:.0f}µs, max={mx:.0f}µs")
    print(f"  StdDev: {std:.0f}µs, CV: {cv:.1f}%")
    if cv > 5:
        print(f"  WARNING: CV={cv:.1f}% > 5% — high variance, consider re-running")
    print(f"  All trials: {[f'{t:.0f}' for t in times]}")
    return avg, mn, peak

def main():
    print("=" * 70)
    print("Rigorous FP8 Backward Benchmark")
    print(f"Shape: T={T}, E={E}, K={K}, H={H}, I={I}, TK={TK}")
    print("=" * 70)

    idle = check_gpu_idle()
    theory = compute_theoretical()
    avg, mn, peak = benchmark_backward()

    # Compare measured vs theoretical
    print("\n=== Measured vs Theoretical ===")
    # From nsys: wgrad_up=3490µs, wgrad_down=387µs
    print(f"wgrad up-proj:   measured ~3490µs, BF16 theory {theory['wgrad_up_bf16']:.0f}µs, "
          f"efficiency={theory['wgrad_up_bf16']/3490*100:.0f}%")
    print(f"wgrad down-proj: measured ~387µs,  BF16 theory {theory['wgrad_down_bf16']:.0f}µs, "
          f"efficiency={theory['wgrad_down_bf16']/387*100:.0f}%")
    print(f"\nIf wgrad → FP8:")
    print(f"  wgrad up-proj:   FP8 theory {theory['wgrad_up_fp8']:.0f}µs (vs measured 3490µs)")
    print(f"  wgrad down-proj: FP8 theory {theory['wgrad_down_fp8']:.0f}µs (vs measured 387µs)")

if __name__ == "__main__":
    main()
