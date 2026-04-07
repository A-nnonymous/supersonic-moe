"""Phase 3.1 End-to-End Validation: precision + memory + performance.

Compares FP8 Frontier (baseline with z dequant) vs FP8 Frontier + Phase 3.1
(fp8 PreAct, no z dequant, no 384MB temp buffer).
"""
import os, sys, torch, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

E, K, H, I = 8, 8, 3072, 1536
T = 8192
SEED = 42
WARMUP, ITERS, TRIALS = 20, 30, 5


def rrmse(a, b):
    return ((a.float()-b.float()).norm() / b.float().norm().clamp(min=1e-8)).item()

def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_mode(phase31: bool):
    """Run full fwd+bwd. Returns (out, dx, dw1, dw2, peak_mem, latency_min)."""
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"

    from sonicmoe.quack_utils.gemm_gated import gemm_gated as _gg
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as _gd
    _gg.compile_cache.clear()
    _gd.compile_cache.clear()

    # TODO: Phase 3.1 is always active now (use_fp8_preact auto-detects z_fp8).
    # To compare, we'd need a way to disable it. For now, just run once.

    torch.manual_seed(SEED)
    moe = MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
    torch.manual_seed(SEED + 1)
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    def run():
        x.grad = None; moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True):
            out, _ = moe(x)
        out.backward(dout)
        return out

    # Warmup
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    # Memory
    torch.cuda.reset_peak_memory_stats()
    out = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    # Timing
    times = []
    for _ in range(TRIALS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)

    dx = x.grad.clone()
    dw1 = moe.c_fc.weight.grad.clone()
    dw2 = moe.c_proj.weight.grad.clone()
    return out.detach(), dx, dw1, dw2, peak / 1024**2, min(times)


def main():
    # Check GPU idle
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    r = subprocess.run(["nvidia-smi", f"--id={gpu_id}", "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    util = int(r.stdout.strip().split("\n")[0].strip())
    print(f"GPU {gpu_id}: util={util}%")
    if util > 5:
        print("WARNING: GPU busy!")

    print("=" * 70)
    print(f"Phase 3.1 End-to-End Report (T={T}, E={E}, K={K}, H={H}, I={I})")
    print("=" * 70)

    print("\nRunning FP8 Frontier + Phase 3.1 (fp8 PreAct)...")
    out, dx, dw1, dw2, peak, latency = run_mode(phase31=True)
    print(f"  Peak memory: {peak:.1f} MiB")
    print(f"  Latency min: {latency:.0f} µs")
    print(f"  out norm: {out.norm().item():.4f}")
    print(f"  dx norm:  {dx.norm().item():.4f}")
    print(f"  dw1 norm: {dw1.norm().item():.4f}")
    print(f"  dw2 norm: {dw2.norm().item():.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
