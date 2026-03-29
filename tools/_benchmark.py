"""E2E benchmark: FP8 vs BF16 latency comparison."""
import torch, os, time, statistics
os.environ["USE_QUACK_GEMM"] = "1"

for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
          "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache
from sonicmoe.functional import clear_all_fp8_weight_caches

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
dout = torch.randn_like(x)
enable_quack_gemm()

def benchmark(mode, warmup=3, iters=10):
    os.environ["SONIC_MOE_FP8_MODE"] = mode
    clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache(); _COMPILE_CACHE.clear()
    for _ in range(warmup):
        x_ = x.detach().clone().requires_grad_(True)
        out, _ = moe(x_)
        out.backward(dout)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        x_ = x.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        out, _ = moe(x_)
        out.backward(dout)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times

print("Benchmarking BF16 (fwd+bwd)...")
bf16_times = benchmark("off")
print("Benchmarking FP8 (fwd+bwd)...")
fp8_times = benchmark("perf")

bf16_med = statistics.median(bf16_times)
fp8_med = statistics.median(fp8_times)
bf16_p25 = sorted(bf16_times)[len(bf16_times)//4]
fp8_p25 = sorted(fp8_times)[len(fp8_times)//4]

print(f"\n=== E2E Latency (fwd+bwd, T={T}, H={H}, I={I}, E={E}, K={K}) ===")
print(f"BF16: median={bf16_med:.2f}ms, p25={bf16_p25:.2f}ms")
print(f"FP8:  median={fp8_med:.2f}ms, p25={fp8_p25:.2f}ms")
print(f"Speedup (median): {bf16_med/fp8_med:.2f}x")
print(f"Speedup (p25):    {bf16_p25/fp8_p25:.2f}x")
print(f"\nRaw BF16: {['%.2f' % t for t in bf16_times]}")
print(f"Raw FP8:  {['%.2f' % t for t in fp8_times]}")
