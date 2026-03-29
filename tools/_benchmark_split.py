"""Split benchmark: fwd-only and bwd-only latency."""
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

def bench_fwd(mode, warmup=3, iters=10):
    os.environ["SONIC_MOE_FP8_MODE"] = mode
    clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache(); _COMPILE_CACHE.clear()
    with torch.no_grad():
        for _ in range(warmup):
            moe(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            moe(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return times

def bench_bwd(mode, warmup=3, iters=10):
    os.environ["SONIC_MOE_FP8_MODE"] = mode
    clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache(); _COMPILE_CACHE.clear()
    # pre-warmup to compile
    for _ in range(warmup):
        x_ = x.detach().clone().requires_grad_(True)
        out, _ = moe(x_)
        out.backward(dout)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        # Forward (not timed)
        x_ = x.detach().clone().requires_grad_(True)
        out, _ = moe(x_)
        torch.cuda.synchronize()
        # Backward (timed)
        t0 = time.perf_counter()
        out.backward(dout)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times

print("=== Forward-only ===")
bf16_fwd = bench_fwd("off")
fp8_fwd = bench_fwd("perf")
print(f"BF16 fwd: {statistics.median(bf16_fwd):.2f}ms")
print(f"FP8  fwd: {statistics.median(fp8_fwd):.2f}ms")
print(f"FP8 fwd speedup: {statistics.median(bf16_fwd)/statistics.median(fp8_fwd):.2f}x")

print("\n=== Backward-only ===")
bf16_bwd = bench_bwd("off")
fp8_bwd = bench_bwd("perf")
print(f"BF16 bwd: {statistics.median(bf16_bwd):.2f}ms")
print(f"FP8  bwd: {statistics.median(fp8_bwd):.2f}ms")
print(f"FP8 bwd speedup: {statistics.median(bf16_bwd)/statistics.median(fp8_bwd):.2f}x")

print(f"\n=== Summary ===")
print(f"FP8 fwd saves {statistics.median(bf16_fwd)-statistics.median(fp8_fwd):.2f}ms vs BF16")
print(f"FP8 bwd saves {statistics.median(bf16_bwd)-statistics.median(fp8_bwd):.2f}ms vs BF16")
