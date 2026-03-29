"""Benchmark FP8 vs BF16 at multiple production shapes."""
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
enable_quack_gemm()

shapes = [
    # (T, H, I, E, K) — progressively larger compute
    (4096, 4096, 1024, 128, 8),   # Current test shape
    (8192, 4096, 1024, 128, 8),   # 2x tokens
    (16384, 4096, 1024, 128, 8),  # 4x tokens
    (32768, 4096, 1024, 128, 8),  # 8x tokens (production)
    (4096, 4096, 2048, 128, 8),   # 2x intermediate
    (32768, 2880, 2880, 64, 8),   # Larger I, fewer E
]

for T, H, I, E, K in shapes:
    torch.manual_seed(42)
    try:
        moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
            activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

        for mode, label in [("off", "BF16"), ("perf", "FP8")]:
            os.environ["SONIC_MOE_FP8_MODE"] = mode
            clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache(); _COMPILE_CACHE.clear()
            # warmup
            with torch.no_grad():
                for _ in range(3): moe(x)
                torch.cuda.synchronize()
                times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    moe(x)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
            med = statistics.median(times)
            if label == "BF16":
                bf16_med = med
            else:
                speedup = bf16_med / med
                print(f"T={T:5d} H={H} I={I:4d} E={E:3d} K={K} | BF16={bf16_med:.2f}ms FP8={med:.2f}ms speedup={speedup:.2f}x")
        del moe, x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"T={T:5d} H={H} I={I:4d} E={E:3d} K={K} | ERROR: {e}")
