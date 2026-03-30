"""Full MoE fwd+bwd comparison: aligned vs non-aligned routing."""
import torch, os, time, sys

os.environ["USE_QUACK_GEMM"] = "1"
for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
          "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches
import sonicmoe.functional as F_mod
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache,
)

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
enable_quack_gemm()

x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)


def _reset_fp8_state():
    """Reset all FP8 caches and alignment tracking."""
    clear_all_fp8_weight_caches()
    clear_blockscaled_fp8_weight_cache()
    _COMPILE_CACHE.clear()
    # Reset alignment streak tracking
    F_mod._ALIGNMENT_STREAK = 0


def run(mode, assume_aligned=False, warmup=8, iters=20):
    if mode == "fp8":
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        if assume_aligned:
            os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
            F_mod._ALIGNMENT_ASSUMED = True
        else:
            os.environ.pop("SONIC_MOE_FP8_ASSUME_ALIGNED", None)
            F_mod._ALIGNMENT_ASSUMED = False
    else:
        os.environ["SONIC_MOE_FP8_MODE"] = "off"
        os.environ.pop("SONIC_MOE_FP8_ASSUME_ALIGNED", None)
        F_mod._ALIGNMENT_ASSUMED = False
    _reset_fp8_state()

    for _ in range(warmup):
        x = x_base.clone().requires_grad_(True)
        out, _ = moe(x)
        out.sum().backward()
    torch.cuda.synchronize()

    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Forward only
    ev_s.record()
    for _ in range(iters):
        with torch.no_grad():
            out, _ = moe(x_base)
    ev_e.record()
    torch.cuda.synchronize()
    fwd_ms = ev_s.elapsed_time(ev_e) / iters

    # Fwd+Bwd
    ev_s.record()
    for _ in range(iters):
        x = x_base.clone().requires_grad_(True)
        out, _ = moe(x)
        out.sum().backward()
    ev_e.record()
    torch.cuda.synchronize()
    total_ms = ev_s.elapsed_time(ev_e) / iters
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}")
print(f"TK={T*K}, tpe_avg={T*K//E}")
print()

# 1. BF16 baseline
bf16_fwd, bf16_bwd, bf16_total = run("bf16")
print(f"BF16 baseline:       fwd={bf16_fwd:.3f}ms  bwd={bf16_bwd:.3f}ms  total={bf16_total:.3f}ms")

# 2. FP8 with non-aligned routing (BF16 fallback)
fp8_nat_fwd, fp8_nat_bwd, fp8_nat_total = run("fp8", assume_aligned=False)
print(f"FP8 (non-aligned):   fwd={fp8_nat_fwd:.3f}ms  bwd={fp8_nat_bwd:.3f}ms  total={fp8_nat_total:.3f}ms")

print(f"\n--- Speedup vs BF16 ---")
print(f"FP8 non-aligned: fwd={bf16_fwd/fp8_nat_fwd:.2f}x  bwd={bf16_bwd/fp8_nat_bwd:.2f}x  total={bf16_total/fp8_nat_total:.2f}x")

# 3. FP8 with assumed-aligned routing (production path with token rounding)
# NOTE: ASSUME_ALIGNED=1 will cause ILLEGAL_INSTRUCTION if routing is not
# actually 128-aligned. Only works with token rounding in production.
try:
    fp8_aln_fwd, fp8_aln_bwd, fp8_aln_total = run("fp8", assume_aligned=True)
    print(f"FP8 (aligned):       fwd={fp8_aln_fwd:.3f}ms  bwd={fp8_aln_bwd:.3f}ms  total={fp8_aln_total:.3f}ms")
    print(f"FP8 aligned:     fwd={bf16_fwd/fp8_aln_fwd:.2f}x  bwd={bf16_bwd/fp8_aln_bwd:.2f}x  total={bf16_total/fp8_aln_total:.2f}x")
except Exception as e:
    print(f"\nFP8 (aligned):       SKIPPED — routing not 128-aligned ({type(e).__name__})")
    print(f"  (Use token rounding in production for aligned FP8 path)")
