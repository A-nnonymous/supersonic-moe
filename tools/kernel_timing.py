"""Kernel-level timing: identifies exactly where FP8 is slower than BF16.

Run:
    CUDA_VISIBLE_DEVICES=5 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
        python tools/kernel_timing.py
"""

import os, sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_FUSED_GATED", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import statistics
import torch
import torch.nn.functional as F

# ── shape parameters ─────────────────────────────────────────────────────────
T, H, I, E, K = 8192, 4096, 1024, 128, 8
WARMUP, ITERS = 5, 10
device = "cuda"

# ── routing setup (realistic, with token-rounding) ───────────────────────────
torch.manual_seed(42)

# Random router scores → top-K selection → per-expert counts
scores = torch.randn(T, E, device=device, dtype=torch.float32).softmax(dim=-1)
_, topk_indices = scores.topk(K, dim=-1)  # (T, K)
expert_freq = torch.bincount(
    topk_indices.reshape(-1), minlength=E
).to(torch.int32)

# Round each expert's count to nearest multiple of 128 (token rounding)
expert_freq_rounded = (torch.round(expert_freq.float() / 128) * 128).to(torch.int32)
expert_freq_rounded = expert_freq_rounded.clamp(min=128)

TK = expert_freq_rounded.sum().item()
expert_frequency_offset = torch.zeros(E + 1, dtype=torch.int32, device=device)
expert_frequency_offset[1:] = expert_freq_rounded.cumsum(0)

# Gather index: maps each of TK positions → original token in [0, T)
x_gather_idx = torch.randint(0, T, (TK,), dtype=torch.int64, device=device)

print(f"Config: T={T}, H={H}, I={I}, E={E}, K={K}, TK={TK}")
print(f"  FP8_MODE={os.environ.get('SONIC_MOE_FP8_MODE', 'off')}, "
      f"FUSED_GATED={os.environ.get('SONIC_MOE_FP8_FUSED_GATED', '0')}")
print(f"  expert_freq min={expert_freq_rounded.min().item()} "
      f"max={expert_freq_rounded.max().item()} "
      f"mean={expert_freq_rounded.float().mean().item():.0f}")
print()

# ── data tensors ─────────────────────────────────────────────────────────────
# Weights MUST be stored in the module's native layout (E, …) and then
# permuted to the (…, E) view that the functional layer expects, so that
# the stride-1 dimension ends up where the CUTLASS kernel needs it.
torch.manual_seed(0)
x_orig = 0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)
x_bf16 = 0.02 * torch.randn(TK, H, device=device, dtype=torch.bfloat16)

w1_param = 0.02 * torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16)
w1 = w1_param.permute(1, 2, 0)           # (2I, H, E) — non-contiguous view

w2_param = 0.02 * torch.randn(E, H, I, device=device, dtype=torch.bfloat16)
w2 = w2_param.permute(1, 2, 0)           # (H, I, E) — non-contiguous view

y1_bf16 = 0.02 * torch.randn(TK, I, device=device, dtype=torch.bfloat16)

# ── imports ──────────────────────────────────────────────────────────────────
from sonicmoe.quack_utils.gemm_interface import gemm_gated_out, gemm_gated_tuned
from quack.gemm_interface import gemm
from sonicmoe.quack_utils import (
    gather_quantize_and_pack_activation,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
    precompute_weight_fp8,
    blockscaled_fp8_gemm_varlen,
)

# ── pre-quantize FP8 inputs (not timed) ─────────────────────────────────────
print("Pre-quantizing FP8 inputs …")
# Up-proj FP8
x_fp8, x_scales = gather_quantize_and_pack_activation(x_orig, x_gather_idx)
w1_fp8, w1_scales = precompute_weight_fp8_for_fused_gated(w1)

# Down-proj FP8
y1_fp8, y1_scales = quantize_and_pack_activation(y1_bf16)
w2_fp8, w2_scales = precompute_weight_fp8(w2)

# ── pre-allocate output buffers ──────────────────────────────────────────────
w1_perm = w1.permute(2, 1, 0)          # (E, H, 2I) — non-contiguous view
w2_perm = w2.permute(2, 1, 0)          # (E, I, H)

z_buf  = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device=device)
y1_buf = torch.empty(TK, I,     dtype=torch.bfloat16, device=device)

z_buf_fp8  = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device=device)
y1_buf_fp8 = torch.empty(TK, I,     dtype=torch.bfloat16, device=device)


# ── timing helper ────────────────────────────────────────────────────────────
def time_kernel(fn, warmup=WARMUP, iters=ITERS):
    """Return median kernel time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


# ── kernel definitions ───────────────────────────────────────────────────────
# 1. BF16 up-proj (gemm_gated_out, autotuned)
def bf16_up_proj():
    gemm_gated_out(
        x_bf16, w1_perm,
        z_buf, y1_buf,
        None, None,                       # C, bias
        "swiglu",
        expert_frequency_offset,
        None,                             # A_idx (already gathered)
        False,                            # dynamic_scheduler
        True,                             # tuned
    )

# 2. FP8 up-proj (gemm_gated_tuned, blockscaled)
def fp8_up_proj():
    gemm_gated_tuned(
        x_fp8, w1_fp8,
        z_buf_fp8, y1_buf_fp8,
        None, None,
        "swiglu",
        expert_frequency_offset,
        None,
        False,
        a_scales=x_scales,
        b_scales=w1_scales,
    )

# 3. BF16 down-proj (gemm, varlen)
def bf16_down_proj():
    return gemm(y1_bf16, w2_perm, cu_seqlens_m=expert_frequency_offset)

# 4. FP8 down-proj (blockscaled_fp8_gemm_varlen)
def fp8_down_proj():
    return blockscaled_fp8_gemm_varlen(
        y1_fp8, w2, expert_frequency_offset,
        a_scales=y1_scales, w_fp8=w2_fp8, w_scales=w2_scales,
        out_dtype=torch.bfloat16,
    )

# 5. gather_quantize_and_pack_activation (x → fp8)
def do_gather_quant():
    return gather_quantize_and_pack_activation(x_orig, x_gather_idx)

# 6. quantize_and_pack_activation (y1 → fp8)
def do_quant():
    return quantize_and_pack_activation(y1_bf16)


# ── run benchmarks ───────────────────────────────────────────────────────────
print("Warming up + autotuning kernels …")
results = {}

print("  [1/6] BF16 gemm_gated_out (up-proj) …")
results["bf16_up"] = time_kernel(bf16_up_proj)

print("  [2/6] FP8  gemm_gated_tuned (up-proj) …")
results["fp8_up"] = time_kernel(fp8_up_proj)

print("  [3/6] BF16 gemm (down-proj) …")
results["bf16_down"] = time_kernel(bf16_down_proj)

print("  [4/6] FP8  blockscaled_fp8_gemm_varlen (down-proj) …")
results["fp8_down"] = time_kernel(fp8_down_proj)

print("  [5/6] gather_quantize_and_pack_activation …")
results["gather_quant"] = time_kernel(do_gather_quant)

print("  [6/6] quantize_and_pack_activation …")
results["quant"] = time_kernel(do_quant)

# ── TFLOPS computation ───────────────────────────────────────────────────────
# Up-proj: (TK, H) × (H, 2I) → 2·TK·H·2I = 4·TK·H·I
# Down-proj: (TK, I) × (I, H) → 2·TK·I·H
up_flops  = 4 * TK * H * I
down_flops = 2 * TK * I * H

def tflops(flops, ms):
    return flops / (ms * 1e9) if ms > 0 else 0.0


# ── results table ────────────────────────────────────────────────────────────
print()
print("=" * 82)
print(f"{'Kernel':<44} {'BF16 ms':>8} {'FP8 ms':>8} {'Ratio':>7} {'BF16 TF':>8} {'FP8 TF':>8}")
print("-" * 82)

bf16_u, fp8_u = results["bf16_up"], results["fp8_up"]
print(f"{'Up-proj  gemm_gated (SwiGLU fused)':<44} "
      f"{bf16_u:>8.3f} {fp8_u:>8.3f} {fp8_u/bf16_u:>6.2f}x "
      f"{tflops(up_flops, bf16_u):>7.1f} {tflops(up_flops, fp8_u):>8.1f}")

bf16_d, fp8_d = results["bf16_down"], results["fp8_down"]
print(f"{'Down-proj varlen GEMM':<44} "
      f"{bf16_d:>8.3f} {fp8_d:>8.3f} {fp8_d/bf16_d:>6.2f}x "
      f"{tflops(down_flops, bf16_d):>7.1f} {tflops(down_flops, fp8_d):>8.1f}")

bf16_tot = bf16_u + bf16_d
fp8_gemm_tot = fp8_u + fp8_d
print(f"{'── Total GEMM only':<44} "
      f"{bf16_tot:>8.3f} {fp8_gemm_tot:>8.3f} {fp8_gemm_tot/bf16_tot:>6.2f}x")

quant_tot = results["gather_quant"] + results["quant"]
fp8_all = fp8_gemm_tot + quant_tot
print(f"{'── Total GEMM + quantization':<44} "
      f"{bf16_tot:>8.3f} {fp8_all:>8.3f} {fp8_all/bf16_tot:>6.2f}x")

print("-" * 82)
print(f"{'FP8 quantization overhead:':<44}")
print(f"{'  gather_quantize_and_pack (x, up-proj)':<44} {'':>8} {results['gather_quant']:>8.3f}")
print(f"{'  quantize_and_pack (y1, down-proj)':<44} {'':>8} {results['quant']:>8.3f}")
print(f"{'  ── Total quantization':<44} {'':>8} {quant_tot:>8.3f}")
print("=" * 82)
