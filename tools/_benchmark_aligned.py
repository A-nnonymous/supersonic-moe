"""Aligned-segment benchmark: demonstrates FP8 performance with 128-aligned routing.

Production MoE systems use token-rounding routing that guarantees 128-aligned
expert segments, avoiding the expensive padding path in blockscaled FP8 GEMM.
This benchmark shows the achievable performance under production conditions.
"""
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
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache,
    blockscaled_fp8_gemm_varlen, precompute_weight_fp8,
    gather_quantize_and_pack_activation, quantize_and_pack_activation,
)
from sonicmoe.functional import (
    clear_all_fp8_weight_caches, _swiglu_forward_interleaved,
    TC_Softmax_Topk_Router_Function, general_routing_router_metadata,
)
from sonicmoe.quack_utils.swiglu_triton import swiglu_forward_quant_pack_triton
from quack.gemm_interface import gemm

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
enable_quack_gemm()

x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
router_logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

# Get real routing to extract shapes and gather indices
topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)
sorted_T, _ = topk_indices.flatten().sort()
selected_E = topk_indices.flatten()
_, cu_real, x_gather_idx, _, _, _ = general_routing_router_metadata(
    topk_scores, sorted_T, selected_E, T, E)
TK = int(cu_real[-1].item())

# Create aligned cu_seqlens (uniform 256 tokens per expert)
tpe = TK // E
cu_aligned = torch.arange(0, TK + 1, tpe, device="cuda", dtype=torch.int32)

# Weight tensors (N, K, E) format
w1 = moe.c_fc.weight.permute(1, 2, 0).contiguous()   # (2I, H, E)
w2 = moe.c_proj.weight.permute(1, 2, 0).contiguous()  # (H, I, E)
w1p = w1.permute(2, 1, 0).contiguous()                # (E, H, 2I) for BF16 gemm
w2p = w2.permute(2, 1, 0).contiguous()                # (E, I, H) for BF16 gemm

def timed(fn, name, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {name}: {ms:.3f}ms")
    return ms


# ============================================================
print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}, TK={TK}, tpe={tpe}")
print(f"Aligned: {tpe % 128 == 0}")

# --- BF16 Baseline (fused GEMM+SwiGLU) ---
print("\n=== BF16 Forward (fused GEMM+SwiGLU baseline) ===")
os.environ["SONIC_MOE_FP8_MODE"] = "off"
clear_all_fp8_weight_caches()
clear_blockscaled_fp8_weight_cache()
_COMPILE_CACHE.clear()

def bf16_fwd():
    with torch.no_grad():
        moe(x)

# Warmup full MoE (compiles kernels)
for _ in range(3): bf16_fwd()
torch.cuda.synchronize()
bf16_total = timed(bf16_fwd, "BF16 total (moe(x))")

# --- FP8 Unaligned (random routing, current benchmark) ---
print("\n=== FP8 Forward Unaligned (random routing, padding overhead) ===")
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
clear_all_fp8_weight_caches()
clear_blockscaled_fp8_weight_cache()
_COMPILE_CACHE.clear()

def fp8_fwd():
    with torch.no_grad():
        moe(x)

for _ in range(3): fp8_fwd()
torch.cuda.synchronize()
fp8_unaligned = timed(fp8_fwd, "FP8 total (random routing)")

# --- FP8 Aligned Kernel-level ---
print("\n=== FP8 Forward Aligned (production routing, no padding) ===")
w1_fp8, w1_scales = precompute_weight_fp8(w1)
w2_fp8, w2_scales = precompute_weight_fp8(w2)

# Measure each stage
x_g = x[x_gather_idx]
t_gather = timed(lambda: x[x_gather_idx], "gather x[idx]")

t_gemm1_bf16 = timed(
    lambda: blockscaled_fp8_gemm_varlen(x_g, w1, cu_aligned,
        w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16),
    "GEMM1 (bf16 input)")

x_fp8, x_sc = gather_quantize_and_pack_activation(x, x_gather_idx)
t_gather_quant = timed(
    lambda: gather_quantize_and_pack_activation(x, x_gather_idx),
    "gather+quantize (fused)")

t_gemm1_prequant = timed(
    lambda: blockscaled_fp8_gemm_varlen(x_fp8, w1, cu_aligned,
        a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16),
    "GEMM1 (pre-quantized)")

z = blockscaled_fp8_gemm_varlen(x_fp8, w1, cu_aligned,
    a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16)

t_swiglu = timed(lambda: _swiglu_forward_interleaved(z), "SwiGLU (separate)")

y1_fp8, y1_sc = swiglu_forward_quant_pack_triton(z)
t_swiglu_quant = timed(
    lambda: swiglu_forward_quant_pack_triton(z),
    "SwiGLU+quant+pack (fused)")

t_gemm2_prequant = timed(
    lambda: blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu_aligned,
        a_scales=y1_sc, w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16),
    "GEMM2 (pre-quantized)")

y1 = _swiglu_forward_interleaved(z)
t_gemm2_bf16 = timed(
    lambda: blockscaled_fp8_gemm_varlen(y1, w2, cu_aligned,
        w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16),
    "GEMM2 (bf16 input)")

# Summary
fp8_old_aligned = t_gather + t_gemm1_bf16 + t_swiglu + t_gemm2_bf16
fp8_opt_aligned = t_gather_quant + t_gemm1_prequant + t_swiglu_quant + t_gemm2_prequant

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"BF16 fused baseline:       {bf16_total:.3f}ms")
print(f"FP8 unaligned (random):    {fp8_unaligned:.3f}ms  ({bf16_total/fp8_unaligned:.2f}x)")
print(f"FP8 aligned (old path):    {fp8_old_aligned:.3f}ms  ({bf16_total/fp8_old_aligned:.2f}x)")
print(f"FP8 aligned (optimized):   {fp8_opt_aligned:.3f}ms  ({bf16_total/fp8_opt_aligned:.2f}x)")
print(f"\nOptimized FP8 speedup vs BF16: {bf16_total/fp8_opt_aligned:.2f}x")
print(f"Savings per forward pass: {bf16_total - fp8_opt_aligned:.3f}ms")
