"""Kernel-level pipeline benchmark: FP8 vs BF16 forward, FP8 act-grad backward."""
import torch, os, time

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
          "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import (
    clear_all_fp8_weight_caches, _swiglu_forward_interleaved,
    TC_Softmax_Topk_Router_Function, general_routing_router_metadata,
)
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache,
    blockscaled_fp8_gemm_varlen, precompute_weight_fp8,
    gather_quantize_and_pack_activation, quantize_and_pack_activation,
)
from sonicmoe.quack_utils.swiglu_triton import (
    swiglu_forward_quant_pack_zsave_triton,
    swiglu_backward_quant_pack_triton,
    dequantize_blockscaled_fp8,
)

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
enable_quack_gemm()

x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
router_logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)
sorted_T, _ = topk_indices.flatten().sort()
selected_E = topk_indices.flatten()
_, cu_real, x_gather_idx, _, _, _ = general_routing_router_metadata(
    topk_scores, sorted_T, selected_E, T, E)
TK = int(cu_real[-1].item())
tpe = TK // E
cu = torch.arange(0, TK + 1, tpe, device="cuda", dtype=torch.int32)

w1 = moe.c_fc.weight.permute(1, 2, 0).contiguous()
w2 = moe.c_proj.weight.permute(1, 2, 0).contiguous()

print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}, TK={TK}, tpe={tpe}")

def bench(fn, warmup=5, iters=30, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {label}: {ms:.3f} ms")
    return ms

# Pre-compute weight FP8
w1_fp8, w1_scales = precompute_weight_fp8(w1)
w2_fp8, w2_scales = precompute_weight_fp8(w2)
w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
w2T_fp8, w2T_scales = precompute_weight_fp8(w2.permute(1, 0, 2))

# ========== FP8 FORWARD ==========
print("\n=== FP8 Forward ===")
t_gq = bench(lambda: gather_quantize_and_pack_activation(x, x_gather_idx), label="gather+quant(x)")
x_fp8, x_sc = gather_quantize_and_pack_activation(x, x_gather_idx)

t_gemm1 = bench(lambda: blockscaled_fp8_gemm_varlen(x_fp8, w1, cu,
    a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16, assume_aligned=True),
    label="GEMM1(up-proj)")

z = blockscaled_fp8_gemm_varlen(x_fp8, w1, cu,
    a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16, assume_aligned=True)

t_swiglu = bench(lambda: swiglu_forward_quant_pack_zsave_triton(z), label="SwiGLU+quant+zsave")
y1_fp8, y1_sc, z_fp8, z_scales = swiglu_forward_quant_pack_zsave_triton(z)
z_raw_u8 = z_scales.view(torch.uint8)

t_gemm2 = bench(lambda: blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu,
    a_scales=y1_sc, w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16, assume_aligned=True),
    label="GEMM2(down-proj)")

fp8_fwd = t_gq + t_gemm1 + t_swiglu + t_gemm2

# ========== BF16 FORWARD ==========
print("\n=== BF16 Forward (fused GEMM+SwiGLU baseline) ===")
os.environ["SONIC_MOE_FP8_MODE"] = "off"
clear_all_fp8_weight_caches()
clear_blockscaled_fp8_weight_cache()
_COMPILE_CACHE.clear()
t_bf16 = bench(lambda: moe(x)[0], label="moe(x)", warmup=3, iters=30)
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

# ========== FP8 BACKWARD (act-grad only, weight-grad is BF16 for both) ==========
print("\n=== FP8 Backward: Act-Grad Pipeline ===")
out = blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu,
    a_scales=y1_sc, w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16, assume_aligned=True)
dout = torch.randn_like(out)
s = torch.randn(TK, device="cuda", dtype=torch.float32).abs()

# Step 1: quant dout
t_qdout = bench(lambda: quantize_and_pack_activation(dout), label="quant(dout)")
dout_fp8, dout_sc = quantize_and_pack_activation(dout)

# Step 2: GEMM dout × w2^T → dy1
t_dy1 = bench(lambda: blockscaled_fp8_gemm_varlen(dout_fp8, w2.permute(1,0,2), cu,
    a_scales=dout_sc, w_fp8=w2T_fp8, w_scales=w2T_scales, out_dtype=torch.bfloat16, assume_aligned=True),
    label="GEMM(dout×w2^T→dy1)")
dy1 = blockscaled_fp8_gemm_varlen(dout_fp8, w2.permute(1,0,2), cu,
    a_scales=dout_sc, w_fp8=w2T_fp8, w_scales=w2T_scales, out_dtype=torch.bfloat16, assume_aligned=True)

# Step 3: dequant z_fp8 → z_bf16
t_dqz = bench(lambda: dequantize_blockscaled_fp8(z_fp8, z_raw_u8), label="dequant(z_fp8)")
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_u8)

# Step 4: dSwiGLU + quant(dz) + dz_bf16
t_dswiglu = bench(lambda: swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=True),
    label="dSwiGLU+quant+dz_bf16")
dz_fp8, dz_sc, y1s, ds, dz = swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=True)

# Step 5: GEMM dz × w1^T → dx
t_dx = bench(lambda: blockscaled_fp8_gemm_varlen(dz_fp8, w1.permute(1,0,2), cu,
    a_scales=dz_sc, w_fp8=w1T_fp8, w_scales=w1T_scales, out_dtype=torch.bfloat16, assume_aligned=True),
    label="GEMM(dz×w1^T→dx)")

fp8_bwd_actgrad = t_qdout + t_dy1 + t_dqz + t_dswiglu + t_dx

# ========== SUMMARY ==========
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"BF16 forward (fused):        {t_bf16:.3f} ms")
print(f"FP8 forward:                 {fp8_fwd:.3f} ms  ({t_bf16/fp8_fwd:.2f}x over BF16)")
print(f"FP8 bwd act-grad:            {fp8_bwd_actgrad:.3f} ms")
print(f"FP8 fwd+bwd act-grad total:  {fp8_fwd + fp8_bwd_actgrad:.3f} ms")
print(f"\nNote: weight-grad GEMMs use BF16 in both paths (identical cost)")
