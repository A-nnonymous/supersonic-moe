"""Profile FP8 pipeline with nsys to find launch gaps."""
import torch, os

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
    clear_all_fp8_weight_caches,
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

w1_fp8, w1_scales = precompute_weight_fp8(w1)
w2_fp8, w2_scales = precompute_weight_fp8(w2)
w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
w2T_fp8, w2T_scales = precompute_weight_fp8(w2.permute(1, 0, 2))
s = torch.randn(TK, device="cuda", dtype=torch.float32).abs()

# Warmup all kernels
for _ in range(3):
    x_fp8, x_sc = gather_quantize_and_pack_activation(x, x_gather_idx)
    z = blockscaled_fp8_gemm_varlen(x_fp8, w1, cu, a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    y1_fp8, y1_sc, z_fp8, z_scales = swiglu_forward_quant_pack_zsave_triton(z)
    z_raw_u8 = z_scales.view(torch.uint8)
    out = blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu, a_scales=y1_sc, w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    dout = torch.randn_like(out)
    dout_fp8, dout_sc = quantize_and_pack_activation(dout)
    dy1 = blockscaled_fp8_gemm_varlen(dout_fp8, w2.permute(1,0,2), cu, a_scales=dout_sc, w_fp8=w2T_fp8, w_scales=w2T_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_u8)
    dz_fp8, dz_sc, y1s, ds, dz = swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=True)
    dx = blockscaled_fp8_gemm_varlen(dz_fp8, w1.permute(1,0,2), cu, a_scales=dz_sc, w_fp8=w1T_fp8, w_scales=w1T_scales, out_dtype=torch.bfloat16, assume_aligned=True)
torch.cuda.synchronize()

# NVTX-marked profile run
print("Starting profiled iterations...")
for i in range(5):
    torch.cuda.nvtx.range_push(f"iter_{i}")

    torch.cuda.nvtx.range_push("fwd_gather_quant")
    x_fp8, x_sc = gather_quantize_and_pack_activation(x, x_gather_idx)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("fwd_gemm1")
    z = blockscaled_fp8_gemm_varlen(x_fp8, w1, cu, a_scales=x_sc, w_fp8=w1_fp8, w_scales=w1_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("fwd_swiglu_quant_zsave")
    y1_fp8, y1_sc, z_fp8, z_scales = swiglu_forward_quant_pack_zsave_triton(z)
    z_raw_u8 = z_scales.view(torch.uint8)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("fwd_gemm2")
    out = blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu, a_scales=y1_sc, w_fp8=w2_fp8, w_scales=w2_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    torch.cuda.nvtx.range_pop()

    dout = out  # use output as gradient for backward

    torch.cuda.nvtx.range_push("bwd_quant_dout")
    dout_fp8, dout_sc = quantize_and_pack_activation(dout)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("bwd_gemm_dy1")
    dy1 = blockscaled_fp8_gemm_varlen(dout_fp8, w2.permute(1,0,2), cu, a_scales=dout_sc, w_fp8=w2T_fp8, w_scales=w2T_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("bwd_dequant_z")
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_u8)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("bwd_dswiglu_quant")
    dz_fp8, dz_sc, y1s, ds, dz = swiglu_backward_quant_pack_triton(dy1, z_bf16, s, return_dz_bf16=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("bwd_gemm_dx")
    dx = blockscaled_fp8_gemm_varlen(dz_fp8, w1.permute(1,0,2), cu, a_scales=dz_sc, w_fp8=w1T_fp8, w_scales=w1T_scales, out_dtype=torch.bfloat16, assume_aligned=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()  # iter

torch.cuda.synchronize()
print("Done. Profile collected.")
