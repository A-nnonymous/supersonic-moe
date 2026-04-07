#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c '
import os,sys,torch;sys.path.insert(0,".")
os.environ["USE_QUACK_GEMM"]="1";os.environ["SONIC_MOE_FP8_MODE"]="perf"
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad, quantize_and_pack_activation,
    blockscaled_fp8_weight_grad_gemm_fast, _auto_capacity,
    _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE, _storage_per_batch, _div_up,
    _gather_isa_packed_scales_kernel, _SF_VEC_SIZE,
)
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8
from tests.bench_warp_dual_quant_v3 import warp_dual_quant_v3
import socket

E, H, I = 8, 3072, 1536; TK = 65536; CAP = TK // E
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2*I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_and_pack_activation(z)  # dummy for preact
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
z_fp8_bs, z_sc_bs = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc_bs.view(torch.uint8)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.full((E,), CAP, dtype=torch.int32, device="cuda").cumsum(0)]).int()
x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
torch.cuda.synchronize()

print(f"Host: {socket.gethostname()}")
print(f"TK={TK} H={H} I={I} E={E} CAP={CAP}")
print("="*70)

W, IT, TR = 5, 10, 5
def bench(fn, name):
    for _ in range(W): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(TR):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(IT): fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000/IT)
    print(f"  {name:<60} min={min(ts):>7.0f}us")
    return min(ts)

print("--- Phase B: y1s transpose quant breakdown ---")
y1s = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)
t_y1s_orig = bench(lambda: fused_transpose_quantize_for_wgrad(y1s, E, CAP, I),
                   "y1s transpose quant (32x32 kernel, I=1536)")

print("\n--- Full wgrad pipeline (current) ---")
dx = torch.empty(TK, 2*I, dtype=torch.bfloat16, device="cuda")
pa = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")

# Step-by-step pipeline
def run_pipeline_separate():
    # 1. Row quant dout
    dout_fp8, dout_scales_t = quantize_and_pack_activation(dout)
    # 2. Scale scatter
    K_bwd = H; k_tiles = _div_up(K_bwd, _SF_TILE_K)
    per_batch = _storage_per_batch(TK, K_bwd)
    dout_scales_tk = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
        dout_scales_t.view(torch.uint8), x_idx, dout_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
    # 3. GemmDGated
    gemm_dgated(dout_fp8, wf, dx, z, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_scales_tk.view(torch.float8_e8m0fnu), b_scales=ws,
                preact_fp8=z_fp8_bs, preact_scales=z_sc_u8)
    # 4. Wgrad
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx)

t_pipeline = bench(run_pipeline_separate, "Full pipeline: row_quant+scatter+GemmDGated+wgrad")

# Optimized pipeline with dual-quant
def run_pipeline_dual():
    # 1. Dual quant dout (row + col in 1 kernel)
    row_fp8, row_sc, col_fp8_a, col_sc_a = warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4)
    # 2. Scale scatter (still needed)
    K_bwd = H; k_tiles = _div_up(K_bwd, _SF_TILE_K)
    per_batch = _storage_per_batch(TK, K_bwd)
    dout_scales_tk = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
        row_sc.view(torch.uint8), x_idx, dout_scales_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
    # 3. GemmDGated
    gemm_dgated(row_fp8, wf, dx, z, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_scales_tk.view(torch.float8_e8m0fnu), b_scales=ws,
                preact_fp8=z_fp8_bs, preact_scales=z_sc_u8)
    # 4. Transpose quant y1s (uses 32x32 kernel)
    col_fp8_b, col_sc_b = fused_transpose_quantize_for_wgrad(pa, E, CAP, I)
    # 5. Wgrad GEMM directly with pre-quantized A and B
    # (Need to call the GEMM part only, but for now measure full pipeline)
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx)

t_dual = bench(run_pipeline_dual, "Dual: dual_quant+scatter+GemmDGated+y1s_quant+wgrad")

# Stream overlap: y1s quant ‖ dout dual-quant
# This only works if y1s is ready before we need it for wgrad
# In practice: GemmDGated → y1s ready → y1s quant on stream2 ‖ wgrad prep on stream1

print(f"\n--- Summary ---")
print(f"Full pipeline (current):  {t_pipeline:.0f}us")
print(f"With dual-quant:          {t_dual:.0f}us ({t_pipeline - t_dual:+.0f}us)")
print(f"y1s quant alone:          {t_y1s_orig:.0f}us (Phase B target to eliminate)")
'
