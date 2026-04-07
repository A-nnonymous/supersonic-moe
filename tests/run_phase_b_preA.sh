#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c '
import os,sys,torch;sys.path.insert(0,".")
os.environ["USE_QUACK_GEMM"]="1";os.environ["SONIC_MOE_FP8_MODE"]="perf"
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_weight_grad_gemm_fast, fused_transpose_quantize_for_wgrad,
    quantize_and_pack_activation, _auto_capacity,
)
from tests.bench_warp_dual_quant_v3 import warp_dual_quant_v3
import socket

E, H, I = 8, 3072, 1536; TK = 65536; CAP = TK // E
torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
y1s = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.full((E,), CAP, dtype=torch.int32, device="cuda").cumsum(0)]).int()
x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
torch.cuda.synchronize()

print(f"Host: {socket.gethostname()}")
print(f"TK={TK} H={H} I={I} E={E}")
print("="*70)

W, IT, TR = 10, 20, 5
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

print("--- Baseline: FP8 wgrad (no pre-quant) ---")
t_base = bench(lambda: blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx),
               "wgrad_fast(dout, y1s) — quants A+B internally")

print("\n--- Phase B: pre-quantize A via dual-quant, skip A quant in wgrad ---")
# Pre-compute A (dout) col-quant via dual-quant
_, _, col_fp8_a, col_sc_a = warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4)
torch.cuda.synchronize()

t_preA = bench(lambda: blockscaled_fp8_weight_grad_gemm_fast(
    dout, y1s, cu, a_gather_idx=x_idx,
    pre_quantized_a=(col_fp8_a, col_sc_a)),
    "wgrad_fast(pre_A, y1s) — skip A quant, B quant internal")

print(f"\n  Baseline: {t_base:.0f}us → Pre-A: {t_preA:.0f}us ({t_base-t_preA:+.0f}us, {t_base/t_preA:.2f}x)")
print(f"  Saved A transpose-quant: ~{t_base-t_preA:.0f}us")

# Verify precision
ref = blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx)
opt = blockscaled_fp8_weight_grad_gemm_fast(dout, y1s, cu, a_gather_idx=x_idx,
                                             pre_quantized_a=(col_fp8_a, col_sc_a))
torch.cuda.synchronize()
rrmse = ((ref.float()-opt.float()).pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
status = "PASS" if rrmse < 1e-5 else "FAIL"
print(f"  Precision: RRMSE={rrmse:.6f} ({status})")
'
