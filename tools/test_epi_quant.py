"""Test epilogue blockscaled quant with scale gmem write."""
import sys, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
import os; os.environ.setdefault("USE_QUACK_GEMM", "1"); os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_and_pack_activation, _quantize_weight_3d_triton
from sonicmoe.quack_utils import gemm_gated as gemm_gated_hl

E, K_top, H, I = 8, 8, 3072, 1536
T, TK = 8192, 8192 * 8
torch.manual_seed(42)
w1 = torch.randn(E, 2*I, H, device="cuda", dtype=torch.bfloat16) * 0.02
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.02
efo = torch.zeros(E+1, device="cuda", dtype=torch.int32)
efo[1:] = torch.full((E,), TK//E, device="cuda", dtype=torch.int32).cumsum(0)
idx = torch.arange(TK, device="cuda", dtype=torch.int32) % T

xf, xs = quantize_and_pack_activation(x)
w1f_enk, w1s = _quantize_weight_3d_triton(w1.contiguous())
w1f = w1f_enk.mT

# Allocate scale output: (TK, 2I//32) int32
n_groups = 2 * I // 32
z_scale_out = torch.zeros(TK, n_groups, device="cuda", dtype=torch.int32)

# Reference: standard gemm_gated WITHOUT scale output
z_ref, y1_ref = gemm_gated_hl(
    xf, w1f, activation="swiglu", out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
    cu_seqlens_m=efo, A_idx=idx, a_scales=xs, b_scales=w1s,
    dynamic_scheduler=False, tuned=False,
)
print(f"Ref:  z nan={torch.isnan(z_ref).any().item()} norm={z_ref.float().norm().item():.4f}")

# Test: with scale output
try:
    z_bq, y1_bq = gemm_gated_hl(
        xf, w1f, activation="swiglu", out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
        cu_seqlens_m=efo, A_idx=idx, a_scales=xs, b_scales=w1s,
        dynamic_scheduler=False, tuned=False,
        z_scale_out=z_scale_out,
    )
    print(f"BQ:   z nan={torch.isnan(z_bq).any().item()} norm={z_bq.float().norm().item():.4f}")
    print(f"y1 diff: {(y1_ref.float() - y1_bq.float()).abs().max().item():.6e}")
    nz = z_scale_out.count_nonzero().item()
    print(f"Scales: nonzero={nz}/{z_scale_out.numel()} ({nz/z_scale_out.numel()*100:.1f}%)")
    print(f"Scale sample [0,:8]: {z_scale_out[0,:8].tolist()}")
    if nz > 0:
        print("EPILOGUE SCALE GMEM WRITE: SUCCESS!")
    else:
        print("Scales are zero — epilogue write may not have executed")
except Exception as e:
    import traceback; traceback.print_exc()
