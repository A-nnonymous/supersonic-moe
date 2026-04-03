"""Bit-exact test: epilogue quant vs standalone quantize_and_pack.

Gets original z from non-quant GemmGated, then compares:
1. Standalone quantize_and_pack_activation(z_orig) → reference fp8 + scales
2. Epilogue quant GemmGated(with scale output) → epilogue fp8 + raw scales
They should be bit-identical (same integer+carry algorithm on same z data).
"""
import sys, torch, struct
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
import os; os.environ.setdefault("USE_QUACK_GEMM", "1"); os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_and_pack_activation, _quantize_weight_3d_triton, _div_up, _SF_VEC_SIZE,
)
from sonicmoe.quack_utils import gemm_gated as gg
import sonicmoe.quack_utils.gemm_sm100_fp8_zeromat as zm
from sonicmoe.quack_utils.gemm_gated import GemmGatedMixin
from quack.gemm_sm100 import GemmSm100
import importlib
gg_mod = importlib.import_module("sonicmoe.quack_utils.gemm_gated")

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

n_groups = 2 * I // 32

# ═══ Step 1: Get original z (non-quant path) ═══
# Temporarily swap ZeroMat to non-quant version
orig_cls = zm.GemmGatedSm100ZeroMat

class GemmGatedSm100ZeroMatNoQuant(GemmGatedMixin, zm._GemmSm100ZeroMatMixin, GemmSm100):
    pass

zm.GemmGatedSm100ZeroMat = GemmGatedSm100ZeroMatNoQuant
gg_mod.gemm_gated.compile_cache.clear()

z_orig, y1_orig = gg(
    xf, w1f, activation="swiglu", out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
    cu_seqlens_m=efo, A_idx=idx, a_scales=xs, b_scales=w1s,
    dynamic_scheduler=False, tuned=False,
)
print(f"z_orig: range=[{z_orig.min():.6f}, {z_orig.max():.6f}] norm={z_orig.float().norm():.4f}")

# Restore quant mixin
zm.GemmGatedSm100ZeroMat = orig_cls
gg_mod.gemm_gated.compile_cache.clear()

# ═══ Step 2: Standalone quant of z_orig ═══
ref_fp8, ref_scales_packed = quantize_and_pack_activation(z_orig)
print(f"Standalone quant: fp8 range=[{ref_fp8.float().min():.2f}, {ref_fp8.float().max():.2f}]")

# Unpack ISA scales to raw (TK, n_groups) format for comparison
# ISA packing is complex — instead, compare at the group level using Python reference
z_cpu = z_orig.float().cpu()
group_size = _SF_VEC_SIZE

# ═══ Step 3: Epilogue quant ═══
z_epi_scales = torch.full((TK, n_groups), -1, device="cuda", dtype=torch.int32)
z_epi, y1_epi = gg(
    xf, w1f, activation="swiglu", out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
    cu_seqlens_m=efo, A_idx=idx, a_scales=xs, b_scales=w1s,
    dynamic_scheduler=False, tuned=False,
    z_scale_out=z_epi_scales,
)
print(f"Epilogue quant: z_epi range=[{z_epi.min():.2f}, {z_epi.max():.2f}]")

# ═══ Step 4: Compare scales ═══
# Compute reference raw UE8M0 from z_orig using Python (same integer+carry)
ref_raw_scales = torch.zeros(TK, n_groups, dtype=torch.int32)
for row in range(min(TK, 100)):  # first 100 rows
    for g in range(n_groups):
        col_s = g * group_size
        col_e = min(col_s + group_size, 2*I)
        amax = max(abs(z_cpu[row, c].item()) for c in range(col_s, col_e))
        bits = struct.unpack('I', struct.pack('f', amax))[0]
        bexp = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        carry = 1 if mant > 0x600000 else 0
        e8m0 = bexp - 8 + carry
        if bexp == 0: e8m0 = 0
        e8m0 = max(e8m0, 0)
        ref_raw_scales[row, g] = e8m0

# Compare epilogue scales vs reference (first 100 rows)
epi_sub = z_epi_scales[:100].cpu()
ref_sub = ref_raw_scales[:100]
scale_match = torch.equal(epi_sub, ref_sub)
if not scale_match:
    diff = (epi_sub - ref_sub).abs()
    n_diff = (diff > 0).sum().item()
    total_checked = epi_sub.numel()
    print(f"\nScale comparison (first 100 rows × {n_groups} groups):")
    print(f"  Mismatches: {n_diff}/{total_checked} ({n_diff/total_checked*100:.2f}%)")
    # Show first mismatches
    for row in range(100):
        for g in range(n_groups):
            if epi_sub[row, g] != ref_sub[row, g]:
                print(f"  Row {row} group {g}: epilogue={epi_sub[row,g].item()} ref={ref_sub[row,g].item()}")
                break
        else:
            continue
        break
else:
    print(f"\nScale comparison: EXACT MATCH (first 100 rows × {n_groups} groups)")

print("\nDONE")
