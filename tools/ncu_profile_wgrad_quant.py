"""NCU profiling script for wgrad colwise/fused-transpose quant kernels.
These are the actual bottleneck kernels in FP8 wgrad path.
"""
import torch
import math

torch.manual_seed(42)

# Ernie shape: TK=65536, H=3072 (hidden), I=1536 (intermediate)
# wgrad needs colwise quant on dz (TK, I) → (I, TK) fp8 + scales
TK = 65536
I_dim = 1536
H_dim = 3072
GROUP_SIZE = 32

# --- Profile colwise_quantize_and_pack on dz shape ---
dz_bf16 = torch.randn(TK, I_dim, dtype=torch.bfloat16, device='cuda')

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    colwise_quantize_and_pack,
    fused_transpose_quantize_and_pack,
    quantize_activation_blockscaled_fast,
)

# Warmup
for _ in range(3):
    colwise_quantize_and_pack(dz_bf16, I_dim, TK)
    fused_transpose_quantize_and_pack(dz_bf16, I_dim, TK)
    quantize_activation_blockscaled_fast(dz_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()

# Profile colwise quant (the original slow kernel)
torch.cuda.nvtx.range_push("colwise_quant")
for _ in range(3):
    colwise_quantize_and_pack(dz_bf16, I_dim, TK)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Profile fused transpose+quant (the optimized version)
torch.cuda.nvtx.range_push("fused_transpose_quant")
for _ in range(3):
    fused_transpose_quantize_and_pack(dz_bf16, I_dim, TK)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Profile rowwise quant (activation quant, for reference)
torch.cuda.nvtx.range_push("rowwise_quant_v2")
for _ in range(3):
    quantize_activation_blockscaled_fast(dz_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Profile script done. Shapes:")
print(f"  dz: ({TK}, {I_dim}) bf16")
print(f"  colwise: groups along axis 0 (TK), BLOCK_DIM along axis 1 (I)")
print(f"  fused_transpose: (TK, I) → (I, TK) fp8")
print(f"  rowwise: standard 1×32 row-major quant")
