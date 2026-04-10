"""NCU profiling script for CuTe DSL vs Triton blockscaled FP8 quant kernels."""
import torch
import math

torch.manual_seed(42)
M, K = 65536, 1536
GROUP_SIZE = 32

x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

# Warmup + run Triton
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
for _ in range(3):
    quantize_activation_blockscaled_fast(x_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()

# Profiling region for Triton
torch.cuda.nvtx.range_push("triton_blockscaled_quant")
for _ in range(5):
    quantize_activation_blockscaled_fast(x_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Warmup + run CuTe
from sonicmoe.quack_utils.cute_blockscaled_quant import blockscaled_quantize_cute
for _ in range(3):
    blockscaled_quantize_cute(x_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()

# Profiling region for CuTe
torch.cuda.nvtx.range_push("cute_blockscaled_quant")
for _ in range(5):
    blockscaled_quantize_cute(x_bf16, group_size=GROUP_SIZE)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Profile script done.")
