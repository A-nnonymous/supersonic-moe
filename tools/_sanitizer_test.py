import os, torch
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
os.environ['SONIC_MOE_FP8_EPILOGUE_QUANT'] = '1'
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
torch.manual_seed(42)
moe = MoE(8, 8, 3072, 1536, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()
x = (0.02 * torch.randn(4096, 3072, device='cuda', dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)
with enable_quack_gemm(True):
    out, _ = moe(x)
torch.cuda.synchronize()
print('Forward OK')
out.backward(dout)
torch.cuda.synchronize()
print('Backward OK')
