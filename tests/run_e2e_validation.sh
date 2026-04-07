#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Unset distributed env vars that may interfere
unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
export NNODES=1 PADDLE_TRAINERS_NUM=1

USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python -c "
import os, sys, torch, gc
sys.path.insert(0, '.')
os.environ['USE_QUACK_GEMM'] = '1'
os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
os.environ['SONIC_MOE_FP8_SAVE_Z_FP8'] = '1'
os.environ['SONIC_MOE_FP8_FUSED_GATED'] = '1'

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import socket

E, K, H, I = 128, 8, 768, 256
T = 256
SEED = 42

def make_moe():
    return MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()

def make_sample():
    torch.manual_seed(SEED)
    x = (0.02 * torch.randn(T, H, device='cuda', dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)
    return x, dout

def rrmse(a, b):
    return ((a.float()-b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()

def cosine(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm()).clamp(min=1e-12)).item()

host = socket.gethostname()
print(f'Host: {host}  GPU: {torch.cuda.get_device_name(0)}')
print(f'Shape: E={E}, K={K}, H={H}, I={I}, T={T}')
print('=' * 70)

# --- BF16 gold standard (no FP8) ---
print('\\n1. BF16 gold standard...')
model_bf16 = make_moe()
x_bf16, dout_bf16 = make_sample()
with enable_quack_gemm():
    y_bf16 = model_bf16(x_bf16)
    y_bf16.backward(dout_bf16)
dx_bf16 = x_bf16.grad.clone()
dw_bf16 = {n: p.grad.clone() for n, p in model_bf16.named_parameters() if p.grad is not None}
print(f'   y: {y_bf16.shape}, dx: {dx_bf16.shape}')

# --- FP8 frontier (without Phase 3.1 — save z as fp8, standalone dequant) ---
print('\\n2. FP8 frontier (z saved fp8, standalone dequant)...')
model_fp8 = make_moe()
model_fp8.load_state_dict(model_bf16.state_dict())
x_fp8, dout_fp8 = make_sample()

# Force z_is_fp8=True but disable the TMA path by unsetting FUSED_GATED
# Actually we want the FUSED path — it's the one that uses preact_fp8
with enable_fp8(), enable_quack_gemm():
    y_fp8 = model_fp8(x_fp8)
    y_fp8.backward(dout_fp8)
dx_fp8 = x_fp8.grad.clone()
dw_fp8 = {n: p.grad.clone() for n, p in model_fp8.named_parameters() if p.grad is not None}

# Precision vs BF16
print(f'   y  RRMSE: {rrmse(y_fp8, y_bf16):.6f}  cosine: {cosine(y_fp8, y_bf16):.6f}')
print(f'   dx RRMSE: {rrmse(dx_fp8, dx_bf16):.6f}  cosine: {cosine(dx_fp8, dx_bf16):.6f}')
for n in dw_bf16:
    if n in dw_fp8:
        r = rrmse(dw_fp8[n], dw_bf16[n])
        c = cosine(dw_fp8[n], dw_bf16[n])
        if r > 0.01:
            print(f'   dw[{n}] RRMSE: {r:.6f}  cosine: {c:.6f}')
print(f'   All dw max RRMSE: {max(rrmse(dw_fp8[n], dw_bf16[n]) for n in dw_bf16 if n in dw_fp8):.6f}')

# --- Memory measurement ---
print('\\n3. Memory comparison...')
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_mem = make_moe()
model_mem.load_state_dict(model_bf16.state_dict())
x_mem, dout_mem = make_sample()
torch.cuda.reset_peak_memory_stats()
with enable_fp8(), enable_quack_gemm():
    y_mem = model_mem(x_mem)
    y_mem.backward(dout_mem)
peak_fp8 = torch.cuda.max_memory_allocated() / 1024**2
del model_mem, x_mem, dout_mem, y_mem
gc.collect(); torch.cuda.empty_cache()

model_mem2 = make_moe()
model_mem2.load_state_dict(model_bf16.state_dict())
x_mem2, dout_mem2 = make_sample()
torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm():
    y_mem2 = model_mem2(x_mem2)
    y_mem2.backward(dout_mem2)
peak_bf16 = torch.cuda.max_memory_allocated() / 1024**2
print(f'   Peak memory BF16: {peak_bf16:.0f} MiB')
print(f'   Peak memory FP8:  {peak_fp8:.0f} MiB')
print(f'   Saving:           {peak_bf16 - peak_fp8:.0f} MiB')

# --- Performance ---
print('\\n4. Performance (fwd+bwd)...')
WARMUP, ITERS, TRIALS = 5, 5, 3
def bench_fwdbwd(ctx_mgr, name):
    model = make_moe()
    model.load_state_dict(model_bf16.state_dict())
    times = []
    for _ in range(WARMUP):
        x_b, dout_b = make_sample()
        x_b = x_b.detach().requires_grad_()
        with ctx_mgr:
            y = model(x_b)
            y.backward(dout_b)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    for _ in range(TRIALS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            x_b, dout_b = make_sample()
            x_b = x_b.detach().requires_grad_()
            with ctx_mgr:
                y = model(x_b)
                y.backward(dout_b)
            model.zero_grad(set_to_none=True)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)
    print(f'   {name:<40} min={min(times):>7.0f}us  all={[f\"{t:.0f}\" for t in times]}')
    return min(times)

import contextlib
class combined_ctx:
    def __init__(self, *cms): self.cms = cms
    def __enter__(self):
        for c in self.cms: c.__enter__()
    def __exit__(self, *a):
        for c in reversed(self.cms): c.__exit__(*a)

t_bf16 = bench_fwdbwd(enable_quack_gemm(), 'BF16 fwd+bwd')
t_fp8 = bench_fwdbwd(combined_ctx(enable_fp8(), enable_quack_gemm()), 'FP8 fwd+bwd')
print(f'\\n   FP8 vs BF16: {t_fp8 - t_bf16:+.0f}us ({(t_fp8/t_bf16 - 1)*100:+.1f}%)')

print('\\nDone.')
"
