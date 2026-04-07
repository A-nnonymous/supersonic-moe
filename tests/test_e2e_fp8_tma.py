"""End-to-end validation: FP8 TMA GemmDGated in full MoE forward+backward.

Tests:
1. Precision: FP8 vs BF16 gold standard (y, dx, dw RRMSE/cosine)
2. Memory: Peak GPU memory comparison
3. Correctness: gradient shapes and non-zero checks
"""
import os, sys, torch, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_SAVE_Z_FP8", "1")
os.environ.setdefault("SONIC_MOE_FP8_FUSED_GATED", "1")

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import socket


# Use small shapes that fit on 1 GPU
E, K, H, I = 8, 8, 768, 256
T = 512
SEED = 42


def make_moe():
    return MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()


def make_sample():
    torch.manual_seed(SEED)
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)
    return x, dout


def rrmse(a, b):
    return ((a.float() - b.float()).pow(2).mean().sqrt() /
            b.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()


def cosine(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm()).clamp(min=1e-12)).item()


host = socket.gethostname()
print(f"Host: {host}  GPU: {torch.cuda.get_device_name(0)}")
print(f"Shape: E={E}, K={K}, H={H}, I={I}, T={T}")
print("=" * 70)

# --- 1. BF16 gold standard ---
print("\n1. BF16 gold standard...")
model_bf16 = make_moe()
x_bf16, dout_bf16 = make_sample()
with enable_quack_gemm():
    y_bf16, _ = model_bf16(x_bf16)
    y_bf16.backward(dout_bf16)
dx_bf16 = x_bf16.grad.clone()
dw_bf16 = {n: p.grad.clone() for n, p in model_bf16.named_parameters() if p.grad is not None}
print(f"   y: {y_bf16.shape}, dx: {dx_bf16.shape}, #dw: {len(dw_bf16)}")

# --- 2. FP8 with Phase 3.1 TMA ---
print("\n2. FP8 (Phase 3.1 TMA-based GemmDGated)...")
model_fp8 = make_moe()
model_fp8.load_state_dict(model_bf16.state_dict())
x_fp8, dout_fp8 = make_sample()
with enable_fp8(), enable_quack_gemm():
    y_fp8, _ = model_fp8(x_fp8)
    y_fp8.backward(dout_fp8)
dx_fp8 = x_fp8.grad.clone()
dw_fp8 = {n: p.grad.clone() for n, p in model_fp8.named_parameters() if p.grad is not None}

print(f"   y  RRMSE: {rrmse(y_fp8, y_bf16):.6f}  cosine: {cosine(y_fp8, y_bf16):.6f}")
print(f"   dx RRMSE: {rrmse(dx_fp8, dx_bf16):.6f}  cosine: {cosine(dx_fp8, dx_bf16):.6f}")

max_dw_rrmse = 0.0
for n in sorted(dw_bf16.keys()):
    if n in dw_fp8:
        r = rrmse(dw_fp8[n], dw_bf16[n])
        c = cosine(dw_fp8[n], dw_bf16[n])
        max_dw_rrmse = max(max_dw_rrmse, r)
        if r > 0.05:  # Only print notable deviations
            print(f"   dw[{n}] RRMSE: {r:.6f}  cosine: {c:.6f}")
print(f"   All dw max RRMSE: {max_dw_rrmse:.6f}")

all_ok = (
    rrmse(y_fp8, y_bf16) < 0.1
    and rrmse(dx_fp8, dx_bf16) < 0.1
    and max_dw_rrmse < 0.15
)
print(f"   PRECISION: {'PASS' if all_ok else 'FAIL'}")

# --- 3. Memory ---
print("\n3. Memory comparison...")
gc.collect(); torch.cuda.empty_cache()

# FP8 peak
model_m = make_moe()
model_m.load_state_dict(model_bf16.state_dict())
x_m, d_m = make_sample()
torch.cuda.reset_peak_memory_stats()
with enable_fp8(), enable_quack_gemm():
    y_m, _ = model_m(x_m)
    y_m.backward(d_m)
peak_fp8 = torch.cuda.max_memory_allocated() / 1024**2
del model_m, x_m, d_m, y_m; gc.collect(); torch.cuda.empty_cache()

# BF16 peak
model_m2 = make_moe()
model_m2.load_state_dict(model_bf16.state_dict())
x_m2, d_m2 = make_sample()
torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm():
    y_m2, _ = model_m2(x_m2)
    y_m2.backward(d_m2)
peak_bf16 = torch.cuda.max_memory_allocated() / 1024**2
del model_m2, x_m2, d_m2, y_m2; gc.collect(); torch.cuda.empty_cache()

print(f"   Peak BF16: {peak_bf16:.0f} MiB")
print(f"   Peak FP8:  {peak_fp8:.0f} MiB")
print(f"   Saving:    {peak_bf16 - peak_fp8:.0f} MiB")

# --- 4. Gradient correctness checks ---
print("\n4. Gradient correctness checks...")
checks_ok = True
for n, p in model_fp8.named_parameters():
    if p.grad is not None:
        if p.grad.abs().max() == 0:
            print(f"   WARNING: {n} has zero gradient!")
            checks_ok = False
        if torch.isnan(p.grad).any():
            print(f"   WARNING: {n} has NaN gradient!")
            checks_ok = False
if dx_fp8.abs().max() == 0:
    print("   WARNING: dx is all zeros!")
    checks_ok = False
if torch.isnan(dx_fp8).any():
    print("   WARNING: dx has NaN!")
    checks_ok = False
print(f"   GRADIENT CHECKS: {'PASS' if checks_ok else 'FAIL'}")

print("\n" + "=" * 70)
print(f"OVERALL: {'ALL PASS' if all_ok and checks_ok else 'ISSUES DETECTED'}")
print("=" * 70)
