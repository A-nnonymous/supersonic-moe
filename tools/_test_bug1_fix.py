"""Test Bug 1 fix: GemmDGatedFP8CLoadSm100ZeroMat — fused path with FUSED_GATED=1.

Runs FP8 fused (default) vs BF16 in separate subprocesses.
Reports per-expert RRMSE for dz (c_fc.weight grad) — the previously broken tensor.
"""
import subprocess, sys, json, os

SCRIPT = r'''
import os, json, torch, gc
import numpy as np

MODE = os.environ["_TEST_MODE"]  # "bf16" or "fp8_fused"

torch.manual_seed(42)
device = "cuda"
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
          intermediate_size=I, activation_function=ActivationType.SWIGLU,
          add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)

x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)

gc.collect(); torch.cuda.empty_cache()

if MODE == "bf16":
    with enable_quack_gemm(True):
        out, aux = moe(x)
elif MODE == "fp8_fused":
    # FUSED_GATED=1 is the DEFAULT — our fix should make this work
    assert os.environ.get("SONIC_MOE_FP8_FUSED_GATED", "1") == "1", "Must be fused"
    with enable_quack_gemm(True), enable_fp8():
        out, aux = moe(x, use_fp8=True)

loss = out.sum() + aux
loss.backward()

# Collect results
results = {}
results["output"] = out.detach().cpu()
results["dx"] = x.grad.detach().cpu()

# Per-expert weight grads
for name, param in moe.named_parameters():
    if param.grad is not None:
        results[name] = param.grad.detach().cpu()

torch.save(results, f"/tmp/_bug1_test_{MODE}.pt")
print(f"DONE:{MODE}")
'''

def run_mode(mode, extra_env=None):
    env = os.environ.copy()
    env["_TEST_MODE"] = mode
    env["USE_QUACK_GEMM"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env, capture_output=True, text=True, timeout=300
    )
    if r.returncode != 0:
        print(f"FAIL [{mode}] stderr:\n{r.stderr[-3000:]}")
        return False
    print(f"OK [{mode}] {r.stdout.strip()}")
    return True

print("=" * 70)
print("Bug 1 Fix Validation: FP8 fused (FUSED_GATED=1) vs BF16")
print("=" * 70)

# Run BF16 baseline
ok1 = run_mode("bf16")
# Run FP8 with FUSED_GATED=1 (DEFAULT — previously broken)
ok2 = run_mode("fp8_fused", {"SONIC_MOE_FP8_MODE": "perf", "SONIC_MOE_FP8_FUSED_GATED": "1"})

if not (ok1 and ok2):
    print("\nSubprocess failed — see errors above")
    sys.exit(1)

import torch, numpy as np

bf16 = torch.load("/tmp/_bug1_test_bf16.pt", weights_only=True)
fp8  = torch.load("/tmp/_bug1_test_fp8_fused.pt", weights_only=True)

def rrmse(a, b):
    a, b = a.float(), b.float()
    diff = (a - b).norm()
    denom = b.norm()
    return (diff / denom * 100).item() if denom > 0 else float('inf')

def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

print("\n" + "=" * 70)
print("PRECISION RESULTS (FP8 fused vs BF16)")
print("=" * 70)

all_pass = True
for key in sorted(bf16.keys()):
    r = rrmse(fp8[key], bf16[key])
    c = cosine(fp8[key], bf16[key])
    status = "PASS" if r < 15 else "FAIL"
    if status == "FAIL":
        all_pass = False
    
    # For c_fc weights, show per-expert breakdown
    if "c_fc" in key and bf16[key].dim() == 3:
        print(f"\n{key}: RRMSE={r:.2f}%, cos={c:.4f}  [{status}]")
        E = bf16[key].shape[0]
        for e in range(E):
            er = rrmse(fp8[key][e], bf16[key][e])
            ec = cosine(fp8[key][e], bf16[key][e])
            es = "PASS" if er < 15 else "FAIL"
            if es == "FAIL":
                all_pass = False
            print(f"  expert {e}: RRMSE={er:.2f}%, cos={ec:.4f}  [{es}]")
    elif "c_proj" in key and bf16[key].dim() == 3:
        print(f"\n{key}: RRMSE={r:.2f}%, cos={c:.4f}  [{status}]")
        E = bf16[key].shape[0]
        for e in range(E):
            er = rrmse(fp8[key][e], bf16[key][e])
            ec = cosine(fp8[key][e], bf16[key][e])
            es = "PASS" if er < 15 else "FAIL"
            if es == "FAIL":
                all_pass = False
            print(f"  expert {e}: RRMSE={er:.2f}%, cos={ec:.4f}  [{es}]")
    else:
        print(f"{key}: RRMSE={r:.2f}%, cos={c:.4f}  [{status}]")

print("\n" + "=" * 70)
if all_pass:
    print("ALL TESTS PASSED — Bug 1 is FIXED!")
else:
    print("SOME TESTS FAILED — Bug 1 NOT fully fixed")
print("=" * 70)
