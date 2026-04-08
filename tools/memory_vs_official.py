#!/usr/bin/env python3
"""Rigorous memory breakdown: Official BF16 vs FP8 frontier.

Runs each case in a fully isolated subprocess (separate Python process)
to avoid any cross-contamination of CUDA context, JIT caches, or global state.

Measures memory at every lifecycle checkpoint:
  base → model_load → input_alloc → pre_fwd → fwd_peak → post_fwd →
  pre_bwd → bwd_peak → post_bwd → cleanup → residual
"""

import json
import os
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────
ERNIE_SHAPE = "T=8192, H=3072, I=1536, E=8, K=8"
T, H, I, E, K = 8192, 3072, 1536, 8, 8

OFFICIAL_BF16_ENV = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16"
OFFICIAL_BF16_CODE = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
FP8_ENV = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer"
FP8_CODE = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"


def _make_script(mode: str) -> str:
    """Generate the inner subprocess script for a given mode."""
    if mode == "official_bf16":
        env_setup = """
import sys, os
sys.path.insert(0, "{code}")
os.environ["USE_QUACK_GEMM"] = "1"
""".format(code=OFFICIAL_BF16_CODE)
        fwd_bwd = """
with enable_quack_gemm(True):
    o, _, _ = moe_TC_softmax_topk_layer(
        x, router_w, w1.permute(1,2,0), None, w2.permute(1,2,0), None,
        K, 0, ActivationType.SWIGLU, False, None,
    )
o.backward(dout)
"""
    elif mode == "fp8_frontier":
        env_setup = """
import sys, os
sys.path.insert(0, "{code}")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
""".format(code=FP8_CODE)
        fwd_bwd = """
with enable_quack_gemm(True):
    o, _, _ = moe_TC_softmax_topk_layer(
        x, router_w, w1.permute(1,2,0), None, w2.permute(1,2,0), None,
        K, 0, ActivationType.SWIGLU, False, None,
    )
o.backward(dout)
"""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return '''
{env_setup}
import gc, json, torch
torch.manual_seed(42)
device = "cuda"
MiB = 1024**2

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.utils import enable_quack_gemm

checkpoints = {{}}

def ck(name):
    torch.cuda.synchronize()
    checkpoints[name] = {{
        "alloc": torch.cuda.memory_allocated() / MiB,
        "peak": torch.cuda.max_memory_allocated() / MiB,
        "reserved": torch.cuda.memory_reserved() / MiB,
    }}

torch.cuda.synchronize()
ck("00_empty")

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02
).to(device=device, dtype=torch.bfloat16)
router_w = moe.router.weight.detach()
w1 = moe.c_fc.weight.detach()
w2 = moe.c_proj.weight.detach()
ck("01_model_loaded")

x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)
ck("02_input_allocated")

# Warmup (2 iters for JIT compilation)
for _ in range(2):
    {fwd_bwd}
    x.grad = None
    for p in moe.parameters():
        if p.grad is not None:
            p.grad = None

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
ck("03_post_warmup")

# Measured forward
{fwd_bwd_forward_only}
torch.cuda.synchronize()
ck("04_fwd_peak")

fwd_alloc = torch.cuda.memory_allocated() / MiB
ck("05_post_fwd")

# Measured backward
torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize()
ck("06_bwd_peak")

# Cleanup
x.grad = None
for p in moe.parameters():
    if p.grad is not None:
        p.grad = None
del o
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
ck("07_post_cleanup")

print(json.dumps(checkpoints))
'''.format(
        env_setup=env_setup,
        T=T, H=H, I=I, E=E, K=K,
        fwd_bwd=fwd_bwd.strip(),
        fwd_bwd_forward_only=fwd_bwd.strip().split("\n")[0] + "\n" +
            fwd_bwd.strip().split("\n")[1] + "\n" +
            fwd_bwd.strip().split("\n")[2],
    )


def run_case(mode: str, gpu: int) -> dict:
    """Run memory measurement in isolated subprocess."""
    env_path = OFFICIAL_BF16_ENV if mode == "official_bf16" else FP8_ENV
    python = os.path.join(env_path, "bin", "python")

    script = _make_script(mode)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Proxy not needed for local execution
    for k in ["http_proxy", "https_proxy"]:
        env.pop(k, None)

    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        print(f"[{mode}] FAILED (exit={result.returncode}):")
        print(result.stderr[-3000:])
        return {}

    # Parse JSON from last line of stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"[{mode}] No JSON output found")
    print(result.stdout[-1000:])
    return {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-bf16", type=int, default=0)
    parser.add_argument("--gpu-fp8", type=int, default=1)
    args = parser.parse_args()

    print(f"=== Memory Breakdown: Official BF16 vs FP8 Frontier ===")
    print(f"Shape: {ERNIE_SHAPE}")
    print(f"Official BF16: {OFFICIAL_BF16_CODE}")
    print(f"FP8 frontier:  {FP8_CODE}")
    print()

    bf16 = run_case("official_bf16", args.gpu_bf16)
    fp8 = run_case("fp8_frontier", args.gpu_fp8)

    if not bf16 or not fp8:
        print("ERROR: One or both cases failed.")
        return

    # Print comparison table
    print(f"\n{'Checkpoint':<25s} {'BF16 alloc':>12s} {'BF16 peak':>12s} {'FP8 alloc':>12s} {'FP8 peak':>12s} {'Δalloc':>10s} {'Δpeak':>10s}")
    print("-" * 95)
    for key in sorted(bf16.keys()):
        ba = bf16[key]["alloc"]
        bp = bf16[key]["peak"]
        fa = fp8.get(key, {}).get("alloc", 0)
        fp_ = fp8.get(key, {}).get("peak", 0)
        da = fa - ba
        dp = fp_ - bp
        print(f"{key:<25s} {ba:>10.1f}M {bp:>10.1f}M {fa:>10.1f}M {fp_:>10.1f}M {da:>+9.1f}M {dp:>+9.1f}M")

    # Key metrics
    bf16_fwd_peak = bf16.get("04_fwd_peak", {}).get("peak", 0)
    bf16_bwd_peak = bf16.get("06_bwd_peak", {}).get("peak", 0)
    fp8_fwd_peak = fp8.get("04_fwd_peak", {}).get("peak", 0)
    fp8_bwd_peak = fp8.get("06_bwd_peak", {}).get("peak", 0)
    bf16_residual = bf16.get("07_post_cleanup", {}).get("alloc", 0)
    fp8_residual = fp8.get("07_post_cleanup", {}).get("alloc", 0)

    print(f"\n=== KEY METRICS ===")
    print(f"{'Metric':<30s} {'BF16':>10s} {'FP8':>10s} {'Delta':>10s} {'%':>8s}")
    print("-" * 70)
    for name, bv, fv in [
        ("Forward peak", bf16_fwd_peak, fp8_fwd_peak),
        ("Backward peak", bf16_bwd_peak, fp8_bwd_peak),
        ("Post-cleanup residual", bf16_residual, fp8_residual),
    ]:
        d = fv - bv
        p = d / bv * 100 if bv > 0 else 0
        print(f"{name:<30s} {bv:>8.1f}M {fv:>8.1f}M {d:>+8.1f}M {p:>+7.1f}%")


if __name__ == "__main__":
    main()
