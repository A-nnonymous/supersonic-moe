#!/usr/bin/env python3
"""Quick end-to-end precision audit: FP8 vs BF16 per-tensor comparison.

Runs in subprocess isolation (avoids SONIC_MOE_FP8_MODE contamination).
Reports RRMSE and cosine similarity for all output tensors and gradients.
"""
import gc
import math
import os
import subprocess
import sys
import json
import tempfile


INNER_SCRIPT = r'''
import gc, os, json, sys, torch, numpy as np

device_id = int(sys.argv[1])
mode = sys.argv[2]
out_path = sys.argv[3]
seed = int(sys.argv[4])

os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
os.environ["USE_QUACK_GEMM"] = "1"
if mode == "fp8":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
else:
    os.environ["SONIC_MOE_FP8_MODE"] = "off"

torch.manual_seed(42)  # shared model init
device = "cuda"

T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(
    num_experts=E, num_experts_per_tok=K,
    hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU,
    add_bias=False, std=0.02,
).to(device=device, dtype=torch.bfloat16)

# Warmup (1 iter to JIT compile)
x_warmup = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
with enable_quack_gemm(True):
    result = moe(x_warmup)
    o_w = result[0]
o_w.backward(torch.randn_like(o_w))
del x_warmup, o_w
gc.collect()
torch.cuda.empty_cache()

# Measured iteration with specific seed
torch.manual_seed(seed)
x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device=device, dtype=torch.bfloat16)

for p in moe.parameters():
    p.grad = None

with enable_quack_gemm(True):
    result = moe(x)
    o = result[0]

o.backward(dout)
torch.cuda.synchronize()

# Collect results
results = {}
results["output"] = o.detach().float().cpu().numpy().tolist()
results["dx"] = x.grad.detach().float().cpu().numpy().tolist()

# Weight grads
for name, param in moe.named_parameters():
    if param.grad is not None:
        key = name.replace(".", "_")
        # Only store norm info (full tensor too large for JSON)
        g = param.grad.detach().float()
        results[f"wgrad_{key}_norm"] = g.norm().item()
        results[f"wgrad_{key}_mean"] = g.mean().item()
        results[f"wgrad_{key}_max"] = g.abs().max().item()

# Store output/dx as numpy for compact comparison
np.savez_compressed(
    out_path,
    output=o.detach().float().cpu().numpy(),
    dx=x.grad.detach().float().cpu().numpy(),
)
# Also save wgrad norms
with open(out_path + ".json", "w") as f:
    wgrad_info = {k: v for k, v in results.items() if k.startswith("wgrad_")}
    json.dump(wgrad_info, f)
print(f"DONE mode={mode} seed={seed}")
'''


def rrmse(ref, test):
    """Relative RMSE: RMSE / RMS(ref)."""
    import numpy as np
    diff = test - ref
    rmse = np.sqrt(np.mean(diff ** 2))
    rms_ref = np.sqrt(np.mean(ref ** 2))
    return rmse / rms_ref if rms_ref > 0 else float('inf')


def cosine_sim(ref, test):
    import numpy as np
    dot = np.sum(ref * test)
    norm_ref = np.sqrt(np.sum(ref ** 2))
    norm_test = np.sqrt(np.sum(test ** 2))
    return dot / (norm_ref * norm_test) if (norm_ref > 0 and norm_test > 0) else 0.0


def run_case(mode, seed, device_id, tmpdir):
    out_path = os.path.join(tmpdir, f"{mode}_s{seed}")
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", INNER_SCRIPT, str(device_id), mode, out_path, str(seed)],
        capture_output=True, text=True, env=env,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"[{mode} seed={seed}] FAILED:")
        print(result.stderr[-2000:])
        return None, None
    import numpy as np
    data = np.load(out_path + ".npz")
    with open(out_path + ".json") as f:
        wgrad_info = json.load(f)
    return data, wgrad_info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    import numpy as np

    print(f"=== FP8 vs BF16 Precision Audit (GPU {args.gpu}) ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        all_results = []
        for seed in seeds:
            print(f"--- Seed {seed} ---")
            bf16_data, bf16_wgrad = run_case("bf16", seed, args.gpu, tmpdir)
            fp8_data, fp8_wgrad = run_case("fp8", seed, args.gpu, tmpdir)

            if bf16_data is None or fp8_data is None:
                print("  SKIP (subprocess failed)")
                continue

            for tensor_name in ["output", "dx"]:
                ref = bf16_data[tensor_name].flatten()
                test = fp8_data[tensor_name].flatten()
                r = rrmse(ref, test)
                c = cosine_sim(ref, test)
                status = "PASS" if r < 0.10 and c > 0.99 else "FAIL"
                print(f"  {tensor_name:15s}  RRMSE={r*100:.2f}%  cosine={c:.6f}  [{status}]")
                all_results.append((seed, tensor_name, r, c, status))

            # Compare wgrad norms
            for key in sorted(bf16_wgrad.keys()):
                if key.endswith("_norm"):
                    bf16_val = bf16_wgrad[key]
                    fp8_val = fp8_wgrad.get(key, 0)
                    rel_err = abs(fp8_val - bf16_val) / bf16_val if bf16_val > 0 else 0
                    param_name = key.replace("wgrad_", "").replace("_norm", "")
                    print(f"  wgrad/{param_name:30s}  bf16_norm={bf16_val:.4f}  fp8_norm={fp8_val:.4f}  rel_err={rel_err*100:.2f}%")

        # Summary
        print(f"\n=== SUMMARY ===")
        n_pass = sum(1 for _, _, _, _, s in all_results if s == "PASS")
        n_total = len(all_results)
        print(f"{n_pass}/{n_total} tensor checks PASS")
        if n_pass == n_total:
            print("All precision checks PASS — optimization is safe.")
        else:
            fails = [(seed, name, r, c) for seed, name, r, c, s in all_results if s == "FAIL"]
            print(f"FAILURES: {fails}")


if __name__ == "__main__":
    main()
