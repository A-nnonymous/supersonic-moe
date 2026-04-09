#!/usr/bin/env python3
"""Session 42 Complete Benchmark: Performance + Memory + Precision.

Runs 3 modes (BF16, FP8, FP8+stash) in separate subprocesses for isolation.
Produces JSON report consumed by visualization scripts.

Usage:
    ssh <idle_host> 'cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe && \
        source ../envs/xfer/bin/activate && \
        CUDA_VISIBLE_DEVICES=1 python tools/session42_benchmark.py'
"""
import json, os, subprocess, sys, time, gc

VENV_PYTHON = sys.executable
DEVICE = "cuda"
SHAPE = {"T": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8}
SEEDS = [42, 123, 777]
WARMUP_ITERS = 5
TIMING_ITERS = 30

# ─── Subprocess template ───────────────────────────────────────────────
_BENCH_SCRIPT = r'''
import torch, gc, os, json, time
os.environ["USE_QUACK_GEMM"] = "1"
MODE = os.environ["BENCH_MODE"]
if MODE != "bf16":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE
from sonicmoe.functional.utils import enable_quack_gemm, enable_fp8
from sonicmoe.enums import ActivationType

MiB = 1024**2
SEED = int(os.environ.get("SEED", "42"))
T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
WARMUP, ITERS = {WARMUP}, {ITERS}

torch.manual_seed(SEED)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

def run_iter(use_stash=False):
    if MODE != "bf16":
        moe.refresh_fp8_shadow_weights()
    if use_stash:
        moe.stash_bf16_to_cpu()
    if MODE == "bf16":
        with enable_quack_gemm(True):
            out, _ = moe(x)
    else:
        with enable_quack_gemm(True), enable_fp8():
            out, _ = moe(x, use_fp8=True)
    out.backward(dout)
    if use_stash:
        moe.unstash_bf16()
    x.grad = None
    moe.zero_grad(set_to_none=True)

use_stash = (MODE == "fp8_stash")

# Warmup
for _ in range(WARMUP):
    run_iter(use_stash)
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ── Memory ──
if MODE != "bf16" and use_stash:
    moe.refresh_fp8_shadow_weights()
    moe.stash_bf16_to_cpu()
elif MODE != "bf16":
    moe.refresh_fp8_shadow_weights()

base = torch.cuda.memory_allocated() / MiB
torch.cuda.reset_peak_memory_stats()
if MODE == "bf16":
    with enable_quack_gemm(True): out, _ = moe(x)
else:
    with enable_quack_gemm(True), enable_fp8(): out, _ = moe(x, use_fp8=True)
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / MiB
fwd_end = torch.cuda.memory_allocated() / MiB
torch.cuda.reset_peak_memory_stats()
out.backward(dout); torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / MiB
bwd_end = torch.cuda.memory_allocated() / MiB

if use_stash:
    moe.unstash_bf16()
x.grad = None; moe.zero_grad(set_to_none=True)
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ── Timing (CUDA events) ──
fwd_times, bwd_times = [], []
for _ in range(ITERS):
    if MODE != "bf16":
        moe.refresh_fp8_shadow_weights()
    if use_stash:
        moe.stash_bf16_to_cpu()
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    e0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    s0.record()
    if MODE == "bf16":
        with enable_quack_gemm(True): out, _ = moe(x)
    else:
        with enable_quack_gemm(True), enable_fp8(): out, _ = moe(x, use_fp8=True)
    e0.record()
    s1.record()
    out.backward(dout)
    e1.record()
    torch.cuda.synchronize()
    fwd_times.append(s0.elapsed_time(e0))
    bwd_times.append(s1.elapsed_time(e1))
    if use_stash:
        moe.unstash_bf16()
    x.grad = None; moe.zero_grad(set_to_none=True)

# ── Precision (vs saved BF16 reference) ──
import numpy as np
tmpdir = os.environ.get("TMPDIR", "/tmp")
prec = {{}}
if MODE != "bf16":
    bf16_out_path = f"{{tmpdir}}/bench_bf16_out_{{SEED}}.npy"
    bf16_dx_path = f"{{tmpdir}}/bench_bf16_dx_{{SEED}}.npy"
    if os.path.exists(bf16_out_path):
        if MODE != "bf16" and use_stash:
            moe.refresh_fp8_shadow_weights()
            moe.stash_bf16_to_cpu()
        elif MODE != "bf16":
            moe.refresh_fp8_shadow_weights()
        if MODE == "bf16":
            with enable_quack_gemm(True): out_p, _ = moe(x)
        else:
            with enable_quack_gemm(True), enable_fp8(): out_p, _ = moe(x, use_fp8=True)
        out_p.backward(dout)
        out_bf16 = torch.from_numpy(np.load(bf16_out_path)).cuda()
        dx_bf16 = torch.from_numpy(np.load(bf16_dx_path)).cuda()
        rr = lambda a,b: ((a-b).norm()/b.norm()).item()*100
        co = lambda a,b: torch.corrcoef(torch.stack([a.flatten(),b.flatten()]))[0,1].item()
        prec = dict(
            out_rrmse=round(rr(out_p.detach().float(), out_bf16), 2),
            out_corr=round(co(out_p.detach().float(), out_bf16), 4),
            dx_rrmse=round(rr(x.grad.detach().float(), dx_bf16), 2),
            dx_corr=round(co(x.grad.detach().float(), dx_bf16), 4),
        )
        if use_stash:
            moe.unstash_bf16()
        x.grad = None; moe.zero_grad(set_to_none=True)
else:
    # Save BF16 reference for other modes
    import numpy as np
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    if MODE != "bf16" and use_stash:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()
    elif MODE != "bf16":
        moe.refresh_fp8_shadow_weights()
    if MODE == "bf16":
        with enable_quack_gemm(True): out_p, _ = moe(x)
    else:
        with enable_quack_gemm(True), enable_fp8(): out_p, _ = moe(x, use_fp8=True)
    out_p.backward(dout)
    np.save(f"{{tmpdir}}/bench_bf16_out_{{SEED}}.npy", out_p.detach().float().cpu().numpy())
    np.save(f"{{tmpdir}}/bench_bf16_dx_{{SEED}}.npy", x.grad.detach().float().cpu().numpy())
    if use_stash:
        moe.unstash_bf16()
    x.grad = None; moe.zero_grad(set_to_none=True)

# Sort and drop outliers (top/bottom 10%)
def trimmed_mean(vals, pct=0.1):
    s = sorted(vals)
    n = len(s)
    lo, hi = int(n*pct), int(n*(1-pct))
    return sum(s[lo:hi]) / max(hi-lo, 1)

result = dict(
    mode=MODE, seed=SEED,
    memory=dict(base=round(base,1), fwd_peak=round(fwd_peak,1), fwd_end=round(fwd_end,1),
                bwd_peak=round(bwd_peak,1), bwd_end=round(bwd_end,1)),
    timing=dict(fwd_ms=round(trimmed_mean(fwd_times), 3),
                bwd_ms=round(trimmed_mean(bwd_times), 3),
                total_ms=round(trimmed_mean([f+b for f,b in zip(fwd_times, bwd_times)]), 3),
                fwd_all=[round(t,3) for t in fwd_times],
                bwd_all=[round(t,3) for t in bwd_times]),
    precision=prec,
    device=torch.cuda.get_device_name(0),
)
print("RESULT:" + json.dumps(result))
'''.format(**SHAPE, WARMUP=WARMUP_ITERS, ITERS=TIMING_ITERS)


def run_mode(mode: str, seed: int, tmpdir: str, gpu: str) -> dict:
    env = {k: v for k, v in os.environ.items() if "FP8" not in k}
    env.update(BENCH_MODE=mode, SEED=str(seed), CUDA_VISIBLE_DEVICES=gpu, TMPDIR=tmpdir)
    # Force our fork over any stale editable installs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = project_root + (":" + env.get("PYTHONPATH", ""))
    if mode != "bf16":
        env["SONIC_MOE_FP8_MODE"] = "perf"
    r = subprocess.run(
        [VENV_PYTHON, "-c", _BENCH_SCRIPT],
        capture_output=True, text=True, env=env, timeout=600,
    )
    for line in r.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    raise RuntimeError(f"{mode}/seed={seed} failed:\n{r.stderr[-800:]}")


def main():
    import argparse, tempfile
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="1", help="GPU index")
    p.add_argument("--output", default="session42_benchmark.json")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        results = []
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Seed {seed}")
            print(f"{'='*60}")
            # BF16 first (saves reference)
            for mode in ["bf16", "fp8", "fp8_stash"]:
                print(f"  Running {mode}...", end=" ", flush=True)
                t0 = time.time()
                r = run_mode(mode, seed, tmpdir, args.gpu)
                dt = time.time() - t0
                results.append(r)
                mem = r["memory"]
                tim = r["timing"]
                print(f"done ({dt:.0f}s) — fwd_peak={mem['fwd_peak']:.0f}M bwd_peak={mem['bwd_peak']:.0f}M "
                      f"total={tim['total_ms']:.1f}ms")
                if r.get("precision"):
                    pr = r["precision"]
                    print(f"    precision: out_rrmse={pr['out_rrmse']:.2f}% dx_rrmse={pr['dx_rrmse']:.2f}%")

    report = {
        "schema": "session42-benchmark-v1",
        "shape": SHAPE,
        "seeds": SEEDS,
        "warmup_iters": WARMUP_ITERS,
        "timing_iters": TIMING_ITERS,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY (trimmed mean across seeds)")
    print(f"{'='*80}")
    modes = ["bf16", "fp8", "fp8_stash"]
    for key in ["fwd_peak", "bwd_peak", "base", "fwd_end", "bwd_end"]:
        vals = {}
        for mode in modes:
            entries = [r for r in results if r["mode"] == mode]
            vals[mode] = sum(r["memory"][key] for r in entries) / len(entries)
        bf16 = vals["bf16"]
        print(f"  {key:12s}  BF16={bf16:7.1f}M  FP8={vals['fp8']:7.1f}M({vals['fp8']-bf16:+.1f})  "
              f"Stash={vals['fp8_stash']:7.1f}M({vals['fp8_stash']-bf16:+.1f})")

    print()
    for key in ["fwd_ms", "bwd_ms", "total_ms"]:
        vals = {}
        for mode in modes:
            entries = [r for r in results if r["mode"] == mode]
            vals[mode] = sum(r["timing"][key] for r in entries) / len(entries)
        bf16 = vals["bf16"]
        print(f"  {key:12s}  BF16={bf16:7.3f}  FP8={vals['fp8']:7.3f}({vals['fp8']/bf16:.3f}x)  "
              f"Stash={vals['fp8_stash']:7.3f}({vals['fp8_stash']/bf16:.3f}x)")


if __name__ == "__main__":
    main()
