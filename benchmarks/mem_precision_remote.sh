#!/bin/bash
set -e
export VSCODE_SHELL_INTEGRATION=0
WORK=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
OUT=$WORK/benchmarks/clean_results
mkdir -p $OUT

FP8_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python
BF16_PY=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/python

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ==================== Memory Breakdown ====================
echo ""
echo "=== Memory Breakdown ==="

# BF16 memory (subprocess isolated)
cat > /tmp/_mem_bf16.py << 'PYEOF'
import torch, os, sys, gc, json
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()

T, H, I, E, K = 8192, 3072, 1536, 8, 8
mem_base = torch.cuda.memory_allocated() / 1024**2

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
moe.train()
mem_model = torch.cuda.memory_allocated() / 1024**2

torch.manual_seed(42)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
dout = 0.02 * torch.randn_like(x)
mem_input = torch.cuda.memory_allocated() / 1024**2

# Warmup (2 iters to stabilize caches)
for i in range(2):
    with enable_quack_gemm(True):
        out, loss = moe(x)
        out.backward(dout)
    moe.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()

# Measured iteration
torch.manual_seed(99)
x2 = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
dout2 = 0.02 * torch.randn_like(x2)
mem_before_fwd = torch.cuda.memory_allocated() / 1024**2
with enable_quack_gemm(True):
    out, loss = moe(x2)
torch.cuda.synchronize()
mem_fwd_peak = torch.cuda.max_memory_allocated() / 1024**2
mem_after_fwd = torch.cuda.memory_allocated() / 1024**2

torch.cuda.reset_peak_memory_stats()
out.backward(dout2)
torch.cuda.synchronize()
mem_bwd_peak = torch.cuda.max_memory_allocated() / 1024**2
mem_after_bwd = torch.cuda.memory_allocated() / 1024**2

result = {
    "mode": "BF16",
    "mem_base_MiB": round(mem_base, 2),
    "mem_model_MiB": round(mem_model, 2),
    "mem_input_MiB": round(mem_input, 2),
    "mem_before_fwd_MiB": round(mem_before_fwd, 2),
    "mem_fwd_peak_MiB": round(mem_fwd_peak, 2),
    "mem_after_fwd_MiB": round(mem_after_fwd, 2),
    "mem_bwd_peak_MiB": round(mem_bwd_peak, 2),
    "mem_after_bwd_MiB": round(mem_after_bwd, 2),
}
print(json.dumps(result, indent=2))
PYEOF

cat > /tmp/_mem_fp8.py << 'PYEOF'
import torch, os, sys, gc, json
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_DOUBLE_QUANT"] = "1"
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()

T, H, I, E, K = 8192, 3072, 1536, 8, 8
mem_base = torch.cuda.memory_allocated() / 1024**2

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
moe.train()
mem_model = torch.cuda.memory_allocated() / 1024**2

torch.manual_seed(42)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
dout = 0.02 * torch.randn_like(x)
mem_input = torch.cuda.memory_allocated() / 1024**2

# Warmup
for i in range(2):
    with enable_quack_gemm(True):
        out, loss = moe(x, use_fp8=True)
        out.backward(dout)
    moe.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()

# Measured iteration
torch.manual_seed(99)
x2 = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
dout2 = 0.02 * torch.randn_like(x2)
mem_before_fwd = torch.cuda.memory_allocated() / 1024**2
with enable_quack_gemm(True):
    out, loss = moe(x2, use_fp8=True)
torch.cuda.synchronize()
mem_fwd_peak = torch.cuda.max_memory_allocated() / 1024**2
mem_after_fwd = torch.cuda.memory_allocated() / 1024**2

torch.cuda.reset_peak_memory_stats()
out.backward(dout2)
torch.cuda.synchronize()
mem_bwd_peak = torch.cuda.max_memory_allocated() / 1024**2
mem_after_bwd = torch.cuda.memory_allocated() / 1024**2

result = {
    "mode": "FP8",
    "mem_base_MiB": round(mem_base, 2),
    "mem_model_MiB": round(mem_model, 2),
    "mem_input_MiB": round(mem_input, 2),
    "mem_before_fwd_MiB": round(mem_before_fwd, 2),
    "mem_fwd_peak_MiB": round(mem_fwd_peak, 2),
    "mem_after_fwd_MiB": round(mem_after_fwd, 2),
    "mem_bwd_peak_MiB": round(mem_bwd_peak, 2),
    "mem_after_bwd_MiB": round(mem_after_bwd, 2),
}
print(json.dumps(result, indent=2))
PYEOF

CUDA_VISIBLE_DEVICES=2 $BF16_PY /tmp/_mem_bf16.py 2>&1 | tee $OUT/mem_bf16.txt
echo ""
CUDA_VISIBLE_DEVICES=3 $FP8_PY /tmp/_mem_fp8.py 2>&1 | tee $OUT/mem_fp8.txt

# ==================== Precision Breakdown ====================
echo ""
echo "=== Precision Breakdown ==="

cat > /tmp/_precision_compare.py << 'PYEOF'
import torch, os, sys, json, subprocess, tempfile

def run_variant(mode, seed, gpu):
    """Run one variant in a subprocess for isolation."""
    script = f'''
import torch, os, sys, json
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/{"official/" if mode == "bf16" else ""}sonic-moe")
os.environ["USE_QUACK_GEMM"] = "1"
{"" if mode == "bf16" else 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"'}
{"" if mode == "bf16" else 'os.environ["SONIC_MOE_FP8_DOUBLE_QUANT"] = "1"'}
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

T, H, I, E, K = 8192, 3072, 1536, 8, 8
device = torch.device("cuda")
torch.manual_seed({seed})
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
moe.train()

# Save initial weights
w_state = {{k: v.clone() for k, v in moe.named_parameters()}}

torch.manual_seed({seed} + 1000)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
dout = 0.02 * torch.randn_like(x)

with enable_quack_gemm(True):
    out, loss = moe(x{", use_fp8=True" if mode == "fp8" else ""})
out.backward(dout)
torch.cuda.synchronize()

result = {{}}
result["out"] = out.detach().float().cpu()
result["dx"] = x.grad.detach().float().cpu()
result["loss"] = loss.detach().float().cpu()
# Collect grad norms per param group
w1_grads, w2_grads, wr_grads = [], [], []
for name, p in moe.named_parameters():
    if p.grad is not None:
        if "w1" in name or "w3" in name:
            w1_grads.append(p.grad.detach().float().cpu().flatten())
        elif "w2" in name:
            w2_grads.append(p.grad.detach().float().cpu().flatten())
        elif "gate" in name or "router" in name:
            wr_grads.append(p.grad.detach().float().cpu().flatten())
if w1_grads: result["dw1"] = torch.cat(w1_grads)
if w2_grads: result["dw2"] = torch.cat(w2_grads)
if wr_grads: result["drouter"] = torch.cat(wr_grads)

torch.save(result, "/tmp/_prec_{mode}_{seed}.pt")
print("saved")
'''
    py = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/" + \
         ("official_bf16" if mode == "bf16" else "xfer") + "/bin/python"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run([py, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"FAIL {mode} seed={seed}: {r.stderr[-500:]}")
        return False
    return True

def compute_metrics(bf16_t, fp8_t):
    bf16_f = bf16_t.flatten().double()
    fp8_f = fp8_t.flatten().double()
    diff = (bf16_f - fp8_f)
    rms_diff = diff.pow(2).mean().sqrt()
    rms_ref = bf16_f.pow(2).mean().sqrt()
    rrmse = (rms_diff / rms_ref.clamp(min=1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(bf16_f.unsqueeze(0), fp8_f.unsqueeze(0)).item()
    abs_diff = diff.abs()
    abs_ref = bf16_f.abs().clamp(min=1e-12)
    max_rel = (abs_diff / abs_ref).max().item()
    return {"rrmse": rrmse, "cosine": cos, "max_rel_err": max_rel}

seeds = [42, 123, 777]
variables = ["out", "dx", "dw1", "dw2", "drouter", "loss"]
all_results = []

for seed in seeds:
    print(f"\\n--- Seed {seed} ---")
    ok_bf16 = run_variant("bf16", seed, 5)
    ok_fp8 = run_variant("fp8", seed, 7)
    if not (ok_bf16 and ok_fp8):
        continue
    bf16_data = torch.load(f"/tmp/_prec_bf16_{seed}.pt", weights_only=True)
    fp8_data = torch.load(f"/tmp/_prec_fp8_{seed}.pt", weights_only=True)
    for var in variables:
        if var in bf16_data and var in fp8_data:
            m = compute_metrics(bf16_data[var], fp8_data[var])
            m["seed"] = seed
            m["variable"] = var
            all_results.append(m)
            status = "PASS" if m["rrmse"] < 0.10 and m["cosine"] > 0.99 else "FAIL"
            print(f"  {var:10s}: RRMSE={m['rrmse']:.4f}  cos={m['cosine']:.6f}  maxrel={m['max_rel_err']:.2f}  [{status}]")

print(f"\\n=== Summary: {len(all_results)} measurements ===")
pass_count = sum(1 for r in all_results if r["rrmse"] < 0.10 and r["cosine"] > 0.99)
print(f"PASS: {pass_count}/{len(all_results)}")
with open("/tmp/_precision_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
PYEOF

$FP8_PY /tmp/_precision_compare.py 2>&1 | tee $OUT/precision.txt

echo ""
echo "=== ALL BENCHMARKS DONE ==="
