"""
Async GPU-side profiling — NO artificial synchronization between phases.

Methodology:
  - CUDA events are stream-ordered markers: zero-overhead, no pipeline stall.
  - ev_fwd_start → moe(x) → ev_fwd_end → out.sum().backward() → ev_bwd_end
  - No sync() between fwd/bwd. The GPU pipeline runs naturally.
  - event.elapsed_time() returns GPU-wall-clock between two events.
  - "bubble" = total - fwd - bwd  (real pipeline gap from CPU dispatch latency).
  - Only ONE sync() per iteration, after ev_bwd_end, to read back times.

For nsys: NVTX push/pop are CPU-side annotations; they do NOT block the GPU.
  We record them purely so that nsys can correlate CPU dispatch → GPU kernels.
"""
import os, sys, json, statistics
os.environ["USE_QUACK_GEMM"] = "1"
for _k in ["PADDLE_ELASTIC_JOB_ID","PADDLE_TRAINER_ENDPOINTS",
           "DISTRIBUTED_TRAINER_ENDPOINTS","FLAGS_START_PORT",
           "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"; os.environ["PADDLE_TRAINERS_NUM"] = "1"

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

import torch, torch.cuda.nvtx as nvtx
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
import sonicmoe.functional as F_mod

T, H, I, E, K = 4096, 4096, 1024, 128, 8
WARMUP = 30
ITERS  = 100

# ── deterministic 128-aligned routing ──────────────────────────────
_uni_idx = _uni_sc = None
def _build_uniform(dev):
    global _uni_idx, _uni_sc
    if _uni_idx is not None: return
    tok = torch.arange(T, device=dev).unsqueeze(1)
    off = torch.arange(K, device=dev).unsqueeze(0)
    _uni_idx = ((tok * K + off) % E).to(torch.int32)
    _uni_sc  = torch.full((T, K), 1.0/K, dtype=torch.float32, device=dev)

class _UniformRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, E_, K_):
        _build_uniform(logits.device)
        ctx.save_for_backward(_uni_sc, _uni_idx)
        ctx.E = E_; ctx.dtype = logits.dtype
        return _uni_sc.clone(), _uni_idx.clone()
    @staticmethod
    def backward(ctx, g_s, _):
        s, _ = ctx.saved_tensors
        return torch.zeros(s.size(0), ctx.E, dtype=ctx.dtype, device=s.device), None, None

F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter

# ── model ──────────────────────────────────────────────────────────
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to("cuda", torch.bfloat16)
x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

def set_fp8():
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
    os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
    os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
    F_mod._ALIGNMENT_ASSUMED = True

def set_bf16():
    os.environ["SONIC_MOE_FP8_MODE"] = "off"
    for k in ("SONIC_MOE_FP8_ASSUME_ALIGNED",
              "SONIC_MOE_FP8_FUSED_SWIGLU_QUANT",
              "SONIC_MOE_FP8_SAVE_Z_FP8"):
        os.environ.pop(k, None)
    F_mod._ALIGNMENT_ASSUMED = False

def zero_g():
    for p in moe.parameters(): p.grad = None

# ── core measurement (fully async, one sync at end) ───────────────
def run_iter(tag=None):
    """Return (fwd_ms, bwd_ms, total_ms) from GPU events. No extra sync."""
    e0 = torch.cuda.Event(enable_timing=True)  # fwd start
    e1 = torch.cuda.Event(enable_timing=True)  # fwd end / pre-bwd
    e2 = torch.cuda.Event(enable_timing=True)  # bwd end

    zero_g()
    x = x_base.clone().requires_grad_(True)

    if tag: nvtx.range_push(f"{tag}_fwd")
    e0.record()
    out, _ = moe(x)
    e1.record()                       # stream-ordered: fires after last fwd kernel
    if tag: nvtx.range_pop()

    # NO synchronize here — backward launches queue behind forward naturally
    if tag: nvtx.range_push(f"{tag}_bwd")
    loss = out.sum()
    loss.backward()
    e2.record()                       # fires after last bwd kernel
    if tag: nvtx.range_pop()

    torch.cuda.synchronize()          # single sync to read back times
    fwd   = e0.elapsed_time(e1)
    bwd   = e1.elapsed_time(e2)
    total = e0.elapsed_time(e2)
    return fwd, bwd, total

# ── warmup ─────────────────────────────────────────────────────────
print(f"[GPU {gpu_id}] warmup BF16 ×{WARMUP}")
set_bf16()
for _ in range(WARMUP): run_iter()

print(f"[GPU {gpu_id}] warmup FP8 ×{WARMUP}")
set_fp8()
for _ in range(WARMUP): run_iter()

# ── timed runs ─────────────────────────────────────────────────────
print(f"[GPU {gpu_id}] bench BF16 ×{ITERS}")
set_bf16()
bf16 = [run_iter() for _ in range(ITERS)]

print(f"[GPU {gpu_id}] bench FP8 ×{ITERS}")
set_fp8()
fp8 = [run_iter() for _ in range(ITERS)]

# ── nsys-friendly profiled iters (no extra sync, just NVTX tags) ──
print(f"[GPU {gpu_id}] nsys capture ×5 each")
torch.cuda.cudart().cudaProfilerStart()
set_bf16()
for i in range(5):
    nvtx.range_push(f"BF16_iter{i}")
    run_iter(tag=f"BF16_i{i}")
    nvtx.range_pop()
set_fp8()
for i in range(5):
    nvtx.range_push(f"FP8_iter{i}")
    run_iter(tag=f"FP8_i{i}")
    nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

# ── precision ──────────────────────────────────────────────────────
print(f"[GPU {gpu_id}] precision check")
set_bf16(); zero_g()
xr = x_base.clone().requires_grad_(True)
or_, _ = moe(xr); or_.sum().backward()
dx_r, dw_r = xr.grad.clone(), moe.c_fc.weight.grad.clone()

set_fp8(); zero_g()
xf = x_base.clone().requires_grad_(True)
of_, _ = moe(xf); of_.sum().backward()
dx_f, dw_f = xf.grad.clone(), moe.c_fc.weight.grad.clone()

def relrmse(a, b):
    return (((a.float()-b.float())**2).mean().sqrt() /
            (b.float()**2).mean().sqrt()).item()
def corr(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a_, b_]))[0,1].item()

prec = dict(
    out_relrmse=relrmse(of_, or_), out_corr=corr(of_, or_),
    dx_relrmse=relrmse(dx_f, dx_r), dx_corr=corr(dx_f, dx_r),
    dw_relrmse=relrmse(dw_f, dw_r), dw_corr=corr(dw_f, dw_r),
)

# ── stats ──────────────────────────────────────────────────────────
def st(vals):
    s = sorted(vals); n = len(s)
    return dict(med=statistics.median(s),
                p5=s[max(0,int(n*.05))], p95=s[min(n-1,int(n*.95))],
                mean=statistics.mean(s),
                std=statistics.stdev(s) if n>1 else 0)

bf16_f = st([t[0] for t in bf16])
bf16_b = st([t[1] for t in bf16])
bf16_t = st([t[2] for t in bf16])
fp8_f  = st([t[0] for t in fp8])
fp8_b  = st([t[1] for t in fp8])
fp8_t  = st([t[2] for t in fp8])

# bubble = total - fwd - bwd (pipeline gap)
bf16_bubble = [t[2]-t[0]-t[1] for t in bf16]
fp8_bubble  = [t[2]-t[0]-t[1] for t in fp8]

result = dict(
    gpu=gpu_id, warmup=WARMUP, iters=ITERS,
    bf16=dict(fwd=bf16_f, bwd=bf16_b, total=bf16_t,
              bubble=st(bf16_bubble)),
    fp8=dict(fwd=fp8_f, bwd=fp8_b, total=fp8_t,
             bubble=st(fp8_bubble)),
    speedup=dict(
        fwd=bf16_f["med"]/fp8_f["med"],
        bwd=bf16_b["med"]/fp8_b["med"],
        total=bf16_t["med"]/fp8_t["med"]),
    precision=prec,
)

with open(f"/tmp/gpu_async_result_{gpu_id}.json","w") as f:
    json.dump(result, f, indent=2)

# ── display ────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"GPU {gpu_id} | {WARMUP}w {ITERS}i | async CUDA-event GPU timing (no artificial sync)")
print(f"{'='*85}")
print(f"{'':18} {'BF16 med':>10} {'p5':>8} {'p95':>8}  {'FP8 med':>10} {'p5':>8} {'p95':>8}  {'Speed':>7}")
print(f"{'-'*85}")
for label, b, f_ in [("Forward (ms)", bf16_f, fp8_f),
                      ("Backward (ms)", bf16_b, fp8_b),
                      ("Total (ms)", bf16_t, fp8_t)]:
    sp = b["med"]/f_["med"]
    print(f"{label:<18} {b['med']:>10.3f} {b['p5']:>8.3f} {b['p95']:>8.3f}"
          f"  {f_['med']:>10.3f} {f_['p5']:>8.3f} {f_['p95']:>8.3f}  {sp:>6.2f}x")

bf16_bub = st(bf16_bubble); fp8_bub = st(fp8_bubble)
print(f"\n{'Pipeline bubble':18} {bf16_bub['med']:>10.3f} {'':>8} {'':>8}"
      f"  {fp8_bub['med']:>10.3f}")
print(f"{'Bubble % of total':18} {bf16_bub['med']/bf16_t['med']*100:>9.1f}%"
      f" {'':>8} {'':>8}"
      f"  {fp8_bub['med']/fp8_t['med']*100:>9.1f}%")

print(f"\nPrecision (FP8 vs BF16):")
for k,v in prec.items():
    if 'rmse' in k: print(f"  {k:<18} {v*100:>7.2f}%  {'✓' if v<0.10 else '✗'}")
    else:           print(f"  {k:<18} {v:>7.5f}  {'✓' if v>0.99 else '✗'}")
print("Done.")
