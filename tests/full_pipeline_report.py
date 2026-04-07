"""Full pipeline performance / precision / memory report.

Covers:
  1. Per-kernel CUDA-event timing (isolated, warm cache)
  2. End-to-end pipeline timing (BF16 baseline vs FP8 optimized)
  3. Precision: every intermediate tensor vs BF16 gold + FP8 frontier
  4. Peak memory comparison
"""
import os, sys, gc, torch, socket, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    blockscaled_fp8_weight_grad_gemm_fast, fused_transpose_quantize_for_wgrad,
    quantize_and_pack_activation, quantize_activation_blockscaled_fast,
    _gather_isa_packed_scales_kernel, _auto_capacity,
    _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE, _SF_VEC_SIZE,
    _storage_per_batch, _div_up,
)
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════
E, H, I = 8, 3072, 1536
TK = 65536
CAP = TK // E
SEED = 42
WARMUP, ITERS, TRIALS = 10, 20, 5

torch.manual_seed(SEED)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2 * I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc.view(torch.uint8)
cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.full((E,), CAP, dtype=torch.int32, device="cuda").cumsum(0)]).int()
x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)
torch.cuda.synchronize()

host = socket.gethostname()
gpu = torch.cuda.get_device_name(0)


def bench(fn, name):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / ITERS)
    mn = min(times)
    med = sorted(times)[len(times) // 2]
    return mn, med, times


def rrmse(a, b):
    a_f, b_f = a.float(), b.float()
    return ((a_f - b_f).pow(2).mean().sqrt() / b_f.pow(2).mean().sqrt().clamp(min=1e-8)).item()


def cosine(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm()).clamp(min=1e-12)).item()


def byte_match(a, b):
    return (a.view(torch.uint8) == b.view(torch.uint8)).float().mean().item()


print(f"Host: {host}")
print(f"GPU: {gpu}")
print(f"Shape: TK={TK}, H={H}, I={I}, E={E}, CAP={CAP}")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════
# Part 1: Isolated per-kernel timing (CUDA events)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART 1: ISOLATED PER-KERNEL TIMING (CUDA events, min of 5 trials × 20 iters)")
print("=" * 78)

dx = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
pa = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")
z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
torch.cuda.synchronize()

results = {}

# 1a. dout row-quant (separate)
mn, med, _ = bench(lambda: quantize_and_pack_activation(dout), "dout_row_quant")
results["dout_row_quant"] = mn
print(f"  dout row-quant (T,H)→fp8+ISA               {mn:>7.0f}us (med {med:.0f})")

# 1b. scale scatter T→TK
dout_fp8, dout_sc_t = quantize_and_pack_activation(dout)
K_bwd = H; k_tiles = _div_up(K_bwd, _SF_TILE_K)
per_batch = _storage_per_batch(TK, K_bwd)
dout_sc_tk = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
def run_scatter():
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
        dout_sc_t.view(torch.uint8), x_idx, dout_sc_tk, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=32, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
mn, med, _ = bench(run_scatter, "scale_scatter")
results["scale_scatter"] = mn
print(f"  ISA scale scatter T→TK                      {mn:>7.0f}us (med {med:.0f})")

# 1c. z dequant (standalone Triton)
mn, med, _ = bench(lambda: dequantize_blockscaled_fp8(z_fp8, z_sc_u8), "z_dequant")
results["z_dequant"] = mn
print(f"  z dequant fp8→bf16 (standalone Triton)       {mn:>7.0f}us (med {med:.0f})")

# 1d. GemmDGated BF16 C (baseline)
def run_dgated_bf16():
    gemm_dgated(dout_fp8, wf, dx, z_bf16, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_sc_tk.view(torch.float8_e8m0fnu), b_scales=ws)
mn, med, _ = bench(run_dgated_bf16, "GemmDGated_bf16")
results["GemmDGated_bf16_only"] = mn
print(f"  GemmDGated BF16 C (kernel only)              {mn:>7.0f}us (med {med:.0f})")

# 1e. GemmDGated FP8 TMA (Phase 3.1)
gemm_dgated.compile_cache.clear()
def run_dgated_fp8():
    gemm_dgated(dout_fp8, wf, dx, z, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=dout_sc_tk.view(torch.float8_e8m0fnu), b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_u8)
mn, med, _ = bench(run_dgated_fp8, "GemmDGated_fp8_tma")
results["GemmDGated_fp8_tma"] = mn
print(f"  GemmDGated FP8 TMA (Int16 C load)            {mn:>7.0f}us (med {med:.0f})")

# 1f. dout dual-quant (Phase A)
try:
    from tests.bench_warp_dual_quant_v3 import warp_dual_quant_v3
    mn, med, _ = bench(lambda: warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4),
                       "dual_quant")
    results["dual_quant"] = mn
    print(f"  dual-quant dout (row+col, 1 kernel)          {mn:>7.0f}us (med {med:.0f})")
except Exception as e:
    print(f"  dual-quant: SKIPPED ({e})")
    results["dual_quant"] = None

# 1g. y1s transpose-quant (col-only)
mn, med, _ = bench(lambda: fused_transpose_quantize_for_wgrad(pa, E, CAP, I),
                   "y1s_transpose_quant")
results["y1s_transpose_quant"] = mn
print(f"  y1s transpose-quant (col, 32x32)             {mn:>7.0f}us (med {med:.0f})")

# 1h. wgrad BF16 A_idx
from sonicmoe.quack_utils.gemm_interface import gemm_gated_tuned
dw2 = torch.empty(H, I, E, dtype=torch.bfloat16, device="cuda")
dw2_base = dw2.permute(2, 0, 1).contiguous()  # (E, H, I)
def run_wgrad_bf16():
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gd
    # Use the BF16 A_idx path directly
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx)
mn, med, _ = bench(lambda: blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx),
                   "wgrad_full_no_preA")
results["wgrad_full_no_preA"] = mn
print(f"  wgrad FP8 full (A+B quant internal)          {mn:>7.0f}us (med {med:.0f})")

# 1i. wgrad with pre-quantized A (Phase B)
_, _, col_a, col_a_sc = warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4)
torch.cuda.synchronize()
mn, med, _ = bench(lambda: blockscaled_fp8_weight_grad_gemm_fast(
    dout, pa, cu, a_gather_idx=x_idx, pre_quantized_a=(col_a, col_a_sc)),
    "wgrad_preA")
results["wgrad_preA"] = mn
print(f"  wgrad FP8 pre-quantized A (Phase B)          {mn:>7.0f}us (med {med:.0f})")

# ═══════════════════════════════════════════════════════════════
# Part 2: Pipeline timing
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART 2: END-TO-END PIPELINE TIMING")
print("=" * 78)

# Pipeline A: BF16 baseline (z_dequant + GemmDGated_bf16 + wgrad_full)
def pipeline_bf16():
    df, ds_t = quantize_and_pack_activation(dout)
    ds_tk = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
        ds_t.view(torch.uint8), x_idx, ds_tk, TK, src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE, BLOCK_ROWS=32,
        GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
    z_b = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
    gemm_dgated(df, wf, dx, z_b, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=ds_tk.view(torch.float8_e8m0fnu), b_scales=ws)
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx)

mn, med, _ = bench(pipeline_bf16, "pipeline_bf16")
results["pipeline_bf16"] = mn
print(f"  BF16 pipeline (dequant+GemmDGated+wgrad)     {mn:>7.0f}us (med {med:.0f})")

# Pipeline B: FP8 optimized (dual_quant + GemmDGated_fp8 + wgrad_preA)
gemm_dgated.compile_cache.clear()
def pipeline_fp8_opt():
    row_fp8, row_sc, col_a, col_a_sc = warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4)
    ds_tk = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
    _gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
        row_sc.view(torch.uint8), x_idx, ds_tk, TK, src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE, BLOCK_ROWS=32,
        GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
    gemm_dgated(row_fp8, wf, dx, z, pa, None, "swiglu", 128, 128, 1, 1,
                cu_seqlens_m=cu, a_scales=ds_tk.view(torch.float8_e8m0fnu), b_scales=ws,
                preact_fp8=z_fp8, preact_scales=z_sc_u8)
    blockscaled_fp8_weight_grad_gemm_fast(dout, pa, cu, a_gather_idx=x_idx,
                                           pre_quantized_a=(col_a, col_a_sc))

mn, med, _ = bench(pipeline_fp8_opt, "pipeline_fp8_opt")
results["pipeline_fp8_opt"] = mn
print(f"  FP8 optimized (dual+TMA+preA)                {mn:>7.0f}us (med {med:.0f})")

delta = results["pipeline_bf16"] - results["pipeline_fp8_opt"]
ratio = results["pipeline_bf16"] / results["pipeline_fp8_opt"]
print(f"\n  Delta: {delta:+.0f}us ({ratio:.2f}x)")

# ═══════════════════════════════════════════════════════════════
# Part 3: Precision analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART 3: PRECISION ANALYSIS (every intermediate tensor)")
print("=" * 78)

# BF16 reference: all intermediates
dout_fp8_ref, dout_sc_ref = quantize_and_pack_activation(dout)
z_bf16_ref = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
dx_ref = torch.empty_like(dx); pa_ref = torch.empty_like(pa)
gemm_dgated.compile_cache.clear()
gemm_dgated(dout_fp8_ref, wf, dx_ref, z_bf16_ref, pa_ref, None, "swiglu", 128, 128, 1, 1,
            cu_seqlens_m=cu, a_scales=dout_sc_tk.view(torch.float8_e8m0fnu), b_scales=ws)
dw2_ref = blockscaled_fp8_weight_grad_gemm_fast(dout, pa_ref, cu, a_gather_idx=x_idx)
torch.cuda.synchronize()

# FP8 optimized: all intermediates
row_fp8_opt, row_sc_opt, col_a_opt, col_a_sc_opt = warp_dual_quant_v3(
    dout, E, CAP, gather_idx=x_idx, gpb=4)
ds_tk_opt = torch.empty((1, per_batch), dtype=torch.uint8, device="cuda")
_gather_isa_packed_scales_kernel[(_div_up(TK, 32), k_tiles)](
    row_sc_opt.view(torch.uint8), x_idx, ds_tk_opt, TK, src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
    SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE, BLOCK_ROWS=32,
    GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE)
dx_opt = torch.empty_like(dx); pa_opt = torch.empty_like(pa)
gemm_dgated.compile_cache.clear()
gemm_dgated(row_fp8_opt, wf, dx_opt, z, pa_opt, None, "swiglu", 128, 128, 1, 1,
            cu_seqlens_m=cu, a_scales=ds_tk_opt.view(torch.float8_e8m0fnu), b_scales=ws,
            preact_fp8=z_fp8, preact_scales=z_sc_u8)
dw2_opt = blockscaled_fp8_weight_grad_gemm_fast(
    dout, pa_opt, cu, a_gather_idx=x_idx, pre_quantized_a=(col_a_opt, col_a_sc_opt))
torch.cuda.synchronize()

print(f"\n  {'Tensor':<35} {'ByteMatch':>10} {'RRMSE':>12} {'Cosine':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10}")

checks = [
    ("dout_fp8 (row quant)", dout_fp8_ref, row_fp8_opt),
    ("dout ISA scales (scattered)", dout_sc_tk.view(torch.uint8), ds_tk_opt),
    ("dx (GemmDGated output)", dx_ref, dx_opt),
    ("y1s / PostAct", pa_ref, pa_opt),
    ("dw2 (wgrad output)", dw2_ref, dw2_opt),
]
all_pass = True
for name, ref, opt in checks:
    bm = byte_match(ref, opt) * 100
    rr = rrmse(ref, opt)
    co = cosine(ref, opt)
    status = "PASS" if (bm > 99.0 or rr < 0.001) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  {name:<35} {bm:>9.2f}% {rr:>12.6f} {co:>10.6f}  {status}")

print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

# ═══════════════════════════════════════════════════════════════
# Part 4: Memory analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART 4: MEMORY ANALYSIS")
print("=" * 78)

mem = {}
mem["z_bf16"] = TK * 2 * I * 2  # bf16
mem["z_fp8"] = TK * 2 * I * 1   # fp8
mem["z_scales"] = TK * (2 * I // 32) * 1  # uint8
mem["dout_fp8"] = TK * H * 1
mem["dout_scales_T"] = _storage_per_batch(TK, H)
mem["dout_scales_TK"] = _storage_per_batch(TK, H)
mem["dx_output"] = TK * 2 * I * 2
mem["y1s_output"] = TK * I * 2
mem["dw2_output"] = H * I * E * 2
mem["col_fp8_A"] = E * H * CAP * 1
mem["col_scales_A"] = E * _storage_per_batch(H, CAP)
mem["col_fp8_B"] = E * I * CAP * 1
mem["col_scales_B"] = E * _storage_per_batch(I, CAP)

# BF16 pipeline peak: z_bf16 + dout_fp8 + dx + y1s + wgrad intermediates
bf16_peak_est = mem["z_bf16"] + mem["dout_fp8"] + mem["dx_output"] + mem["y1s_output"] + mem["col_fp8_A"] + mem["col_fp8_B"]
# FP8 pipeline peak: z_fp8+scales + dout_fp8 + col_fp8_A + dx + y1s + col_fp8_B
fp8_peak_est = mem["z_fp8"] + mem["z_scales"] + mem["dout_fp8"] + mem["col_fp8_A"] + mem["dx_output"] + mem["y1s_output"] + mem["col_fp8_B"]

print(f"\n  {'Tensor':<35} {'BF16 (MiB)':>12} {'FP8 (MiB)':>12} {'Delta':>10}")
print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

def mib(x): return x / 1024 / 1024

tensors = [
    ("z preactivation",     mem["z_bf16"],   mem["z_fp8"] + mem["z_scales"]),
    ("dout_fp8",            mem["dout_fp8"], mem["dout_fp8"]),
    ("col_fp8_A (wgrad A)", mem["col_fp8_A"], mem["col_fp8_A"]),
    ("col_fp8_B (wgrad B)", mem["col_fp8_B"], mem["col_fp8_B"]),
    ("dx output",           mem["dx_output"], mem["dx_output"]),
    ("y1s output",          mem["y1s_output"], mem["y1s_output"]),
    ("dw2 output",          mem["dw2_output"], mem["dw2_output"]),
]
total_bf16 = total_fp8 = 0
for name, bf16, fp8 in tensors:
    total_bf16 += bf16; total_fp8 += fp8
    d = fp8 - bf16
    print(f"  {name:<35} {mib(bf16):>11.1f} {mib(fp8):>11.1f} {mib(d):>+9.1f}")
print(f"  {'TOTAL':<35} {mib(total_bf16):>11.1f} {mib(total_fp8):>11.1f} {mib(total_fp8-total_bf16):>+9.1f}")

# ═══════════════════════════════════════════════════════════════
# Part 5: Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART 5: EXECUTIVE SUMMARY")
print("=" * 78)

print(f"""
  Host: {host}
  GPU: {gpu}
  Shape: TK={TK}, H={H}, I={I}, E={E}

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    PERFORMANCE (CUDA events, µs)                   │
  ├─────────────────────────────────────┬───────────┬───────────┬──────┤
  │ Kernel                              │   BF16    │   FP8 opt │  Δ   │
  ├─────────────────────────────────────┼───────────┼───────────┼──────┤
  │ dout quant                          │ {results['dout_row_quant']:>7.0f}   │ {results.get('dual_quant', 0) or 0:>7.0f}   │ {(results.get('dual_quant',0) or 0)-results['dout_row_quant']:>+4.0f} │
  │ z dequant                           │ {results['z_dequant']:>7.0f}   │       0   │ {-results['z_dequant']:>+4.0f} │
  │ GemmDGated                          │ {results['GemmDGated_bf16_only']:>7.0f}   │ {results['GemmDGated_fp8_tma']:>7.0f}   │ {results['GemmDGated_fp8_tma']-results['GemmDGated_bf16_only']:>+4.0f} │
  │ wgrad FP8                           │ {results['wgrad_full_no_preA']:>7.0f}   │ {results['wgrad_preA']:>7.0f}   │ {results['wgrad_preA']-results['wgrad_full_no_preA']:>+4.0f} │
  ├─────────────────────────────────────┼───────────┼───────────┼──────┤
  │ END-TO-END PIPELINE                 │ {results['pipeline_bf16']:>7.0f}   │ {results['pipeline_fp8_opt']:>7.0f}   │ {results['pipeline_fp8_opt']-results['pipeline_bf16']:>+4.0f} │
  └─────────────────────────────────────┴───────────┴───────────┴──────┘

  MEMORY: z_bf16 {mib(mem['z_bf16']):.0f}MiB → z_fp8 {mib(mem['z_fp8']+mem['z_scales']):.0f}MiB = -{mib(mem['z_bf16']-mem['z_fp8']-mem['z_scales']):.0f}MiB
  PRECISION: {'ALL PASS — bit-exact or near-exact vs frontier' if all_pass else 'FAILURES DETECTED'}
""")
