"""Per-expert streaming dequant: benchmark and validate.

Instead of dequanting all 384MB z_fp8 → z_bf16 at once, dequant one expert
at a time into a reusable buffer (~48MB), overlapping with GemmDGated.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.gemm_dgated import gemm_dgated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast, quantize_and_pack_activation,
)
from sonicmoe.quack_utils import precompute_weight_fp8_for_direct_fused_dgated
from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

E, H, I = 8, 3072, 1536
TK = 65536  # production size
CAP = TK // E  # 8192 tokens per expert (uniform routing)

torch.manual_seed(42)
dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
w2 = 0.02 * torch.randn(H, I, E, device="cuda", dtype=torch.bfloat16)
z = 0.02 * torch.randn(TK, 2 * I, device="cuda", dtype=torch.bfloat16)
z_fp8, z_sc = quantize_activation_blockscaled_fast(z)
z_sc_u8 = z_sc.view(torch.uint8)

cu = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    torch.full((E,), CAP, dtype=torch.int32, device="cuda").cumsum(0),
]).int()

df, ds = quantize_and_pack_activation(dout)
wf, ws = precompute_weight_fp8_for_direct_fused_dgated(w2)

WARMUP, ITERS, TRIALS = 5, 10, 5


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
    print(f"  {name:<50} min={mn:>7.0f}µs  all={[f'{t:.0f}' for t in times]}")
    return mn


print("=" * 70)
print(f"Per-Expert Streaming Dequant (TK={TK}, E={E}, CAP={CAP})")
print("=" * 70)

# --- Baseline: full dequant + single varlen GemmDGated ---
dx_base = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
pa_base = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")


def run_baseline():
    z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_sc_u8)
    gemm_dgated(
        df, wf, dx_base, z_bf16, pa_base,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
    )


t_base = bench(run_baseline, "Baseline: full dequant + varlen GemmDGated")

# --- Per-expert: streaming dequant with reusable buffer ---
# Allocate small buffer for 1 expert's z_bf16
z_buf = torch.empty(CAP, 2 * I, dtype=torch.bfloat16, device="cuda")
dx_stream = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
pa_stream = torch.empty(TK, I, dtype=torch.bfloat16, device="cuda")

# Pre-create per-expert cu_seqlens
cu_single = torch.tensor([0, CAP], dtype=torch.int32, device="cuda")

dequant_stream = torch.cuda.Stream()


def run_streaming():
    for e in range(E):
        start = e * CAP
        end = (e + 1) * CAP
        # Dequant this expert's z_fp8 → buffer
        dequantize_blockscaled_fp8(
            z_fp8[start:end], z_sc_u8[start:end], out=z_buf
        )
        # GemmDGated for this expert (single-expert varlen)
        gemm_dgated(
            df[start:end].unsqueeze(0),
            wf[e:e + 1],
            dx_stream[start:end].unsqueeze(0),
            z_buf.unsqueeze(0),
            pa_stream[start:end].unsqueeze(0),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            "swiglu", 128, 128, 1, 1,
            a_scales=ds,  # TODO: need per-expert scales
            b_scales=ws,
        )


# This needs per-expert scale slicing which is complex with ISA-packed format.
# For now, test with a simpler approach: sequential dequant + single varlen launch
# but using the smaller buffer.

def run_sequential_dequant():
    """Dequant experts one at a time into rolling buffer, then single varlen launch."""
    # Allocate full z_bf16 but from CAP-sized chunks
    z_bf16_full = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        start = e * CAP
        end = (e + 1) * CAP
        dequantize_blockscaled_fp8(
            z_fp8[start:end], z_sc_u8[start:end],
            out=z_bf16_full[start:end]
        )
    gemm_dgated(
        df, wf, dx_stream, z_bf16_full, pa_stream,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        "swiglu", 128, 128, 1, 1,
        cu_seqlens_m=cu, a_scales=ds, b_scales=ws,
    )


t_seq = bench(run_sequential_dequant, "Sequential dequant (still 384MB, reference)")

# --- Overlapped: dequant[e+1] ‖ GemmDGated[e] on different streams ---
# For this we need per-expert GemmDGated calls, which requires per-expert
# A/B scale slicing. Skip for now — measure just the overhead.

print(f"\n--- Memory Analysis ---")
print(f"Baseline z_bf16: {TK * 2 * I * 2 / 1024**2:.0f} MiB (full 384MB)")
print(f"Per-expert buffer: {CAP * 2 * I * 2 / 1024**2:.0f} MiB (reusable)")
print(f"Saving: {(TK - CAP) * 2 * I * 2 / 1024**2:.0f} MiB")
print(f"\n--- Timing ---")
print(f"Baseline:            {t_base:.0f}µs")
print(f"Sequential dequant:  {t_seq:.0f}µs (Δ={t_seq - t_base:+.0f}µs)")
