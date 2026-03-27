"""Rigorous memory analysis: FP8 vs BF16, excluding master weights."""
import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

T,H,I,E,K = 8192,4096,1024,128,8
TK = T * K  # 65536

def mib(b): return b / (1024**2)

print("=" * 70)
print(f"Memory Analysis: T={T}, H={H}, I={I}, E={E}, K={K}, TK={TK}")
print("=" * 70)

# Master weights (SAME for both bf16 and fp8 training)
w1_master = E * 2*I * H * 2  # bf16 (2I, H, E)
w2_master = E * H * I * 2    # bf16 (H, I, E)
router_master = E * H * 2    # bf16 (E, H)
master_total = w1_master + w2_master + router_master

# Optimizer state (Adam: 2x fp32 per param, or mixed precision: fp32 master + fp32 m + fp32 v)
opt_state = (w1_master + w2_master + router_master) * 2 * 3  # m, v, master all fp32 = 3x bf16 param size in fp32
# Actually for mixed precision: master=fp32(4B), m=fp32(4B), v=fp32(4B) per param
# vs bf16 param which is 2B, so opt state = (4+4+4)/2 * param_size_bf16 = 6x

print("\n--- Shared (identical for BF16 and FP8 training) ---")
print(f"w1 master (bf16):     {mib(w1_master):.0f} MiB")
print(f"w2 master (bf16):     {mib(w2_master):.0f} MiB")
print(f"Router master (bf16): {mib(router_master):.1f} MiB")
print(f"Total master weights: {mib(master_total):.0f} MiB")

# FP8 weight cache (ONLY in FP8 training, not in BF16)
w1_fp8_data = E * 2*I * H * 1  # float8_e4m3fn
w1_fp8_scales_raw = E * (2*I // 32) * H * 1  # E8M0 1x32 blockscaled
w1_fp8_isa = E * ((2*I // 128 + 1) * (H // 128 + 1)) * 512  # ISA packed
w2_fp8_data = E * H * I * 1
w2_fp8_scales_raw = E * (H // 32) * I * 1
w2_fp8_isa = E * ((H // 128 + 1) * (I // 128 + 1)) * 512

print("\n--- FP8 weight cache (additional, perf mode only) ---")
print(f"w1 fp8 data:          {mib(w1_fp8_data):.0f} MiB")
print(f"w1 ISA-packed scales: {mib(w1_fp8_isa):.0f} MiB")
print(f"w2 fp8 data:          {mib(w2_fp8_data):.0f} MiB")
print(f"w2 ISA-packed scales: {mib(w2_fp8_isa):.0f} MiB")
w_cache_total = w1_fp8_data + w1_fp8_isa + w2_fp8_data + w2_fp8_isa
print(f"Total FP8 weight cache: {mib(w_cache_total):.0f} MiB")

# Forward activations (training mode — saved for backward)
print("\n--- Forward activations (saved for backward) ---")
# BF16 forward saves: z (for SwiGLU backward), y1 (via gemm_gated postact)
z_bf16 = TK * 2*I * 2  # (TK, 2I) bf16 — preact, saved
y1_bf16 = TK * I * 2   # (TK, I) bf16 — postact
x_saved = T * H * 2    # original x for backward weight grad

print(f"  z (preact, saved):    {mib(z_bf16):.0f} MiB (bf16, both paths)")
print(f"  y1 (postact):         {mib(y1_bf16):.0f} MiB (bf16)")

# FP8: y1 can be fp8 with fused SwiGLU+quant (separate path only)
y1_fp8 = TK * I * 1
y1_fp8_scales = TK * (I // 32) * 1
print(f"  y1 (fp8+scales):      {mib(y1_fp8 + y1_fp8_scales):.0f} MiB (fp8 separate path)")

# Transient activations during forward
print("\n--- Transient activations (not saved, peak only) ---")
x_gathered_bf16 = TK * H * 2  # gathered x for GEMM
x_fp8 = TK * H * 1
x_scales_isa = (TK // 128 + 1) * (H // 128 + 1) * 512

print(f"  x_gathered (bf16):    {mib(x_gathered_bf16):.0f} MiB")
print(f"  x_fp8 + scales:       {mib(x_fp8 + x_scales_isa):.0f} MiB")
print(f"  Savings:              {mib(x_gathered_bf16 - x_fp8 - x_scales_isa):.0f} MiB")

# Summary: delta memory (excluding master weights + optimizer)
print("\n" + "=" * 70)
print("SUMMARY: Memory delta (excluding shared master weights + optimizer)")
print("=" * 70)

bf16_fwd_peak = x_gathered_bf16 + z_bf16 + y1_bf16  # gather + preact + postact
fp8_fused_fwd_peak = x_fp8 + x_scales_isa + z_bf16 + y1_bf16 + w_cache_total  # +cache
fp8_separate_fwd_peak = x_fp8 + x_scales_isa + z_bf16 + y1_fp8 + y1_fp8_scales + w_cache_total

print(f"\nBF16 forward peak activations:     {mib(bf16_fwd_peak):.0f} MiB")
print(f"FP8 fused forward peak:            {mib(fp8_fused_fwd_peak):.0f} MiB (+{mib(fp8_fused_fwd_peak - bf16_fwd_peak):.0f} from weight cache)")
print(f"FP8 separate forward peak:         {mib(fp8_separate_fwd_peak):.0f} MiB")

# Without weight cache (mem mode)
fp8_mem_fwd_peak = x_fp8 + x_scales_isa + z_bf16 + y1_bf16  # no cache, re-quantize each time
print(f"FP8 fused forward (mem mode):      {mib(fp8_mem_fwd_peak):.0f} MiB ({mib(fp8_mem_fwd_peak - bf16_fwd_peak):+.0f} vs BF16)")

print(f"\n--- Activation memory savings (fp8 mem mode vs bf16) ---")
print(f"x: bf16 {mib(x_gathered_bf16):.0f} → fp8 {mib(x_fp8 + x_scales_isa):.0f} MiB = {mib(x_gathered_bf16 - x_fp8 - x_scales_isa):.0f} MiB saved")
print(f"y1: bf16 {mib(y1_bf16):.0f} → fp8+scales {mib(y1_fp8 + y1_fp8_scales):.0f} MiB = {mib(y1_bf16 - y1_fp8 - y1_fp8_scales):.0f} MiB saved (separate path)")
print(f"z: always bf16 {mib(z_bf16):.0f} MiB (needed for SwiGLU backward)")
