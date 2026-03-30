"""Benchmark BF16 backward kernels for comparison."""
import torch, os, time

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "off"
for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
          "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import (
    clear_all_fp8_weight_caches, _swiglu_forward_interleaved,
    TC_Softmax_Topk_Router_Function, general_routing_router_metadata,
)
from quack.gemm_interface import gemm, gemm_dgated

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to("cuda", torch.bfloat16)
enable_quack_gemm()

x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
router_logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)
sorted_T, _ = topk_indices.flatten().sort()
selected_E = topk_indices.flatten()
_, cu_real, x_gather_idx, _, _, _ = general_routing_router_metadata(
    topk_scores, sorted_T, selected_E, T, E)
TK = int(cu_real[-1].item())
tpe = TK // E
cu = torch.arange(0, TK + 1, tpe, device="cuda", dtype=torch.int32)

w1 = moe.c_fc.weight.permute(1, 2, 0).contiguous()  # (2I, H, E)
w2 = moe.c_proj.weight.permute(1, 2, 0).contiguous() # (H, I, E)
w1p = w1.permute(2, 1, 0).contiguous()  # (E, H, 2I) for BF16 gemm
w2p = w2.permute(2, 1, 0).contiguous()  # (E, I, H) for BF16 gemm

x_g = x[x_gather_idx[:TK]]
s = torch.randn(TK, device="cuda", dtype=torch.float32).abs()

def bench(fn, warmup=5, iters=30, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        fn()
    ev_e.record()
    torch.cuda.synchronize()
    ms = ev_s.elapsed_time(ev_e) / iters
    print(f"  {label}: {ms:.3f} ms")
    return ms

print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}, TK={TK}, tpe={tpe}")

# BF16 Forward
print("\n=== BF16 Forward ===")
t_bf16_gemm1 = bench(lambda: gemm(x_g, w1p, cu_seqlens_m=cu, dynamic_scheduler=False),
    label="GEMM1 (up-proj, fused SwiGLU)")

z = gemm(x_g, w1p, cu_seqlens_m=cu, dynamic_scheduler=False)

t_bf16_swiglu = bench(lambda: _swiglu_forward_interleaved(z), label="SwiGLU (unfused)")
y1 = _swiglu_forward_interleaved(z)

t_bf16_gemm2 = bench(lambda: gemm(y1, w2p, cu_seqlens_m=cu, dynamic_scheduler=False),
    label="GEMM2 (down-proj)")

# Actually BF16 uses fused GEMM+SwiGLU via gemm_dgated in forward
# Let me check if that's available for forward too
print("\nNote: BF16 forward uses fused GemmGated (GEMM+SwiGLU in one kernel)")
print("      The `moe(x)` forward is the real BF16 baseline")

# BF16 Backward: down-proj
print("\n=== BF16 Backward: Down-Proj ===")
out = gemm(y1, w2p, cu_seqlens_m=cu, dynamic_scheduler=False)
dout = torch.randn_like(out)
dz = torch.empty_like(z)

# gemm_dgated: fused dout × w2^T with dSwiGLU
def bf16_dgated():
    return gemm_dgated(
        dout, w2.permute(2, 0, 1), PreAct=z, activation="swiglu",
        dx_out=dz, colvec_scale=s.float(), colvec_reduce=True,
        cu_seqlens_m=cu, A_idx=x_gather_idx[:TK], dynamic_scheduler=False,
    )

t_dgated = bench(bf16_dgated, label="gemm_dgated (GEMM+dSwiGLU fused)")
_, y1s, ds = bf16_dgated()

# Weight grad
t_wgrad = bench(lambda: gemm(dout.T, y1s, out=torch.empty_like(w2), cu_seqlens_k=cu, A_idx=x_gather_idx[:TK]),
    label="weight_grad(dw2)")

print("\n=== BF16 Backward: Up-Proj ===")
t_wgrad1 = bench(lambda: gemm(x.T, dz, out=torch.empty_like(w1.permute(0,1,2)), cu_seqlens_k=cu, A_idx=x_gather_idx[:TK]),
    label="weight_grad(dw1)")

t_actgrad1 = bench(lambda: gemm(dz, w1.permute(2, 0, 1), cu_seqlens_m=cu, dynamic_scheduler=False),
    label="GEMM(dz×w1^T→dx)")

print(f"\n{'='*60}")
print(f"SUMMARY: BF16 kernel-level")
print(f"{'='*60}")
bf16_bwd_down = t_dgated + t_wgrad
bf16_bwd_up = t_wgrad1 + t_actgrad1
print(f"  BF16 bwd down (dgated+wgrad): {bf16_bwd_down:.3f} ms")
print(f"  BF16 bwd up (wgrad+actgrad):  {bf16_bwd_up:.3f} ms")
print(f"  BF16 bwd total:               {bf16_bwd_down + bf16_bwd_up:.3f} ms")
