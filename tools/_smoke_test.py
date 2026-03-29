"""Minimal smoke test for decomposed FP8 path changes."""
import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

# Unset distributed env
for k in ["PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
          "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
          "PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils.blockscaled_fp8_gemm import clear_blockscaled_fp8_weight_cache
from sonicmoe.functional import clear_all_fp8_weight_caches

T, H, I, E, K = 256, 768, 256, 128, 8
torch.manual_seed(42)

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
          intermediate_size=I, activation_function=ActivationType.SWIGLU,
          add_bias=False, std=0.02).to("cuda", torch.bfloat16)

x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

enable_quack_gemm()

# --- BF16 baseline ---
os.environ["SONIC_MOE_FP8_MODE"] = "off"
clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache()
out_bf16, _ = moe(x)
out_bf16.backward(dout)
gold_dx = x.grad.detach().clone()
gold_dw1 = moe.c_fc.weight.grad.detach().clone()
gold_dw2 = moe.c_proj.weight.grad.detach().clone()
x.grad = None
moe.zero_grad(set_to_none=True)

print(f"BF16 forward: shape={out_bf16.shape}, nan={out_bf16.isnan().any()}")

# --- FP8 ---
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache()

x_fp8 = x.detach().clone().requires_grad_()
try:
    out_fp8, _ = moe(x_fp8)
    print(f"FP8 forward: shape={out_fp8.shape}, nan={out_fp8.isnan().any()}")

    # Forward accuracy
    rmse_fwd = ((out_fp8.float()-out_bf16.float()).pow(2).mean().sqrt() /
                out_bf16.float().pow(2).mean().sqrt()).item()
    corr_fwd = ((out_fp8.float()*out_bf16.float()).sum() /
                (out_fp8.float().norm()*out_bf16.float().norm())).item()
    print(f"Forward RelRMSE: {rmse_fwd*100:.2f}%, Corr: {corr_fwd:.6f}")

    # Backward
    out_fp8.backward(dout)
    cand_dx = x_fp8.grad.detach().clone()
    cand_dw2 = moe.c_proj.weight.grad.detach().clone()

    rmse_dx = ((cand_dx.float()-gold_dx.float()).pow(2).mean().sqrt() /
               gold_dx.float().pow(2).mean().sqrt()).item()
    rmse_dw2 = ((cand_dw2.float()-gold_dw2.float()).pow(2).mean().sqrt() /
                gold_dw2.float().pow(2).mean().sqrt()).item()
    print(f"dx RelRMSE: {rmse_dx*100:.2f}%")
    print(f"dw2 RelRMSE: {rmse_dw2*100:.2f}%")

    # Pass/fail
    ok = rmse_fwd < 0.15 and rmse_dx < 0.15 and rmse_dw2 < 0.15
    print(f"\n{'PASS' if ok else 'FAIL'}: all RelRMSE < 15%")
    sys.exit(0 if ok else 1)

except Exception as e:
    print(f"FP8 FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
