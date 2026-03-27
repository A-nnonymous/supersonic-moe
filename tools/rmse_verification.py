"""Per-step RMSE verification: FP8 blockscaled vs BF16 baseline.

Measures RelRMSE for every intermediate tensor in forward and backward.
"""
import os, sys, torch
os.environ["USE_QUACK_GEMM"] = "1"
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional import moe_general_routing_inputs, count_cumsum
import torch.nn.functional as F


def rel_rmse(test: torch.Tensor, ref: torch.Tensor) -> float:
    """RelRMSE = ||test - ref|| / ||ref||"""
    diff = (test.float() - ref.float())
    return (diff.norm() / ref.float().norm()).item()


def run_comparison(fp8_mode: str, fused_gated: str, label: str):
    """Run one forward+backward and compare with BF16 baseline."""
    T, H, I, E, K = 8192, 4096, 1024, 128, 8
    torch.manual_seed(42)

    moe = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(torch.bfloat16).cuda()
    x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = 0.2 * torch.randn_like(x)
    w1, w2, rw = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight

    # Token rounding routing (deterministic)
    router_logits = F.linear(x, rw)
    scores = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    topk_values, topk_indices = scores.topk(K, dim=-1)
    expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
    expert_freq_rounded = torch.round(expert_freq / 128).to(torch.int32) * 128
    topk_values /= topk_values.sum(dim=-1, keepdim=True)
    sc = scores.detach().clone() - 1
    sc.scatter_(1, topk_indices, topk_values)
    idx2 = sc.argsort(dim=0, descending=True).int()
    mask = torch.arange(T, device="cuda", dtype=torch.int32)[:, None].expand(-1, E) < expert_freq_rounded[None, :]
    ti = idx2[mask]
    ei = torch.arange(E, device="cuda", dtype=torch.int32)[None, :].expand(T, -1)[mask]
    order = ti.argsort().int()
    ti = ti[order]; ei = ei[order]
    rs = scores[ti, ei].contiguous()

    def fwd_bwd(mode, fused):
        os.environ["SONIC_MOE_FP8_MODE"] = mode
        os.environ["SONIC_MOE_FP8_FUSED_GATED"] = fused

        # Clear caches to ensure fresh state
        from sonicmoe.functional import clear_all_fp8_weight_caches
        clear_all_fp8_weight_caches()

        # Forward
        o, ef = moe_general_routing_inputs(
            x, rs, ti, ei,
            w1.permute(1, 2, 0), None,
            w2.permute(1, 2, 0), None,
            E, moe.stream_id, ActivationType.SWIGLU, False
        )
        # Backward
        grads = torch.autograd.grad(o, [x, rs, w1, w2], dout, retain_graph=False)
        return o.detach().clone(), [g.detach().clone() for g in grads]

    # BF16 baseline
    print(f"\n{'='*60}")
    print(f"  {label}: FP8_MODE={fp8_mode}, FUSED_GATED={fused_gated}")
    print(f"  Shape: T={T}, H={H}, I={I}, E={E}, K={K}, TK={rs.shape[0]}")
    print(f"{'='*60}")

    o_bf16, grads_bf16 = fwd_bwd("off", "0")
    o_fp8, grads_fp8 = fwd_bwd(fp8_mode, fused_gated)

    print(f"\n  Forward output RelRMSE:  {rel_rmse(o_fp8, o_bf16):.6f}")
    print(f"  dx RelRMSE:              {rel_rmse(grads_fp8[0], grads_bf16[0]):.6f}")
    print(f"  d(router_scores) RelRMSE:{rel_rmse(grads_fp8[1], grads_bf16[1]):.6f}")
    print(f"  dw1 RelRMSE:             {rel_rmse(grads_fp8[2], grads_bf16[2]):.6f}")
    print(f"  dw2 RelRMSE:             {rel_rmse(grads_fp8[3], grads_bf16[3]):.6f}")

    # Pass/fail thresholds
    threshold = 0.10  # 10% RelRMSE
    results = {
        "output": rel_rmse(o_fp8, o_bf16),
        "dx": rel_rmse(grads_fp8[0], grads_bf16[0]),
        "d_scores": rel_rmse(grads_fp8[1], grads_bf16[1]),
        "dw1": rel_rmse(grads_fp8[2], grads_bf16[2]),
        "dw2": rel_rmse(grads_fp8[3], grads_bf16[3]),
    }
    all_pass = True
    for name, rmse in results.items():
        status = "PASS" if rmse < threshold else "FAIL"
        if rmse >= threshold:
            all_pass = False
        print(f"  {name}: {rmse:.6f} [{status}]")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'} (threshold={threshold})")
    return results


if __name__ == "__main__":
    # Test the separate backward path with blockscaled weight grads
    run_comparison("perf", "0", "FP8 Separate BWD + Blockscaled Weight Grad")

    # Test the fused gated path
    run_comparison("perf", "1", "FP8 Fused Gated + Blockscaled Weight Grad")
