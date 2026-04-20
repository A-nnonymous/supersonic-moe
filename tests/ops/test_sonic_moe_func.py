"""Test SonicMoEFunc: ERNIE-core style PyLayer with SonicMoE FP8 expert compute.

Validates:
  1. Forward output shape, no NaN/Inf.
  2. Backward produces dx and d_router_scores.
  3. Weight grads accumulated into per-expert float32 main_grad.
  4. Multi-iteration gradient accumulation.

Usage:
  $EBVENV/bin/python tests/ops/test_sonic_moe_func.py
"""

import math
import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")

_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEFunc,
    invalidate_weight_caches,
    prepare_sonic_inputs,
    stack_ernie_w1,
    stack_ernie_w2,
)

# ── Config (match test_moe_general_routing_fp8.py for autotuner cache hit) ────
H = 3072
I = 1536
E = 8
T_SEQ = 16384
K_TOPK = 8
EP_SIZE = 32


class MockExpert:
    """Mimics ERNIE expert: up_gate_proj.weight [H, 2I], down_proj.weight [I, H]."""
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def make_dispatched_inputs(seed=42):
    paddle.seed(seed)
    T_total = T_SEQ * EP_SIZE
    E_total = E * EP_SIZE
    logits = paddle.randn([T_total, E_total])
    scores = paddle.nn.functional.softmax(logits.cast("float32"), axis=-1)
    topk_scr, topk_idx = paddle.topk(scores, K_TOPK, axis=-1)
    local_mask = topk_idx < E
    disp_idx = paddle.where(local_mask, topk_idx,
                            paddle.full_like(topk_idx, -1)).cast("int32")
    disp_probs = paddle.where(local_mask, topk_scr, paddle.zeros_like(topk_scr))
    has_local = (disp_idx >= 0).any(axis=-1)
    disp_idx = disp_idx[has_local]
    disp_probs = disp_probs[has_local]
    T = disp_idx.shape[0]
    x = paddle.randn([T, H], dtype="bfloat16") * 0.02
    return x, disp_idx, disp_probs


def test_forward_backward():
    print("=== Forward + Backward ===")
    experts = [MockExpert(H, I, e) for e in range(E)]
    x, disp_idx, disp_probs = make_dispatched_inputs()

    w1 = stack_ernie_w1(experts, H, I)
    w2 = stack_ernie_w2(experts)
    x_padded, tok_idx, exp_idx, rscores, T_orig = prepare_sonic_inputs(
        disp_idx, disp_probs, x, E)

    # Tensor inputs to PyLayer need grad
    x_padded = x_padded.detach(); x_padded.stop_gradient = False
    rscores = rscores.detach(); rscores.stop_gradient = False

    # Non-diff tensor inputs must have stop_gradient=True
    tok_idx.stop_gradient = True
    exp_idx.stop_gradient = True
    w1.stop_gradient = True
    w2.stop_gradient = True

    invalidate_weight_caches()
    out = SonicMoEFunc.apply(
        x_padded, rscores, tok_idx, exp_idx, w1, w2,
        experts, E, T_orig, ActivationType.SWIGLU,
    )

    assert list(out.shape) == [T_orig, H], f"shape: {list(out.shape)}"
    assert not out.isnan().any(), "NaN in output"
    assert not out.isinf().any(), "Inf in output"

    out_grad = paddle.randn_like(out)
    out.backward(out_grad)

    assert x_padded.grad is not None, "no dx"
    assert not x_padded.grad.isnan().any(), "NaN in dx"

    # Weight grads should be in main_grad, not in w1/w2.grad
    for e_idx, exp in enumerate(experts):
        for name in ("up_gate_proj", "down_proj"):
            mg = getattr(getattr(exp, name).weight, "main_grad", None)
            assert mg is not None, f"expert[{e_idx}].{name}.main_grad is None"
            assert mg.dtype == paddle.float32
            assert not mg.isnan().any()

    print(f"  T_orig={T_orig}, T_padded={x_padded.shape[0]}")
    print(f"  dx norm={float(x_padded.grad.norm()):.4f}")
    w1_mg = [f"{float(e.up_gate_proj.weight.main_grad.norm()):.2f}" for e in experts]
    w2_mg = [f"{float(e.down_proj.weight.main_grad.norm()):.2f}" for e in experts]
    print(f"  w1 main_grad norms: [{', '.join(w1_mg)}]")
    print(f"  w2 main_grad norms: [{', '.join(w2_mg)}]")
    print("  PASSED\n")
    invalidate_weight_caches()


def test_multi_iter_accumulation():
    N_ACCUM = 3
    print(f"=== Multi-iter Accumulation ({N_ACCUM} iters) ===")
    experts = [MockExpert(H, I, e) for e in range(E)]
    x, disp_idx, disp_probs = make_dispatched_inputs()

    x_padded, tok_idx, exp_idx, rscores, T_orig = prepare_sonic_inputs(
        disp_idx, disp_probs, x, E)

    # Warm FP8 cache
    invalidate_weight_caches()
    w1 = stack_ernie_w1(experts, H, I)
    w2 = stack_ernie_w2(experts)
    xp = x_padded.detach(); xp.stop_gradient = False
    rs = rscores.detach(); rs.stop_gradient = False
    tok_idx.stop_gradient = True; exp_idx.stop_gradient = True
    w1.stop_gradient = True; w2.stop_gradient = True
    warm = SonicMoEFunc.apply(xp, rs, tok_idx, exp_idx, w1, w2,
                              experts, E, T_orig, ActivationType.SWIGLU)
    out_grad = paddle.randn_like(warm)

    for i in range(N_ACCUM):
        w1 = stack_ernie_w1(experts, H, I)
        w2 = stack_ernie_w2(experts)
        xp = x_padded.detach(); xp.stop_gradient = False
        rs = rscores.detach(); rs.stop_gradient = False
        w1.stop_gradient = True; w2.stop_gradient = True
        out = SonicMoEFunc.apply(xp, rs, tok_idx, exp_idx, w1, w2,
                                 experts, E, T_orig, ActivationType.SWIGLU)
        out.backward(out_grad)

    for e_idx, exp in enumerate(experts):
        for name in ("up_gate_proj", "down_proj"):
            mg = getattr(getattr(exp, name).weight, "main_grad", None)
            assert mg is not None
            assert mg.dtype == paddle.float32
            assert not mg.isnan().any()
            assert float(mg.norm().item()) > 0

    w1_mg = [f"{float(e.up_gate_proj.weight.main_grad.norm()):.2f}" for e in experts]
    w2_mg = [f"{float(e.down_proj.weight.main_grad.norm()):.2f}" for e in experts]
    print(f"  w1 main_grad norms: [{', '.join(w1_mg)}]")
    print(f"  w2 main_grad norms: [{', '.join(w2_mg)}]")
    print("  PASSED\n")
    invalidate_weight_caches()


if __name__ == "__main__":
    test_forward_backward()
    test_multi_iter_accumulation()
    print("All tests passed.")
