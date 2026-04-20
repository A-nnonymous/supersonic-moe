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
    flush_native_grads,
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

    # Flush native-layout grad accumulators into per-expert main_grad
    flush_native_grads()

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

    flush_native_grads()

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


## ── DeepEP metadata + SonicMoEMlpNode tests ──────────────────────────────────

from sonicmoe.ernie_compat import deepep_to_sonic_metadata, SonicMoEMlpNode


def test_deepep_metadata_correctness():
    """Verify deepep_to_sonic_metadata produces correct routing tensors."""
    print("=== DeepEP Metadata Correctness ===")

    tokens_per_expert = [512, 300, 0, 700, 128, 1, 256, 150]
    T = sum(tokens_per_expert)
    (
        efo, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
        naept_offset, router_scores, TK_padded, total_pad_rows,
    ) = deepep_to_sonic_metadata(tokens_per_expert, T, E, device="cuda")

    # 1. expert_frequency_offset has E+1 entries, starts at 0
    assert efo.shape[0] == E + 1, f"efo shape: {efo.shape}"
    assert int(efo[0].item()) == 0, f"efo[0]: {efo[0]}"
    assert int(efo[-1].item()) == TK_padded, f"efo[-1]={efo[-1]} != TK_padded={TK_padded}"

    # 2. Each expert segment is 128-aligned
    for i in range(E):
        segment_len = int(efo[i + 1].item()) - int(efo[i].item())
        if tokens_per_expert[i] == 0:
            assert segment_len == 0, f"expert {i}: empty but segment_len={segment_len}"
        else:
            assert segment_len % 128 == 0, f"expert {i}: segment_len={segment_len} not 128-aligned"
            assert segment_len >= tokens_per_expert[i], \
                f"expert {i}: segment_len={segment_len} < real={tokens_per_expert[i]}"

    # 3. Identity scatter
    assert (s_scatter_idx == paddle.arange(TK_padded, dtype="int32")).all()
    assert (s_reverse_scatter_idx == paddle.arange(TK_padded, dtype="int32")).all()

    # 4. naept_offset is arange(T+1)
    assert naept_offset.shape[0] == T + 1
    assert (naept_offset == paddle.arange(T + 1, dtype="int32")).all()

    # 5. router_scores: real tokens have 1.0, padding has 0.0
    assert int(router_scores.sum().item()) == T, \
        f"real score sum={router_scores.sum().item()} != T={T}"

    # 6. x_gather_idx: real tokens map to [0, T), padding to [T, T+pad)
    real_mask = router_scores > 0
    real_indices = x_gather_idx[real_mask]
    assert int(real_indices.min().item()) == 0
    assert int(real_indices.max().item()) == T - 1

    print(f"  T={T}, TK_padded={TK_padded}, total_pad_rows={total_pad_rows}")
    print(f"  efo: {efo.tolist()}")
    print("  PASSED\n")


def test_deepep_metadata_edge_cases():
    """Test edge cases: all-zero experts, perfectly aligned counts."""
    print("=== DeepEP Metadata Edge Cases ===")

    # All experts have 128-aligned counts → zero padding
    tokens_per_expert = [128, 256, 384, 128, 256, 128, 384, 256]
    T = sum(tokens_per_expert)
    (efo, _, _, _, _, router_scores, TK_padded, total_pad_rows,
     ) = deepep_to_sonic_metadata(tokens_per_expert, T, E, device="cuda")
    assert TK_padded == T, f"perfectly aligned: TK_padded={TK_padded} != T={T}"
    assert total_pad_rows == 0
    print(f"  Perfectly aligned: T={T}, TK_padded={TK_padded}, pad=0")

    # Single expert has all tokens
    tokens_per_expert = [1024, 0, 0, 0, 0, 0, 0, 0]
    T2 = 1024
    (efo2, _, _, _, _, _, TK2, pad2,
     ) = deepep_to_sonic_metadata(tokens_per_expert, T2, E, device="cuda")
    assert int(efo2[1].item()) == 1024  # first expert gets 1024 (already aligned)
    assert TK2 == 1024
    print(f"  Single expert: T={T2}, TK_padded={TK2}")

    print("  PASSED\n")


def make_deepep_inputs(seed=42):
    """Simulate DeepEP dispatch: tokens already sorted by expert."""
    paddle.seed(seed)
    # Generate random tokens_per_expert
    tpe = [int(x) for x in paddle.randint(100, 600, [E]).tolist()]
    T = sum(tpe)
    x = paddle.randn([T, H], dtype="bfloat16") * 0.02
    return x, tpe


def test_mlpnode_forward_backward():
    """Test SonicMoEMlpNode forward + backward + main_grad."""
    print("=== SonicMoEMlpNode Forward + Backward ===")
    experts = [MockExpert(H, I, e) for e in range(E)]
    x, tpe = make_deepep_inputs()
    T = x.shape[0]

    node = SonicMoEMlpNode(
        experts=experts,
        n_experts=E,
        hidden_size=H,
        intermediate_size=I,
    )

    invalidate_weight_caches()
    out = node.forward(x, tpe)

    assert list(out.shape) == [T, H], f"shape: {list(out.shape)}"
    assert not out.isnan().any(), "NaN in output"
    assert not out.isinf().any(), "Inf in output"

    out_grad = paddle.randn_like(out)
    out.backward(out_grad)

    # Weight grads should be in main_grad (float32)
    for e_idx, exp in enumerate(experts):
        for name in ("up_gate_proj", "down_proj"):
            mg = getattr(getattr(exp, name).weight, "main_grad", None)
            assert mg is not None, f"expert[{e_idx}].{name}.main_grad is None"
            assert mg.dtype == paddle.float32
            assert not mg.isnan().any()

    print(f"  T={T}, out norm={float(out.norm()):.4f}")
    w1_mg = [f"{float(e.up_gate_proj.weight.main_grad.norm()):.2f}" for e in experts]
    w2_mg = [f"{float(e.down_proj.weight.main_grad.norm()):.2f}" for e in experts]
    print(f"  w1 main_grad norms: [{', '.join(w1_mg)}]")
    print(f"  w2 main_grad norms: [{', '.join(w2_mg)}]")
    print("  PASSED\n")
    invalidate_weight_caches()


def test_mlpnode_multi_iter():
    """Test SonicMoEMlpNode multi-iteration gradient accumulation."""
    N_ACCUM = 3
    print(f"=== SonicMoEMlpNode Multi-iter ({N_ACCUM} iters) ===")
    experts = [MockExpert(H, I, e) for e in range(E)]
    x, tpe = make_deepep_inputs()
    T = x.shape[0]

    node = SonicMoEMlpNode(
        experts=experts,
        n_experts=E,
        hidden_size=H,
        intermediate_size=I,
    )

    # Warm FP8 cache
    invalidate_weight_caches()
    warm = node.forward(x, tpe)
    out_grad = paddle.randn_like(warm)

    for i in range(N_ACCUM):
        invalidate_weight_caches()
        out = node.forward(x, tpe)
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
    # Original tests (ERNIE top-K dispatch path)
    test_forward_backward()
    test_multi_iter_accumulation()

    # DeepEP metadata tests
    test_deepep_metadata_correctness()
    test_deepep_metadata_edge_cases()

    # SonicMoEMlpNode tests (DeepEP path)
    test_mlpnode_forward_backward()
    test_mlpnode_multi_iter()

    print("All tests passed.")
