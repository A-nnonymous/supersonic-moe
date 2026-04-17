#!/usr/bin/env python3
"""
Single-GPU ERNIE-core MoE Expert Computation Test (FP8 on Blackwell)

Validates the ERNIE-core FP8 expert computation pipeline on a single GPU
WITHOUT any distributed communication (dispatch/combine removed).

Pipeline under test:
  1. Router: gate_score → topk → routing metadata
  2. Unzip: permute tokens into per-expert contiguous layout, 128-align
  3. Gate-up FP8 GEMM: x @ w1^T via split_group_gemm (deep_gemm per expert)
  4. SwiGLU: silu(gate) * up  (split-half convention)
  5. Prob scaling: o2 = swiglu(o1) * probs  (BEFORE down-proj)
  6. Down FP8 GEMM: o2 @ w2^T via split_group_gemm
  7. Zip: scatter expert outputs back, weighted sum

Gold reference: pure BF16 matmul in float32 (no FP8, no fused ops).

Usage:
  CUDA_VISIBLE_DEVICES=0 python tests/ernie_core_moe_single_gpu.py
"""
import os
import sys

# Add ernie-core to path
ERNIE_ROOT = "/root/paddlejob/share-storage/gpfs/system-public/liangshuhao/erniebot_test_speed/third_party/ernie-core/src"
if ERNIE_ROOT not in sys.path:
    sys.path.insert(0, ERNIE_ROOT)

os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import paddle
import paddle.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
FP8_ALIGN = 128
IS_BLACKWELL = paddle.device.cuda.get_device_capability()[0] == 10


# ═══════════════════════════════════════════════════════════════════════════
# Gold Reference (pure BF16/FP32, no FP8)
# ═══════════════════════════════════════════════════════════════════════════

def swiglu_gold(x):
    """Split-half SwiGLU: silu(first_half) * second_half."""
    gate, up = paddle.chunk(x, chunks=2, axis=-1)
    return F.silu(gate) * up


def moe_gold_forward(x, w1_list, w2_list, topk_indices, topk_probs):
    """Full MoE forward in float32 with split-half SwiGLU (ERNIE convention).

    Args:
        x: (S, H) bfloat16 — input hidden states
        w1_list: list of E tensors, each (H, 2*I) bfloat16 — gate_up weights
        w2_list: list of E tensors, each (I, H) bfloat16 — down weights
        topk_indices: (S, K) int32 — selected expert indices per token
        topk_probs: (S, K) float32 — routing probabilities per token

    Returns:
        output: (S, H) float32
    """
    S, H = x.shape
    E = len(w1_list)
    K = topk_indices.shape[1]
    I = w2_list[0].shape[0]

    x_f32 = paddle.cast(x, "float32")
    output = paddle.zeros([S, H], dtype="float32")

    for k in range(K):
        expert_ids = topk_indices[:, k].numpy()  # (S,)
        probs_k = topk_probs[:, k]  # (S,)

        for e in range(E):
            indices_np = np.where(expert_ids == e)[0]
            if len(indices_np) == 0:
                continue
            indices = paddle.to_tensor(indices_np, dtype="int64")
            x_e = paddle.gather(x_f32, indices, axis=0)  # (n, H)
            p_e = paddle.gather(probs_k, indices)  # (n,)

            # gate_up proj
            w1_f32 = paddle.cast(w1_list[e], "float32")
            o1 = paddle.matmul(x_e, w1_f32)  # (n, 2I)

            # SwiGLU
            o2 = swiglu_gold(o1)  # (n, I)

            # prob scaling (BEFORE down-proj — ERNIE convention)
            o2 = o2 * p_e.unsqueeze(-1)

            # down proj
            w2_f32 = paddle.cast(w2_list[e], "float32")
            o3 = paddle.matmul(o2, w2_f32)  # (n, H)

            # scatter back — manual accumulation
            for j in range(len(indices_np)):
                output[indices_np[j]] += o3[j]

    return output


# ═══════════════════════════════════════════════════════════════════════════
# Local Routing (no distributed)
# ═══════════════════════════════════════════════════════════════════════════

def make_deterministic_routing(S, E, K, seed=42):
    """Generate deterministic topk routing (round-robin).

    Returns:
        topk_indices: (S, K) int32
        topk_probs: (S, K) float32 (softmax-normalized)
    """
    paddle.seed(seed)
    topk_indices = paddle.zeros([S, K], dtype="int32")
    for s in range(S):
        for k in range(K):
            topk_indices[s, k] = (s * K + k) % E

    logits = paddle.randn([S, E])
    # Gather scores for selected experts
    gathered = paddle.stack([
        paddle.gather(logits[s], topk_indices[s].cast("int64"))
        for s in range(S)
    ])  # (S, K)
    topk_probs = F.softmax(gathered, axis=-1)
    return topk_indices, topk_probs


def unzip_tokens(x, topk_indices, topk_probs, E):
    """Permute tokens into per-expert contiguous layout with 128-alignment.

    Replicates the Unzip step from ERNIE-core MoE without distributed communication.

    Returns:
        unzipped_tokens: (M_total, H) — contiguous expert segments, 128-padded
        unzipped_probs: (M_total,) — per-row routing probability
        tokens_per_expert: list of int — 128-aligned count per expert
        original_tokens_per_expert: list of int — actual count per expert (before padding)
        sorted_indices: (M_total,) — mapping from sorted pos to (token, k) index
    """
    S, K = topk_indices.shape
    H = x.shape[1]
    flat_expert_ids = topk_indices.reshape([-1])  # (S*K,)
    flat_token_ids = paddle.arange(S).unsqueeze(1).expand([S, K]).reshape([-1])  # (S*K,)
    flat_probs = topk_probs.reshape([-1])  # (S*K,)

    # Sort by expert id
    sorted_order = paddle.argsort(flat_expert_ids, stable=True)
    sorted_experts = paddle.gather(flat_expert_ids, sorted_order)
    sorted_token_ids = paddle.gather(flat_token_ids, sorted_order)
    sorted_probs = paddle.gather(flat_probs, sorted_order)

    # Compute per-expert counts
    original_counts = []
    for e in range(E):
        count = int((sorted_experts == e).sum())
        original_counts.append(count)

    # 128-align counts
    aligned_counts = []
    for c in original_counts:
        aligned_counts.append(((c + FP8_ALIGN - 1) // FP8_ALIGN) * FP8_ALIGN if c > 0 else 0)

    M_total = sum(aligned_counts)

    # Build contiguous buffer with padding
    unzipped_tokens = paddle.zeros([M_total, H], dtype=x.dtype)
    unzipped_probs = paddle.zeros([M_total], dtype="float32")
    reverse_map = paddle.full([M_total], -1, dtype="int64")

    src_offset = 0
    dst_offset = 0
    for e in range(E):
        n = original_counts[e]
        n_aligned = aligned_counts[e]
        if n == 0:
            continue
        # Copy real tokens
        src_indices = sorted_token_ids[src_offset:src_offset + n].cast("int64")
        unzipped_tokens[dst_offset:dst_offset + n] = paddle.gather(x, src_indices, axis=0)
        unzipped_probs[dst_offset:dst_offset + n] = sorted_probs[src_offset:src_offset + n]
        reverse_map[dst_offset:dst_offset + n] = sorted_order[src_offset:src_offset + n].cast("int64")
        # Padding rows (dst_offset+n : dst_offset+n_aligned) stay zero
        src_offset += n
        dst_offset += n_aligned

    return unzipped_tokens, unzipped_probs, aligned_counts, original_counts, reverse_map


def zip_tokens(expert_output, reverse_map, S, K, original_counts, aligned_counts, E, topk_probs):
    """Scatter expert outputs back to original token order.

    Reverses the unzip. Probs were already applied in fwd_down, so just accumulate.
    """
    H = expert_output.shape[1]
    output_np = np.zeros([S, H], dtype="float32")

    reverse_np = reverse_map.numpy()
    expert_np = expert_output.cast("float32").numpy()

    dst_offset = 0
    for e in range(E):
        n = original_counts[e]
        n_aligned = aligned_counts[e]
        if n == 0:
            continue
        for i in range(n):
            flat_idx = int(reverse_np[dst_offset + i])
            if flat_idx < 0:
                continue
            token_id = flat_idx // K
            output_np[token_id] += expert_np[dst_offset + i]
        dst_offset += n_aligned

    return paddle.to_tensor(output_np)


# ═══════════════════════════════════════════════════════════════════════════
# ERNIE-core FP8 Expert Computation (exact replication)
# ═══════════════════════════════════════════════════════════════════════════

def ernie_fp8_expert_forward(
    unzipped_tokens,
    unzipped_probs,
    tokens_per_expert,
    w1_list,
    w2_list,
    use_ue8m0=True,
):
    """Replicate ExpertsGroupGemmContiguousNode.forward on single GPU.

    Uses ernie-core's exact functions: fused_stack_transpose_quant,
    kitchen_quant, split_group_gemm, fuse_weighted_swiglu_fp8_quant.

    Args:
        unzipped_tokens: (M_total, H) bf16
        unzipped_probs: (M_total,) float32
        tokens_per_expert: list of int (128-aligned)
        w1_list: list of E weight tensors, each (H, 2I) bf16
        w2_list: list of E weight tensors, each (I, H) bf16
        use_ue8m0: bool — use UE8M0 scales on Blackwell

    Returns:
        o3: (M_total, H) bf16
    """
    import kitchen
    import deep_gemm
    from ernie_core.models.moe.token_dispatcher.fp8_utils import (
        fused_stack_transpose_quant,
        split_group_gemm,
        kitchen_quant,
        swiglu,
    )
    from ernie_core.models.utils import TDU

    num_expert = len(w1_list)
    M_total = unzipped_tokens.shape[0]
    H = w1_list[0].shape[0]
    two_I = w1_list[0].shape[1]
    I = two_I // 2

    # ── Step 1: gate_up GEMM (fwd_gate_up_fp8) ──
    # Quantize weights: stack + transpose + quant
    w1_t_quant, w1_t_scale = fused_stack_transpose_quant(w1_list, use_ue8m0=use_ue8m0)
    w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
    w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

    # Quantize activations
    x_fp8, x_scale = kitchen_quant(
        unzipped_tokens,
        backend=kitchen.ops.Backend.CUTLASS,
        is_1d_scaled=True,
        return_transpose=False,
    )
    x_scale = paddle.transpose(paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0])

    # GEMM
    o1 = paddle.empty([M_total, two_I], dtype=w1_list[0].dtype)
    if np.prod(x_fp8.shape) != 0:
        split_group_gemm(x_fp8, x_scale, w1_t_quant, w1_t_scale, tokens_per_expert, o1, use_ue8m0=use_ue8m0)

    # ── Step 2: SwiGLU + prob scaling + quant (fwd_down_fp8) ──
    # Quantize w2
    w2_quant, w2_scale = fused_stack_transpose_quant(w2_list, use_ue8m0=use_ue8m0)
    w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
    w2_scale = w2_scale.reshape([num_expert, -1, w2_scale.shape[-1]])

    # Fused SwiGLU + prob scaling + FP8 quant (spaq)
    if hasattr(TDU, "fuse_weighted_swiglu_fp8_quant"):
        o2_fp8, o2_scale = TDU.fuse_weighted_swiglu_fp8_quant(
            o1, unzipped_probs, using_pow2_scaling=True, use_ue8m0=use_ue8m0
        )
    else:
        # Fallback: unfused
        o2 = swiglu(o1)
        o2 = (o2 * unzipped_probs.unsqueeze(-1)).cast(paddle.bfloat16)
        o2_fp8, o2_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
    o2_scale = paddle.transpose(paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0])

    # ── Step 3: down GEMM ──
    o3 = paddle.empty([M_total, H], dtype=w1_list[0].dtype)
    if np.prod(o2_fp8.shape) != 0:
        split_group_gemm(o2_fp8, o2_scale, w2_quant, w2_scale, tokens_per_expert, o3, use_ue8m0=use_ue8m0)

    return o3


# ═══════════════════════════════════════════════════════════════════════════
# BF16 Expert Computation (reference without FP8)
# ═══════════════════════════════════════════════════════════════════════════

def bf16_expert_forward(unzipped_tokens, unzipped_probs, tokens_per_expert, w1_list, w2_list):
    """BF16 expert forward — same pipeline as FP8 but without quantization.

    Uses matmul directly in bfloat16 for each expert.
    """
    num_expert = len(w1_list)
    M_total = unzipped_tokens.shape[0]
    H = w1_list[0].shape[0]

    o3_parts = []
    start = 0
    for e in range(num_expert):
        n = tokens_per_expert[e]
        if n == 0:
            continue
        x_e = unzipped_tokens[start:start + n]  # (n, H) bf16
        p_e = unzipped_probs[start:start + n].unsqueeze(-1)  # (n, 1)

        # gate_up
        o1_e = paddle.matmul(x_e, w1_list[e])  # (n, 2I)

        # SwiGLU (split-half)
        gate, up = paddle.chunk(o1_e, 2, axis=-1)
        o2_e = F.silu(gate) * up  # (n, I)

        # prob scaling (BEFORE down-proj)
        o2_e = (o2_e * p_e).cast("bfloat16")

        # down proj
        o3_e = paddle.matmul(o2_e, w2_list[e])  # (n, H)

        o3_parts.append(o3_e)
        start += n

    if o3_parts:
        return paddle.concat(o3_parts, axis=0)
    return paddle.zeros([M_total, H], dtype="bfloat16")


# ═══════════════════════════════════════════════════════════════════════════
# Precision metrics
# ═══════════════════════════════════════════════════════════════════════════

def rrmse(a, b):
    a_f = paddle.cast(a.reshape([-1]), "float32")
    b_f = paddle.cast(b.reshape([-1]), "float32")
    diff = a_f - b_f
    return float(paddle.sqrt((diff * diff).mean() / (b_f * b_f).mean()))


def cosine_sim(a, b):
    a_f = paddle.cast(a.reshape([-1]), "float32")
    b_f = paddle.cast(b.reshape([-1]), "float32")
    return float(paddle.sum(a_f * b_f) / (paddle.norm(a_f) * paddle.norm(b_f)))


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_test(S, H, I, E, K, seed=42):
    """Run MoE forward test at given shape."""
    print(f"\n{'='*60}")
    print(f"Shape: S={S}, H={H}, I={I}, E={E}, K={K}, seed={seed}")
    print(f"{'='*60}")

    paddle.seed(seed)
    np.random.seed(seed)

    # Generate data
    x = (paddle.randn([S, H]) * 0.02).cast("bfloat16")
    w1_list = [paddle.randn([H, 2 * I]).cast("bfloat16") * 0.02 for _ in range(E)]
    w2_list = [paddle.randn([I, H]).cast("bfloat16") * 0.02 for _ in range(E)]

    topk_indices, topk_probs = make_deterministic_routing(S, E, K, seed)

    # ── Gold reference ──
    o_gold = moe_gold_forward(x, w1_list, w2_list, topk_indices, topk_probs)
    print(f"Gold output: shape={list(o_gold.shape)}, norm={float(o_gold.norm()):.6f}")

    # ── Unzip (local permutation) ──
    unzipped_tokens, unzipped_probs, aligned_counts, orig_counts, reverse_map = \
        unzip_tokens(x, topk_indices, topk_probs, E)
    print(f"Routing: orig={orig_counts}, aligned={aligned_counts}, M_total={sum(aligned_counts)}")

    # ── BF16 expert forward ──
    o3_bf16 = bf16_expert_forward(unzipped_tokens, unzipped_probs, aligned_counts, w1_list, w2_list)
    o_bf16 = zip_tokens(o3_bf16, reverse_map, S, K, orig_counts, aligned_counts, E, topk_probs)

    r_bf16 = rrmse(o_bf16, o_gold)
    c_bf16 = cosine_sim(o_bf16, o_gold)
    print(f"BF16 vs Gold:  RRMSE={r_bf16:.6f}, cosine={c_bf16:.6f}", end="")
    bf16_pass = r_bf16 < 0.01 and c_bf16 > 0.999
    print(f"  {'PASS' if bf16_pass else 'FAIL'}")

    # ── FP8 expert forward ──
    if IS_BLACKWELL:
        o3_fp8 = ernie_fp8_expert_forward(
            unzipped_tokens, unzipped_probs, aligned_counts,
            w1_list, w2_list, use_ue8m0=True,
        )
        o_fp8 = zip_tokens(o3_fp8, reverse_map, S, K, orig_counts, aligned_counts, E, topk_probs)

        r_fp8 = rrmse(o_fp8, o_gold)
        c_fp8 = cosine_sim(o_fp8, o_gold)
        r_fp8_bf16 = rrmse(o_fp8, o_bf16)
        c_fp8_bf16 = cosine_sim(o_fp8, o_bf16)
        print(f"FP8 vs Gold:   RRMSE={r_fp8:.6f}, cosine={c_fp8:.6f}", end="")
        fp8_pass = r_fp8 < 0.10 and c_fp8 > 0.99
        print(f"  {'PASS' if fp8_pass else 'FAIL'}")
        print(f"FP8 vs BF16:   RRMSE={r_fp8_bf16:.6f}, cosine={c_fp8_bf16:.6f}")
    else:
        print("SKIP FP8 (not Blackwell)")
        fp8_pass = True

    return bf16_pass and fp8_pass


def main():
    print(f"Paddle: {paddle.__version__}")
    print(f"GPU: {paddle.device.cuda.get_device_name()}, SM{paddle.device.cuda.get_device_capability()}")
    print(f"Blackwell: {IS_BLACKWELL}")
    print(f"use_ue8m0: {IS_BLACKWELL}")

    shapes = [
        # (S, H, I, E, K)
        (256, 1024, 512, 8, 4),    # small
        (1024, 2048, 1024, 8, 4),  # medium
        (2048, 3072, 1536, 8, 8),  # production-like
    ]

    all_pass = True
    for S, H, I, E, K in shapes:
        for seed in [42, 123]:
            ok = run_test(S, H, I, E, K, seed)
            all_pass = all_pass and ok

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
