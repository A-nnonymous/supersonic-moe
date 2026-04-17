
import gc, json, os, sys
import numpy as np
ERNIE_ROOT = "/root/paddlejob/share-storage/gpfs/system-public/liangshuhao/erniebot_test_speed/third_party/ernie-core/src"
if ERNIE_ROOT not in sys.path:
    sys.path.insert(0, ERNIE_ROOT)
os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import paddle, paddle.nn.functional as F

data_dir, S, H, I, E, K = "/tmp/moe_xfw_lt5yebcj/T8192_H3072_I1536_E32_K8/seed42", 8192, 3072, 1536, 32, 8
FP8_ALIGN = 128
mode = "nsys"
warmup, iters = 5, 20
USE_FP8 = False  # True for FP8 path, False for BF16 path
OUTPUT_PREFIX = "paddle_bf16"

from ernie_core.models.moe.token_dispatcher.fp8_utils import ExpertsGroupGemmContiguousNode

# ── Mock custom_map for ExpertsGroupGemmContiguousNode ──
class _W:
    def __init__(self, t): self.weight = t
class _Expert:
    def __init__(self, w1, w2):
        self.up_gate_proj = _W(w1); self.down_proj = _W(w2)
class _Cfg:
    fp8_fused_ops_configs = {"spaq": True, "stack_quant": True, "swiglu_probs_bwd": True, "transpose_split_quant": True}
    # BF16 path: moe_grouped_gemm=False (ERNIE BF16 doesn't support grouped GEMM)
    # FP8 path: moe_grouped_gemm controlled by flag (default True = production training)
    moe_grouped_gemm = True if USE_FP8 else False
class _Map:
    def __init__(self, experts): self.experts = experts; self.config = _Cfg()

# ── Load data ──
x_all = paddle.to_tensor(np.load(os.path.join(data_dir, "x.npy")), dtype="bfloat16")
w1_list = [paddle.to_tensor(np.load(os.path.join(data_dir, f"w1_e{e}.npy")), dtype="bfloat16") for e in range(E)]
w2_list = [paddle.to_tensor(np.load(os.path.join(data_dir, f"w2_e{e}.npy")), dtype="bfloat16") for e in range(E)]
for w in w1_list + w2_list:
    w.stop_gradient = False
    w.main_grad = None
topk_indices = paddle.to_tensor(np.load(os.path.join(data_dir, "topk_indices.npy")), dtype="int32")
topk_probs = paddle.to_tensor(np.load(os.path.join(data_dir, "topk_scores.npy")), dtype="float32")

# ── Build node ──
experts = [_Expert(w1_list[e], w2_list[e]) for e in range(E)]
custom_map = _Map(experts)
node = ExpertsGroupGemmContiguousNode(
    custom_map,
    fp8="e4m3" if USE_FP8 else None,
    use_ue8m0=True if USE_FP8 else False,
)

# ── Precompute tokens_per_expert and ali_cnt (routing is deterministic) ──
flat_eid = topk_indices.reshape([-1])
tokens_per_expert = [int((flat_eid == e).sum()) for e in range(E)]
ali_cnt = [((c + FP8_ALIGN - 1) // FP8_ALIGN) * FP8_ALIGN if c > 0 else 0 for c in tokens_per_expert]

def do_permute():
    """F.moe_permute — called every iteration (matches production)."""
    with paddle.amp.auto_cast(False):
        ut, rowmap, up, _ = F.moe_permute(
            x_all, None, topk_indices, topk_probs,
            num_experts=E, tokens_per_expert=tokens_per_expert,
            padding_alignment=FP8_ALIGN, do_gather=True,
        )
    return ut, rowmap, up

def do_unpermute(o3, rowmap, up):
    """F.moe_unpermute — called every iteration (matches production)."""
    with paddle.amp.auto_cast(False):
        output, _ = F.moe_unpermute(o3, rowmap, topk_indices, up, total_zipped_tokens=S, num_experts=E)
    return output

def run_fwd():
    """Forward: permute → ExpertsGroupGemmContiguousNode.forward → unpermute."""
    ut, rowmap, up = do_permute()
    o3 = node.forward(ut, up, ali_cnt, tokens_per_expert)
    output = do_unpermute(o3, rowmap, up)
    return o3, output, up

def run_bwd(o3, grad_expert_out, up):
    """Backward: ExpertsGroupGemmContiguousNode.backward (production ERNIE bwd)."""
    for w in w1_list + w2_list:
        w.main_grad = None
    dx, probs_grad = node.backward(grad_expert_out, up)
    return dx

def run_fwd_bwd():
    """Full forward + backward iteration (permute → fwd → unpermute → bwd)."""
    o3, output, up = run_fwd()
    grad_expert_out = paddle.ones_like(o3)
    dx = run_bwd(o3, grad_expert_out, up)
    return output, dx

if mode == "precision":
    # Warmup
    for _ in range(2):
        _ = run_fwd()
        node.reset_statue()
    paddle.device.synchronize(); gc.collect(); paddle.device.cuda.empty_cache()
    paddle.device.synchronize()

    # ── Staged memory measurement (every MiB accounted) ──
    paddle.device.cuda.reset_max_memory_allocated()
    mem_baseline = paddle.device.cuda.memory_allocated() / (1024**2)  # weights + routing tensors

    # Forward (includes saved tensors for backward)
    o3, output, up_measured = run_fwd()
    paddle.device.synchronize()
    mem_post_fwd = paddle.device.cuda.memory_allocated() / (1024**2)
    peak_fwd = paddle.device.cuda.max_memory_allocated() / (1024**2)

    # Backward (weight grads accumulated into main_grad)
    M = o3.shape[0]
    for w in w1_list + w2_list:
        w.main_grad = None
    grad_expert_out = paddle.ones([M, H], dtype="bfloat16")
    dx = node.backward(grad_expert_out, up_measured)
    paddle.device.synchronize()
    mem_post_bwd = paddle.device.cuda.memory_allocated() / (1024**2)
    peak_total = paddle.device.cuda.max_memory_allocated() / (1024**2)

    # Save output
    np.save(os.path.join(data_dir, f"{OUTPUT_PREFIX}_output.npy"), output.cast("float32").numpy())
    # Save gradients
    for e in range(E):
        if w1_list[e].main_grad is not None:
            np.save(os.path.join(data_dir, f"{OUTPUT_PREFIX}_dw1_e{e}.npy"), w1_list[e].main_grad.numpy())
        if w2_list[e].main_grad is not None:
            np.save(os.path.join(data_dir, f"{OUTPUT_PREFIX}_dw2_e{e}.npy"), w2_list[e].main_grad.numpy())
    print(json.dumps({
        "status": "ok",
        "mem_baseline_mib": round(mem_baseline, 1),
        "mem_post_fwd_mib": round(mem_post_fwd, 1),
        "mem_post_bwd_mib": round(mem_post_bwd, 1),
        "peak_fwd_mib": round(peak_fwd, 1),
        "peak_total_mib": round(peak_total, 1),
    }))

elif mode == "nsys":
    # Full fwd+bwd pipeline under nsys capture
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    for _ in range(warmup):
        run_fwd_bwd()
        node.reset_statue()
    paddle.device.synchronize(); gc.collect()
    libcudart.cudaProfilerStart()
    for _ in range(iters):
        run_fwd_bwd()
        node.reset_statue()
    paddle.device.synchronize()
    libcudart.cudaProfilerStop()
    print("NSYS_DONE", flush=True)
