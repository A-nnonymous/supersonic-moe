"""NSYS profiling script for Ernie DeepEPMOELayer (PaddlePaddle).

Run via nsys:
    CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda,nvtx --gpu-metrics-device=0 \
        --cuda-memory-usage=false -f true \
        -o /tmp/ernie_moe --export=sqlite \
        python tools/nsys_profile_ernie_moe.py

Shape: T=4096, H=4096, I=1024, E=128, K=2 (Ernie top-2 routing)
Note: Ernie uses K=2 (DeepEPTop2Gate), not K=8 like SonicMoE.
      Per-expert compute is scaled differently.
"""
import os
import sys

for _k in [
    "PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "True"

import paddle
from paddle import nn
import paddle.nn.functional as F

T, H, I, E = 4096, 4096, 1024, 128
K_ERNIE = 2  # DeepEPTop2Gate supports top-2
WARMUP = 5
PROFILE_ITERS = 3

# Add ernie-core to path
ernie_src = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/baidu/ernie/ernie-core/src"
sys.path.insert(0, ernie_src)


class SwiGLUExpert(nn.Layer):
    """SwiGLU expert matching SonicMoE architecture."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias_attr=False)

    def forward(self, x):
        gu = self.gate_up(x)
        gate, up = paddle.chunk(gu, 2, axis=-1)
        return self.down_proj(F.silu(gate) * up)


class FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.rank = 0
        self.world_size = nranks
    def get_world_size(self): return self.nranks
    def get_rank(self): return self.rank


def main():
    from ernie_core.models.moe.moe_layer import DeepEPMOELayer, MoEStatics
    from ernie_core.models.moe.top2_gate import DeepEPTop2Gate
    from ernie_core.models.ernie5_moe.configuration import ErniemmMoEConfig

    paddle.seed(42)

    config = ErniemmMoEConfig(
        hidden_size=H,
        n_routed_experts=E,
        intermediate_size=I,
        num_experts_per_tok=K_ERNIE,
        moe_capacity=T * K_ERNIE // E,
        scoring_func="softmax",
        router_aux_loss_coef=0.01,
        moe_gate="deepep_top2_fused",
        n_group=8,
        topk_group=4,
    )

    experts = nn.LayerList([SwiGLUExpert(H, I) for _ in range(E)])
    gate = DeepEPTop2Gate(config, layer_idx=0, group=None)
    moe_statics = MoEStatics(config, layer_idx=0)

    moe_layer = DeepEPMOELayer(
        gate=gate,
        experts=experts,
        layer_idx=0,
        group=FakeGroup(1),
        moe_statics=moe_statics,
    )

    x_base = paddle.randn([T, H], dtype='bfloat16')
    input_ids = paddle.arange(T, dtype='int64')

    # Warmup
    print(f"Warming up Ernie MoE ({WARMUP} iters)...")
    for _ in range(WARMUP):
        moe_layer.clear_gradients()
        x_ = x_base.clone()
        x_.stop_gradient = False
        out, _, loss, _ = moe_layer(x_, input_ids=input_ids)
        total_loss = out.sum() + loss
        total_loss.backward()
    paddle.device.cuda.synchronize()
    print("Warmup done.")

    # NVTX profiling using paddle's CUDA profiler
    # Paddle doesn't have built-in NVTX, so we use ctypes
    import ctypes
    try:
        libnvtx = ctypes.CDLL("libnvToolsExt.so")
        def nvtx_push(name):
            libnvtx.nvtxRangePushA(name.encode())
        def nvtx_pop():
            libnvtx.nvtxRangePop()
    except OSError:
        print("WARNING: libnvToolsExt.so not found, NVTX markers disabled")
        def nvtx_push(name): pass
        def nvtx_pop(): pass

    for i in range(PROFILE_ITERS):
        nvtx_push(f"ernie_iter_{i}")

        nvtx_push("zero_grad")
        moe_layer.clear_gradients()
        nvtx_pop()

        nvtx_push("clone_input")
        x_ = x_base.clone()
        x_.stop_gradient = False
        nvtx_pop()

        paddle.device.cuda.synchronize()
        nvtx_push("forward")
        out, _, loss, _ = moe_layer(x_, input_ids=input_ids)
        paddle.device.cuda.synchronize()
        nvtx_pop()

        nvtx_push("backward")
        total_loss = out.sum() + loss
        total_loss.backward()
        paddle.device.cuda.synchronize()
        nvtx_pop()

        nvtx_pop()  # iter

    paddle.device.cuda.synchronize()
    print(f"Ernie MoE profiling complete: {PROFILE_ITERS} iters, T={T} H={H} I={I} E={E} K={K_ERNIE}")


if __name__ == "__main__":
    main()
