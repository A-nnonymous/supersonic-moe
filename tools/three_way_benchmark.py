"""Three-way MoE benchmark: Official SonicMoE BF16 vs Fork FP8 vs Ernie DeepEPMOELayer.

Produces per-module timing breakdown, memory profiling, and NSYS timeline commands.

Usage:
    # Run all three benchmarks (wall-clock + memory):
    CUDA_VISIBLE_DEVICES=0 python tools/three_way_benchmark.py --all

    # Run individual:
    CUDA_VISIBLE_DEVICES=0 python tools/three_way_benchmark.py --mode official_bf16
    CUDA_VISIBLE_DEVICES=0 python tools/three_way_benchmark.py --mode fork_fp8
    CUDA_VISIBLE_DEVICES=0 python tools/three_way_benchmark.py --mode ernie

    # Generate NSYS profiling commands (with --gpu-metrics-device):
    python tools/three_way_benchmark.py --nsys-commands --gpu-id 0

Shape: T=4096, H=4096, I=1024, E=128, K=8 (uniform 128-aligned routing)
"""
import argparse
import gc
import json
import os
import sys
import time

# ========================== Shape ==========================
T, H, I, E, K = 4096, 4096, 1024, 128, 8
TK = T * K
TPE = TK // E  # tokens per expert = 256
WARMUP = 10
BENCH_ITERS = 50

# ========================== Utilities ==========================

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def format_mem(bytes_val):
    return f"{bytes_val / (1024**3):.3f} GiB"


def print_results(mode, fwd_ms, bwd_ms, total_ms, peak_mem_bytes):
    print(f"\n--- {mode} Results ---")
    print(f"  Forward:    {fwd_ms:.3f} ms")
    print(f"  Backward:   {bwd_ms:.3f} ms")
    print(f"  Total:      {total_ms:.3f} ms")
    print(f"  Peak memory: {format_mem(peak_mem_bytes)}")
    return {
        "mode": mode,
        "forward_ms": round(fwd_ms, 3),
        "backward_ms": round(bwd_ms, 3),
        "total_ms": round(total_ms, 3),
        "peak_memory_gib": round(peak_mem_bytes / (1024**3), 3),
    }


# ========================== Official BF16 ==========================

def bench_official_bf16():
    """Benchmark official SonicMoE BF16 (from upstream repo)."""
    print_header("Official SonicMoE BF16")

    # Use official sonic-moe
    official_path = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
    sys.path.insert(0, official_path)

    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ.pop("SONIC_MOE_FP8_MODE", None)

    import importlib
    # Force reimport from official path
    for mod_name in list(sys.modules.keys()):
        if "sonicmoe" in mod_name:
            del sys.modules[mod_name]

    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_quack_gemm
    import sonicmoe.functional as F_mod

    import torch
    import torch.cuda

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)

    with enable_quack_gemm():
        # Uniform routing
        _scores, _indices = _build_uniform_routing_torch(torch.device("cuda"))

        class _UniformRouter(torch.autograd.Function):
            @staticmethod
            def forward(ctx, router_logits, E_arg, K_arg):
                ctx.save_for_backward(_scores, _indices)
                ctx.E = E_arg; ctx.dtype = router_logits.dtype
                return _scores.clone(), _indices.clone()
            @staticmethod
            def backward(ctx, grad_scores, _grad_indices):
                scores, _ = ctx.saved_tensors
                return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None

        orig_router = F_mod.TC_Softmax_Topk_Router_Function
        F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter

        x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

        # Warmup
        print(f"Warming up ({WARMUP} iters)...")
        for _ in range(WARMUP):
            for p in moe.parameters(): p.grad = None
            x_ = x_base.clone().requires_grad_(True)
            out, _ = moe(x_)
            out.sum().backward()
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        fwd_times, bwd_times = [], []

        for _ in range(BENCH_ITERS):
            for p in moe.parameters(): p.grad = None
            x_ = x_base.clone().requires_grad_(True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, _ = moe(x_)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t2 - t1) * 1000)

        peak_mem = torch.cuda.max_memory_allocated()

        # Restore
        F_mod.TC_Softmax_Topk_Router_Function = orig_router

    # Remove official path
    sys.path.remove(official_path)
    for mod_name in list(sys.modules.keys()):
        if "sonicmoe" in mod_name:
            del sys.modules[mod_name]

    fwd_ms = sum(sorted(fwd_times)[5:-5]) / (len(fwd_times) - 10)
    bwd_ms = sum(sorted(bwd_times)[5:-5]) / (len(bwd_times) - 10)
    return print_results("Official BF16", fwd_ms, bwd_ms, fwd_ms + bwd_ms, peak_mem)


# ========================== Fork FP8 ==========================

def bench_fork_fp8():
    """Benchmark fork SonicMoE FP8 (our optimized path)."""
    print_header("Fork SonicMoE FP8 (FUSED_GATED=1, WGRAD=0)")

    fork_path = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
    if fork_path not in sys.path:
        sys.path.insert(0, fork_path)

    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["SONIC_MOE_FP8_FUSED_GATED"] = "1"
    os.environ["SONIC_MOE_FP8_WGRAD"] = "0"
    os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
    os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"

    import importlib
    for mod_name in list(sys.modules.keys()):
        if "sonicmoe" in mod_name:
            del sys.modules[mod_name]

    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F_mod

    import torch
    import torch.cuda

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    # Uniform routing + alignment
    _scores, _indices = _build_uniform_routing_torch(torch.device("cuda"))

    class _UniformRouter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, router_logits, E_arg, K_arg):
            ctx.save_for_backward(_scores, _indices)
            ctx.E = E_arg; ctx.dtype = router_logits.dtype
            return _scores.clone(), _indices.clone()
        @staticmethod
        def backward(ctx, grad_scores, _grad_indices):
            scores, _ = ctx.saved_tensors
            return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None

    orig_router = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter
    F_mod._ALIGNMENT_ASSUMED = True

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # Warmup
    print(f"Warming up ({WARMUP} iters)...")
    for _ in range(WARMUP):
        for p in moe.parameters(): p.grad = None
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    fwd_times, bwd_times = [], []

    for _ in range(BENCH_ITERS):
        for p in moe.parameters(): p.grad = None
        x_ = x_base.clone().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = moe(x_)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    peak_mem = torch.cuda.max_memory_allocated()
    F_mod.TC_Softmax_Topk_Router_Function = orig_router

    fwd_ms = sum(sorted(fwd_times)[5:-5]) / (len(fwd_times) - 10)
    bwd_ms = sum(sorted(bwd_times)[5:-5]) / (len(bwd_times) - 10)
    return print_results("Fork FP8", fwd_ms, bwd_ms, fwd_ms + bwd_ms, peak_mem)


# ========================== Ernie DeepEPMOELayer ==========================

def bench_ernie():
    """Benchmark Ernie DeepEPMOELayer (PaddlePaddle)."""
    print_header("Ernie DeepEPMOELayer (PaddlePaddle)")

    try:
        import paddle
        from paddle import nn
        import paddle.nn.functional as F
    except ImportError:
        print("  [SKIP] PaddlePaddle not available in this environment")
        return {"mode": "Ernie", "status": "skipped", "reason": "no paddle"}

    ernie_src = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/baidu/ernie/ernie-core/src"
    if ernie_src not in sys.path:
        sys.path.insert(0, ernie_src)

    try:
        from ernie_core.models.moe.moe_layer import DeepEPMOELayer, MoEStatics
        from ernie_core.models.moe.top2_gate import DeepEPTop2Gate
        from ernie_core.models.ernie5_moe.configuration import ErniemmMoEConfig
    except ImportError as e:
        print(f"  [SKIP] Cannot import ernie-core: {e}")
        return {"mode": "Ernie", "status": "skipped", "reason": str(e)}

    # Match SonicMoE shape: T=4096, H=4096, I=1024, E=128, K=8
    # Ernie uses top-2 routing (K=2 max for DeepEPTop2Gate)
    # We'll use K=2 with E=128 experts, same H and I
    ernie_K = 2  # DeepEPTop2Gate only supports top-2

    class SimpleExpert(nn.Layer):
        """SwiGLU expert matching SonicMoE architecture."""
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            # gate_proj + up_proj fused
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

    paddle.seed(42)

    config = ErniemmMoEConfig(
        hidden_size=H,
        n_routed_experts=E,
        intermediate_size=I,
        num_experts_per_tok=ernie_K,
        moe_capacity=TPE,
        scoring_func="softmax",
        router_aux_loss_coef=0.01,
        moe_gate="deepep_top2_fused",
        n_group=8,
        topk_group=4,
    )

    experts = nn.LayerList([SimpleExpert(H, I) for _ in range(E)])
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
    print(f"Warming up ({WARMUP} iters)...")
    for _ in range(WARMUP):
        moe_layer.clear_gradients()
        x_ = x_base.clone()
        x_.stop_gradient = False
        out, _, loss, _ = moe_layer(x_, input_ids=input_ids)
        total_loss = out.sum() + loss
        total_loss.backward()
    paddle.device.cuda.synchronize()

    # Benchmark
    paddle.device.cuda.max_memory_allocated()  # reset
    fwd_times, bwd_times = [], []

    for _ in range(BENCH_ITERS):
        moe_layer.clear_gradients()
        x_ = x_base.clone()
        x_.stop_gradient = False

        paddle.device.cuda.synchronize()
        t0 = time.perf_counter()
        out, _, loss, _ = moe_layer(x_, input_ids=input_ids)
        paddle.device.cuda.synchronize()
        t1 = time.perf_counter()
        total_loss = out.sum() + loss
        total_loss.backward()
        paddle.device.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    peak_mem = paddle.device.cuda.max_memory_allocated()

    fwd_ms = sum(sorted(fwd_times)[5:-5]) / (len(fwd_times) - 10)
    bwd_ms = sum(sorted(bwd_times)[5:-5]) / (len(bwd_times) - 10)
    return print_results("Ernie DeepEPMOELayer", fwd_ms, bwd_ms, fwd_ms + bwd_ms, peak_mem)


# ========================== Helpers ==========================

def _build_uniform_routing_torch(device):
    import torch
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def generate_nsys_commands(gpu_id):
    """Print nsys commands for timeline collection."""
    print_header("NSYS Profiling Commands")

    output_dir = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output"
    sonic_dir = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
    official_dir = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official/sonic-moe"
    env_activate = "source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate"

    print(f"\n# 1. Official BF16 timeline (with GPU metrics)")
    print(f"cd {official_dir}")
    print(f"{env_activate}")
    print(f"CUDA_VISIBLE_DEVICES={gpu_id} \\")
    print(f"  nsys profile -t cuda,nvtx --gpu-metrics-device={gpu_id} \\")
    print(f"  --cuda-memory-usage=false -f true \\")
    print(f"  -o {output_dir}/official_bf16_timeline --export=sqlite \\")
    print(f"  python {sonic_dir}/tools/nsys_profile_official_bf16.py")

    print(f"\n# 2. Fork FP8 timeline (with GPU metrics)")
    print(f"cd {sonic_dir}")
    print(f"{env_activate}")
    print(f"CUDA_VISIBLE_DEVICES={gpu_id} SONIC_MOE_FP8_FUSED_GATED=1 \\")
    print(f"  nsys profile -t cuda,nvtx --gpu-metrics-device={gpu_id} \\")
    print(f"  --cuda-memory-usage=false -f true \\")
    print(f"  -o {output_dir}/fork_fp8_timeline --export=sqlite \\")
    print(f"  python tools/nsys_profile_comprehensive.py --mode fp8")

    print(f"\n# 3. Ernie DeepEPMOELayer timeline (with GPU metrics)")
    print(f"CUDA_VISIBLE_DEVICES={gpu_id} \\")
    print(f"  nsys profile -t cuda,nvtx --gpu-metrics-device={gpu_id} \\")
    print(f"  --cuda-memory-usage=false -f true \\")
    print(f"  -o {output_dir}/ernie_moe_timeline --export=sqlite \\")
    print(f"  python {sonic_dir}/tools/nsys_profile_ernie_moe.py")


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(description="Three-way MoE benchmark")
    parser.add_argument("--mode", choices=["official_bf16", "fork_fp8", "ernie", "all"],
                        help="Which implementation to benchmark")
    parser.add_argument("--nsys-commands", action="store_true",
                        help="Print nsys profiling commands instead of running benchmarks")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for nsys commands")
    parser.add_argument("--output", type=str,
                        default="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.nsys_commands:
        generate_nsys_commands(args.gpu_id)
        return

    if not args.mode:
        parser.print_help()
        return

    print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}")
    print(f"Warmup={WARMUP}, Bench iters={BENCH_ITERS}")

    results = []

    if args.mode in ("official_bf16", "all"):
        results.append(bench_official_bf16())

    if args.mode in ("fork_fp8", "all"):
        # Need fresh process for module isolation
        if args.mode == "all":
            print("\n[NOTE] For --all mode, run each mode in a separate process for clean module isolation.")
            print("       Use: python tools/three_way_benchmark.py --mode official_bf16")
            print("       Use: python tools/three_way_benchmark.py --mode fork_fp8")
            print("       Use: python tools/three_way_benchmark.py --mode ernie")
            return
        results.append(bench_fork_fp8())

    if args.mode in ("ernie",):
        results.append(bench_ernie())

    # Save results
    if results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # Append to existing results if present
        existing = []
        if os.path.exists(args.output):
            try:
                with open(args.output) as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.extend(results)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
