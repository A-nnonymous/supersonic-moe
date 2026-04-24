<!-- ********************************************************************************
Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
******************************************************************************** -->

# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080)

**SonicMoE** is a blazing-fast Mixture-of-Experts (MoE) implementation optimized for NVIDIA Hopper and Blackwell GPUs, leveraging [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html).

![image](./assets/mem.png)
![image](./assets/tput.png)

## Prerequisites

- NVIDIA Hopper GPUs (H100, H200) or Blackwell GPUs (GB200, B200, B300)
- CUDA 12.9+ (13.0+ for B300)
- Python 3.12+, PyTorch 2.7+
- `USE_QUACK_GEMM=1` for Blackwell kernels

## Paddle Integration (ERNIE / PaddleFleet)

The `session60-ds-fix` branch integrates SonicMoE into PaddleFleet via `paddle.compat.enable_torch_proxy`. Production entry point: `SonicMoEMlpNode`.

### Training Loop

```python
from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads

node = SonicMoEMlpNode(experts, n_experts=E, hidden_size=H, intermediate_size=I)

# ── Cold start: first fwd+bwd triggers JIT compilation (~42s) ──
# For explicit warmup before training:
#   from sonicmoe.jit_warmup import warmup_jit
#   warmup_jit(E=8, H=3072, I=1536, device="cuda")

for step in range(num_steps):
    for mb in microbatches:
        out = node(x, tokens_per_expert, dispatched_indices, dispatched_probs)
        out.backward(grad)
    optimizer.step()
    node.step()          # flush wgrad → main_grad + invalidate weight caches
    optimizer.zero_grad()
```

### Cold Start vs Hot Start

| Phase | What happens | Time |
|-------|-------------|:----:|
| **Import** | CUDA topk metadata kernel compiled | ~4s |
| **1st fwd+bwd** | CuTe CUTLASS GEMM + all Triton kernels JIT compiled | ~42s |
| **2nd fwd+bwd** | All caches hit, steady-state | 0.05s |
| **New seqlen** | CuTe GEMM: **0 recompile** (dynamic dims via `mark_layout_dynamic`). Triton: ~2.5s per new TK (cached in `~/.triton/cache` across sessions) | 0-2.5s |
| **After optimizer.step()** | Call `node.step()` → flushes native-layout wgrad to per-expert `main_grad`, clears weight/FP8/topk caches | <1ms |

### JIT Cache Architecture

Three tiers of caching, each with different invalidation strategies:

| Cache | Key includes | Invalidated by | Max size |
|-------|-------------|---------------|:--------:|
| **CuTe compile cache** (`_COMPILE_CACHE*`) | Static model dims (H, I, E, dtype, tile config) only. **No token counts.** | Never (persistent for model lifetime) | Unbounded (typically 3-8 entries) |
| **Fast-path runtime cache** (`_GEMM_FAST_PATH*`) | Exact problem shape (total_M/K + all tensor dims) | Auto-eviction at 64 entries | 64 |
| **FP8 weight cache** (`_WEIGHT_CACHE` etc.) | `data_ptr + _inplace_version + shape + stride` | `node.step()` / `invalidate_weight_caches()` | 8 per cache |
| **Triton JIT cache** (`~/.triton/cache/`) | Full kernel source hash | `rm -rf ~/.triton/cache` | Unbounded (disk) |

**Design principle**: `compile_key` contains only static model dimensions — never `TK`, `total_M`, `total_K`, `capacity`, or any token-count-dependent value. Dynamic token dimensions are handled at runtime via CuTe's `mark_layout_dynamic`. This ensures **zero CuTe recompilation** when batch size or routing distribution changes.

### Gradient Contract

| Gradient | Mechanism | Verified |
|----------|-----------|:--------:|
| **dx** (d/d hidden_states) | Paddle autograd through `_SonicMoEDeepEPFunc.backward` | cos=0.9975 |
| **ds** (d/d dispatched_probs) | Triton `_build_score_src_idx_kernel` → differentiable fancy-index → autograd | cos=0.9972 |
| **dw1, dw2** | CUTLASS wgrad accumulate directly into `_NATIVE_W{1,2}_GRAD`; `flush_native_grads()` transposes to per-expert `main_grad` at step time | cos=0.9975/0.9972 |

### Precision (Session 65, FP8 vs BF16 gold, TMA Reduce-Add epilogue)

| N | K | E | I | out | dx | dw1 | dw2 |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 128 | 4 | 4 | 384 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 128 | 8 | 8 | 384 | 0.9979 | 0.9975 | 0.9975 | 0.9971 |
| 512 | 4 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 512 | 8 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 1024 | 8 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 256 | 8 | 32 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9971 |

ds gradient verified via `test_cold_start_e2e.py`: cos=0.9972 across all 6 shapes (1024/8192/4096/2048/512/16384 tokens).

All cosine > 0.99, RRMSE < 7.6%.  Shapes include E=32 (production), varying topk (4/8), small/large token counts.

### Performance (nsys GPU-projection, B30Z, TMA Reduce-Add)

Session 65 (`SonicMoEMlpNode`, TMA reduce-add wgrad epilogue, default config):

| Shape (I=1536 K=8) | S53 BF16 (µs) | Paddle FP8 (µs) | Speedup vs BF16 |
|---|:---:|:---:|:---:|
| T=8192 E=8 | 3644 | 2820 | **1.29x** |
| T=8192 E=32 | 3844 | 3283 | **1.17x** |
| T=16384 E=8 | 7953 | 5548 | **1.43x** |
| T=16384 E=32 | 8129 | 5916 | **1.37x** |

TMA reduce-add optimization (Session 65): replaced fused `D=A@B+1.0*C` wgrad epilogue
(86 regs/thread) with TMA hardware atomic add on store (50 regs/thread). Improvement:
- E=8: -65 µs/iter (-2.3%)
- E=32: -138 µs/iter (-4.0%)
- BF16 wgrad GEMM per-call: -16µs (E=8), -33µs (E=32), 5-7.7% faster

To fall back to the legacy fused beta=1.0 epilogue: `SONIC_MOE_FP8_WGRAD_BETA_ACCUM=1`

ERNIE-shape detail (E=32 H=3072 I=1536 K=8 EP=8 SEQ=4096):
- **Forward GPU-proj: 625 µs** (CUTLASS GEMM 65%, FP8 quant 10%, router 14%)
- **Backward GPU-proj: 1904 µs** (wgrad 78%, actgrad 13%, quant 5%)
- **Total: 2530 µs/iter** (CV < 0.3%)

See `HANDOFF.md` for full kernel breakdown and Session 53 baseline comparison.

### Key Files

| File | Purpose |
|------|---------|
| `sonicmoe/ernie_compat/mlp_node_v2.py` | `SonicMoEMlpNode`: production MlpNode with `.forward()`, `.step()`, `.warmup()` |
| `sonicmoe/jit_warmup.py` | `warmup_jit(E, H, I)`: pre-compiles all CuTe + Triton kernels |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 GEMM wrappers (CUTLASS + Triton quant), cache key design |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU Triton kernels (5 production + 2 legacy bf16 variants) |
| `sonicmoe/quack_utils/_validate.py` | Low-overhead input validation (dtype/stride/shape, zero GPU sync) |
| `sonicmoe/ernie_compat/deepep_metadata.py` | DeepEP topk → SonicMoE routing metadata conversion |
| `sonicmoe/functional/__init__.py` | `_UpProjection`, `_DownProjection` autograd Functions |

### Test Files

| Test | What it validates | Run command |
|------|-------------------|-------------|
| `test_cold_start_e2e.py` | Cache clear → JIT → 6-shape precision (out/dx/ds/dw1/dw2) | `CUDA_VISIBLE_DEVICES=2 python tests/ops/test_cold_start_e2e.py` |
| `test_jit_optimization.py --quick` | Correctness (cos>0.99), zero JIT recompile, memory | `CUDA_VISIBLE_DEVICES=0 python tests/ops/test_jit_optimization.py --quick` |
| `test_mlpnode_precision.py` | Multi-topk precision audit | `CUDA_VISIBLE_DEVICES=0 python tests/ops/test_mlpnode_precision.py` |
| `bench_mlpnode_mem.py` | E=32 fwd+bwd memory benchmark (ERNIE shape) | `CUDA_VISIBLE_DEVICES=1 python tests/ops/bench_mlpnode_mem.py` |
| `bench_wgrad_epilogue.py` | A/B wgrad epilogue benchmark (TMA add vs fused beta) | `CUDA_VISIBLE_DEVICES=2 python tests/ops/bench_wgrad_epilogue.py` |
| `bench_mlpnode_topk_nsys.py` | nsys GPU-projection benchmark | Wrap with `nsys profile --resolve-symbols=false` |

### Read First (for next developer/agent)

| Priority | Resource | Path |
|:---:|----------|------|
| 1 | **This README** | Root `README.md` — architecture, cache design, training loop, test matrix |
| 2 | **Handoff** | Root `HANDOFF.md` — bugs found, constraints, what works / what doesn't |
| 3 | **Knowledge Base** | `docs/KNOWLEDGE_BASE.md` — deep expert reference |
| 4 | **Environment** | `/panzhaowu/env.md` — machine setup, Paddle compat pitfalls, perf methodology |

## Native PyTorch Quick Start

```python
import torch
from sonicmoe import MoE, SonicMoEConfig
from sonicmoe.enums import ActivationType

moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
           intermediate_size=1536, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

x = torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)

cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True)
with cfg.activate():
    output, aux_loss = moe(x, use_fp8=True)
```

## Citation

```bibtex
@misc{guo2025sonicmoeacceleratingmoeio,
      title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations},
      author={Wentao Guo and Mayank Mishra and Xinle Cheng and Ion Stoica and Tri Dao},
      year={2025},
      eprint={2512.14080},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.14080},
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).
