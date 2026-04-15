<!-- ********************************************************************************
Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
******************************************************************************** -->

# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080)

**SonicMoE** is a simple but blazing-fast Mixture-of-Experts (MoE) implementation optimized for NVIDIA Hopper and Blackwell (beta stage) architecture GPUs. It mainly leverages [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html) to deliver state-of-the-art performance through IO-aware optimizations. These 2 figures provide an overview of activation memory usage and training throughput. 

![image](./assets/mem.png)
![image](./assets/tput.png)

## 📦 Installation

### Prerequisites

- NVIDIA Hopper GPUs (H100, H200, etc.), Blackwell GPUs (GB200, B200). **For B300, please manually upgrade the triton version to 3.6.0**. We need to manually set environment variable `USE_QUACK_GEMM=1` to use the Blackwell kernels.
- CUDA 12.9+ (13.0+ for B300 GPUs)
- Python 3.12 or 3.13
- PyTorch 2.7+ (2.9.1 recommended)

### Install from pip
```bash
pip install sonic-moe
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Dao-AILab/sonic-moe.git
cd sonic-moe

# Install dependencies
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
pip install -r requirements.txt

# Install SonicMoE
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

# Create MoE layer
moe = MoE(
    num_experts=128,                           # Number of experts
    num_experts_per_tok=8,                     # Top-k experts per token
    hidden_size=4096,                          # Hidden dimension
    intermediate_size=1536,                    # Expert intermediate size
    activation_function=ActivationType.SWIGLU, # SwiGLU activation
    add_bias=False,                            # Add bias to linear layers
    std=0.02,                                  # Weight initialization std
).to(device="cuda", dtype=torch.bfloat16)

# Forward pass
x = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
```

## 🧪 Testing

Run the test suite to verify correctness:

```bash
make test
```

For a Blackwell-only QuACK smoke test, run:

```bash
make test-blackwell
```

For the current Blackwell-focused regression set, run:

```bash
make test-blackwell-full
```

For an opt-in multi-process run on an idle machine, run:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

This parallel entry is intentionally opt-in. On a single busy GPU it may not speed up the heaviest QuACK/CuTe tests, so keep comparing it against the serial path.

On this machine, a better option is to shard the Blackwell regression files across separate GPUs:

```bash
make test-blackwell-multigpu BLACKWELL_TEST_GPUS=0,1,2
```

This avoids multiple workers contending on one GPU.

If the machine is saturated, validate the shard mapping without launching pytest:

```bash
python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run
```

For FP8 accuracy and memory reporting against the official bf16 baseline, run:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

The reporting policy for every FP8 step is:

- accuracy baseline: official bf16
- memory baseline: official bf16
- performance baselines: previous commit and official bf16

## 🔥 FP8 Blockscaled Status (2026-04-15, Session 54)

The `native-fp8-exploration` branch has a fully functional **zero-materialization** blockscaled FP8 training path for Blackwell (B30Z) with **32×32 isotropic weight quantization**, optional **weight stash** memory optimization, native **CUTLASS / QuACK** FP8 kernels, **Pythonic config API** (`SonicMoEConfig`), **unaligned FP8 padding** (forward only), **epilogue FP8 D output** (z written directly as fp8 by CUTLASS), **NCU-guided quant kernel optimization** (num_warps=1 → 2.3× colwise speedup), **shape-based wgrad FP8 auto-tuning**, and **fused dual row+col quantization**. No TK-sized FP8 activation is materialized.

### Session 54 Additions

- **MoE module-level test suite** (`tests/ops/test_moe_module.py`): 59 tests validating the full MoE pipeline (permute → up-proj → SwiGLU → down-proj → unpermute) against a pure-torch float32 gold reference. BF16 RRMSE=0.0044, FP8 RRMSE=0.065.
- **Cross-framework weight conversion**: Split-half (ERNIE) ↔ interleaved (SonicMoE) SwiGLU convention conversion, verified bit-exact round-trip.
- **0-size expert audit**: All forward/backward paths handle 0-token experts correctly (verified with up to 7 empty experts).
- **Edge-case tests**: Empty experts, deterministic runs, large tensors (T=4096), routing metadata correctness, numerical stability.

For the full handoff document, see [`docs/HANDOFF.md`](docs/HANDOFF.md).

### MoE Module Tests

```bash
# Full module-level precision test (59 tests, ~4 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/test_moe_module.py -v --tb=short

# Edge cases only (empty experts, determinism, stability)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/ops/test_moe_module.py -v -k "empty or deterministic or stability"
```

### Session 53 Highlights

- **VARLEN weight cache fix**: eliminated ~360µs/iter re-quantization → FP8 now **14.2% faster** than BF16.
- **3× repeated nsys GPU-projection** (CV=0.09%): FP8 is **14.2% faster** than BF16 at Ernie shape.
- **FP8+Stash validated** (3 GPUs × 60 trials): 2.8% faster (CUDA events) + **24.5-25.9% less peak memory**.
- **NCU kernel analysis** (clock-control=none): Triton colwise 1.51× faster than CuTe colwise. Row quant at 73% HBM throughput (near limit).
- **introspect.py enhanced**: auto Python resolution, `ncu-bench` and `wgrad-force` modes, torch.cuda.memory_stats breakdown.

### Performance (Session 53 — nsys GPU-Projection, 12 iters, B30Z)

**BF16 baseline: official SonicMoE** (`/lab/official/sonic-moe`, env `official_bf16`)

**Full 27-shape grid** (3T × 3E × 3I, H=3072, K=8):

| T | I | E | BF16 (µs) | FP8 (µs) | Speedup | FP8 Bwd (MiB) | MemΔ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **8192** | **1536** | **8** | **3644** | **2715** | **1.34×** | **1547** | +6.0% |
| 8192 | 2048 | 8 | 4958 | 3387 | **1.46×** | 1992 | +6.3% |
| 8192 | 3072 | 8 | 8110 | 4774 | **1.70×** | 2884 | +6.5% |
| 8192 | 1536 | 32 | 3844 | 2922 | **1.32×** | 2909 | +7.5% |
| 8192 | 2048 | 32 | 5263 | 3709 | **1.42×** | 3678 | +7.9% |
| 8192 | 3072 | 32 | 8124 | 5318 | **1.53×** | 5218 | +8.3% |
| 8192 | 1536 | 128 | 5009 | 3897 | **1.29×** | 8700 | +10.3% |
| 8192 | 2048 | 128 | 6967 | 4995 | **1.39×** | 11385 | +10.3% |
| 8192 | 3072 | 128 | 10839 | 7267 | **1.49×** | 16756 | +10.3% |
| 16384 | 1536 | 8 | 7953 | 5227 | **1.52×** | 2819 | +7.9% |
| 16384 | 2048 | 8 | 10832 | 6765 | **1.60×** | 3622 | +8.2% |
| 16384 | 3072 | 8 | 16172 | 10065 | **1.61×** | 5232 | +8.5% |
| 16384 | 1536 | 32 | 8129 | 5432 | **1.50×** | 3891 | +6.1% |
| 16384 | 2048 | 32 | 10860 | 7039 | **1.54×** | 4794 | +6.5% |
| 16384 | 3072 | 32 | 16558 | 10166 | **1.63×** | 6863 | +4.8% |
| 16384 | 1536 | 128 | 9099 | 6360 | **1.43×** | 9688 | +9.4% |
| 16384 | 2048 | 128 | 12348 | 8198 | **1.51×** | 12506 | +9.6% |
| 16384 | 3072 | 128 | 19216 | 11862 | **1.62×** | 18142 | +9.7% |
| 32768 | 1536 | 8 | 16287 | 10652 | **1.53×** | 5359 | +8.9% |
| 32768 | 2048 | 8 | 21753 | 13587 | **1.60×** | 6882 | +9.3% |
| 32768 | 3072 | 8 | 33278 | 20010 | **1.66×** | 9927 | +9.7% |
| 32768 | 1536 | 32 | 16829 | 10753 | **1.56×** | 6176 | +6.7% |
| 32768 | 2048 | 32 | 22584 | 13967 | **1.62×** | 7965 | +6.9% |
| 32768 | 3072 | 32 | 33504 | 19761 | **1.70×** | 11549 | +7.2% |
| 32768 | 1536 | 128 | 17635 | 11509 | **1.53×** | 11669 | +8.3% |
| 32768 | 2048 | 128 | 23312 | 14956 | **1.56×** | 14751 | +8.5% |
| 32768 | 3072 | 128 | 35627 | 22026 | **1.62×** | 20919 | +8.8% |

**Speedup range: 1.29× – 1.70×, mean 1.53×.** Memory overhead: +4.8% to +10.3%.

Scaling rules:
- **T scaling**: larger T → higher speedup (1.34× at T=8k → 1.53× at T=32k for E=8,I=1536)
- **I scaling**: larger I → higher speedup (1.34× at I=1536 → 1.70× at I=3072 for T=8k,E=8)
- **E scaling**: minimal impact at fixed T×I (E=8 vs E=128 differ by <0.15×)
- **Memory**: FP8 uses 5-10% more peak backward memory (FP8 shadow weight caches)

Precision (3 seeds, FP8 vs BF16 on identical routing — RRMSE %):

| T | E | output | dx | dw1 | dw2 | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 8192 | 8 | 6.52 | 6.53 | 4.71 | 4.90 | PASS |
| 8192 | 32 | 6.52 | 6.51 | 5.47 | 5.88 | PASS |
| 8192 | 128 | 6.52 | 6.52 | 6.01 | 6.50 | PASS |
| 32768 | 8 | 6.55 | 6.55 | 4.12 | 4.20 | PASS |
| 32768 | 32 | 6.55 | 6.54 | 4.60 | 4.84 | PASS |
| 32768 | 128 | 6.55 | 6.55 | 5.40 | 5.81 | PASS |

All RRMSE < 10%. Precision tested on same code path as performance.

### Memory Optimization (optional, for memory-constrained scenarios)

```python
# CPU optimizer offload: master weights + Adam on CPU, only FP8 on GPU
# Saves ~3.4 GB at E=128 base, costs ~500µs/iter from CPU↔GPU transfers
moe.setup_cpu_optimizer(torch.optim.Adam, lr=1e-3)
for batch in dataloader:
    out, aux = moe(x, use_fp8=True)
    loss.backward()
    moe.cpu_optimizer_step()
```

> **Methodology:** nsys GPU-projection, 12-20 iters after 5 warmup. Each shape×mode in isolated subprocess (`CUDA_VISIBLE_DEVICES` per GPU). BF16 baseline verified within <1% of official SonicMoE. E>8 FP8 uses official token rounding (Mtile=128). nsys-rep files: `panzhaowu/output/nsys/`.

### How to Reproduce All Results

All measurements use `tools/introspect.py`. Activate the environment first:

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
```

#### 1. Performance (nsys GPU-projection — gold standard)

```bash
# Single shape: T=8192, H=3072, I=1536, E=8, K=8
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py \
  --mode nsys --gpu 0 --nsys-iters 12 --nsys-warmup 3 \
  --nsys-shapes 8192,3072,1536,8,8

# Output includes:
#   - BF16 and FP8 GPU-projection µs/iter (merged overlapping kernel intervals)
#   - Paired memory: base, peak_fwd, peak_bwd for both modes
#   - Per-category budget breakdown (GEMM savings vs FP8 overhead)
#   - nsys-rep files saved to panzhaowu/output/nsys/ for GUI inspection
```

#### 2. Full 27-shape grid benchmark (3T × 3E × 3I on 8 GPUs)

```bash
# Reproduces the full 27-shape performance table (LPT load-balanced across GPUs):
python tools/introspect.py --mode grid --gpu 8 \
  --nsys-warmup 3 --nsys-iters 12

# Output: reports/grid_session53/session53_grid_full.json
# Per-GPU logs: reports/grid_session53/logs/gpu{0-7}.log
# Grid shapes: T∈{8192,16384,32768} × E∈{8,32,128} × I∈{1536,2048,3072}
# Each GPU runs its shapes sequentially; GPUs run in parallel.
# Typical runtime: ~15 min on 8 idle B30Z GPUs.
```

#### 3. Multi-shape sweep on a single GPU

```bash
# Multiple shapes on one GPU (sequential within subprocess):
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py \
  --mode nsys --gpu 0 --nsys-iters 12 --nsys-warmup 3 \
  --nsys-shapes 8192,3072,1536,8,8 8192,3072,2048,8,8 8192,3072,3072,8,8

# Manual parallel across GPUs (one shape per GPU):
for g in 0 1 2 3 4 5; do
  shapes=("8192,3072,1536,8,8" "8192,3072,1536,32,8" "8192,3072,1536,128,8" \
          "32768,3072,1536,8,8" "32768,3072,1536,32,8" "32768,3072,1536,128,8")
  CUDA_VISIBLE_DEVICES=$g python tools/introspect.py \
    --mode nsys --gpu 0 --nsys-iters 12 --nsys-warmup 3 \
    --nsys-shapes ${shapes[$g]} \
    2>&1 > /tmp/nsys_gpu${g}.log &
done
wait
```

#### 4. Precision audit (FP8 vs BF16, multi-seed)

```bash
# Single shape:
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py \
  --mode precision --gpu 0 \
  --nsys-shapes 8192,3072,1536,8,8 \
  --precision-seeds 42,123,777

# Parallel multi-shape precision:
for g in 0 1 2 3 4 5; do
  shapes=("8192,3072,1536,8,8" "8192,3072,1536,32,8" "8192,3072,1536,128,8" \
          "32768,3072,1536,8,8" "32768,3072,1536,32,8" "32768,3072,1536,128,8")
  CUDA_VISIBLE_DEVICES=$g python tools/introspect.py \
    --mode precision --gpu 0 \
    --nsys-shapes ${shapes[$g]} \
    --precision-seeds 42,123,777 \
    2>&1 > /tmp/prec_gpu${g}.log &
done
wait
for g in 0 1 2 3 4 5; do
  grep "output=" /tmp/prec_gpu${g}.log | tail -1
done
```

#### 5. Memory-only measurement

Memory is automatically paired with nsys runs. For standalone memory:

```bash
# From Python (uses subprocess isolation internally):
python -c "
import sys; sys.path.insert(0, '.')
from tools.introspect import _run_memory_measure
for mode in ('bf16', 'fp8'):
    r = _run_memory_measure(mode, {'T':8192,'H':3072,'I':1536,'E':8,'K':8}, 0)
    print(f'{mode}: base={r[\"base_mib\"]:.0f}  fwd={r[\"peak_fwd_mib\"]:.0f}  bwd={r[\"peak_bwd_mib\"]:.0f} MiB')
"
```

#### 6. View nsys timelines

```bash
# nsys-rep files are saved to persistent storage:
ls /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/nsys/*.nsys-rep
# Open in Nsight Systems GUI for SM utilization, kernel timeline, etc.
```

### Measurement Rules

- **GPU must be idle** (`nvidia-smi` util=0%) before any measurement
- **Each shape×mode runs in its own subprocess** (avoids CUTLASS JIT cache cross-contamination between different shapes)
- **BF16 uses `moe_TC_softmax_topk_layer` directly** (same API as official benchmark, verified <1% gap)
- **FP8 E≤8** uses stash mode (bf16 weights → CPU during fwd/bwd)
- **FP8 E>8** uses official token rounding (`forward_token_choice_rounding`, Mtile=128) + `moe_general_routing_inputs`
- **Expert segments must be 128-aligned** (SM100 ISA scale tile hardware constraint; non-aligned raises `RuntimeError`)

### Quick Start (Pythonic Config — no env vars needed)

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

Alternatively, env vars still work: `USE_QUACK_GEMM=1` and `SONIC_MOE_FP8_MODE=perf`.

### Precision (Session 53, 5 seeds: 42, 123, 777, 999, 2024, verified on 3 GPUs)

| Tensor | RRMSE (%) | Std (%) | Cosine Sim | Status |
|--------|:---------:|:-------:|:----------:|:------:|
| output | 6.52 | 0.002 | 0.9979 | PASS |
| dx | 6.53 | 0.001 | 0.9979 | PASS |
| dw1 | 4.27 | 0.001 | 0.9991 | PASS |
| dw2 | 4.72 | 0.044 | 0.9989 | PASS |

All within guardrails: **RRMSE < 10%**, **cosine > 0.99**. Results identical across 3 GPUs.

### Weight Stash Training Loop

```python
optimizer.step()
moe.refresh_fp8_shadow_weights()  # bf16 → FP8 shadow caches
moe.stash_bf16_to_cpu()           # -216 MiB GPU (bf16 → CPU)
with cfg.activate():
    output, aux_loss = moe(x, use_fp8=True)
output.backward(dout)
moe.unstash_bf16()                # +216 MiB GPU (CPU → bf16)
```

#### Session 53 Performance (HANDOFF)

27-shape grid — speedup range: **1.29× – 1.70×**, mean **1.53×**.

| T | I | E | BF16 µs | FP8 µs | Speedup | FP8 Bwd MiB | MemΔ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| 8192 | 1536 | 8 | 3644 | 2715 | **1.34×** | 1547 | +6% |
| 8192 | 3072 | 8 | 8110 | 4774 | **1.70×** | 2884 | +7% |
| 32768 | 3072 | 32 | 33504 | 19761 | **1.70×** | 11549 | +7% |
| 32768 | 3072 | 128 | 35627 | 22026 | **1.62×** | 20919 | +9% |

> Full 27-shape table: see Performance section above or `reports/grid_session53/session53_grid_full.json`.

### Read first (for next developer/agent)

| Priority | Resource | Path | Why |
|:---:|----------|------|-----|
| 1 | **Handoff** | `docs/HANDOFF.md` | **Start here** — complete project state, architecture, 27-shape data, bugs fixed, lessons, next steps |
| 2 | **Grid data** | `reports/grid_session53/session53_grid_full.json` | Raw 27-shape benchmark JSON (performance + memory per shape) |
| 3 | **Breakdown** | `reports/session53_breakdown.md` | Performance/memory table with scaling rules |
| 4 | **Engineering log** | `reports/fp8_upgrade/engineering_log.md` | Historical development log (Phases 1-18), useful for understanding design decisions |
| 5 | **BF16 baseline** | `/lab/official/sonic-moe` (env: `official_bf16`) | The ONLY valid BF16 baseline for comparison |
| 6 | **Environment** | `/panzhaowu/env.md` | Machine setup, compilation, cluster tools |
| 7 | Introspect tool | `tools/introspect.py` | All-in-one profiling: `--mode nsys/grid/precision/report` |
| 8 | Contract tests | `tests/fp8_large_project_contract_test.py` | 34-test contract gate (+20 subtests) |

> **Note:** `reports/fp8_upgrade/HANDOFF.md` is **stale** (Session 52 data). Use `docs/HANDOFF.md` only.

## 📊 Architecture & Dataflow Visualization

Publication-quality Session 53 frontier figures auto-generated from fresh nsys profiling data.
Run `python -m visualization` to regenerate all figures into `assets/`.

### Session 53 Frontier Figures

| # | Figure | What it shows |
|---|--------|---------------|
| 1 | Kernel Runtime Breakdown | 2×2 grid: E2E latency, speedup decomposition, per-category kernel budget, quant micro-benchmark |
| 2 | Memory Breakdown | Memory lifecycle waterfall, peak comparison across shapes, delta vs BF16 |
| 3 | Computation Data Flow | Side-by-side BF16 vs FP8 zero-materialization data flow diagram |

#### Kernel Runtime Breakdown
![Kernel Runtime Breakdown](./assets/fig11_kernel_runtime_breakdown.png)

#### Memory Breakdown
![Memory Breakdown](./assets/fig12_memory_breakdown.png)

#### Computation Data Flow
![Computation Data Flow](./assets/fig13_computation_dataflow.png)

### Introspection Pipeline

The visualization suite is powered by a zero-code-change introspection engine:

```bash
# 1. Fresh nsys GPU-projection benchmarks (gold standard)
python tools/introspect.py --mode nsys --gpu 0 \
  --nsys-iters 12 --nsys-warmup 3 \
  --nsys-shapes 8192,3072,1536,8,8

# 2. Aggregate all data into summary JSON (no GPU needed)
python tools/introspect.py --mode compile-session53

# 3. Render all figures
python -m visualization
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SonicMoE in your research, please cite:

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
