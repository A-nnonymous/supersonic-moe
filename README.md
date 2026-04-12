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

## 🔥 FP8 Blockscaled Status (2026-04-13, Session 46)

The `native-fp8-exploration` branch has a fully functional **zero-materialization** blockscaled FP8 training path for Blackwell (B200) with **32×32 isotropic weight quantization**, optional **weight stash** memory optimization, native **CUTLASS / QuACK** FP8 kernels, **Pythonic config API** (`SonicMoEConfig`), and **unaligned FP8 padding** for non-128-aligned expert segments. No TK-sized FP8 activation is materialized — the FP8 path matches SonicMoE's core BF16 design principle.

### Sessions 45–46 Highlights

- **Pythonic Config API:** `SonicMoEConfig` dataclass replaces env-var flags. Thread-local, context-manager-based. Priority: config > context manager > env var.
- **wgrad FP8 default-ON** at all shapes with stream-overlapped quant pipeline.
- **NCU-driven quant analysis:** All 4 hot quant kernels at 89–99% HBM bandwidth — ceiling reached.
- **Unaligned FP8 padding:** `_padded_blockscaled_gated_forward()` pads expert segments to 128 for FP8 GEMM.
- **TILE_ROWS tuning:** 32→128 for quantize/gather kernels, 16→32 for pad kernel.
- **Idle-GPU benchmarks** on truly idle B200 (0% util) with CUDA events and 5-seed precision audit.

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

### Performance (idle B200, CUDA events, 12 runs trimmed mean)

| Shape | BF16 | FP8 | Speedup | Takeaway |
|-------|:----:|:---:|:-------:|----------|
| **Ernie** (T=8192, H=3072, I=1536, E=8, K=8) | 4.97 ± 0.02 ms | 5.00 ± 0.03 ms | **0.993×** | Break-even at small I |
| **I=2048** (T=8192, H=3072, I=2048, E=8, K=8) | 6.56 ± 0.01 ms | **5.82 ± 0.01 ms** | **1.127×** | Speedup grows with I |

> FP8 speedup scales with intermediate size. At I=1536, quant overhead matches GEMM savings. At I=2048+, FP8 wins clearly.

### Memory (subprocess-isolated peak)

| Shape | BF16 Peak | FP8 Peak | Delta |
|-------|-----------|----------|-------|
| **Ernie** | 1460 MiB | 1851 MiB | +391 MiB (+27%) |
| **I=2048** | 1876 MiB | 2331 MiB | +455 MiB (+24%) |

FP8 uses more memory due to weight caches. Use **FP8 + stash** (moves bf16 weights to CPU) for net GPU memory savings.

### Precision (5 seeds, idle GPU)

| Tensor | Ernie RRMSE | I=2048 RRMSE | Correlation | Status |
|--------|:-----------:|:------------:|:-----------:|:------:|
| output | 6.51% | 6.51% | 0.9979 | ✓ PASS |
| dx | 6.52% | 6.54% | — | ✓ PASS |
| dw1 | 4.69% | 4.72% | — | ✓ PASS |
| dw2 | 4.84% | 4.88% | — | ✓ PASS |

All within guardrails: **RRMSE < 10%**, **correlation > 0.99**. `tests/fp8_large_project_contract_test.py` passes **33/34 tests + 20 subtests** (1 memory test expected — FP8 trades memory for speed).

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

#### Executive Summary

![Session 42 Executive Summary](./assets/session42_executive_summary.png)

#### Memory Waterfall

![Memory Waterfall](./assets/session42_memory_waterfall.png)

#### Kernel Breakdown (nsys GPU Projection)

![Kernel Breakdown](./assets/session42_kernel_breakdown.png)

### Read first

| Resource | Path | Why |
|----------|------|-----|
| **Handoff** | `reports/fp8_upgrade/HANDOFF.md` | Complete project state, bugs, measurements, next steps |
| **Benchmark report** | `reports/fp8_upgrade/FP8_BENCHMARK_REPORT.md` | Detailed performance/precision/memory analysis (Chinese) |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Phase-by-phase development history |
| Frontier tests | `tests/fp8_large_project_contract_test.py` | 34-test contract gate (+20 subtests) |

## 📊 Architecture & Dataflow Visualization

Eleven publication-quality figures + unified scoreboard auto-generated from profiling data.
Run `python -m visualization` to regenerate all figures into `assets/`.

### Key Figures

| # | Figure | What it shows |
|---|--------|---------------|
| 1 | Executive Summary | 3-panel hero: latency (1.12× GPU-proj), memory (stash −8.3% fwd), precision (all tracked tensors PASS) |
| 2 | Performance Waterfall | BF16 → GEMM savings → quant overhead → FP8 breakdown |
| 3 | Memory Lifecycle | 4-checkpoint BF16 vs FP8 memory trajectory |
| 4 | Backward Peak Breakdown | Tensor-level audit of the backward-memory envelope |
| 5 | Kernel-Level Comparison | Per-kernel BF16 vs FP8 timing (forward + backward) |
| 6 | Precision State Matrix | Dtype heatmap: every tensor × every phase, BF16 vs FP8 |
| 7 | Precision Profile | RRMSE + cosine similarity with pass/fail thresholds |
| 8 | Optimization Design Space | Shipped gains vs dead ends (memory impact) |
| **9** | **Buffer Lifecycle Gantt** | **Per-buffer lifetime bars, dtype-coloured, event markers, peak MiB** |
| **10** | **Dtype Transformation Flow** | **Operator-level FP8 quantization pipeline with I/O dtype boxes** |
| **11** | **Unified Scoreboard** | **Twin BF16/FP8 Gantt + memory envelope + DAG flow + operator R/W table** |

#### Buffer Lifecycle (fig 9) — per-tensor lifetime, dtype & memory
![Buffer Lifecycle](./assets/fig9_buffer_lifecycle.png)

#### Dtype Transformation Flow (fig 10) — operator-level FP8 pipeline
![Dtype Flow](./assets/fig10_dtype_flow.png)

#### Precision State Matrix (fig 6) — tensor dtype at each execution phase
![Precision Flow](./assets/fig6_precision_flow.png)

#### Unified Buffer Scoreboard (fig 11) — lifecycle × operator × memory DAG
![Scoreboard](./assets/scoreboard_unified.png)

### Introspection Pipeline

The visualization suite is powered by a zero-code-change introspection engine:

```bash
# 1. Full refresh: trace + repeated benchmark + GPU-projection + memory artifacts
python tools/introspect.py --mode full \
  --precision-seeds 42,123,777 \
  --bench-repeats 3 \
  --profile-trials 2

# 2. Optional: trace-only refresh of manifest/scoreboard-compatible artifacts
python tools/introspect.py --mode trace

# 3. Refresh the executive summary triptych fed by benchmark/profiler JSON
python visualization/session42_viz.py

# 4. Render all figures (reads manifest + scoreboard when available)
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
