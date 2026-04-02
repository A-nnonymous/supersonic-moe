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

## 🔥 FP8 Blockscaled Status (2026-04-02, Session 34)

The repository has a working **zero-materialization** blockscaled FP8 training path for Blackwell.
No TK-sized FP8 activation is materialized — follows SonicMoE's core "no materialization" design.

### Quick Start (Programmatic)

```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType

moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
           intermediate_size=1536, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

with enable_quack_gemm(True), enable_fp8():
    output, aux_loss = moe(x, use_fp8=True)
```

### Performance (CUDA GPU projection, Ernie shape T=8192 H=3072 I=1536 E=8 K=8)

| Config | CUDA µs/iter | vs Official BF16 |
|--------|-------------|-----------------|
| Official BF16 (quack 0.2.5) | 3932 | baseline |
| **FP8 frontier (quack 0.3.7)** | **3690** | **1.066× faster** |

> Measured on fully idle node 0342 (8/8 GPUs, 0% utilization). GEMM savings 21%, FP8 overhead 532µs/iter.

### Correctness

31/31 frontier + 12/12 native FP8 tests pass (run separately, see below).

### Memory

| Metric | BF16 | FP8 | Delta |
|--------|------|-----|-------|
| Peak (FWD+BWD) | 1411.8 MiB | 1913.8 MiB | +502 MiB |

FP8 peak is higher due to weight caches (~650 MiB). Z FP8 save reduces activation memory by 186 MiB.

### Native FP8 (experimental, `native-fp8-exploration` branch)

A prototype for "native FP8 params" (x arrives as FP8, weights stored as FP8) exists on the `native-fp8-exploration` branch. Current status:
- **Forward/backward works** with `enable_native_fp8()` context manager
- **Functionally identical** to frontier FP8 (same kernels, same performance)
- **Not truly native** — still quantizes inside MoE. True native requires pre-quantized x input + persistent FP8 weight buffers (design plan in `.claude/plans/shiny-beaming-knuth.md`)

### Read first

| Resource | Path | Why |
|----------|------|-----|
| **Handoff** | `reports/fp8_upgrade/HANDOFF.md` | Complete project state, performance, architecture, next steps |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Historical milestone timeline |
| Frontier tests | `tests/fp8_large_project_contract_test.py` | 31-test correctness gate |
| Native FP8 tests | `tests/fp8_native_params_test.py` | 12-test precision gate (run separately) |
| Native FP8 plan | `.claude/plans/shiny-beaming-knuth.md` | True native FP8 implementation plan |

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
