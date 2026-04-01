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

## 🔥 FP8 Blockscaled Status (2026-04-01, Session 25)

The repository has a working **zero-materialization** blockscaled FP8 training path for Blackwell.
No TK-sized FP8 activation is materialized — follows SonicMoE's core "no materialization" design.

### Quick Start (Programmatic)

```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm

moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
           intermediate_size=1536, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

with enable_quack_gemm(True):
    output, aux_loss = moe(x, use_fp8=True)
```

### Performance (Wall-clock, idle GPU, uniform 128-aligned routing)

| Shape | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| I=1024 (T=4096,H=4096,E=128,K=8) | 4.90 | 3.77 | **1.30×** |
| I=2048 (T=4096,H=4096,E=128,K=8) | 9.83 | 4.18 | **2.35×** |
| Ernie (T=8192,H=3072,I=1536,E=8,K=8) | 4.45 | 3.42 | **1.30×** |

### Performance (CUDA GPU projection, Ernie shape)

| Config | CUDA µs/iter | vs BF16 |
|--------|-------------|---------|
| Official BF16 | 3937 | baseline |
| **FP8 frontier** | **3324** | **1.18× faster** |
### Correctness

15/15 contract tests pass: 11 original + 4 aligned tests (E=8, K=8) including production shape T=8192.

### Known limitations

- **Memory:** FP8 uses 1.38–1.48× more memory due to FP8 weight caches (3 copies)
- **FP8 wgrad:** Disabled by default — colwise quant SM contention makes it net-negative
- **FP8 down-proj:** Only at I≥2048 — quant overhead ≈ GEMM savings at I=1536

### Read first

| Resource | Path | Why |
|----------|------|-----|
| **Handoff** | `reports/fp8_upgrade/HANDOFF.md` | Complete project state, root causes, next steps |
| Agent quick-start | `AGENTS.md` | Cold-start for AI agents |
| Contract tests | `tests/fp8_large_project_contract_test.py` | Current correctness gate |

### Practical guidance

- Use **official BF16** as the only authoritative baseline.
- Exclude `elementwise_kernel` from BF16 numbers (QuACK 0.3.7 layout bug).
- Use **NSYS GPU projection** for performance claims.
- FP8 advantage scales with intermediate size — I≥2048 sees the biggest wins.

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
