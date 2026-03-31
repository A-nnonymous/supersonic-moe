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

## 🔥 FP8 Blockscaled Status (2026-03-31)

**Full-chain blockscaled FP8 (1×32 UE8M0) for Blackwell SM100.** Forward + backward via decomposed CUTLASS blockscaled GEMM + Triton SwiGLU. 8/8 contract tests pass. Precision: RelRMSE 5.3-6.6%, Correlation 0.998.

**⚠️ Current FP8 is 18% slower than official BF16 baseline** (2930µs vs 2475µs, nsys GPU projection). The gap comes from Triton quant/SwiGLU overhead (~850µs) exceeding FP8 GEMM savings (~218µs), plus inability to use fused GemmGated with FP8. See `reports/fp8_upgrade/HANDOFF.md` for full three-way kernel breakdown and optimization roadmap.

| Resource | Path |
|----------|------|
| **Handoff** (start here) | `reports/fp8_upgrade/HANDOFF.md` |
| Agent context | `agent.md` |
| Contract tests | `tests/fp8_large_project_contract_test.py` |
| E2E benchmark | `tools/bench_aligned_e2e.py` |
| nsys breakdown | `tools/nsys_full_breakdown.py` |

### Quick start

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 small pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# E2E benchmark (production shape)
CUDA_VISIBLE_DEVICES=0 python tools/bench_aligned_e2e.py

# nsys kernel breakdown analysis
python tools/nsys_full_breakdown.py reports/sonic_fork_fp8_v4.sqlite
```

### Performance — Three-way nsys comparison (B200 SM100a, production shape)

| Mode | Forward | Backward | Total | vs Official BF16 |
|------|---------|----------|-------|-------------------|
| **Official BF16** (quack 0.2.5) | 777µs | 1698µs | **2475µs** | 1.00x (baseline) |
| Fork BF16 (quack 0.3.7) | 800µs | 3781µs | 4581µs | 0.54x ❌ |
| Fork FP8 (quack 0.3.7) | 935µs | 1995µs | 2930µs | 0.84x ❌ |

> ⚠️ Fork BF16 is inflated by 2101µs contiguous copy overhead from quack 0.3.7 layout differences. Always use official BF16 as the baseline.

### Current state

| Component | Status | Notes |
|-----------|--------|-------|
| FP8 forward (decomposed) | ✅ Working | 6.56% RelRMSE, 0.998 corr |
| FP8 backward (decomposed) | ✅ Working | dx 6.54%, dw2 5.35% RelRMSE, 0.998 corr |
| BF16 fallback (non-aligned) | ✅ Working | Auto-detects alignment, zero perf penalty |
| Fused FP8 GEMM+SwiGLU | 🔴 Blocked | CUTLASS DSL `recast_layout` bug |
| FP8 wgrad (varlen_k) | ⚠️ Integrated | Per-op faster, but layout copy overhead → E2E regression. Default OFF |

### Next steps (priority order)

1. **P0**: Triton SwiGLU kernel optimization — 5-8x headroom vs theoretical (biggest ROI)
2. **P1**: Restore fused forward GEMM+SwiGLU for FP8 (~160µs saving)
3. **P2**: Multi-stream overlap (act-grad ∥ weight-grad)
4. **P3**: Eliminate FP8 wgrad layout copy overhead

### Example usage
- SonicMoE with TC top-K choice routing (SwiGLU activation) on Hopper GPUs
```python
python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --activation swiglu
```

- SonicMoE with TC top-K token-choice routing (SwiGLU activation) on Blackwell GPUs. **This feature is currently in beta and supports only SwiGLU.** Full Blackwell kernel coverage will be available in the next release.
```python
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --activation swiglu
```

- SonicMoE with token rounding routing (SwiGLU activation)
```python
python benchmarks/moe-token-rounding.py --routing nr --thiekq 16384,4096,1024,256,8,128
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
