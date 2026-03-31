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

The repository now has a working aligned blockscaled FP8 training path for Blackwell, but the project is **not yet at the final target state**.

The old `2930µs / 455µs gap` narrative is stale for the current fused branch. After the fused gated/dgated integration and the FP8 wgrad grad-layout-copy fix, the real frontier has changed.

### Current truth

- Best current training path in this repo: `SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0`
- Full-chain FP8 (`SONIC_MOE_FP8_WGRAD=1`) is functionally working but still **too slow**
- `tests/fp8_large_project_contract_test.py`: **8 passed, 3 deselected** with `SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1`
- Current local aligned inference and peak-memory measurements still lose to BF16, so do **not** claim an inference or memory win yet

### Authoritative training baseline

These are the numbers to use in serious comparisons. They come from **NSYS NVTX GPU projection with sync barriers** on the aligned production shape (`T=4096 H=4096 I=1024 E=128 K=8`).

| Path | Forward | Backward | Total |
|------|---------|----------|-------|
| **Official BF16** | `777.3us` | `1697.9us` | `2475.2us` |
| **Current fused FP8 + BF16 wgrad** | `812.1us` | `1788.2us` | `2600.3us` |
| **Current fused FP8 + FP8 wgrad** | `812.0us` | `4838.4us` | `5650.4us` |

### Local aligned perf + memory snapshot

Use this for local peak-memory checks and rough trend checks only. On the shared machine, event timings can drift materially:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/measure_aligned_perf_memory.py
```

Current local aligned snapshot:

| Mode | Total | Peak memory |
|------|-------|-------------|
| Train BF16 | `4.894ms` | `7.051 GiB` |
| Train FP8 + BF16 wgrad | `3.136ms` | `10.746 GiB` |
| Train FP8 + FP8 wgrad | `3.325ms` | `10.808 GiB` |
| Infer BF16 | `1.019ms` | `7.526 GiB` |
| Infer FP8 | `4.638ms` | `9.760 GiB` |

### Read first

| Resource | Path | Why |
|----------|------|-----|
| **Handoff** | `reports/fp8_upgrade/HANDOFF.md` | complete current project state, authoritative metrics, lessons, next frontier |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | cleaned milestone log with only the conclusions that still survive revalidation |
| Agent quick-start | `agent.md` | short cold-start summary |
| Contract tests | `tests/fp8_large_project_contract_test.py` | current correctness gate |
| Local perf/memory helper | `tools/measure_aligned_perf_memory.py` | reproducible local train / infer perf + memory snapshot |
| NSYS harness | `tools/nsys_profile_comprehensive.py` | authoritative aligned training profiling |

### Practical guidance

- Use **official BF16** as the only authoritative baseline.
- Use **NSYS GPU projection** for performance claims.
- Treat the current training frontier as **FP8 weight-grad redesign**, not another round of small forward SwiGLU tweaks.
- Do not enable `SONIC_MOE_FP8_WGRAD=1` by default yet.

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
