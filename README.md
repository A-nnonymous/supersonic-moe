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

## 🔥 FP8 Blockscaled Status (2026-04-01)

The repository has a working aligned blockscaled FP8 training path for Blackwell that **significantly outperforms BF16** across all tested shapes.

### Performance

**NSYS GPU projection (authoritative, iter_2, excluding elementwise_kernel):**

| Shape | BF16 (µs) | FP8 (µs) | Speedup |
|-------|-----------|----------|---------|
| I=1024 (T=4096,H=4096,E=128,K=8) | 2511 | 2137 | **14.9%** |
| I=2048 (T=4096,H=4096,E=128,K=8) | 7654 | 4403 | **42.5%** |
| I=4096 (T=4096,H=4096,E=128,K=8) | 20711 | 10479 | **49.4%** |

**Wall-clock:**

| Shape | BF16 (ms) | FP8 (ms) | Speedup |
|-------|-----------|----------|---------|
| T=4096,H=4096,I=1024 | 8.98 | 5.40 | **1.66×** |
| T=4096,H=4096,I=2048 | 18.43 | 8.58 | **2.15×** |
| T=4096,H=4096,I=4096 | 41.50 | 17.48 | **2.37×** |
| T=8192,H=4096,I=2048 | 31.03 | 15.33 | **2.02×** |
| T=4096,H=7168,I=2048 | 37.08 | 15.74 | **2.36×** |

### Correctness

44/44 contract tests pass across seeds 42, 123, 777, 2024 (all 11 tests × 4 seeds). RelRMSE <10%, correlation >0.99.

### Best training path

```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=0
```

### Known limitations

- **Memory:** FP8 uses 1.38–1.48× more memory due to FP8 weight caches (fixed per-layer overhead)
- **Inference:** FP8 inference path is not yet optimized (re-quantization overhead)
- **FP8 wgrad:** Disabled by default — net-negative at standard shapes (K_per_expert too small)

### Read first

| Resource | Path | Why |
|----------|------|-----|
| **Handoff** | `reports/fp8_upgrade/HANDOFF.md` | Complete project state, measurements, lessons, next frontier |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Cleaned milestone log with conclusions |
| Agent quick-start | `AGENTS.md` / `agent.md` | Short cold-start summary |
| Contract tests | `tests/fp8_large_project_contract_test.py` | Current correctness gate |
| Perf/memory helper | `tools/measure_aligned_perf_memory.py` | Wall-clock + memory snapshot |
| NSYS harness | `tools/nsys_profile_comprehensive.py` | Authoritative aligned training profiling |

### Practical guidance

- Use **official BF16** as the only authoritative baseline.
- Exclude `elementwise_kernel` from BF16 numbers (QuACK 0.3.7 layout bug).
- Use **NSYS GPU projection** for performance claims.
- FP8 advantage scales with intermediate size — production shapes (I≥2048) see the biggest wins.

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
