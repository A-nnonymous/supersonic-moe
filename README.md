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

## 🔥 FP8 Upgrade Status (2026-03-26 Session 2)

**Current state**: 6/8 GEMM operators use **blockscaled 1x32 UE8M0** FP8 (forward + activation-grad). 2/8 weight-grad operators use per-tensor FP8. Fused SwiGLU+quantize Triton kernels integrated. **Inference forward 43% faster than BF16** (3.9ms→2.2ms). **Training E2E 4x slower** — root cause: `blockscaled_fp8_gemm_varlen` pack/unpack/quantize overhead at E=128 experts. **Priority: eliminate varlen GEMM overhead to unlock training speedup.**

| Resource | Path |
|----------|------|
| **Handoff** (start here) | `reports/fp8_upgrade/HANDOFF.md` |
| Engineering log | `reports/fp8_upgrade/ENGINEERING_LOG.md` |
| Contract tests (8/8 pass) | `tests/fp8_large_project_contract_test.py` |
| Benchmark | `benchmarks/moe-cute.py` |

### Quick start

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Run contract tests (8/8 pass; exclude large_shape for pre-existing NaN)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# BF16 baseline benchmark
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py \
  --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test

# FP8 blockscaled benchmark
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python benchmarks/moe-cute.py \
  --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
```

### Next steps (priority order)

1. **P0**: Eliminate blockscaled varlen GEMM pack/unpack overhead (training performance — see HANDOFF.md §4.1)
2. **P1**: Weight-grad blockscaled quantization (if precision requires)
3. **P2**: Memory optimization — unify weight caches

## 📋 FP8 Upgrade TODOs

- [x] Bootstrap the Blackwell-capable Python environment
- [x] Merge latest upstream `main` and validate local Blackwell behavior
- [x] Add Blackwell QuACK pytest entry (`make test-blackwell`)
- [x] FP8 protocol layer (`fp8_protocol.py`, `fp8_quant.py`, `fp8_reference.py`)
- [x] All 6 GEMM operators: FP8 tensor core forward + backward (per-tensor)
- [x] FP8 weight cache with version-aware invalidation
- [x] `blockscaled_fp8_gemm_varlen` prototype with rank-aware CUTLASS monkey-patch
- [x] 11 contract tests covering forward, backward, gradients, small+large shapes
- [x] **Forward + act-grad: blockscaled 1x32 UE8M0 integrated as default**
- [x] **Fused SwiGLU+quantize Triton kernels integrated**
- [x] **Weight cache eviction when blockscaled path activates**
- [ ] **Eliminate blockscaled varlen GEMM pack/unpack overhead** (P0, training 4x regression)
- [ ] **Weight-grad blockscaled quantization** (P1, currently per-tensor)
- [ ] **Unify FP8 weight cache layout** (P2)
- [ ] FP8 optimizer for master weight elimination (long-term)

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
