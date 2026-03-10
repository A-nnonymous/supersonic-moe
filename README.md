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

## Control Plane

This repository includes a local multi-agent control plane under `control_plane/fp8/` for planning, launching, and monitoring FP8 delivery work.

### Control Plane Deployment

The default deployment target is a Linux machine with NVIDIA Hopper or Blackwell GPUs already available, with the full SonicMoE runtime environment installed and ready for direct testing.

- recommended hardware: H100, H200, B200, or GB200
- recommended runtime state: CUDA, PyTorch, Triton, and SonicMoE dependencies already installed in the active environment
- recommended workflow: launch the control plane from the same provisioned repository checkout that will run FP8 tests and benchmarks

If you are testing Blackwell kernels directly on B200 or GB200, export `USE_QUACK_GEMM=1` in the worker environment before running Blackwell-specific commands.

### Control Plane Quickstart

1. Copy the runtime template:

```bash
cp control_plane/fp8/runtime/config_template.yaml control_plane/fp8/runtime/local_config.yaml
```

2. Fill `local_config.yaml` with:

- resource pool API keys
- provider and model assignments
- Paddle absolute path
- worker worktree paths and branches
- per-worker git commit identities when different agents should submit under different names
- real `test_command` values that can run immediately on the local Hopper or Blackwell host

3. Start only the local dashboard backend and frontend from the fully provisioned project environment:

```bash
python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser
```

4. Start the dashboard and launch configured workers in one step:

```bash
python control_plane/fp8/runtime/control_plane.py up --config control_plane/fp8/runtime/local_config.yaml --open-browser
```

5. Override bind address when needed:

```bash
python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --host 0.0.0.0 --port 8233
```

6. Optional compatibility path for non-GPU manager machines:

```bash
uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser
```

### Control Plane Behavior

- the first screen shows every manager and worker agent as a status card
- the overview page stays focused on agent dashboards and overall program progress
- the dashboard can save config, launch workers, restart workers, stop workers, and copy startup commands
- worker launch decisions use static pool priority plus runtime connection-quality and work-quality scoring
- each worker can carry its own git identity, and A0 owns final merge into the integration branch
- every worker may declare a `resource_pool_queue` for fallback ordering
- provider queue, runtime topology, heartbeats, and validation errors remain available in the dashboard

### Control Plane Notes

- on a real Hopper or Blackwell deployment host, the direct `python ...` commands above are the default path
- keep worker `test_command` and benchmark commands pointed at the same CUDA-capable environment that will run real validation on the target GPU
- the control plane source of truth remains `control_plane/fp8/README.md`

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
