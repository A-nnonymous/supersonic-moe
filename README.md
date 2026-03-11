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

The FP8 multi-agent control plane has been migrated out of this repository into the standalone sibling repository `../warp`.

### Control Plane Deployment

The default deployment target is a Linux machine with NVIDIA Hopper or Blackwell GPUs already available, with the full SonicMoE runtime environment installed and ready for direct testing.

- recommended hardware: H100, H200, B200, or GB200
- recommended runtime state: CUDA, PyTorch, Triton, and SonicMoE dependencies already installed in the active environment
- recommended workflow: launch the standalone `warp` control plane against this SonicMoE checkout on the same provisioned machine

If you are testing Blackwell kernels directly on B200 or GB200, export `USE_QUACK_GEMM=1` in the worker environment before running Blackwell-specific commands.

### Control Plane Quickstart

Use the external `warp` checkout as the canonical entrypoint. If you keep it as a sibling repository, these commands cover the normal manager workflow and match the runtime defaults:

```bash
python3 ../warp/runtime/control_plane.py serve
python3 ../warp/runtime/control_plane.py up
python3 ../warp/runtime/control_plane.py stop-agents
python3 ../warp/runtime/control_plane.py silent
python3 ../warp/runtime/control_plane.py stop-all
```

Or launch the same commands from this repository through the convenience wrapper:

```bash
make warp-serve
make warp-up
make warp-stop-agents
make warp-silent
make warp-stop-all
```

- `serve` starts the dashboard only
- `up` starts the dashboard and launches all configured workers immediately
- `stop-agents` stops workers but keeps the dashboard available
- `silent` closes only the dashboard listener and leaves workers running
- `stop-all` stops both the listener and the worker fleet

For the default deployment, these commands already do the reliable thing:

- they use `runtime/local_config.yaml` when it exists
- they bind the dashboard to `0.0.0.0:8233` unless you override host or port
- `serve` detaches by default so the control plane keeps running after the shell returns

Before using `up`, fill `../warp/runtime/local_config.yaml` with:

- resource pool API keys
- provider and model assignments
- Paddle absolute path
- shared worker defaults for environment path, sync command, test command, submit strategy, and default git identity
- per-worker worktree paths and branches
- only the per-worker overrides that truly differ from the shared defaults

If you want browser auto-open during local debugging, add `--open-browser` to either launch command:

```bash
python3 ../warp/runtime/control_plane.py serve --open-browser
python3 ../warp/runtime/control_plane.py up --open-browser
```

Use additional parameters only when you actually need them:

- `--foreground`: keep `serve` attached to the current shell instead of detaching
- `--bootstrap`: force template-backed cold-start mode when you want to edit from the default template
- `--host 127.0.0.1`: do not expose the dashboard on all interfaces
- `--port 9000`: move the listener to a different port
- `--config <path>`: load a non-default runtime config file
- `--log-file <path>`: change the detached log path
- `--detach`: force detach on a non-`serve` command

For lightweight manager machines that do not carry the full CUDA stack, keep the same command shape and only swap the launcher:

```bash
uv run --no-project --with 'PyYAML>=6.0.2' python ../warp/runtime/control_plane.py serve
```

That fallback launcher also supports the same optional parameters, for example:

```bash
uv run --no-project --with 'PyYAML>=6.0.2' python ../warp/runtime/control_plane.py serve --bootstrap --open-browser
```

### Control Plane Behavior

- the first screen shows every manager and worker agent as a status card
- the overview page stays focused on agent dashboards and overall program progress
- the dashboard now uses a compiled React frontend served from `../warp/runtime/web/static/`
- the dashboard can save validated form-based config, launch workers, restart workers, enter silent mode, stop agents, stop all, and copy startup commands
- the Settings page now uses `worker_defaults` plus lean per-worker overrides so common fields are filled once instead of repeated on every worker
- resource pools are shown in a horizontal strip so provider routing stays visible without a long vertical form
- the launch bar supports first-run Copilot, explicit provider/model pinning, and elastic provider selection
- worker launch decisions use static pool priority plus runtime connection-quality and work-quality scoring
- each worker can still override git identity, test command, environment, and routing when it needs a non-default path, and A0 owns final merge into the integration branch
- every worker may still declare its own `resource_pool_queue` for fallback ordering when the default queue is not enough
- provider queue, runtime topology, heartbeats, and validation errors remain available in the dashboard

If you change the frontend source under `../warp/runtime/web/src/`, rebuild it with:

```bash
cd ../warp/runtime/web
npm install
npm run build
```

### Control Plane Notes

- on a real Hopper or Blackwell deployment host, the direct `python ...` commands above are the default path
- keep `worker_defaults.test_command` and benchmark commands pointed at the same CUDA-capable environment that will run real validation on the target GPU
- the control plane source of truth is now `../warp/README.md`
- use `../warp/README.md` for the full manager workflow, including cold-start, start, pause, and resume guidance
- stop commands are available separately for agents, listener, or both: `stop-agents`, `stop-listener`, and `stop-all`

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
