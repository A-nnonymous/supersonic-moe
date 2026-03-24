# Next-Agent Handoff

This is the minimum context a new agent needs to continue work without replaying the entire history.

## 0. 最新中文结论

- Blackwell + QuACK 的 `fp8_protocol` 默认前向路径已经切到 pre-SwiGLU 融合量化：

```text
z(pre-SwiGLU) -> fused_weighted_swiglu_act_quant_best -> fused_act_dequant_best -> y1
```

- 这一步已经通过：
  - `tests/fp8_protocol_test.py` 的 targeted fused-path 回归
  - `tests/moe_blackwell_test.py` smoke 回归
- 这一步的收益来源：
  - 替换掉了旧的 post-SwiGLU torch-side quant/dequant。
  - 还没有改 backward 主 kernel，所以收益主要来自 forward。
- 这一步的遗留问题：
  - 精度相对旧 reference path 有回退。
  - 共享机小 shape 数据显示仍慢于 bf16。
  - `prob/topk_scores` 还没有被真正前移进 fused epilogue。

## 1. Environment

- activate: `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
- Python: `3.13.x`
- GPU target here: Blackwell (`sm_100a`)
- Blackwell route in SonicMoE: QuACK (`USE_QUACK_GEMM=1`)

## 2. Repository state

- fork remote contains the latest pushed work on `main`
- use the current branch head as the source of truth for protocol work; the last pushed pre-protocol doc commit was `57d7faa`
- key files changed so far:
  - `Makefile`
  - `README.md`
  - `tests/moe_test.py`
  - `tests/moe_blackwell_test.py`
  - `sonicmoe/functional/fp8_protocol.py`
  - `sonicmoe/functional/fp8_quant.py`
  - `sonicmoe/functional/fp8_reference.py`
  - `tests/fp8_protocol_test.py`

## 3. Validated command

Run this before doing new FP8 work:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Expected result at handoff time:

```text
18 passed, 91 skipped
```

Faster opt-in command on this machine:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

Observed runtime for the parallel entry:

```text
18 passed, 91 skipped in 168.14s
```

Multi-GPU shard launcher:

```bash
make test-blackwell-multigpu BLACKWELL_TEST_GPUS=0,1,2
python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run
```

Metric-reporting command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

共享机小 shape 对照命令（本轮已经跑过）：

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

## 4. What has already been settled

- the Blackwell smoke path is real and regression-tested
- the main MoE test matrix no longer tries to force unsupported non-QuACK SonicMoE execution on Blackwell
- the FP8 protocol is now code, not just planning
- the current protocol scope is fixed to `e4m3` activations + `e8m0` scales + `1x128` granularity
- the protocol is now threaded through `MoE.forward(..., fp8_protocol=...)` and `moe_TC_softmax_topk_layer(..., fp8_protocol=...)`
- the current functional-boundary implementation is intentionally correctness-first and is still slower than the baseline
- a first memory optimization pass already landed:
  - fp8 boundary peak memory: `15312.85 MiB -> 13017.85 MiB`
  - fp8 boundary Fwd+Bwd: `119.803 ms -> 109.117 ms`
- a new adapter landing point now exists in `sonicmoe/functional/fp8_cutely_fused.py`
- the first real high-performance forward step is now landed in the same file:
  - default Blackwell/QuACK forward path consumes `z` instead of `y1`
  - quant path uses `cutify.fused_weighted_swiglu_act_quant_best`
  - dequant path uses `cutify.fused_act_dequant_best`
- CUDA graph capture compatibility had to be fixed:
  - direct `ue8m0` packing inside the incubator path broke capture
  - current workaround is: kernel emits float32 dequant scale, SonicMoE re-encodes it to `e8m0`
- the incubator fused quant kernel does **not** match the current SonicMoE boundary 1:1:
  - incubator input contract: pre-SwiGLU `(T, 2H)`
  - current SonicMoE boundary: post-SwiGLU `(TK, I)`
  - this mismatch is now handled by the pre-SwiGLU bridge logic in `fp8_cutely_fused.py`
- the next kernel target is the Hopper FP8 up-projection epilogue, not a standalone gather kernel and not a monolithic full-graph rewrite

## 5. The next concrete edits

### Stage 1: 把 prob/topk_scores 真正吃进融合 epilogue

The protocol/reference modules are already wired through:

- `sonicmoe/moe.py`
- `sonicmoe/functional/__init__.py`
- `sonicmoe/functional/fp8_reference.py`
- `sonicmoe/functional/fp8_cutely_fused.py`

The next step should remove the remaining semantic gap for:

- optional router probability weighting
- post-router scaling placement
- backward cache consumption
- backward fused kernel contract

### Stage 2: first fused kernel

Implement the Hopper-side fused up-projection epilogue equivalent to:

```text
grouped_gemm(varlen/gather-A) -> SwiGLU -> optional prob -> 1x128 quant
```

Reuse:

- SonicMoE routing metadata and grouped GEMM structure
- Paddle fused op semantics
- operator-incubator CuTe prototypes

## 6. File map for fast navigation

- SonicMoE execution path:
  - `sonicmoe/moe.py`
  - `sonicmoe/functional/__init__.py`
  - `sonicmoe/functional/forward.py`
  - `sonicmoe/functional/backward.py`
- Blackwell adapter:
  - `sonicmoe/quack_utils/gemm_gated.py`
  - `sonicmoe/quack_utils/gemm_dgated.py`
- operator incubator kernels:
  - `operator-incubator/cutify/ops/cute/fused_weighted_swiglu_act_quant.py`
  - `operator-incubator/cutify/ops/cute/fused_act_dequant.py`
  - `operator-incubator/cutify/ops/cute/fused_swiglu_weighted_bwd.py`

## 7. Do not re-discover these points

- `dev_b` is not the authoritative environment for SonicMoE because it is on Python 3.10
- `xfer` is the environment to use for SonicMoE work
- Blackwell currently relies on QuACK; do not expect the default Hopper-only path to compile on `sm_100a`
- current torch already provides both `torch.float8_e4m3fn` and `torch.float8_e8m0fnu`; a nightly upgrade is not currently required
- the benchmark gate for performance-facing work is now documented in `reports/fp8_upgrade/ENGINEERING_LOG.md`
- `pytest-xdist` is now installed in `xfer`; keep the worker count conservative (`2`) for the single-GPU Blackwell regression path
- the reporting policy is fixed:
  - accuracy baseline: official bf16
  - memory baseline: official bf16
  - performance baselines: previous commit and official bf16
- from this turn onward, engineering records should be written in Chinese and the metric annotations must come first
- keep `reports/` up to date whenever the branch, validation command, or next target changes
