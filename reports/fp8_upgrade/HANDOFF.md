# Next-Agent Handoff

This is the minimum context a new agent needs to continue work without replaying the entire history.

## 1. Environment

- activate: `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
- Python: `3.13.x`
- GPU target here: Blackwell (`sm_100a`)
- Blackwell route in SonicMoE: QuACK (`USE_QUACK_GEMM=1`)

## 2. Repository state

- fork remote contains the latest pushed work on `main`
- pushed commit: `aaa8bfb`
- key files changed so far:
  - `Makefile`
  - `README.md`
  - `tests/moe_test.py`
  - `tests/moe_blackwell_test.py`

## 3. Validated command

Run this before doing new FP8 work:

```bash
python -m pytest -q tests/moe_test.py tests/moe_blackwell_test.py
```

Expected result at handoff time:

```text
14 passed, 91 skipped
```

## 4. What has already been settled

- the Blackwell smoke path is real and regression-tested
- the main MoE test matrix no longer tries to force unsupported non-QuACK SonicMoE execution on Blackwell
- the next kernel target is the Hopper FP8 up-projection epilogue, not a standalone gather kernel and not a monolithic full-graph rewrite

## 5. The next concrete edits

### Stage 0: protocol freeze

Add:

- `sonicmoe/functional/fp8_protocol.py`
- `sonicmoe/functional/fp8_quant.py`
- `sonicmoe/functional/fp8_reference.py`

Define:

- `fp8_dtype`
- `scale_encoding`
- `scale_granularity`
- cache tensors required by backward
- backend adapter contract for Hopper vs. Blackwell

### Stage 1: reference path

Wire a reference FP8 flow through:

- `sonicmoe/functional/forward.py`
- `sonicmoe/functional/backward.py`
- `sonicmoe/functional/__init__.py`

The reference path should be able to express:

- grouped-gemm output activation
- SwiGLU
- optional router probability weighting
- blockwise quant/dequant
- backward cache consumption

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
  - `sonicmoe/functional/forward.py`
  - `sonicmoe/functional/backward.py`
  - `sonicmoe/functional/grouped_gemm.py`
  - `sonicmoe/functional/moe_config.py`
- Blackwell adapter:
  - `sonicmoe/quack_utils/gemm_interface.py`
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
- keep `reports/` up to date whenever the branch, validation command, or next target changes
