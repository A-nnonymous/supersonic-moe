# Next-Agent Handoff

This is the minimum context a new agent needs to continue work without replaying the entire history.

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
- the next kernel target is the Hopper FP8 up-projection epilogue, not a standalone gather kernel and not a monolithic full-graph rewrite

## 5. The next concrete edits

### Stage 1: replace the torch-side boundary path with a fused epilogue

The protocol/reference modules are already wired through:

- `sonicmoe/functional/forward.py`
- `sonicmoe/functional/backward.py`
- `sonicmoe/functional/__init__.py`

The next step should remove the current torch-side quant/dequant scaffold for:

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
- current torch already provides both `torch.float8_e4m3fn` and `torch.float8_e8m0fnu`; a nightly upgrade is not currently required
- the benchmark gate for performance-facing work is now documented in `reports/fp8_upgrade/ENGINEERING_LOG.md`
- `pytest-xdist` is now installed in `xfer`; keep the worker count conservative (`2`) for the single-GPU Blackwell regression path
- keep `reports/` up to date whenever the branch, validation command, or next target changes
