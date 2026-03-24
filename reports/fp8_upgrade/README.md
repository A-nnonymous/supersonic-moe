# FP8 Upgrade Progress

This file tracks the real state of the SonicMoE FP8 upgrade, not a wish list.

## Validated baseline

- environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- upstream base merged: `e9190f9`
- fork head carrying the latest validated work: the current `fork-main-sync` branch head
- Blackwell capability on this machine: `sm_100a`
- validated command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

- validated result: `18 passed, 91 skipped`
- opt-in faster command:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

- observed parallel runtime: `168.14s` vs. `187.28s` for the serial command

## Completed work

### Blackwell test path

- added `tests/moe_blackwell_test.py`
- added `make test-blackwell`
- documented the Blackwell smoke path in `README.md`

### Main test matrix cleanup

- fixed the upstream missing comma in `tests/moe_test.py`
- skip unsupported `KernelBackendMoE.sonicmoe + use_quack_gemm=False` cases on Blackwell
- keep `enable_quack_gemm(use_quack_gemm)` active around backward as well as forward

### FP8 planning conclusion

The next implementation step is **not** a monolithic end-to-end FP8 mega-kernel.

The first kernel milestone should be the Hopper up-projection epilogue:

```text
grouped_gemm(varlen/gather-A) -> SwiGLU -> optional prob -> 1x128 blockwise quant
```

Blackwell should continue to consume the same protocol through the QuACK adapter path rather than through a separate public API.

## Newly landed protocol scope

The current implementation intentionally supports only:

- activation dtype: `torch.float8_e4m3fn`
- scale encoding: `torch.float8_e8m0fnu`
- scale granularity: `1x128`
- runtime target: Blackwell with QuACK enabled

Implemented files:

- `sonicmoe/functional/fp8_protocol.py`
- `sonicmoe/functional/fp8_quant.py`
- `sonicmoe/functional/fp8_reference.py`
- `tests/fp8_protocol_test.py`

Important behavior:

- scale encoding is rounded **up** to the next power-of-two before storing in `e8m0`
- this avoids `e4m3` overflow-to-`nan` during activation quantization
- non-divisible tail blocks are padded internally to preserve `1x128` protocol semantics without rejecting widths like `2880`

## Newly landed functional-boundary wiring

The protocol is now visible at the current SonicMoE functional boundary:

- `MoE.forward(..., fp8_protocol=...)`
- `moe_TC_softmax_topk_layer(..., fp8_protocol=...)`

Current behavior:

- the boundary path quantizes/dequantizes the activation between `_UpProjection` and `_DownProjection`
- backward uses a straight-through estimator so the training graph stays usable
- this is intentionally a correctness-first step; it is not yet the final performance path

See `reports/fp8_upgrade/ENGINEERING_LOG.md` for the benchmark deltas from this change.

The latest optimization pass already improved the current boundary path:

- fp8 boundary peak memory: `15312.85 MiB -> 13017.85 MiB`
- fp8 boundary Fwd+Bwd time: `119.803 ms -> 109.117 ms`

This is useful progress, but it is still not enough; the torch-side boundary path remains slower than the bf16 baseline.

## Immediate next implementation targets

1. Replace the torch-side boundary quant/dequant with a fused up-projection epilogue
2. Land the paired backward kernel and cache contract
3. Reuse the same protocol for Blackwell QuACK activation paths
4. Keep the current benchmark command as the mandatory regression gate

## Source-of-truth inputs

- operator incubator workspace: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/operator-incubator`
- Paddle semantic references:
  - `Paddle_B/paddle/phi/kernels/fusion/gpu/fused_weighted_swiglu_act_quant_kernel.cu`
  - `Paddle_B/paddle/phi/kernels/fusion/gpu/fused_act_dequant_kernel.cu`
  - `Paddle_B/paddle/phi/kernels/fusion/gpu/fused_swiglu_weighted_bwd_kernel.cu`
  - `Paddle_B/paddle/phi/kernels/fusion/gpu/quant_utils.h`
- end-to-end FP8 reference behavior:
  - `supersonic-moe/tests/reference_layers/standalone_moe_layer/moe_standalone/moe/deep_ep_moe_layer.py`
  - `supersonic-moe/tests/reference_layers/standalone_moe_layer/moe_standalone/token_dispatcher/fp8_utils.py`

## Maintenance rule

Update this file whenever one of these changes:

- validated test command or result
- active branch/commit carrying the work
- immediate next implementation target
- authoritative source files for the next stage
