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

- validated result: `17 passed, 91 skipped`

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

## Immediate next implementation targets

1. Wire the new protocol into `forward.py` / `backward.py` call boundaries
2. Land the Hopper fused up-projection epilogue
3. Land the paired backward kernel and cache contract
4. Reuse the same protocol for Blackwell QuACK activation paths

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
