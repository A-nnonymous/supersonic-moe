# SonicMoE Work Reports

This directory is the live work log for the FP8 upgrade effort. It is not meant to hold speculative essays; it should hold the current state, the validated commands, and the next handoff targets.

## Current status

- authoritative Python environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- Blackwell path: QuACK-enabled (`USE_QUACK_GEMM=1`)
- latest validated fork state: the current `fork-main-sync` working tree carrying the Blackwell FP8 protocol changes
- latest targeted validation:
  - serial: `python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py`
  - serial result: `18 passed, 91 skipped`
  - opt-in parallel: `make test-blackwell-parallel PYTEST_WORKERS=2`
  - parallel result: `18 passed, 91 skipped in 168.14s`

## What is already done

- upstream `main` was merged into the local branch before the fork push
- `tests/moe_test.py` was made Blackwell-aware
- `tests/moe_blackwell_test.py` was added as a dedicated QuACK smoke/regression test
- `Makefile` now exposes `make test-blackwell`
- `README.md` now carries the active TODO list and the FP8 roadmap instead of stale control-plane material
- a Blackwell-only FP8 protocol layer now exists in:
  - `sonicmoe/functional/fp8_protocol.py`
  - `sonicmoe/functional/fp8_quant.py`
  - `sonicmoe/functional/fp8_reference.py`
- the protocol is wired through `MoE.forward(..., fp8_protocol=...)` and `moe_TC_softmax_topk_layer(..., fp8_protocol=...)`
- current protocol scope is intentionally constrained to:
  - activation dtype: `torch.float8_e4m3fn`
  - scale encoding: `torch.float8_e8m0fnu`
  - scale granularity: `1x128`
  - runtime target: Blackwell + QuACK enabled
- current FP8 boundary path is still slower than baseline; see `reports/fp8_upgrade/ENGINEERING_LOG.md` for exact numbers

## What the next agent should do first

1. Read `reports/fp8_upgrade/HANDOFF.md`
2. Confirm the environment with `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
3. Re-run `python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py`
4. Start the first fused FP8 kernel work at the up-projection epilogue

## Working rule

Whenever the active plan changes, update this directory first so the next agent does not need to re-discover state from commit history or chat logs.
