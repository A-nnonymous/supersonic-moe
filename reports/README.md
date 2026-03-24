# SonicMoE Work Reports

This directory is the live work log for the FP8 upgrade effort. It is not meant to hold speculative essays; it should hold the current state, the validated commands, and the next handoff targets.

## Current status

- authoritative Python environment: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- Blackwell path: QuACK-enabled (`USE_QUACK_GEMM=1`)
- latest validated fork commit: `aaa8bfb` on `fork/main`
- latest targeted validation:
  - `python -m pytest -q tests/moe_test.py tests/moe_blackwell_test.py`
  - result: `14 passed, 91 skipped`

## What is already done

- upstream `main` was merged into the local branch before the fork push
- `tests/moe_test.py` was made Blackwell-aware
- `tests/moe_blackwell_test.py` was added as a dedicated QuACK smoke/regression test
- `Makefile` now exposes `make test-blackwell`
- `README.md` now carries the active TODO list and the FP8 roadmap instead of stale control-plane material

## What the next agent should do first

1. Read `reports/fp8_upgrade/HANDOFF.md`
2. Confirm the environment with `source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate`
3. Re-run `python -m pytest -q tests/moe_test.py tests/moe_blackwell_test.py`
4. Start Stage 0 / Stage 1 FP8 protocol work in `sonicmoe/functional/`

## Working rule

Whenever the active plan changes, update this directory first so the next agent does not need to re-discover state from commit history or chat logs.
