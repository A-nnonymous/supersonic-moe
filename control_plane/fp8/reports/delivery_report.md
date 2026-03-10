# SonicMoE FP8 Delivery Report

Owner: A0
Program: sonicmoe_fp8
Status: control_plane ready

## Scope

The active operational root is `control_plane/fp8/`.

This control plane keeps only current program state:

- `strategy/`: current plan and baseline mapping
- `governance/`: current operating rules and decisions
- `state/`: backlog, gates, heartbeats, and locks
- `status/agents/`: live worker state
- `checkpoints/`: current resumable state
- `experiments/`: experiment ledger
- `reports/`: manager view and operator notes

## Current Readiness

- control_plane structure is in place
- heartbeat and lock control are active
- manager and worker checkpoints are present
- new machine startup is defined
- implementation has not started

## Baseline Facts

- final delivery remains torch-only
- `tests/reference_layers/standalone_moe_layer` is the main correctness baseline
- `tests/reference_layers/standalone_moe_layer/moe_standalone/compat.py` is the key compatibility shim
- Paddle remains semantic reference plus compatibility input
- G0 protocol freeze is still the next gate

## Active Entry Points

- current overview: `reports/manager_report.md`
- interrupted-session resume: `RESUME.md`
- new-machine startup: `new_machine_prompt.md`

## Next Action

1. Restore from `control_plane/fp8/RESUME.md` or `control_plane/fp8/new_machine_prompt.md`.
2. Report gates, blockers, heartbeats, and locks.
3. Freeze A1 protocol and A6 baseline contract before implementation.
