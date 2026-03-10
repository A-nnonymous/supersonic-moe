# SonicMoE FP8 Control Plane

This directory is the operational control plane for SonicMoE FP8 agentic delivery.

Fork repository name: `supersonic-moe`

## Structure

- `strategy/`: program intent, scope, and baseline mapping
- `governance/`: operating rules, decisions, and machine policy
- `state/`: live backlog, gates, heartbeats, and lock state
- `status/agents/`: live worker status feeds
- `checkpoints/`: resumable manager and worker snapshots
- `experiments/`: experiment registry
- `reports/`: production-facing reporting and delivery artifacts

## Minimum files to read

1. `checkpoints/manager/latest.md`
2. `reports/manager_report.md`
3. `state/backlog.yaml`
4. `state/gates.yaml`
5. `state/heartbeats.yaml`
6. `state/edit_locks.yaml`
7. `state/agent_runtime.yaml`
8. `status/agents/`
9. `checkpoints/agents/`
10. `experiments/registry.yaml`
11. `strategy/integration_plan.md`
12. `strategy/baseline_trace.md`

## Rule

No work is considered active unless it is reflected here.

## Startup entrypoints

- cold start on a new machine: `new_machine_prompt.md`
- resume an interrupted session: `RESUME.md`
- launch workers: `governance/worker_launch_playbook.md`
- run integrated frontend and backend: `runtime/control_plane.py`
- current production snapshot: `reports/manager_report.md`

## Deployment assumption

The default target is a Linux machine with Hopper or Blackwell GPUs and a fully provisioned SonicMoE runtime environment.

- expected hardware: H100, H200, B200, or GB200
- expected environment: CUDA, PyTorch, Triton, and SonicMoE dependencies are already installed and usable from the active shell
- expected behavior: worker `test_command` values can run immediately on the same machine without extra bootstrap steps

If you are validating Blackwell kernels directly on B200 or GB200, set `USE_QUACK_GEMM=1` in the worker environment before launching those jobs.

## Quickstart

1. Copy `runtime/config_template.yaml` to `runtime/local_config.yaml`.
2. Fill resource pool api keys, provider/model assignments, worktree paths, `paddle_repo_path`, and real GPU test commands.
3. Run `python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start only the local web control plane.
4. Run `python control_plane/fp8/runtime/control_plane.py up --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start the web control plane and launch workers in one process.
5. Override the bind address if needed with `--host` and `--port`, for example `--host 0.0.0.0 --port 9000`.
6. Open `http://127.0.0.1:8233` if the browser does not open automatically.

### Compatibility fallback

If you need to run the control plane from a lightweight manager machine that does not carry the full CUDA stack, use the standalone path below instead of the default GPU-host deployment path above:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser`

## Dashboard capabilities

The local webpage now provides:

- a compact top bar for save, launch, restart, stop, refresh, and command copy actions
- a default first screen that shows every manager and worker agent as a status card
- summary cards for visible agents, attention-needed agents, validation count, and last event
- an editable local YAML config in a collapsible editor
- commands, validation, and provider queue directly below the agent overview
- low-frequency sections such as worker config, project/processes, operational state, and reference data folded behind details panels
- provider priority queue with runtime connection-quality and work-quality scoring

## Interaction model

The dashboard is intentionally simple:

1. check the agent wall first to see who is active, parked, stale, or offline
2. confirm the summary cards and validation block
3. fix config issues in the collapsible editor if needed
4. copy the `serve` or `up` command when you need terminal control
5. use `Launch`, `Restart`, or `Stop` from the top bar for normal operations
6. open the folded sections only when you need deeper runtime inspection

## Execution topology rule

Every real worker must be recorded in `state/agent_runtime.yaml` before it is counted as active.

The runtime record must capture at least:

- repository name
- resource pool
- provider
- model
- worktree path
- branch name
- environment path or `uv` environment root
- test command
- submit path back to the integration branch

## Concurrency rule

High-conflict control files are single-writer files. An agent must claim them in `state/edit_locks.yaml` before editing them.

## Agent checkpoint rule

Each worker agent must maintain both:

- `status/A*.md` for current live status
- `checkpoints/agents/A*.md` for resumable checkpoint state

## Heartbeat rule

Agent liveness is tracked in `state/heartbeats.yaml`.

- `healthy`: agent heartbeat is within the expected window
- `stale`: agent previously reported but missed the expected window
- `not-started`: agent has no runtime heartbeat yet
- `offline`: agent was intentionally stopped or released

The manager must include heartbeat state in every production status report.