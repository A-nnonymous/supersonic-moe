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

## Quickstart

1. Copy `runtime/config_template.yaml` to `runtime/local_config.yaml`.
2. For the control plane itself, you can run a standalone runtime with `uv run --no-project --with 'PyYAML>=6.0.2' python ...` even on machines that cannot install the full CUDA stack.
3. Fill resource pool api keys, provider/model assignments, worktree paths, and `paddle_repo_path`.
4. Run `uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start only the local web control plane.
5. Run `uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py up --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start the web control plane and launch workers in one process.
6. Override the bind address if needed with `--host` and `--port`, for example `--port 9000`.
7. Open `http://127.0.0.1:8233` if the browser does not open automatically.

## Dashboard capabilities

The local webpage now provides:

- editable local YAML config for resource pool keys, provider/model selection, Paddle path, worktrees, branches, and commands
- one-click save for config changes
- one-click launch, restart, and stop for configured workers
- generated main-agent startup commands for both `serve` and `up` modes
- provider priority queue with runtime connection-quality and work-quality scoring
- live views for runtime topology, heartbeats, backlog, gates, worker config, and manager report

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