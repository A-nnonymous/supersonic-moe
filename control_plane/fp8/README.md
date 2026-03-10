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
	Also set `project.integration_branch`, optional `project.manager_git_identity`, and any per-worker `git_identity` values you want the runtime to apply inside worker worktrees.
3. Run `python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start only the local web control plane.
4. Run `python control_plane/fp8/runtime/control_plane.py up --config control_plane/fp8/runtime/local_config.yaml --open-browser` to start the web control plane and launch workers in one process.
5. Override the bind address if needed with `--host` and `--port`, for example `--host 0.0.0.0 --port 9000`.
6. If you bind to `0.0.0.0`, open `http://<server-hostname-or-ip>:8233` from another machine instead of `127.0.0.1`.
7. On startup, the runtime now prints the effective listen address and a remote access URL hint to stderr.

### Fire-and-forget mode

If you want the control plane to keep running after the shell returns, use detached mode:

`python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --host 0.0.0.0 --port 8233 --detach`

The detached process writes combined stdout and stderr to `control_plane/fp8/runtime/control_plane.log` by default. You can override that path with `--log-file`.

If your working foreground command uses `uv run ... python`, keep that same launcher and append `--detach` at the end instead of switching interpreters.

### Compatibility fallback

If you need to run the control plane from a lightweight manager machine that does not carry the full CUDA stack, use the standalone path below instead of the default GPU-host deployment path above:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser`

### Remote access note

If your deployment hostname resolves to IPv6 first, an IPv6-only listener can still look like `ERR_CONNECTION_REFUSED` from the browser when your clients reach the node over IPv4. The runtime now always brings up an IPv4 listener first when you pass `--host 0.0.0.0`, then adds IPv6 as a secondary listener when possible.

## Dashboard capabilities

The local webpage now provides:

- a compact top bar for launch, restart, stop, refresh, and command copy actions
- an `Overview` page that shows agent dashboards, overall delivery progress, and branch merge status at a glance
- an `Operations` page for commands, validation, provider queue, merge queue, runtime state, heartbeats, backlog, and manager report
- a `Settings` page for API keys, provider routing, Paddle path, worktrees, worker commands, and git submission identities
- an editable local YAML config with supporting project, resource-pool, and worker summaries
- provider priority queue with runtime connection-quality and work-quality scoring
- per-worker git commit identities that are applied inside each worker worktree before launch
- a manager-owned merge queue that tracks which worker branch should be integrated into the target branch

## Interaction model

The dashboard is intentionally simple:

1. start on `Overview` to judge agent health, overall delivery progress, and which worker branches are waiting for manager merge review
2. move to `Operations` when you need runtime inspection, launch commands, validation, or provider scheduling detail
3. move to `Settings` when you need to edit API keys, provider assignments, Paddle paths, worker commands, or per-worker git identities
4. let A0 own merge timing and final integration into `project.integration_branch`
5. use `Launch`, `Restart`, or `Stop` from the top bar for normal operations
6. copy the `serve` or `up` command when you need terminal control

## Execution topology rule

Every real worker must be recorded in `state/agent_runtime.yaml` before it is counted as active.

The runtime record must capture at least:

- repository name
- resource pool
- provider
- model
- worktree path
- branch name
- merge target branch
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