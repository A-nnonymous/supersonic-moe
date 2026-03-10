# SonicMoE FP8 Control Plane

This directory is the multi-agent control plane for upgrading SonicMoE with FP8 support.

Its job is to keep the FP8 program executable as coordinated manager and worker work, not as ad hoc local notes. The control plane owns:

- agent planning and backlog state
- worker launch and stop orchestration
- provider and resource-pool routing
- branch and merge visibility for manager-owned integration
- resumable checkpoints, heartbeats, and runtime status

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

## Operating modes

Use the control plane in one of these modes:

- `serve --bootstrap`: the explicit cold-start path; it accepts template-backed editing immediately and detaches automatically on remote-first startup unless you pass `--foreground`
- `serve`: open the dashboard with your saved config, but do not launch workers until you explicitly press `Launch` or `Restart`
- `up`: start the dashboard and immediately start all configured workers

If `runtime/local_config.yaml` is missing, the runtime automatically falls back to `runtime/config_template.yaml` and enters cold-start mode. If you later save a real config from the Settings page, it is written to `runtime/local_config.yaml` automatically.

## Frontend architecture

The dashboard frontend is now served as compiled static assets from `runtime/web/static/`.

- source code lives in `runtime/web/src/`
- the runtime serves `index.html`, `app.js`, and `app.css` directly instead of embedding HTML and JavaScript inside `control_plane.py`
- UI state is managed in React so tabs, top-bar actions, settings editing, and refresh behavior share one predictable state model

When you change the frontend source, rebuild it with:

`cd control_plane/fp8/runtime/web && npm install && npm run build`

The built assets in `runtime/web/static/` are what the Python runtime serves in production.

## Quickstart

### 1. Default remote startup

Use this as the primary bring-up path on a remote machine. It starts the control plane in cold-start mode, listens on all interfaces, and returns the shell immediately:

`python control_plane/fp8/runtime/control_plane.py serve --bootstrap`

If you do not want background mode, add `--foreground`.

### 2. Prepare real multi-agent execution

Fill `runtime/local_config.yaml` from the Settings page or by editing YAML directly. At minimum, replace placeholder values for:

- `project.local_repo_root`
- `project.paddle_repo_path`
- worker `worktree_path`
- worker `environment_path`
- resource pool credentials or credential env vars
- real worker `test_command` values

Also set `project.integration_branch`, optional `project.manager_git_identity`, and any per-worker `git_identity` values you want the runtime to apply inside worker worktrees.

### 3. Start the dashboard only

Use this when you want the manager to inspect config, queues, and branch state first, then launch agents from the UI:

`python control_plane/fp8/runtime/control_plane.py serve --open-browser`

### 4. Start the dashboard and launch all configured agents

Use this when your config is ready and the manager wants to begin active multi-agent execution immediately:

`python control_plane/fp8/runtime/control_plane.py up --open-browser`

### 5. Pause running multi-agent work

Use one of these paths:

1. Preferred: click `Stop Agents` in the top bar. This stops all worker processes while keeping the dashboard online.
2. Remote or scripted control: `python control_plane/fp8/runtime/control_plane.py stop-agents`
3. Full shutdown: stop the foreground process with `Ctrl-C`, or stop the detached control-plane process from the shell.

### 6. Resume paused work

After a pause, use one of these paths:

1. Click `Launch` to start workers again with the current config.
2. Click `Restart` if you want a full stop-and-relaunch cycle.
3. Re-run `python control_plane/fp8/runtime/control_plane.py up --open-browser` if the control plane itself is not running.

## Launch parameters

Use the short form by default:

- `serve` opens the control plane only
- `up` opens the control plane and launches workers
- `--bootstrap` forces template-backed cold-start mode
- `--open-browser` opens the dashboard automatically
- `--foreground` disables the auto-detach behavior for remote cold-start `serve`

Add these only when needed:

- `--host 0.0.0.0` to override the listener explicitly
- `--port 9000` to change the dashboard port
- `--detach` to keep the control plane running after the shell returns
- `--log-file <path>` to change the detached log path
- `--config <path>` only if you do not want the default `runtime/local_config.yaml`

## Stop commands

Use these shell commands when you want to control a running detached session from the same machine. The runtime now records per-port session state, so `--port 8233` targets the listener you actually care about even if another control-plane instance was started elsewhere.

- stop worker agents only: `python control_plane/fp8/runtime/control_plane.py stop-agents`
- stop the dashboard listener only: `python control_plane/fp8/runtime/control_plane.py stop-listener`
- stop both the dashboard listener and all worker agents: `python control_plane/fp8/runtime/control_plane.py stop-all`

`stop-all` is the hard stop path:

- it terminates worker process groups, not just parent PIDs
- it waits for the target listener port to be released before reporting success
- when the dashboard is running on `8233`, the command verifies that `8233` is actually clean before it returns

### Fire-and-forget mode

If you want the control plane to keep running after the shell returns, use detached mode:

`python control_plane/fp8/runtime/control_plane.py serve --bootstrap`

The detached process writes combined stdout and stderr to `control_plane/fp8/runtime/control_plane.log` by default. You can override that path with `--log-file`.

If your working foreground command uses `uv run ... python`, keep that same launcher and append `--detach` at the end instead of switching interpreters.

### Compatibility fallback

If you need to run the control plane from a lightweight manager machine that does not carry the full CUDA stack, use the standalone path below instead of the default GPU-host deployment path above:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --open-browser`

That same standalone path also supports cold-start startup with no `local_config.yaml` present:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --bootstrap --open-browser`

### Remote access note

If your deployment hostname resolves to IPv6 first, an IPv6-only listener can still look like `ERR_CONNECTION_REFUSED` from the browser when your clients reach the node over IPv4. The runtime now always brings up an IPv4 listener first when you pass `--host 0.0.0.0`, then adds IPv6 as a secondary listener when possible.

## Dashboard capabilities

The local webpage now provides:

- a compact top bar for launch, restart, stop agents, stop all, refresh, and command copy actions
- an `Overview` page that shows agent dashboards, overall delivery progress, and branch merge status at a glance
- an `Operations` page for commands, validation, provider queue, merge queue, runtime state, heartbeats, backlog, and manager report
- a `Settings` page for API keys, provider routing, Paddle path, worktrees, worker commands, and git submission identities
- an editable local YAML config with supporting project, resource-pool, and worker summaries
- provider priority queue with runtime connection-quality and work-quality scoring
- per-worker git commit identities that are applied inside each worker worktree before launch
- a manager-owned merge queue that tracks which worker branch should be integrated into the target branch
- React-managed UI state so page navigation and actions stay interactive under refresh and cold-start edits

## Real usage pattern

For actual SonicMoE FP8 multi-agent delivery, the normal manager loop is:

1. bring the control plane up with `serve --bootstrap` or `serve --open-browser`
2. verify Settings and validation output are clean
3. press `Launch` or run `up --open-browser`
4. monitor agent health, backlog progress, and branch merge status from `Overview`
5. inspect provider routing, runtime topology, and heartbeats in `Operations`
6. press `Stop Agents` or run `stop-agents` when you want to pause the worker fleet without losing dashboard state
7. press `Stop All` or run `stop-all --port 8233` when you need the listener port and every worker process fully released
8. let A0 merge finished worker branches into `project.integration_branch`

## Interaction model

The dashboard is intentionally simple:

1. start on `Overview` to judge agent health, overall delivery progress, and which worker branches are waiting for manager merge review
2. move to `Operations` when you need runtime inspection, launch commands, validation, or provider scheduling detail
3. move to `Settings` when you need to edit API keys, provider assignments, Paddle paths, worker commands, or per-worker git identities
4. let A0 own merge timing and final integration into `project.integration_branch`
5. use `Launch`, `Restart`, `Stop Agents`, or `Stop All` from the top bar for normal operations
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