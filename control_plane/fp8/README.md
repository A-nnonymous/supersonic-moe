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

Keep the high-frequency path simple:

- `serve`: start the dashboard only
- `up`: start the dashboard and launch all configured workers immediately
- `stop-agents`: stop workers and keep the dashboard running
- `silent`: close only the dashboard listener and keep workers running
- `stop-all`: stop both the listener and the worker fleet

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

### 1. Primary shell commands

Use these first. They match the runtime defaults and cover the normal manager loop:

`python control_plane/fp8/runtime/control_plane.py serve`

`python control_plane/fp8/runtime/control_plane.py up`

`python control_plane/fp8/runtime/control_plane.py stop-agents`

`python control_plane/fp8/runtime/control_plane.py silent`

`python control_plane/fp8/runtime/control_plane.py stop-all`

What these do by default:

- `serve` starts the dashboard only and detaches automatically
- `up` starts the dashboard and launches all configured workers
- `stop-agents` pauses the worker fleet without removing the dashboard
- `silent` removes only the listener and leaves workers alone
- `stop-all` performs the full shutdown path
- the default listener is `0.0.0.0:8233` unless you override it

### 2. Prepare real multi-agent execution

Fill `runtime/local_config.yaml` from the Settings page or by editing YAML directly. At minimum, replace placeholder values for:

- `project.local_repo_root`
- `project.paddle_repo_path`
- resource pool credentials or credential env vars
- `worker_defaults.environment_path`
- `worker_defaults.test_command`
- every worker `worktree_path`

Then use the settings workflow in this order:

1. set Project and Merge Policy once
2. tune Resource Pools in the horizontal pool strip
3. fill `worker_defaults` for the shared environment, sync, test, submit, and git identity behavior
4. add workers with only agent identity, branch, worktree, and any routing overrides that differ from the defaults
5. open the advanced override panel for a worker only when that worker truly needs a different environment path, test command, submit strategy, or git identity

Also set `project.integration_branch`, optional `project.manager_git_identity`, and any per-worker `git_identity` overrides you want the runtime to apply inside worker worktrees.

### 3. Optional launch flags

Add parameters only when the default path is not enough:

- `--open-browser`: open the dashboard automatically after startup
- `--foreground`: keep `serve` attached to the current shell instead of detaching
- `--bootstrap`: force template-backed cold-start mode
- `--host 127.0.0.1`: bind to a specific address instead of the default `0.0.0.0`
- `--port 9000`: move the listener to a different port
- `--config <path>`: load a non-default runtime config file
- `--log-file <path>`: change the detached log path
- `--detach`: force detach on a non-`serve` command

Examples:

`python control_plane/fp8/runtime/control_plane.py serve --open-browser`

`python control_plane/fp8/runtime/control_plane.py up --open-browser`

`python control_plane/fp8/runtime/control_plane.py serve --bootstrap --foreground`

### 4. Resume paused work

After `stop-agents`, use one of these paths:

1. Click `Launch` to start workers again with the current config.
2. Click `Restart` if you want a full stop-and-relaunch cycle.
3. Re-run `python control_plane/fp8/runtime/control_plane.py up` if the control plane itself is not running.

## Launch parameters

Use the short form by default:

- `serve` opens the control plane only
- `up` opens the control plane and launches workers

Add these only when needed:

- `--open-browser` opens the dashboard automatically
- `--foreground` disables the default fire-and-forget behavior of `serve`
- `--bootstrap` forces template-backed cold-start mode
- `--port 9000` to change the dashboard port
- `--host 127.0.0.1` or another address if you do not want the default `0.0.0.0`
- `--detach` only if you want to force detach on a non-serve command
- `--log-file <path>` to change the detached log path
- `--config <path>` only if you do not want the default `runtime/local_config.yaml`

## Stop commands

Use these shell commands when you want to control a running detached session from the same machine. The runtime now records per-port session state, so `--port 8233` targets the listener you actually care about even if another control-plane instance was started elsewhere.

- stop worker agents only: `python control_plane/fp8/runtime/control_plane.py stop-agents`
- enter silent mode by closing the dashboard listener only: `python control_plane/fp8/runtime/control_plane.py silent`
- compatibility alias for silent mode: `python control_plane/fp8/runtime/control_plane.py stop-listener`
- stop both the dashboard listener and all worker agents: `python control_plane/fp8/runtime/control_plane.py stop-all`

`silent` is the soft stop path:

- it closes the HTTP listener but leaves worker processes running
- it updates session state so later stop commands still know the listener is already gone
- it is the right option when you want to reduce exposure to scans without interrupting work

`stop-all` is the hard stop path:

- it terminates worker process groups, not just parent PIDs
- it waits for the target listener port to be released before reporting success
- when the dashboard is running on `8233`, the command verifies that `8233` is actually clean before it returns

### Fire-and-forget mode

`python control_plane/fp8/runtime/control_plane.py serve` is already the default detached path. It binds `0.0.0.0:8233` unless you override host or port. Use `--foreground` only when you want logs in the current shell.

The detached process writes combined stdout and stderr to `control_plane/fp8/runtime/control_plane.log` by default. You can override that path with `--log-file`.

If your working foreground command uses `uv run ... python`, keep that same launcher and append `--detach` at the end instead of switching interpreters.

### Compatibility fallback

If you need to run the control plane from a lightweight manager machine that does not carry the full CUDA stack, keep the same command shape and only swap the launcher:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve`

That same standalone path supports the same optional flags, for example:

`uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --bootstrap --open-browser`

### Remote access note

If your deployment hostname resolves to IPv6 first, an IPv6-only listener can still look like `ERR_CONNECTION_REFUSED` from the browser when your clients reach the node over IPv4. The runtime now brings up an IPv4 listener on `0.0.0.0` by default, then adds IPv6 as a secondary listener when possible.

## Dashboard capabilities

The local webpage now provides:

- a compact top bar for launch, restart, stop agents, stop all, refresh, and command copy actions
- a `Silent Mode` action that closes the listener without stopping workers
- launch modes for first-run Copilot, explicit provider/model pinning, or elastic provider selection
- an `Overview` page that shows agent dashboards, overall delivery progress, and branch merge status at a glance
- an `Operations` page for commands, validation, provider queue, merge queue, runtime state, heartbeats, backlog, and manager report
- a `Settings` page with responsive editable project, pool, merge-policy, worker-default, and worker-override forms
- a horizontal resource-pool strip so pool routing stays visible without a long vertical stack
- a `worker_defaults` layer that fills common environment, sync, test, submit, and git identity values once for the whole fleet
- lean worker cards that focus on identity and branch routing, with advanced overrides hidden until needed
- strict config validation before save, including path checks via `ls` and host checks via `ping`
- provider priority queue with runtime connection-quality and work-quality scoring
- per-worker git commit identities that are applied inside each worker worktree before launch
- a manager-owned merge queue that tracks which worker branch should be integrated into the target branch
- React-managed UI state so page navigation and actions stay interactive under refresh and cold-start edits

## Real usage pattern

For actual SonicMoE FP8 multi-agent delivery, the normal manager loop is:

1. bring the control plane up with `serve`
2. verify Settings and validation output are clean, starting with shared `worker_defaults`
3. use `Initial Copilot` for the first fleet launch, then switch to `Selected Model` or `Elastic` as needed before pressing `Launch`
4. monitor agent health, backlog progress, and branch merge status from `Overview`
5. inspect provider routing, runtime topology, and heartbeats in `Operations`
6. press `Stop Agents` or run `stop-agents` when you want to pause the worker fleet without losing dashboard state
7. press `Stop All` or run `stop-all --port 8233` when you need the listener port and every worker process fully released
8. let A0 merge finished worker branches into `project.integration_branch`

## Interaction model

The dashboard is intentionally simple:

1. start on `Overview` to judge agent health, overall delivery progress, and which worker branches are waiting for manager merge review
2. move to `Operations` when you need runtime inspection, launch commands, validation, or provider scheduling detail
3. move to `Settings` when you need to edit API keys, provider assignments, Paddle paths, shared worker defaults, or a small number of per-worker overrides
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