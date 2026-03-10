# Worker Launch Playbook

## Goal

Launch multiple workers from different providers in parallel without sharing mutable source trees, branches, or Python environments.

Fork repository name: `supersonic-moe`

## Core Rule

One worker equals one provider session, one worktree, one branch, and one environment.

No worker is considered active until all of the following are true:

1. its runtime entry exists in `control_plane/fp8/state/agent_runtime.yaml`
2. its heartbeat is visible in `control_plane/fp8/state/heartbeats.yaml`
3. its status file is initialized in `control_plane/fp8/status/agents/`
4. its checkpoint file is initialized in `control_plane/fp8/checkpoints/agents/`

## Manager Orchestrator

If you want the manager to create worktrees and launch workers automatically, use:

```bash
cp control_plane/fp8/runtime/config_template.yaml control_plane/fp8/runtime/local_config.yaml
uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py up --config control_plane/fp8/runtime/local_config.yaml --open-browser

```

If you only want the control webpage without launching workers immediately, use:

```bash
uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser
```

The orchestrator reads:

- resource pool api keys
- provider and model selection
- provider priority queue and per-worker fallback queue
- Paddle absolute path
- worktree and branch assignments
- environment and test commands

The dashboard is served on port `8233` by default.

The runtime is a single Python process that serves both the frontend and backend. The webpage can edit and save the local config, then launch, restart, or stop workers directly.

All configured resource pools are evaluated continuously into a priority queue. The queue combines static priority with runtime connection quality and work quality, then the manager chooses the best eligible pool for each worker from its declared fallback queue.

Before first launch, make sure `uv` is available. The control plane can run with `--no-project` plus `PyYAML`, which avoids pulling the full CUDA dependency stack onto manager machines such as macOS.

## Recommended Defaults

- Python environment manager: `uv`
- submission mode: `patch_handoff` unless the worker is trusted to produce clean commits
- branch pattern: `a<agent>_<task>`
- worktree root pattern: sibling directories next to the manager workspace

## Launch Sequence

### 1. Choose the worker slot

Pick an agent id and one concrete task from `state/backlog.yaml`.

Examples:

- `A1` for protocol freeze
- `A3` for Blackwell audit
- `A5` for baseline-aligned tests

### 2. Create the worktree

Example for `A1`:

```bash
git worktree add ../supersonic_moe_a1 -b a1_protocol_freeze
```

Guidelines:

- place the worktree outside the manager root when possible
- use a distinct branch for every worker
- do not reuse an existing dirty worktree for another worker

### 3. Create the environment

Inside the worker worktree:

```bash
uv sync
```

If the worker needs extra commands for test or build preparation, record them exactly in `state/agent_runtime.yaml`.

### 4. Register the runtime

Before coding, fill the worker entry in `control_plane/fp8/state/agent_runtime.yaml`.

Required fields to set:

- `repository_name: supersonic-moe`
- `resource_pool`
- `provider`
- `model`
- `launch_owner`
- `local_workspace_root`
- `worktree_path`
- `branch`
- `environment_type`
- `environment_path`
- `sync_command`
- `test_command`
- `submit_strategy`
- `status`

### 5. Initialize status and checkpoint

Update:

- `control_plane/fp8/status/agents/A*.md`
- `control_plane/fp8/checkpoints/agents/A*.md`

Minimum fields to write:

- provider
- worktree
- branch
- environment
- current task
- next test command

### 6. Start the provider session

Allowed providers include:

- Copilot
- Claude Code
- OpenCode

The provider does not matter to the control plane as long as the runtime, heartbeat, status, and checkpoint are recorded.

If you use the manager orchestrator in `runtime/control_plane.py`, it may generate the worktree, prompt, and launch command automatically from the local config.

### 7. Test inside the worker boundary

Run only the tests that match the worker scope first.

Examples:

- protocol worker: config and API checks
- backend audit worker: focused smoke or audit scripts
- test worker: the specific test module being added or modified

### 8. Submit back to the manager

Allowed submit modes:

- `patch_handoff`
- `cherry_pick`
- `merge_commit`

Manager rules:

- A0 owns final integration
- A0 decides whether to cherry-pick or merge
- no worker should integrate directly into the manager branch

## Provider Matrix Template

Use this pattern when planning concurrent workers:

| Agent | Provider | Worktree | Branch | Env | Submit |
|------|----------|----------|--------|-----|--------|
| A1 | copilot | `../supersonic_moe_a1` | `a1_protocol_freeze` | `uv` | `patch_handoff` |
| A3 | claude_code | `../supersonic_moe_a3` | `a3_blackwell_audit` | `uv` | `cherry_pick` |
| A5 | opencode | `../supersonic_moe_a5` | `a5_fp8_tests` | `uv` | `patch_handoff` |

## What The Manager Must Control

The manager should control:

- task assignment
- gate ordering
- runtime registration
- lock discipline
- merge timing
- final integration
- experiment approval on shared machines

The manager should not silently assume:

- that a worker actually launched
- that a provider session has a clean worktree
- that a branch name implies a valid environment
- that a heartbeat without runtime metadata is sufficient

## When To Stop And Re-Plan

Stop and re-plan if any of the following occurs:

- two workers need the same editable worktree
- two workers need the same branch
- a worker cannot declare its test command
- a worker cannot declare its submit mode
- a provider requires sharing mutable state with another worker
- a shared machine run is requested without a runtime entry