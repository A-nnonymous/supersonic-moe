# Control Plane Playbook

## Goal

Operate or evolve the FP8 control plane without losing the simplified workflow that recent iterations established.

## Use This Playbook When

- bringing the control plane up on a new machine
- resuming control-plane work in a new session
- changing launch, stop, settings, validation, or worker-planning behavior
- reviewing whether a proposed change matches the intended operator workflow

## Read Order

1. `control_plane/fp8/README.md`
2. `control_plane/fp8/governance/worker_launch_playbook.md`
3. `control_plane/fp8/runtime/control_plane.py`
4. `control_plane/fp8/runtime/config_template.yaml`
5. `control_plane/fp8/runtime/web/src/App.tsx`
6. `control_plane/fp8/state/backlog.yaml`
7. `control_plane/fp8/state/agent_runtime.yaml`

## Working Rules

1. Preserve the simplest reliable operator path first.
2. Do not reintroduce dry-run-first or YAML-first workflow unless explicitly requested.
3. Keep shared defaults in `worker_defaults`; keep worker-specific cards lean.
4. A0 should auto-fill safe values like task IDs, branch names, worktree paths, test commands, and default routing instead of forcing manual entry.
5. Let operators validate and save one settings block at a time.
6. Keep worker roster logic synchronized with backlog/runtime plan.
7. Use provider availability plus persisted quality history when A0 recommends or locks a resource pool.
8. Keep docs and runtime behavior aligned in the same change.

## High-Frequency Operator Path

1. Start the dashboard with `python control_plane/fp8/runtime/control_plane.py serve`.
2. Open Settings and fill only the project paths and pool credentials A0 cannot infer.
3. Configure Resource Pools.
4. Let A0 hydrate `worker_defaults` and worker roster from runtime/backlog first.
5. Change per-worker fields only when a worker is a real exception to the default path.
6. Validate and save only the section being edited.
7. Launch with the default flow or the selected launch policy.
8. Use `stop-agents`, `silent`, or `stop-all` according to how much of the system should remain alive.

## Settings Checklist

### Project

- set `repository_name`
- set `local_repo_root`
- set `reference_workspace_root` only if the project has a shared reference repo or baseline workspace
- confirm dashboard host/port behavior if changed
- verify worktree auto-derivation still makes sense from the project root

### Task Policies

- prefer explicit backlog `task_type` fields over title-based inference
- keep `task_policies.rules` and `task_policies.types` aligned so A0 routing stays data-driven
- set a shared default test command only in `task_policies.defaults` or `worker_defaults` when every task really shares it

### Merge Policy

- set `integration_branch`
- set optional `manager_git_identity` only when needed
- confirm merge behavior still leaves final integration under A0 / manager control

### Resource Pools

- verify provider names and credentials
- verify priority/quality logic still maps to intended routing
- keep layout horizontally scannable

### Worker Defaults

- set shared environment type/path
- set shared sync command
- set shared test command only if you need to override A0's task-aware default selection
- set shared submit strategy
- set shared git identity if workers should inherit one

### Workers

- sync from backlog/runtime if roster should match plan
- confirm all expected agents are present, not just template leftovers
- verify A0-generated branch naming is plausible
- verify A0-generated worktree paths are unique
- verify A0-selected test commands and pool locks only where the task is unusual
- open advanced overrides only for real exceptions

## Change Checklist For Agents

### If Changing Backend Runtime

- check how the change affects `worker_defaults` merge behavior
- check how the change affects A0 task policy resolution and inferred worker defaults
- check whether validation errors still map cleanly to settings sections
- check whether launch blockers remain consistent with runtime launch behavior
- check whether stop/listener semantics still honor per-port session state

### If Changing Frontend Settings

- check for wasted whitespace on a laptop-sized viewport
- check whether the common path got simpler or accidentally more verbose
- check whether advanced overrides stayed available but de-emphasized
- rebuild static assets after source edits

### If Changing Docs

- put the shortest reliable commands first
- move optional flags and fallback launchers later
- update both control-plane docs and any repo-level guidance that now became stale

## Validation Checklist

1. Run Python compile checks on touched runtime/test files when feasible.
2. Run `npm run build` in `control_plane/fp8/runtime/web` after frontend edits.
3. Run the live control-plane integration test for meaningful workflow changes.
4. If pre-commit reformats files, restage and continue.

## Regression Questions

- Did this make the common launch/stop flow longer or harder to trust?
- Did this push users back toward repetitive per-worker entry?
- Did this make the operator enter information that A0 already knew from backlog/runtime/provider state?
- Did this reintroduce large blank areas in Settings?
- Did docs drift away from current runtime behavior?
- Did worker roster behavior fall back to stale examples instead of live plan state?
- Did we make worktree handling stricter than runtime actually requires?

## Expected Output Quality

- Changes should improve operational trust, not just add knobs.
- The default workflow should stay obvious to a manager returning after a pause.
- A new agent should be able to read this file and avoid undoing already-settled workflow decisions.