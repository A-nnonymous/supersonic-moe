# Shared Machine Policy

## Goal

Allow experiments on a shared machine without disrupting other users or corrupting project signal.

## Preflight checklist

An experiment may be inserted only if all checks pass.

### Resource checks

- target GPU is free or explicitly reserved for this project
- enough free VRAM exists for the planned run
- enough host RAM and disk space exist
- no higher-priority long-running job will be interrupted

### Ownership checks

- the device or slot owner is known
- the expected runtime is recorded
- the experiment has an owner and an experiment ID
- the worker's `state/agent_runtime.yaml` entry exists and matches the submitting branch and environment

### Value checks

- the result is needed for a current gate or blocker
- the run has a clear stop condition
- retry count is bounded before launch

## Retry policy

- infrastructure flake: at most 2 automatic retries
- OOM: no blind retry; adjust shape, batch, or placement first
- hang or deadlock: stop and escalate
- unexplained numerical diff: stop and escalate

## Recording requirements

Every experiment must be recorded in `experiments/registry.yaml` with:

- experiment ID
- owner
- commit or patch basis
- machine or device
- config summary
- start time
- end time
- result
- whether retry is allowed
- worktree path
- branch name
- environment path

## Safe insertion rule

If any of the following is unknown, do not insert the experiment:

- who currently owns the target device
- what gate the experiment supports
- how to stop the run safely
- whether enough resources are available

## Worktree rule

Shared machines should use explicit git worktrees for concurrent workers.

- one worktree per worker
- one branch per worktree
- one environment per worktree
- worktree path and branch must match the runtime registry

If two workers need the same branch or the same editable environment, stop and re-plan instead of sharing state.