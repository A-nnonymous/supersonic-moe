# Governance

## Goal

Run multiple agents in parallel without relying on any single agent's conversational memory.

## Model

### Roles

- Manager: owns planning, gating, integration, escalation, and user reporting
- Worker agents: own bounded implementation tracks
- Gates: explicit pass/fail control points between stages

### Current worker ownership

- A1: API, dtype, backend, scale protocol
- A2: Hopper FP8 backend
- A3: Blackwell FP8 backend
- A4: torch functional path and autograd
- A5: correctness, gradient, and regression tests
- A6: baseline traceability and Paddle compatibility
- A7: performance, profiling, and delivery docs

## Rules

1. A1 freezes protocol before downstream parallel work starts.
2. A2 and A3 may work in parallel after protocol freeze.
3. A4 integrates only against frozen protocol and reviewed backend adapters.
4. A5 may build test skeletons early, but tolerance changes require justification.
5. A6 maintains baseline traceability continuously.
6. A7 does not request invasive optimization before functional gates pass.

## Heartbeat control

Agent startup is not assumed. The manager must distinguish between planned agents and agents that are actually alive.

Heartbeat state is recorded in `state/heartbeats.yaml`.

### Heartbeat states

- `healthy`: heartbeat received within the service-level window
- `stale`: agent was active but has not checked in within the service-level window
- `not-started`: no valid heartbeat has ever been recorded for the current phase
- `offline`: agent intentionally stopped, merged away, or not assigned

### Heartbeat sources

A heartbeat may be inferred from any of the following, in descending order of confidence:

1. explicit manager update in `state/heartbeats.yaml`
2. fresh write to `status/agents/A*.md`
3. fresh write to `checkpoints/agents/A*.md`
4. fresh lock activity in `state/edit_locks.yaml` for the owning agent

### Service-level window

For preflight and planning work, an agent is `stale` if it has not produced any valid heartbeat within one manager review cycle.

For active implementation or experiment phases, the manager should tighten the window and record the chosen threshold in `reports/manager_report.md`.

If no worker agents are active, the manager must report that only A0 is alive instead of pretending parallel execution exists.

## Document concurrency control

### Single-writer files

The following files are high-conflict and must be treated as single-writer files:

- `control_plane/fp8/strategy/integration_plan.md`
- `control_plane/fp8/strategy/baseline_trace.md`
- `control_plane/fp8/state/backlog.yaml`
- `control_plane/fp8/state/gates.yaml`
- `control_plane/fp8/reports/manager_report.md`
- `control_plane/fp8/governance/decisions.md`
- `control_plane/fp8/checkpoints/manager/latest.md`
- `control_plane/fp8/experiments/registry.yaml`

Only the current owner recorded in `state/edit_locks.yaml` may edit a single-writer file.

### Low-conflict files

The following files are multi-writer by design, but each file still has one primary owner:

- `control_plane/fp8/status/agents/A*.md`
- `control_plane/fp8/checkpoints/agents/A*.md`

One agent should not edit another agent's status or checkpoint file unless acting as manager during explicit recovery.

### Lock protocol

Before editing a single-writer file, an agent must:

1. claim the file in `state/edit_locks.yaml`
2. record intent and timestamp
3. complete the edit
4. release the lock or transfer ownership

If the lock owner is stale or unknown, escalate to the manager instead of writing through it.

### Merge windows

The manager should define narrow merge windows for high-conflict files. Workers should prefer updating their own status and checkpoint files over editing shared control files directly.

## Memory stomp prevention

To avoid concurrent overwrite of evolving plans or reports:

- workers write findings to their own status and checkpoint files first
- manager folds accepted changes into shared control files
- no worker should rewrite `reports/manager_report.md` or `checkpoints/manager/latest.md`
- large rewrites of `control_plane/fp8/strategy/integration_plan.md` require explicit manager lock

## Agent checkpoint model

Every worker agent needs a resumable checkpoint independent of chat history.

Each `checkpoints/agents/A*.md` file must capture:

- scope owned by the agent
- current task
- current branch or patch basis
- assumptions in force
- artifacts produced
- open blockers
- safe next step
- rollback note if work is partial

The manager checkpoint is `checkpoints/manager/latest.md`. Worker checkpoints must be sufficient for another agent to take over the same scope with minimal ambiguity.

## Required update format

Every worker status update must include:

- Current task
- Branch or patch scope
- Completed since last update
- Blockers
- Requested unlocks
- Next check-in condition

Every worker checkpoint update must include:

- Snapshot timestamp
- Owned scope
- Last known good state
- Pending change set
- Dependencies and assumptions
- Resume instruction

Every heartbeat update must include:

- Agent id
- State
- Last seen timestamp
- Evidence source
- Expected next check-in
- Escalation note if stale

## Gate semantics

- A gate is passed only when required artifacts exist and pass criteria are met.
- A gate may be conditionally passed only if remaining risk is recorded in `reports/manager_report.md` and `governance/decisions.md`.
- No agent should advance a dependency-bound task before the upstream gate is passed.

## Manager loop

The manager should run this loop at every handoff or resume:

1. Load checkpoint
2. Read backlog and gates
3. Read heartbeats
4. Read edit locks
5. Poll agent status files
6. Poll agent checkpoints
7. Poll experiment registry
8. Recompute blockers, alive agents, and next parallel set
9. Update manager report
10. Refresh checkpoint if anything material changed

## Escalation cases

Escalate to the user when:

- a frozen protocol must change
- a baseline diff cannot be explained
- a shared-machine experiment would disrupt another owner
- two agents need the same high-conflict file at the same time
- the gate criteria need to be relaxed