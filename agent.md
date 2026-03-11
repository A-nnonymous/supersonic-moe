# Control Plane Migration Context

This file is the handoff context for a new A0 that needs to keep evolving the external `warp` FP8 control plane without reconstructing prior intent from scratch.

## Current Role

- The control plane is the operator-facing runtime for backlog-aware multi-agent execution.
- A0 is expected to behave as scheduler of record, not just a process launcher.
- High-frequency commands are `serve`, `up`, `stop-agents`, `silent`, and `stop-all`.

## Current Architecture

- Standalone repo location is expected at sibling path `../warp` unless `WARP_ROOT` overrides it.
- Backend runtime: `../warp/runtime/control_plane.py`
- Frontend source: `../warp/runtime/web/src`
- Served frontend assets: `../warp/runtime/web/static`
- Primary regression suite: `../warp/runtime/test_control_plane_integration.py`

## Design Decisions To Preserve

- Settings are structured forms, not YAML-primary UX.
- Shared config belongs in `worker_defaults`; per-worker cards should stay sparse.
- A0 plan should be shown as the default derived state; manual overrides are exceptional and should be easy to reset.
- `task_policies` plus backlog `task_type` metadata own task-aware routing and test-command selection.
- `project.reference_workspace_root`, `project.reference_inputs`, and `project.prompt_context_files` replaced older task-specific reference fields.
- Session targeting depends on per-port files like `session_state_<port>.json`; do not regress this.

## Stability Fixes Already Landed

- Unknown `/api/*` routes return JSON 404 payloads instead of HTML error pages.
- Frontend API parsing now reports request path plus non-JSON response snippets for easier diagnosis.
- `stop-all` no longer kills the control-plane server via process-group termination; it terminates the listener PID directly while still using process-tree shutdown for worker jobs.
- Settings save flow rehydrates from server state after section saves.
- Worker defaults are split into common defaults and advanced defaults.
- Worker cards expose `Reset to A0` so derived plan values can take over again.

## Regression Coverage That Exists Now

The integration suite currently covers:

- project section validate/save through `/api/config/validate-section` and `/api/config/section`
- unknown API routes returning JSON 404 bodies
- `stop-agents` stopping workers while leaving the listener alive
- `silent` releasing the listener while preserving worker processes
- `stop-all` fully reclaiming the listener and shutting down the runtime cleanly
- launch policy behavior and multi-agent heartbeat/runtime transitions across `initial_copilot`, `selected_model`, and `elastic`

## Commands To Trust During Maintenance

- `python3 -m py_compile ../warp/runtime/control_plane.py ../warp/runtime/test_control_plane_integration.py`
- `python3 ../warp/runtime/test_control_plane_integration.py`
- `cd ../warp/runtime/web && npm run build`

## Maintenance Heuristics

- Prefer fewer operator inputs over more flexible but repetitive configuration.
- If a value can be inferred safely, let A0 derive it and preserve an override path.
- Keep docs, frontend labels, and backend behavior synchronized in the same change.
- Validate lifecycle changes against real process behavior, not just static logic inspection.
- When touching launch/stop behavior, think in terms of worker process trees versus listener PID scope.

## Likely Next Tests Worth Adding

- detached `serve`/`up` lifecycle with session recovery
- degraded provider paths such as missing API keys or missing binaries
- section-save coverage for `worker_defaults` and `workers` if UI logic changes there again

## Delivery Convention

- Default remote target for this repo is `origin main`.