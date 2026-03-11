# Control Plane Agent Context

Use this as the cold-start context for any new agent working in `control_plane/`.

## Scope

- The FP8 control plane is the operator-facing runtime and dashboard for worker launch, stop, validation, backlog visibility, provider routing, and manager-owned merge flow.
- Backend entry: `control_plane/fp8/runtime/control_plane.py`.
- Frontend source: `control_plane/fp8/runtime/web/src`.
- Frontend build output served in production: `control_plane/fp8/runtime/web/static`.
- Live integration test: `control_plane/fp8/runtime/test_control_plane_integration.py`.

## Non-Negotiable Workflow Decisions

- Keep the high-frequency path minimal and reliable.
- Do not drift back to dry-run-heavy, bootstrap-heavy, or YAML-primary UX unless explicitly requested.
- `serve`, `up`, `stop-agents`, `silent`, and `stop-all` are the main operating commands and should stay easy to trust.
- Structured settings forms replaced the old raw-YAML-first flow on purpose.
- Shared worker config belongs in top-level `worker_defaults`; per-worker cards should stay lean by default.
- Resource pools should use horizontal space efficiently; avoid tall sparse layouts.
- Settings should support section-scoped validate/save so one block can change without revalidating everything.
- Worktree paths should be auto-derived when safe, then overridable.
- Worker roster logic should reflect real plan/runtime state, not stale sample entries.
- Stop behavior depends on per-port session files like `session_state_<port>.json`; do not regress that targeting model.

## User Preferences

- Simplest reliable path first; advanced options later.
- Optimize for repeated operator use, not abstract flexibility.
- Prefer workflow improvements over cosmetic-only changes.
- Prefer automatic defaults over repetitive manual entry.
- Keep override power, but hide infrequent knobs behind advanced sections.
- Avoid large blank areas; use horizontal space well without making laptop layouts fragile.
- Keep UI labels, comments, README guidance, and runtime behavior aligned.
- When workflow changes, update docs in the same pass.

## Architecture Facts

- Config supports top-level `worker_defaults`, merged into each worker for validation and launch.
- Backlog/runtime files are part of the planning model. Use `control_plane/fp8/state/backlog.yaml` and `control_plane/fp8/state/agent_runtime.yaml` when deriving or syncing worker roster.
- Missing-but-plausible worktree paths are acceptable if runtime can create the worktree at launch time.
- Default deployment assumption is a Hopper/Blackwell Linux machine with a provisioned SonicMoE environment; the `uv --no-project` path is fallback only.
- SonicMoE public path is still BF16-first; FP8 work is integration-heavy, and QuACK / Blackwell paths already exist behind `USE_QUACK_GEMM`.

## Default Implementation Heuristics

- First decide whether a new setting belongs in `project`, `worker_defaults`, or a per-worker override.
- If a value can be inferred safely, auto-fill it and leave override room instead of forcing manual entry.
- Keep worker cards focused on identity, routing, and common controls; treat env/test/sync/submit/git overrides as advanced.
- Preserve the merge model between defaults and workers in both validation and launch.
- If docs mention commands, lead with the high-frequency path and move optional flags later.

## Read First

- `control_plane/fp8/README.md`
- `control_plane/fp8/governance/worker_launch_playbook.md`
- `control_plane/fp8/governance/control_plane_playbook.md`
- `control_plane/fp8/runtime/control_plane.py`
- `control_plane/fp8/runtime/config_template.yaml`
- `control_plane/fp8/runtime/web/src/App.tsx`
- `control_plane/fp8/runtime/web/src/api.ts`
- `control_plane/fp8/runtime/web/src/types.ts`
- `control_plane/fp8/runtime/web/src/styles.css`

## Validation Expectations

- For backend changes, run Python compile checks on touched runtime/test files when feasible.
- For frontend changes, rebuild `control_plane/fp8/runtime/web` and keep `runtime/web/static` in sync.
- For meaningful workflow changes, run the live control-plane integration test when feasible.
- If pre-commit reformats files, restage and continue.

## Operating Principle

- Reconstruct workflow intent before changing behavior.
- Prefer root-cause workflow fixes over cosmetic patches.
- Optimize for fewer steps, fewer repeated fields, fewer stale docs, and fewer surprises.