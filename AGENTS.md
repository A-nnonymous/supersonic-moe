# Control Plane Workflow Notes

This repository contains a long-running control-plane workflow under `control_plane/`.

When working in `control_plane/**`, follow these rules:

- Optimize for the highest-frequency operator path first. Keep launch and stop flows minimal, reliable, and obvious before adding optional flags or advanced modes.
- Do not reintroduce dry-run-oriented UX or documentation as the primary workflow unless the user explicitly asks for it.
- Prefer automatic defaults over repetitive manual entry. If a value can be derived safely, derive it and still leave room for explicit override.
- Separate fleet-wide defaults from per-worker overrides. Keep worker cards lean by default and hide infrequent knobs behind advanced sections.
- Reduce wasted space in the Settings UI. Favor dense, horizontally efficient layouts as long as readability remains intact on laptop screens.
- Treat worktree paths as planned runtime inputs. Validate for plausibility and uniqueness, and allow the runtime to create missing worktrees when appropriate.
- Keep the worker roster aligned with the actual backlog/runtime plan instead of stale sample entries.
- For settings changes, prefer section-scoped validation and persistence so operators can change one block without revalidating unrelated blocks.
- Keep README and control-plane docs synchronized with the actual runtime behavior and UI model whenever the workflow changes.
- Before finishing nontrivial control-plane changes, run the relevant validation path when feasible: Python compile checks, frontend build in `control_plane/fp8/runtime/web`, and the live control-plane integration test.

User preferences captured from recent sessions:

- Simplest reliable path first, advanced options later.
- Strong bias toward workflow-level UX improvements, not cosmetic tweaks alone.
- Defaults should remove repetitive operator input while preserving targeted override capability.
- Layouts should avoid large blank areas and make better use of horizontal space.
- Explanations and docs should stay consistent with the actual code and current behavior.