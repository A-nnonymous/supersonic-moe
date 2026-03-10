# FP8 Control Plane Engineering Report

Date: 2026-03-10
Owner: A0
Repository: supersonic-moe
Status: production-ready control plane handoff

## Scope

This report covers the repository-resident control plane and manager runtime under `control_plane/fp8/`.

## Delivered

- integrated single-process dashboard and API server
- editable local runtime config from the webpage
- one-click save, launch, restart, stop, and refresh actions
- provider priority queue with runtime connection-quality and work-quality scoring
- per-worker provider fallback queue support
- runtime-safe startup path for manager machines using `uv --no-project`
- root README startup instructions for main-agent operations
- `.gitignore` coverage for local uv and pip artifacts

## Interaction model

- operator edits `control_plane/fp8/runtime/local_config.yaml` through the dashboard or a local editor
- manager starts with `serve` for control only, or `up` for control plus worker launch
- dashboard polls current runtime state, provider queue, heartbeats, gates, backlog, and manager report
- launch actions update runtime topology and heartbeat files in `control_plane/fp8/state/`

## Production hardening

- manager runtime no longer depends on installing the full CUDA project stack on macOS
- API responses now serialize YAML-native date values safely
- dashboard action buttons are serialized client-side to avoid overlapping operations
- refresh errors are surfaced without breaking the page state
- provider availability is checked from both CLI presence and API key presence

## Verification

Completed on this machine:

- `python3 -m py_compile control_plane/fp8/runtime/control_plane.py`
- editor diagnostics for runtime script, config template, and operator docs
- portable runtime startup with `uv run --no-project --with 'PyYAML>=6.0.2' python ...`
- live API smoke test on `/api/state`

Observed API result summary:

- dashboard endpoint bound successfully on `127.0.0.1:8244`
- provider queue returned correctly
- `copilot_pool` detected local CLI availability
- template config returned the expected placeholder-path validation errors
- launch API rejected invalid template configuration with structured `400` JSON instead of attempting worktree creation

## Residual constraints

- full project dependency installation is still not possible on this macOS manager machine because CUDA/CUTLASS runtime wheels are Linux-only
- real worker execution for GPU tasks still requires Linux/CUDA worktrees and environments
- template config intentionally has placeholder API keys, so launch from the template will not succeed until local secrets are filled

## Recommended operator command

```bash
uv run --no-project --with 'PyYAML>=6.0.2' python control_plane/fp8/runtime/control_plane.py serve --config control_plane/fp8/runtime/local_config.yaml --open-browser
```

## Release note

The control plane is ready to be committed and shipped as a manager-side orchestration surface. SonicMoE FP8 implementation work remains gated behind protocol and backend delivery tasks.
