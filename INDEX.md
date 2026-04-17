# Directory Index: `/`

> Repository root with source, docs, reports, benchmarks, tests, and agent guidance.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Local notes
- Canonical project state lives in `docs/HANDOFF.md`.
- Use these indexes before broad file searches to reduce token consumption.

## Shallow overview
| Path | Role | Notes |
| --- | --- | --- |
| `.claude/` | Local Claude editor / workflow metadata used during iterative development. | — |
| `assets/` | Static figures used by the root README and related documentation. | — |
| `benchmarks/` | One-off and repeatable benchmark entrypoints for FP8, BF16, and routing experiments. | — |
| `docs/` | Canonical architecture, handoff, and design documentation. | — |
| `reports/` | Collected benchmark outputs, summaries, and historical experiment artifacts. | — |
| `sonicmoe/` | Primary Python package implementing SonicMoE kernels, configuration, and module entrypoints. | — |
| `tests/` | Repository-level regression, integration, and contract tests. | — |
| `tools/` | Developer tooling for profiling, benchmarking, validation, orchestration, and audits. | — |
| `visualization/` | Plotting and visualization entrypoints plus image assets. | — |
| `.git/` | Volatile / generated subtree. | Git internals; never index or edit manually. |
| `.pytest_cache/` | Volatile / generated subtree. | Pytest cache; disposable. |
| `build/` | Volatile / generated subtree. | Generated build output from native extension compilation; do not track a local index here. |
| `sonic_moe.egg-info/` | Volatile / generated subtree. | Generated packaging metadata; disposable and usually recreated by install commands. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `.clang-format` | clang-format style configuration for native code. | — |
| `.gitignore` | Git ignore rules, including generated profiling and build artifacts. | — |
| `.gitmodules` | Git submodule configuration. | — |
| `.pre-commit-config.yaml` | Pre-commit hook definitions. | — |
| `agent.md` | Compatibility alias that redirects readers to `AGENTS.md`. | compatibility alias to `AGENTS.md` |
| `AGENTS.md` | Canonical agent bootstrap note for this repository's FP8 workstream. | canonical agent bootstrap |
| `LICENSE` | Repository license text. | — |
| `Makefile` | Convenience commands for tests and common developer workflows. | — |
| `manifest.json` | Manifest-style metadata file used by repository tooling. | — |
| `pyproject.toml` | Primary Python packaging and tool configuration. | — |
| `README.md` | Top-level project overview, installation, testing, and current FP8 status summary. | — |
| `requirements.txt` | Pinned Python runtime dependencies for local development. | — |
| `scoreboard.json` | Scoreboard data consumed by visualization or reporting helpers. | — |
| `setup.cfg` | Setuptools and style configuration. | — |
| `setup.py` | Setuptools installation entrypoint. | — |

## Redundancy and cleanup watchlist
- `agent.md` should remain a thin compatibility alias to `AGENTS.md`, not a second independently edited bootstrap document.
- Generated directories (`build/`, `sonic_moe.egg-info/`, caches) are intentionally summarized in parent indexes instead of receiving their own tracked index files.
