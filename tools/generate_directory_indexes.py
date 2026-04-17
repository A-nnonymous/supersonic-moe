#!/usr/bin/env python3
"""Generate human- and agent-friendly INDEX.md files for stable directories."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
INDEX_NAME = "INDEX.md"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".venv",
    "dist",
    "env",
    "venv",
}

EXCLUDED_DIR_PATHS = {
    Path("build"),
    Path("sonic_moe.egg-info"),
    Path("benchmarks/nsys_clean"),
    Path("reports/grid_session53/logs"),
}

DIR_SUMMARY_OVERRIDES = {
    ".": "Repository root with source, docs, reports, benchmarks, tests, and agent guidance.",
    ".claude": "Local Claude editor / workflow metadata used during iterative development.",
    ".claude/plans": "Ad hoc planning notes created during previous development sessions.",
    "assets": "Static figures used by the root README and related documentation.",
    "benchmarks": "One-off and repeatable benchmark entrypoints for FP8, BF16, and routing experiments.",
    "benchmarks/clean_results": "Saved clean-run benchmark text outputs used as reference snapshots.",
    "benchmarks/nsys_run": "Minimal benchmark entrypoints tailored for nsys profiling runs.",
    "docs": "Canonical architecture, handoff, and design documentation.",
    "reports": "Collected benchmark outputs, summaries, and historical experiment artifacts.",
    "reports/fp8_upgrade": "Historical FP8-upgrade notes; partly superseded by newer docs and reports.",
    "reports/grid_session53": "Session 53 grid benchmark shards and consolidated JSON output.",
    "reports/nsys_final": "Final consolidated nsys-derived breakdowns for Session 53.",
    "sonicmoe": "Primary Python package implementing SonicMoE kernels, configuration, and module entrypoints.",
    "sonicmoe/count_cumsum": "CUDA extension for count / cumsum helpers used by routing code.",
    "sonicmoe/functional": "Core forward and backward orchestration, routing helpers, and FP8 protocol flow.",
    "sonicmoe/functional/triton_kernels": "Imported Triton helper kernels and license material.",
    "sonicmoe/include": "C/C++ headers shared by compiled extensions.",
    "sonicmoe/quack_utils": "QuACK / CUTLASS / Triton utilities for BF16 and FP8 GEMM paths.",
    "tests": "Repository-level regression, integration, and contract tests.",
    "tests/ops": "Focused operator and module-level tests, including the newer MoE module suite.",
    "tests/reference_layers": "Reference implementations vendored for compatibility and behavior checks.",
    "tests/reference_layers/standalone_moe_layer": "Standalone reference MoE package used for interface and behavior comparison.",
    "tests/reference_layers/standalone_moe_layer/moe_standalone": "Importable reference package namespace.",
    "tests/reference_layers/standalone_moe_layer/moe_standalone/moe": "Reference MoE layer and gating components.",
    "tests/reference_layers/standalone_moe_layer/moe_standalone/token_dispatcher": "Reference token-dispatch helpers and FP8 utility code.",
    "tests/reference_layers/standalone_moe_layer/scripts": "Small scripts for validating reference-layer compatibility.",
    "tests/reference_layers/standalone_moe_layer/tests": "Reference package smoke and interface tests.",
    "tools": "Developer tooling for profiling, benchmarking, validation, orchestration, and audits.",
    "visualization": "Plotting and visualization entrypoints plus image assets.",
}

DIR_NOTES = {
    ".": [
        "Canonical project state lives in `docs/HANDOFF.md`.",
        "Use these indexes before broad file searches to reduce token consumption.",
    ],
    "reports/fp8_upgrade": [
        "`reports/fp8_upgrade/HANDOFF.md` is stale and explicitly superseded by `docs/HANDOFF.md`.",
    ],
}

VOLATILE_DIR_NOTES = {
    ".git": "Git internals; never index or edit manually.",
    ".pytest_cache": "Pytest cache; disposable.",
    "__pycache__": "Python bytecode cache; disposable.",
    "build": "Generated build output from native extension compilation; do not track a local index here.",
    "sonic_moe.egg-info": "Generated packaging metadata; disposable and usually recreated by install commands.",
    "benchmarks/nsys_clean": "Raw nsys capture artifacts (`.sqlite`, `.nsys-rep`); high churn and not useful for durable indexing.",
    "reports/grid_session53/logs": "Per-GPU log outputs from benchmark runs; append-only runtime artifacts.",
}

PATH_NOTES = {
    "docs/HANDOFF.md": ["canonical handoff", "authoritative current state"],
    "reports/fp8_upgrade/HANDOFF.md": ["stale", "superseded by `docs/HANDOFF.md`"],
    "reports/fp8_upgrade/engineering_log.md": ["historical reference"],
    "AGENTS.md": ["canonical agent bootstrap"],
    "agent.md": ["compatibility alias to `AGENTS.md`"],
    "reports/quant_bench_final.json": ["legacy snapshot", "compare with `reports/quant_bench.json` before reusing"],
    "reports/wgrad_fp8_benchmark_legacy.json": ["legacy benchmark snapshot", "kept for historical comparison with `reports/wgrad_bench.json`"],
    "reports/README.md": ["keep aligned with `docs/HANDOFF.md`"],
    ".claude/settings.local.json": ["local editor settings"],
    ".claude/plans/native_fp8_params.md": ["local planning artifact"],
}

WATCHLIST = {
    ".": [
        "`agent.md` should remain a thin compatibility alias to `AGENTS.md`, not a second independently edited bootstrap document.",
        "Generated directories (`build/`, `sonic_moe.egg-info/`, caches) are intentionally summarized in parent indexes instead of receiving their own tracked index files.",
    ],
    "reports": [
        "`reports/README.md` should stay aligned with `docs/HANDOFF.md` whenever the authoritative handoff changes.",
        "`quant_bench.json` and `quant_bench_final.json` look like structured-vs-legacy variants of the same benchmark family; verify the intended canonical file before adding new results.",
        "`wgrad_fp8_benchmark_legacy.json` is historical only; new wgrad report outputs should stay structured and live under `reports/`.",
    ],
    "reports/fp8_upgrade": [
        "This subtree is historical by default; new authoritative state should go to `docs/` or newer `reports/` summaries unless there is a strong reason otherwise.",
    ],
}

FILE_SUMMARY_OVERRIDES = {
    "README.md": "Top-level project overview, installation, testing, and current FP8 status summary.",
    "AGENTS.md": "Canonical agent bootstrap note for this repository's FP8 workstream.",
    "agent.md": "Compatibility alias that redirects readers to `AGENTS.md`.",
    ".clang-format": "clang-format style configuration for native code.",
    ".gitmodules": "Git submodule configuration.",
    "pyproject.toml": "Primary Python packaging and tool configuration.",
    "requirements.txt": "Pinned Python runtime dependencies for local development.",
    "setup.py": "Setuptools installation entrypoint.",
    "setup.cfg": "Setuptools and style configuration.",
    "Makefile": "Convenience commands for tests and common developer workflows.",
    ".gitignore": "Git ignore rules, including generated profiling and build artifacts.",
    ".pre-commit-config.yaml": "Pre-commit hook definitions.",
    "LICENSE": "Repository license text.",
    "manifest.json": "Manifest-style metadata file used by repository tooling.",
    "scoreboard.json": "Scoreboard data consumed by visualization or reporting helpers.",
    "reports/wgrad_bench.json": "Structured wgrad benchmark report under `reports/`.",
    "reports/wgrad_fp8_benchmark_legacy.json": "Legacy full-replacement wgrad benchmark snapshot kept for history.",
    "reports/quant_bench.json": "Structured quant benchmark report with per-kernel summaries and metadata.",
    "reports/quant_bench_final.json": "Legacy flat quant benchmark snapshot still emitted by `tools/introspect.py`.",
    "reports/fp8_frontier_path_analysis.json": "Compiled BF16-vs-FP8 path-comparison report consumed by the new visualization module.",
    "docs/HANDOFF.md": "Canonical handoff with current performance, architecture, and validation state.",
    "reports/README.md": "High-level map of report outputs and profiling artifacts.",
    "sonicmoe/__init__.py": "Package export surface for SonicMoE.",
    "sonicmoe/config.py": "Pythonic `SonicMoEConfig` context manager and configuration helpers.",
    "sonicmoe/moe.py": "Main MoE module implementation and FP8 stash / optimizer helpers.",
    "sonicmoe/enums.py": "Shared enums used across module configuration and dispatch.",
    "sonicmoe/jit.py": "JIT and compilation helpers.",
    "sonicmoe/utils.py": "General package-level utility helpers.",
    "sonicmoe/functional/__init__.py": "Core FP8/BF16 forward-backward orchestration entrypoints.",
    "sonicmoe/quack_utils/blockscaled_fp8_gemm.py": "Hot-path Triton FP8 quantization, packing, and cache utilities.",
    "sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py": "Zero-materialization SM100 FP8 GEMM kernels.",
    "tools/introspect.py": "Main profiling harness for nsys, precision, grid, and memory experiments.",
    "tests/ops/test_moe_module.py": "MoE module-level regression suite against a pure-torch reference.",
    "visualization/path_compare_viz.py": "Renders the BF16-vs-FP8 frontier path and contribution comparison figures.",
    "assets/fig15_fp8_bf16_path_compare.png": "BF16-vs-FP8 operator/dataflow comparison figure with bridge tensors and phase memory envelope.",
    "assets/fig16_fp8_frontier_contributions.png": "BF16-vs-FP8 frontier contribution and scaling figure built from the Session 53 grid.",
}

TOKEN_MAP = {
    "fp8": "FP8",
    "bf16": "BF16",
    "nsys": "nsys",
    "ncu": "NCU",
    "moe": "MoE",
    "swiglu": "SwiGLU",
    "topk": "top-k",
    "quack": "QuACK",
    "cuda": "CUDA",
    "triton": "Triton",
    "cute": "CuTe",
    "dgated": "DGated",
    "gemm": "GEMM",
    "wgrad": "wgrad",
    "dout": "dout",
    "y1": "y1",
    "y1s": "y1s",
    "rcp": "RCP",
    "varlen": "varlen",
    "colwise": "colwise",
    "rowwise": "rowwise",
    "dequant": "dequant",
    "zy1": "z+y1",
}


@dataclass(frozen=True)
class DirEntry:
    path: Path
    stable_children: tuple[Path, ...]
    volatile_children: tuple[Path, ...]
    files: tuple[Path, ...]


def rel(path: Path) -> str:
    value = path.relative_to(ROOT).as_posix()
    return value or "."


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_tracked_files() -> set[str]:
    output = run_git("ls-files")
    return {line for line in output.splitlines() if line}


def should_skip_dir(path: Path) -> bool:
    rel_path = rel(path)
    return path.name in EXCLUDED_DIR_NAMES or Path(rel_path) in EXCLUDED_DIR_PATHS


def scan_tree(path: Path, entries: dict[str, DirEntry]) -> None:
    stable_children: list[Path] = []
    volatile_children: list[Path] = []
    files: list[Path] = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.name == INDEX_NAME:
            continue
        if child.is_dir():
            if should_skip_dir(child):
                volatile_children.append(child)
            else:
                stable_children.append(child)
        else:
            files.append(child)
    entries[rel(path)] = DirEntry(path=path, stable_children=tuple(stable_children), volatile_children=tuple(volatile_children), files=tuple(files))
    for child in stable_children:
        scan_tree(child, entries)


def markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped
    return None


def first_sentence(text: str, *, limit: int = 180) -> str:
    flat = " ".join(text.split())
    if not flat:
        return ""
    match = re.search(r"(?<=[.!?])\s", flat)
    if match:
        flat = flat[: match.start()].strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def extract_python_docstring(path: Path) -> str | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    doc = ast.get_docstring(module)
    if not doc:
        return None
    return " ".join(doc.split())


def extract_shell_comment(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#!"):
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            break
    except OSError:
        return None
    return None


def humanize_stem(stem: str) -> str:
    parts = [part for part in re.split(r"[_\-]+", stem) if part]
    words = [TOKEN_MAP.get(part.lower(), part.upper() if len(part) <= 3 else part.capitalize()) for part in parts]
    return " ".join(words) or stem


def summarize_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return f"JSON artifact for {humanize_stem(path.stem).lower()}."
    if isinstance(data, dict):
        keys = list(data.keys())[:4]
        if keys:
            return f"JSON artifact with top-level keys: {', '.join(f'`{key}`' for key in keys)}."
        return "Empty JSON object artifact."
    if isinstance(data, list):
        return f"JSON list artifact with {len(data)} entries."
    return f"JSON scalar artifact of type {type(data).__name__}."


def generic_file_summary(path: Path) -> str:
    rel_path = rel(path)
    if rel_path in FILE_SUMMARY_OVERRIDES:
        return FILE_SUMMARY_OVERRIDES[rel_path]
    if path.name in FILE_SUMMARY_OVERRIDES:
        return FILE_SUMMARY_OVERRIDES[path.name]

    suffix = path.suffix.lower()
    parent_parts = set(path.parent.parts)
    stem_human = humanize_stem(path.stem)

    if suffix == ".md":
        title = markdown_title(path.read_text(encoding="utf-8", errors="ignore"))
        return f"Markdown note: {title}." if title else f"Markdown note about {stem_human.lower()}."
    if suffix == ".py":
        if path.name == "__init__.py":
            if "tests" in parent_parts:
                return "Package marker for test discovery."
            return "Package marker and re-export surface."
        doc = extract_python_docstring(path)
        if doc:
            return first_sentence(doc).rstrip(".") + "."
        if path.name.startswith("test_") or path.name.endswith("_test.py") or "tests" in parent_parts:
            topic = stem_human.removeprefix("Test ").removesuffix(" Test")
            return f"Pytest coverage for {topic.lower()}."
        if "benchmarks" in parent_parts:
            return f"Benchmark entrypoint for {stem_human.lower()}."
        if "tools" in parent_parts:
            return f"Developer utility for {stem_human.lower()}."
        if "visualization" in parent_parts:
            return f"Visualization entrypoint for {stem_human.lower()}."
        if "sonicmoe" in parent_parts:
            return f"Python module for {stem_human.lower()}."
        return f"Python source for {stem_human.lower()}."
    if suffix == ".json":
        return summarize_json(path)
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return f"Image asset for {stem_human.lower()}."
    if suffix in {".sh", ".bash"}:
        comment = extract_shell_comment(path)
        return f"Shell helper: {comment}." if comment else f"Shell helper for {stem_human.lower()}."
    if suffix in {".cu", ".cuh"}:
        return f"CUDA source for {stem_human.lower()}."
    if suffix in {".h", ".hpp"}:
        return f"C/C++ header for {stem_human.lower()}."
    if suffix == ".txt":
        return f"Text artifact for {stem_human.lower()}."
    if suffix == ".toml":
        return f"TOML configuration for {stem_human.lower()}."
    if suffix == ".yaml" or suffix == ".yml":
        return f"YAML configuration for {stem_human.lower()}."
    if suffix == ".log":
        return f"Log artifact for {stem_human.lower()}."
    if suffix == ".sqlite":
        return f"SQLite profiling artifact for {stem_human.lower()}."
    if suffix == ".nsys-rep":
        return f"nsys report artifact for {stem_human.lower()}."
    if suffix == ".ncu-rep":
        return f"NCU report artifact for {stem_human.lower()}."
    return f"Project file for {stem_human.lower()}."


def path_notes(path: Path, tracked_files: set[str]) -> list[str]:
    rel_path = rel(path)
    notes = list(PATH_NOTES.get(rel_path, ()))
    if rel_path not in tracked_files:
        notes.append("untracked in git")
    return notes


def dir_summary(path: Path) -> str:
    return DIR_SUMMARY_OVERRIDES.get(rel(path), f"Directory for {humanize_stem(path.name).lower()}.")


def dir_notes(path: Path) -> list[str]:
    return list(DIR_NOTES.get(rel(path), ()))


def volatile_dir_note(path: Path) -> str:
    rel_path = rel(path)
    return VOLATILE_DIR_NOTES.get(rel_path, VOLATILE_DIR_NOTES.get(path.name, "Volatile generated directory; summarize in parent instead of tracking a local index."))


def format_notes(notes: Iterable[str]) -> str:
    clean = [note for note in notes if note]
    return ", ".join(clean) if clean else "—"


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def maintenance_rules() -> list[str]:
    return [
        f"Before opening many files under this directory, read this `{INDEX_NAME}` first to narrow the search space.",
        f"Any create / delete / rename / move in this directory must update the summaries in this `{INDEX_NAME}`.",
        f"Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.",
        f"If a change crosses directory boundaries, update this `{INDEX_NAME}` and the nearest affected ancestor `{INDEX_NAME}` files together.",
        "Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.",
    ]


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(escape_cell(cell) for cell in row) + " |")
    return out


def root_overview(entry: DirEntry) -> list[str]:
    rows = []
    for child in entry.stable_children:
        rows.append([f"`{child.name}/`", dir_summary(child), format_notes(dir_notes(child))])
    for child in entry.volatile_children:
        rows.append([f"`{child.name}/`", "Volatile / generated subtree.", volatile_dir_note(child)])
    return ["## Shallow overview", *render_table(["Path", "Role", "Notes"], rows), ""]


def render_index(entry: DirEntry, tracked_files: set[str]) -> str:
    current_rel = rel(entry.path)
    title_path = "/" if current_rel == "." else f"/{current_rel}/"
    lines: list[str] = [
        f"# Directory Index: `{title_path}`",
        "",
        f"> {dir_summary(entry.path)}",
        f"> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.",
        "",
        "## Maintenance rules",
    ]
    lines.extend(f"- {rule}" for rule in maintenance_rules())
    lines.append("")

    notes = dir_notes(entry.path)
    if notes:
        lines.append("## Local notes")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    if current_rel == ".":
        lines.extend(root_overview(entry))

    if current_rel != ".":
        stable_rows = [
            [f"`{child.name}/`", dir_summary(child), format_notes(dir_notes(child))]
            for child in entry.stable_children
        ]
        if stable_rows:
            lines.append("## Stable child directories")
            lines.extend(render_table(["Path", "Summary", "Notes"], stable_rows))
            lines.append("")

        volatile_rows = [
            [f"`{child.name}/`", "Volatile / generated subtree.", volatile_dir_note(child)]
            for child in entry.volatile_children
        ]
        if volatile_rows:
            lines.append("## Volatile / generated child directories")
            lines.extend(render_table(["Path", "Summary", "Notes"], volatile_rows))
            lines.append("")

    file_rows = [
        [f"`{file.name}`", generic_file_summary(file), format_notes(path_notes(file, tracked_files))]
        for file in entry.files
    ]
    if file_rows:
        lines.append("## Files")
        lines.extend(render_table(["File", "Summary", "Notes"], file_rows))
        lines.append("")

    watchlist = WATCHLIST.get(current_rel, ())
    if watchlist:
        lines.append("## Redundancy and cleanup watchlist")
        lines.extend(f"- {item}" for item in watchlist)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    tracked_files = load_tracked_files()
    entries: dict[str, DirEntry] = {}
    scan_tree(ROOT, entries)
    for key in sorted(entries):
        entry = entries[key]
        content = render_index(entry, tracked_files)
        (entry.path / INDEX_NAME).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
