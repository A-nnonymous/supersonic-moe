#!/usr/bin/env python3
"""
SonicMoE Buffer Scoreboard — Publication-Quality Visualization
==============================================================

Three-panel figure consuming ``scoreboard.json`` produced by
``tools/scoreboard.py``:

  Panel A — **Scoreboard Matrix**: buffers × phases heatmap showing
            Read / Write / Live states, per-cell MiB annotations.
  Panel B — **Peak Memory Waterfall**: stacked bar per phase, colour-coded
            by buffer class.
  Panel C — **Optimisation Hints**: ranked bar chart of savings opportunities.

Usage::

    python visualization/scoreboard_viz.py            # both modes
    python visualization/scoreboard_viz.py --mode fp8  # FP8 only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ─── paths ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SCOREBOARD_PATH = ROOT / "scoreboard.json"

# ─── palette ──────────────────────────────────────────────────────────
STATE_CMAP = {"R": "#2196F3", "W": "#F44336", "L": "#BDBDBD", "-": "#FAFAFA"}
CLASS_CMAP = {
    "data":   "#42A5F5",
    "weight": "#66BB6A",
    "grad":   "#EF5350",
    "meta":   "#AB47BC",
    "quant":  "#FF7043",
    "aux":    "#78909C",
}
HINT_CMAP = {
    "recompute":       "#E53935",
    "early_free":      "#43A047",
    "storage_release": "#FB8C00",
    "dtype_opt":       "#1E88E5",
}


def _load_scoreboard() -> dict:
    if not SCOREBOARD_PATH.exists():
        print(f"[scoreboard_viz] {SCOREBOARD_PATH} not found — run tools/scoreboard.py first", file=sys.stderr)
        sys.exit(1)
    return json.loads(SCOREBOARD_PATH.read_text())


# ═════════════════════════════════════════════════════════════════════════
# Panel A — Scoreboard Matrix
# ═════════════════════════════════════════════════════════════════════════

def _draw_scoreboard_matrix(ax: plt.Axes, mode_data: dict, title: str) -> None:
    bufs = mode_data["bufs"]
    board = mode_data["board"]
    n_phases = len(board)

    # Sort buffers: by class (data > weight > grad > quant > meta > aux), then by MiB desc
    class_order = {"data": 0, "weight": 1, "grad": 2, "quant": 3, "meta": 4, "aux": 5}
    buf_names = sorted(
        bufs.keys(),
        key=lambda b: (class_order.get(bufs[b]["class"], 9), -bufs[b]["mib"]),
    )
    # Filter out tiny buffers (< 0.1 MiB) for readability
    buf_names = [b for b in buf_names if bufs[b]["mib"] >= 0.1]
    n_bufs = len(buf_names)

    # Build matrix: 0=inactive, 1=Live, 2=Read, 3=Write
    state_val = {"-": 0, "L": 1, "R": 2, "W": 3}
    mat = np.zeros((n_bufs, n_phases), dtype=int)
    for j, ph in enumerate(board):
        for i, bname in enumerate(buf_names):
            s = ph["state"].get(bname, "-")
            mat[i, j] = state_val.get(s, 0)

    # Custom colormap: white, light grey, blue, red
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#FAFAFA", "#E0E0E0", "#42A5F5", "#EF5350"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)

    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # Annotate MiB in cells with activity
    for i, bname in enumerate(buf_names):
        mib = bufs[bname]["mib"]
        for j in range(n_phases):
            if mat[i, j] >= 2:  # R or W
                ax.text(j, i, f"{mib:.0f}" if mib >= 1 else f"{mib:.1f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white", fontweight="bold")

    # Axes
    phase_labels = [f"P{ph['phase']}\n{ph['name']}" for ph in board]
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phase_labels, fontsize=6.5, rotation=0)
    ax.set_yticks(range(n_bufs))

    # Colour buf labels by class
    ylabels = []
    for bname in buf_names:
        cls = bufs[bname]["class"]
        ylabels.append(bname)
    ax.set_yticklabels(ylabels, fontsize=6, fontfamily="monospace")
    for i, bname in enumerate(buf_names):
        cls = bufs[bname]["class"]
        ax.get_yticklabels()[i].set_color(CLASS_CMAP.get(cls, "#333"))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(length=0)

    # Grid
    for i in range(n_bufs + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(n_phases + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#42A5F5", label="Read"),
        mpatches.Patch(facecolor="#EF5350", label="Write"),
        mpatches.Patch(facecolor="#E0E0E0", label="Live (idle)"),
        mpatches.Patch(facecolor="#FAFAFA", edgecolor="#CCC", label="Inactive"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=5.5,
              framealpha=0.9, ncol=2, handlelength=1.2, handletextpad=0.4)


# ═════════════════════════════════════════════════════════════════════════
# Panel B — Per-Phase Memory Stack (stacked bar by buffer class)
# ═════════════════════════════════════════════════════════════════════════

def _draw_memory_stack(ax: plt.Axes, mode_data: dict, title: str) -> None:
    bufs = mode_data["bufs"]
    board = mode_data["board"]
    n_phases = len(board)
    classes = ["data", "weight", "grad", "quant", "meta", "aux"]

    # Per-phase, per-class MiB
    stack = {c: np.zeros(n_phases) for c in classes}
    for j, ph in enumerate(board):
        for bname, state in ph["state"].items():
            if state == "-" or bname not in bufs:
                continue
            cls = bufs[bname]["class"]
            if cls in stack:
                stack[cls][j] += bufs[bname]["mib"]

    x = np.arange(n_phases)
    bottom = np.zeros(n_phases)
    for cls in classes:
        vals = stack[cls]
        if vals.max() < 0.01:
            continue
        ax.bar(x, vals, bottom=bottom, width=0.7,
               color=CLASS_CMAP[cls], label=cls.capitalize(), edgecolor="white", linewidth=0.3)
        bottom += vals

    # Peak line
    totals = sum(stack[c] for c in classes)
    peak_idx = int(np.argmax(totals))
    ax.axhline(totals[peak_idx], color="#E53935", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(f"Peak: {totals[peak_idx]:.0f} MiB",
                xy=(peak_idx, totals[peak_idx]),
                xytext=(peak_idx + 0.3, totals[peak_idx] + 30),
                fontsize=6.5, color="#E53935", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=0.8))

    phase_labels = [f"P{ph['phase']}" for ph in board]
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=7)
    ax.set_ylabel("Allocated (MiB)", fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.85,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    sns.despine(ax=ax)


# ═════════════════════════════════════════════════════════════════════════
# Panel C — Optimisation Hints (horizontal bar chart)
# ═════════════════════════════════════════════════════════════════════════

def _draw_hints(ax: plt.Axes, mode_data: dict, title: str) -> None:
    hints = mode_data["hints"]
    if not hints:
        ax.text(0.5, 0.5, "No hints", ha="center", va="center", fontsize=9, color="#999")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        return

    # Top-N
    top = hints[:8]
    top = list(reversed(top))  # bottom-to-top for barh

    labels = [f"{h['buf']}" for h in top]
    savings = [h.get("save_mib", h.get("mib", 0)) for h in top]
    colours = [HINT_CMAP.get(h["kind"], "#999") for h in top]
    y = np.arange(len(top))

    ax.barh(y, savings, color=colours, height=0.6, edgecolor="white", linewidth=0.3)

    for i, h in enumerate(top):
        ax.text(savings[i] + 2, i, h["note"], va="center", fontsize=5.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5, fontfamily="monospace")
    ax.set_xlabel("Potential Savings (MiB)", fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Legend for hint kinds
    seen = {}
    for h in top:
        if h["kind"] not in seen:
            seen[h["kind"]] = HINT_CMAP.get(h["kind"], "#999")
    legend_patches = [mpatches.Patch(facecolor=c, label=k.replace("_", " ").title())
                      for k, c in seen.items()]
    ax.legend(handles=legend_patches, fontsize=5.5, loc="lower right",
              framealpha=0.85, handlelength=1.2, handletextpad=0.4)
    sns.despine(ax=ax)


# ═════════════════════════════════════════════════════════════════════════
# Composite Figure
# ═════════════════════════════════════════════════════════════════════════

def render_scoreboard(mode: str = "both") -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sb = _load_scoreboard()
    modes = []
    if mode in ("both", "bf16") and "bf16" in sb:
        modes.append(("bf16", sb["bf16"]))
    if mode in ("both", "fp8") and "fp8" in sb:
        modes.append(("fp8", sb["fp8"]))

    for mode_name, mode_data in modes:
        fig = plt.figure(figsize=(18, 14), dpi=150)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2],
                              hspace=0.35, wspace=0.28,
                              left=0.06, right=0.97, top=0.93, bottom=0.05)

        # A: Scoreboard matrix (top, full width)
        ax_matrix = fig.add_subplot(gs[0, :])
        _draw_scoreboard_matrix(ax_matrix, mode_data,
                                f"Buffer Scoreboard — {mode_name.upper()} Path")

        # B: Memory stack (bottom-left)
        ax_mem = fig.add_subplot(gs[1, 0])
        _draw_memory_stack(ax_mem, mode_data,
                           f"Phase Memory Breakdown — {mode_name.upper()}")

        # C: Hints (bottom-right)
        ax_hints = fig.add_subplot(gs[1, 1])
        _draw_hints(ax_hints, mode_data,
                    f"Optimisation Opportunities — {mode_name.upper()}")

        # Suptitle
        n_bufs = len(mode_data["bufs"])
        n_dag = len(mode_data["dag"])
        peak = mode_data["peak"]
        fig.suptitle(
            f"SonicMoE Buffer Scoreboard  ·  {mode_name.upper()}  ·  "
            f"{n_bufs} buffers  ·  {n_dag} causal edges  ·  "
            f"peak {peak['alloc_mib']:.0f} MiB @ P{peak['phase']}",
            fontsize=12, fontweight="bold", y=0.98,
        )

        out = ASSETS / f"scoreboard_{mode_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SonicMoE Scoreboard Visualization")
    parser.add_argument("--mode", choices=["both", "bf16", "fp8"], default="both")
    args = parser.parse_args()
    render_scoreboard(args.mode)


if __name__ == "__main__":
    main()
