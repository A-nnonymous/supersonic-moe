#!/usr/bin/env python3
"""
SonicMoE FP8 Blockscaled — Publication-Quality Visualization Suite
===================================================================

Ten data-driven figures for the zero-materialization FP8 training path
on Blackwell (B200), aligned with Session 41 HANDOFF.md measurements.

Figures
-------
  1. Executive Summary        — 3-panel: speedup, memory, precision
  2. Performance Waterfall    — BF16 -> GEMM savings -> quant overhead -> FP8
  3. Memory Lifecycle         — 4-checkpoint BF16 vs FP8 trajectory
  4. Backward Peak Breakdown  — 100% tensor-level audit (1367 MiB)
  5. Kernel-Level Comparison  — per-kernel BF16 vs FP8 grouped bars
  6. Precision State Matrix   — dtype heatmap at each tensor x phase cell
  7. Precision Profile        — RRMSE + cosine with pass thresholds
  8. Optimization Design Space — shipped gains vs dead ends
  9. Buffer Lifecycle Gantt   — per-buffer bars, dtype-coloured, event markers
 10. Dtype Transformation Flow — operator-level FP8 quantization pipeline

Usage
-----
    python -m visualization
    python visualization/sonicmoe_dataflow.py

Output: <repo_root>/assets/
All data: Session 41, idle B200, subprocess-isolated, nsys GPU Projection.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt                              # noqa: E402
import matplotlib.patches as mpatches                        # noqa: E402
import matplotlib.ticker as mticker                          # noqa: E402
from matplotlib.patches import Patch                         # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.font_manager import FontProperties           # noqa: E402
import numpy as np                                           # noqa: E402
import seaborn as sns                                        # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Global Configuration
# ═══════════════════════════════════════════════════════════════════════════

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# ── Ernie shape ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoEShape:
    T: int = 8192; H: int = 3072; I: int = 1536; E: int = 8; K: int = 8

    @property
    def TK(self) -> int:
        return self.T * self.K

    @property
    def label(self) -> str:
        return f"T={self.T}, H={self.H}, I={self.I}, E={self.E}, K={self.K}"

SHAPE = MoEShape()
_SUB = (f"Ernie MoE  ($T$={SHAPE.T}, $H$={SHAPE.H}, $I$={SHAPE.I}, "
        f"$E$={SHAPE.E}, $K$={SHAPE.K})")
_HW = "Blackwell B200, idle GPU, subprocess-isolated"

# ── Publication colour palette ────────────────────────────────────────────

C_BF16    = "#2563EB"   # blue
C_FP8     = "#EA580C"   # warm orange
C_SAVE    = "#059669"   # emerald — savings / pass
C_COST    = "#DC2626"   # red — overhead / fail
C_NEUTRAL = "#6B7280"   # gray
C_ACCENT  = "#7C3AED"   # violet — highlights
C_LIGHT   = "#F3F4F6"   # background gray
C_AMBER   = "#F59E0B"   # amber — FP8 dtype / warning
C_SCALE   = "#10B981"   # teal  — SCALE dtype

# Font for Unicode glyphs (DejaVu Sans has full Unicode coverage)
_GLYPH_FP = FontProperties(family="DejaVu Sans", weight="bold")

# ── Phase system (for precision flow) ─────────────────────────────────────

PHASES = ["Router\n& Meta", "UpProj\nFwd", "DnProj\nFwd",
          "DnProj\nBwd", "UpBwd\n(wgrad)", "UpBwd\n(actgrad)"]
N_PH = len(PHASES)

# ── Shared style ──────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Set publication-quality matplotlib defaults (STIX mathtext, serif)."""
    sns.set_style("whitegrid", {
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.edgecolor": "#D1D5DB",
    })
    plt.rcParams.update({
        "font.family":          "serif",
        "font.serif":           ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset":     "stix",
        "font.size":            10,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "figure.dpi":           200,
        "savefig.dpi":          250,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "savefig.pad_inches":   0.18,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    out = ASSETS / name
    fig.savefig(str(out))
    plt.close(fig)
    print(f"  -> {out}")


def _pct(base: float, new: float) -> str:
    d = (new - base) / base * 100
    return f"{d:+.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Executive Summary (3-panel)
# ═══════════════════════════════════════════════════════════════════════════

def fig1_executive_summary() -> None:
    """Three-panel hero figure: speedup, memory, precision."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5),
                              gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(top=0.82, bottom=0.10)

    # ── (a) Latency / Speedup ────────────────────────────────────────────
    ax = axes[0]
    vals = [3840, 3442]
    colors = [C_BF16, C_FP8]
    bars = ax.bar(["BF16", "FP8"], vals, width=0.52, color=colors,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60,
                f"{v}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1F2937")
    # speedup bracket — draw to the right of FP8 bar
    ax.annotate("", xy=(1.15, 3442), xytext=(1.15, 3840),
                arrowprops=dict(arrowstyle="<->", color=C_SAVE, lw=2),
                annotation_clip=False)
    ax.text(1.32, (3840 + 3442) / 2, "$\\mathbf{1.12\\times}$",
            fontsize=13, color=C_SAVE, va="center", ha="left",
            clip_on=False)
    ax.set_ylabel("GPU Kernel Time ($\\mu$s / iter)")
    ax.set_title("(a) Latency", fontweight="bold", pad=8)
    ax.set_ylim(0, 4700)
    ax.set_xlim(-0.5, 1.8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))

    # ── (b) Peak Memory ──────────────────────────────────────────────────
    ax = axes[1]
    x = np.arange(2)
    w = 0.28
    bf16_mem = [1386, 1412]
    fp8_mem  = [1263, 1367]
    b1 = ax.bar(x - w / 2, bf16_mem, w, label="BF16", color=C_BF16,
                edgecolor="white", linewidth=1.2, zorder=3)
    b2 = ax.bar(x + w / 2, fp8_mem,  w, label="FP8",  color=C_FP8,
                edgecolor="white", linewidth=1.2, zorder=3)
    # bar-top values — stagger vertically so they don't collide
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 18,
                f"{b.get_height():.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=C_BF16)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 18,
                f"{b.get_height():.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=C_FP8)
    # delta annotations — placed well above bars
    for i, (bf, fp) in enumerate(zip(bf16_mem, fp8_mem)):
        ax.text(i, 1560, f"{fp - bf:+.0f} MiB ({_pct(bf, fp)})",
                ha="center", fontsize=8.5, fontweight="bold", color=C_SAVE)
    ax.set_xticks(x)
    ax.set_xticklabels(["Forward Peak", "Backward Peak"])
    ax.set_ylabel("Peak Memory (MiB)")
    ax.set_title("(b) Memory", fontweight="bold", pad=8)
    ax.set_ylim(0, 1700)
    ax.legend(loc="upper left", framealpha=0.9)

    # ── (c) Precision ────────────────────────────────────────────────────
    ax = axes[2]
    metrics = ["output\nRRMSE", "$\\partial x$\nRRMSE",
               "$\\partial w_1$\nrel err", "$\\partial w_2$\nrel err"]
    values = [6.60, 7.48, 0.45, 0.50]
    threshold = 10.0
    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, values, width=0.52, color=C_FP8, edgecolor="white",
                  linewidth=1.2, zorder=3, alpha=0.85)
    ax.axhline(threshold, color=C_COST, ls="--", lw=1.5, zorder=2,
               label=f"Threshold ({threshold}%)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=C_SAVE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("(c) Precision", fontweight="bold", pad=8)
    ax.set_ylim(0, 14)
    # place legend and badge so they don't overlap
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.text(3, 12.5, "31/31 PASS", fontsize=10, fontweight="bold",
            color=C_SAVE, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=C_SAVE,
                      lw=1.2, alpha=0.95))

    fig.suptitle(f"SonicMoE FP8 Blockscaled  --  Executive Summary\n"
                 f"{_SUB}  |  {_HW}",
                 fontsize=12.5, fontweight="bold", y=0.97)
    _save(fig, "fig1_executive_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Performance Waterfall
# ═══════════════════════════════════════════════════════════════════════════

def fig2_performance_waterfall() -> None:
    """Waterfall: BF16 total -> GEMM savings -> quant overhead -> FP8 total."""
    steps = [
        ("BF16\nBaseline",      3840,  C_NEUTRAL, True),
        ("GEMM\nSavings",       -921,  C_SAVE,    False),
        ("FP8 Quant\nOverhead", +559,  C_COST,    False),
        ("Other\n$\\Delta$",    -36,   C_NEUTRAL, False),
        ("FP8\nFrontier",       3442,  C_FP8,     True),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.subplots_adjust(top=0.82, bottom=0.15)

    running = 0
    bar_w = 0.55
    tops = []  # track bar top for connectors

    for i, (_, val, color, is_abs) in enumerate(steps):
        if is_abs:
            bottom, height = 0, val
            if i == 0:
                running = val
        else:
            if val < 0:
                bottom = running + val
                height = -val
            else:
                bottom = running
                height = val
            running += val

        ax.bar(i, height, bar_w, bottom=bottom, color=color,
               edgecolor="white", linewidth=1.5, zorder=3, alpha=0.9)
        tops.append(bottom + height)

        # value label — above for positive, below for negative deltas
        val_str = f"{val:+d}" if not is_abs else str(val)
        ax.text(i, bottom + height + 50, f"{val_str} $\\mu$s",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=color if not is_abs else "#1F2937")

    # connector lines
    levels = [3840, 3840, 2919, 3478, 3442]  # running after each step
    for i in range(3):
        y = levels[i + 1]
        ax.plot([i + bar_w / 2 + 0.04, i + 1 - bar_w / 2 - 0.04],
                [y, y], color="#9CA3AF", lw=1.0, ls=":", zorder=2)

    # speedup callout — positioned in the empty space below, left of FP8 bar
    ax.annotate(
        "$\\mathbf{1.12\\times}$ speedup\n$-398\\;\\mu$s ($-10.4\\%$)",
        xy=(4, 3442), xytext=(3.0, 1200),
        fontsize=11, color=C_SAVE, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_SAVE, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec=C_SAVE,
                  lw=1.0, alpha=0.95))

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=10)
    ax.set_ylabel("GPU Kernel Time ($\\mu$s / iter)")
    ax.set_ylim(0, 4600)
    ax.set_xlim(-0.5, 4.8)
    ax.set_title(f"Performance Waterfall  --  nsys GPU Projection\n{_SUB}",
                 fontsize=12.5, fontweight="bold", pad=10)
    _save(fig, "fig2_performance_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Memory Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def fig3_memory_lifecycle() -> None:
    """Memory trajectory across 4 checkpoints: BF16 vs FP8."""
    ckpts = ["Post-Warmup\nBase", "Forward\nPeak",
             "Backward\nPeak", "Post-Cleanup"]
    bf16 = np.array([376, 1386, 1412, 328])
    fp8  = np.array([488, 1263, 1367, 440])
    x = np.arange(len(ckpts))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=0.83, bottom=0.12)

    # filled area between curves
    ax.fill_between(x, fp8, bf16, where=(bf16 > fp8),
                    alpha=0.12, color=C_SAVE, interpolate=True, zorder=1)
    ax.fill_between(x, fp8, bf16, where=(fp8 > bf16),
                    alpha=0.12, color=C_COST, interpolate=True, zorder=1)

    ax.plot(x, bf16, "o-", color=C_BF16, lw=2.5, ms=9, zorder=4,
            label="BF16 Baseline", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, fp8, "s-", color=C_FP8, lw=2.5, ms=9, zorder=4,
            label="FP8 Frontier", markeredgecolor="white", markeredgewidth=1.5)

    # point labels — BF16 always above its point, FP8 always below
    for i in range(len(ckpts)):
        b, f = bf16[i], fp8[i]
        # Decide which is on top at this checkpoint
        if b >= f:
            # BF16 above, FP8 below
            ax.text(i, b + 40, str(b), ha="center", va="bottom", fontsize=8.5,
                    color=C_BF16, fontweight="bold")
            ax.text(i, f - 40, str(f), ha="center", va="top", fontsize=8.5,
                    color=C_FP8, fontweight="bold")
        else:
            # FP8 above, BF16 below
            ax.text(i, f + 40, str(f), ha="center", va="bottom", fontsize=8.5,
                    color=C_FP8, fontweight="bold")
            ax.text(i, b - 40, str(b), ha="center", va="top", fontsize=8.5,
                    color=C_BF16, fontweight="bold")

    # delta annotations — placed offset to the right, clear of point labels
    deltas = fp8 - bf16
    for i in range(len(ckpts)):
        d = deltas[i]
        color = C_SAVE if d < 0 else C_COST
        mid_y = (bf16[i] + fp8[i]) / 2
        ax.text(i + 0.28, mid_y, f"{d:+.0f}\nMiB",
                fontsize=8, fontweight="bold", color=color,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                          lw=0.6, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(ckpts, fontsize=10)
    ax.set_ylabel("GPU Memory (MiB)")
    ax.set_ylim(100, 1700)
    ax.set_xlim(-0.4, 3.7)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_title(f"Memory Lifecycle  --  BF16 vs FP8 Training Iteration\n{_SUB}",
                 fontsize=12.5, fontweight="bold", pad=10)
    _save(fig, "fig3_memory_lifecycle.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Backward Peak Breakdown (100% audit)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_backward_breakdown() -> None:
    """Horizontal bar: 100% backward peak tensor audit (1367 MiB)."""
    categories = [
        # (short_label, mib, pct, kind, note)
        ("dz (TK,2I) bf16",             384, 28, "fixed",   "CUTLASS d_dtype=32"),
        ("y1s (TK,I) bf16",             192, 14, "fixed",   "CUTLASS constraint"),
        ("z_fp8 (TK,2I) ctx",           192, 14, "shipped", "Already FP8 (was 384)"),
        ("w1 bf16 params",              144, 11, "limited", "4-layout ceiling"),
        ("w1T+w2 fp8 caches",           111,  8, "limited", "Can defer w1T"),
        ("dw2 bf16 pre-alloc",           72,  5, "limited", "Can defer"),
        ("w2 bf16 params",               72,  5, "limited", "4-layout ceiling"),
        ("dout_fp8+w2_fp8 input",        66,  5, "fixed",   "Needed by GEMM"),
        ("x+dout+meta",                  49,  4, "fixed",   "Interface contract"),
        ("scales+colvec+autograd",        37,  3, "fixed",   "Overhead"),
    ]

    labels = [c[0] for c in categories]
    sizes  = [c[1] for c in categories]
    pcts   = [c[2] for c in categories]
    kinds  = [c[3] for c in categories]
    notes  = [c[4] for c in categories]

    kind_color = {"fixed": C_NEUTRAL, "shipped": C_SAVE, "limited": C_AMBER}
    kind_label = {
        "fixed":   "Fixed (CUTLASS / interface)",
        "shipped": "Already optimized",
        "limited": "Limited headroom",
    }
    colors = [kind_color[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.85, bottom=0.08)

    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, sizes, height=0.65, color=colors,
            edgecolor="white", linewidth=1.2, zorder=3, alpha=0.88)

    # annotations — all right-aligned to consistent x to avoid overlap
    ann_x = 410
    for i, (sz, pct, note) in enumerate(zip(sizes, pcts, notes)):
        y = y_pos[i]
        # inline label at end of bar
        if sz >= 60:
            ax.text(sz - 4, y, f"{sz}", ha="right", va="center",
                    fontsize=8.5, fontweight="bold", color="white")
        else:
            ax.text(sz + 4, y, f"{sz}", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color="#1F2937")
        # percentage + note at fixed right column
        ax.text(ann_x, y, f"{pct}%  {note}", va="center", ha="left",
                fontsize=8, color="#4B5563", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Memory (MiB)")
    ax.set_xlim(0, 580)

    # legend — bottom left, away from bars
    handles = [Patch(facecolor=kind_color[k], edgecolor="white", label=kind_label[k])
               for k in ("shipped", "limited", "fixed")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.95)

    # total box — top right, clear of top bar
    ax.text(0.97, 0.97,
            "Total: 1368 MiB (theoretical)\n"
            "Measured: 1367 MiB  |  Gap: 0.1%",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            fontweight="bold", color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F5F3FF", ec=C_ACCENT,
                      lw=1.0, alpha=0.95))

    ax.set_title(f"Backward Peak Breakdown  --  100% Tensor-Level Audit\n{_SUB}",
                 fontsize=12.5, fontweight="bold", pad=10)
    _save(fig, "fig4_backward_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Kernel-Level Comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig5_kernel_comparison() -> None:
    """Grouped bar: per-kernel BF16 vs FP8 timing (forward + backward)."""
    kernels = [
        ("GemmGated (up)",       779,  461, "fwd"),
        ("GemmDefault (down)",   387,  235, "fwd"),
        ("FP8 Quant (fwd)",       0,  265, "fwd"),
        ("Other (fwd)",          222,  217, "fwd"),
        ("Wgrad GEMMs x3",     1978, 1578, "bwd"),
        ("DGated (bwd)",         508,  410, "bwd"),
        ("FP8 Quant (bwd)",       0,  328, "bwd"),
        ("SwiGLU bwd",            0,  157, "bwd"),
        ("Other (bwd)",          163,  161, "bwd"),
    ]

    names = [k[0] for k in kernels]
    bf16  = np.array([k[1] for k in kernels], dtype=float)
    fp8   = np.array([k[2] for k in kernels], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=0.20, right=0.88, top=0.85, bottom=0.08)

    y_pos = np.arange(len(names))[::-1]
    h = 0.34

    ax.barh(y_pos + h / 2, bf16, h, label="BF16", color=C_BF16,
            edgecolor="white", linewidth=1.0, zorder=3, alpha=0.88)
    ax.barh(y_pos - h / 2, fp8,  h, label="FP8",  color=C_FP8,
            edgecolor="white", linewidth=1.0, zorder=3, alpha=0.88)

    # speedup / status labels — fixed right column
    ann_x = max(bf16.max(), fp8.max()) * 1.04
    for i, (b16, f8) in enumerate(zip(bf16, fp8)):
        y = y_pos[i]
        if b16 > 0 and f8 > 0:
            ratio = b16 / f8
            color = C_SAVE if ratio >= 1.05 else (C_COST if ratio < 0.95 else C_NEUTRAL)
            ax.text(ann_x, y, f"{ratio:.2f}x",
                    va="center", fontsize=9, fontweight="bold", color=color)
        elif b16 == 0:
            ax.text(ann_x, y, "FP8 only", va="center", fontsize=8,
                    color=C_COST, style="italic")

    # phase separator line
    sep_y = (y_pos[3] + y_pos[4]) / 2
    ax.axhline(sep_y, color="#D1D5DB", lw=1.2, ls="-", zorder=1)
    # phase labels — placed at left margin using axes transform
    ax.text(0.01, 0.72, "FORWARD", fontsize=8.5, fontweight="bold",
            color=C_BF16, va="center", transform=ax.transAxes)
    ax.text(0.01, 0.28, "BACKWARD", fontsize=8.5, fontweight="bold",
            color=C_COST, va="center", transform=ax.transAxes)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Kernel Time ($\\mu$s)")
    ax.set_xlim(0, max(bf16.max(), fp8.max()) * 1.18)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))

    ax.set_title(f"Kernel-Level Performance  --  BF16 vs FP8\n"
                 f"{_SUB}  |  nsys GPU Projection",
                 fontsize=12.5, fontweight="bold", pad=10)
    _save(fig, "fig5_kernel_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Precision State Matrix (heatmap)
# ═══════════════════════════════════════════════════════════════════════════

_DTYPE_ENC = {0: "", 1: "BF16", 2: "FP8", 3: "FP32", 4: "SCALE"}
_TENSOR_NAMES = [
    "x", "router_w", "w1", "w2", "topk_scores", "gather_idx",
    "z / z_fp8", "y1 / y1_fp8", "y2",
    "dout / dout_fp8", "dz / dz_fp8", "y1s (recomp)", "dx",
]


def _precision_matrices():
    """Build (bf16, fp8) numeric matrices + annotation dicts."""
    N = len(_TENSOR_NAMES)
    #                        Rtr  UpF  DnF  DnB  UpW  UpA
    bf16 = np.zeros((N, N_PH))
    bf16[0]  = [1, 1, 1, 1, 1, 1]  # x
    bf16[1]  = [1, 0, 0, 0, 0, 0]  # router_w
    bf16[2]  = [0, 1, 1, 1, 1, 1]  # w1
    bf16[3]  = [0, 0, 1, 1, 0, 0]  # w2
    bf16[4]  = [3, 3, 3, 3, 0, 0]  # topk_scores  FP32
    bf16[5]  = [0, 1, 1, 1, 1, 1]  # gather_idx   (INT32, vis as BF16)
    bf16[6]  = [0, 1, 1, 1, 0, 0]  # z
    bf16[7]  = [0, 1, 1, 1, 0, 0]  # y1
    bf16[8]  = [0, 0, 1, 0, 0, 0]  # y2
    bf16[9]  = [0, 0, 0, 1, 1, 1]  # dout
    bf16[10] = [0, 0, 0, 1, 1, 1]  # dz
    bf16[11] = [0, 0, 0, 1, 0, 0]  # y1s
    bf16[12] = [0, 0, 0, 0, 0, 1]  # dx

    fp8 = np.zeros((N, N_PH))
    fp8[0]  = [1, 1, 1, 1, 1, 1]   # x            BF16
    fp8[1]  = [1, 0, 0, 0, 0, 0]   # router_w     BF16
    fp8[2]  = [0, 1, 1, 1, 1, 1]   # w1           BF16
    fp8[3]  = [0, 0, 1, 1, 0, 0]   # w2           BF16
    fp8[4]  = [3, 3, 3, 3, 0, 0]   # topk_scores  FP32
    fp8[5]  = [0, 1, 1, 1, 1, 1]   # gather_idx
    fp8[6]  = [0, 2, 2, 2, 0, 0]   # z_fp8
    fp8[7]  = [0, 2, 2, 0, 0, 0]   # y1_fp8
    fp8[8]  = [0, 0, 1, 0, 0, 0]   # y2           BF16
    fp8[9]  = [0, 0, 0, 1, 1, 1]   # dout         BF16
    fp8[10] = [0, 0, 0, 2, 1, 2]   # dz: FP8->BF16->FP8
    fp8[11] = [0, 0, 0, 1, 0, 0]   # y1s          BF16
    fp8[12] = [0, 0, 0, 0, 0, 1]   # dx           BF16

    # Annotations: position carefully. We compute an offset direction
    # to avoid overlaps — if the cell above or to the right has a note, shift.
    bf16_ann = {
        (0, 1): ("gather\nvia A_idx",   0.42, -0.35),
        (6, 1): ("384 MiB\n(TK,2I)",    0.42, -0.35),
        (7, 1): ("192 MiB\n(TK,I)",     0.42,  0.35),
        (6, 3): ("used by\ndgated",      0.42, -0.35),
        (10, 3): ("384 MiB",            0.42, -0.30),
        (11, 3): ("recomp",             0.42,  0.35),
    }
    fp8_ann = {
        (0, 1):  ("quant->FP8\nT-sized",    0.42, -0.35),
        (6, 1):  ("192M z_fp8\n(bf16 freed)", 0.42, -0.35),
        (7, 1):  ("96M y1_fp8",             0.42,  0.35),
        (7, 2):  ("prequant\ncache",        0.42,  0.35),
        (9, 3):  ("quant T-sized",          0.42, -0.35),
        (10, 3): ("dz FP8 192M",           0.42, -0.35),
        (10, 4): ("resize_(0)\ndz bf16",    0.42, -0.35),
        (10, 5): ("dz_fp8\nprequant",      -0.42, -0.35),
        (11, 3): ("from\nz_fp8+e8m0",       0.42,  0.35),
    }
    return bf16, fp8, bf16_ann, fp8_ann


def fig6_precision_flow() -> None:
    """Heatmap of tensor precision state across execution phases."""
    colors_map = ["#F9FAFB", C_BF16, C_AMBER, C_COST, C_SCALE]
    cmap = ListedColormap(colors_map)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    bf16, fp8, bf16_ann, fp8_ann = _precision_matrices()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8.5),
                                    gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(top=0.87, bottom=0.10)

    for ax, mat, ann, title in [
        (ax1, bf16, bf16_ann, "BF16 Baseline"),
        (ax2, fp8,  fp8_ann,  "FP8 Frontier"),
    ]:
        # re-enable spines for grid heatmap
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, N_PH, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(_TENSOR_NAMES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(range(N_PH))
        ax.set_xticklabels(PHASES, fontsize=8)
        ax.set_yticks(range(len(_TENSOR_NAMES)))
        ax.set_yticklabels(_TENSOR_NAMES, fontsize=8.5)

        # dtype text in cells
        for r in range(len(_TENSOR_NAMES)):
            for c in range(N_PH):
                v = int(mat[r, c])
                if v == 0:
                    continue
                fc = "white" if v in (1, 2, 3) else "#1F2937"
                ax.text(c, r, _DTYPE_ENC[v], ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=fc)

        # annotations — with per-item offset
        for (r, c_), (note, dx, dy) in ann.items():
            ax.annotate(
                note, xy=(c_, r), xytext=(c_ + dx, r + dy),
                fontsize=5.5, color="#374151", ha="left" if dx > 0 else "right",
                va="top" if dy < 0 else "bottom",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#9CA3AF"),
                bbox=dict(boxstyle="round,pad=0.12", fc="#FFFBEB",
                          ec="#FCD34D", lw=0.5, alpha=0.9),
            )

        # fwd/bwd divider
        ax.axvline(2.5, color=C_COST, lw=1.5, ls="--", alpha=0.6)
        ax.text(2.5, -0.85, "fwd | bwd", ha="center", fontsize=7.5,
                color=C_COST, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8,
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc=C_LIGHT, ec="#D1D5DB", lw=0.8))

    handles = [
        mpatches.Patch(color=C_BF16,  label="BF16"),
        mpatches.Patch(color=C_AMBER, label="FP8 (e4m3fn)"),
        mpatches.Patch(color=C_COST,  label="FP32"),
        mpatches.Patch(color="#F9FAFB", ec="#D1D5DB", lw=0.8,
                       label="Not present"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, edgecolor="#D1D5DB",
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Tensor Precision State per Execution Phase  --  {_SUB}",
        fontsize=12.5, fontweight="bold", y=0.96)
    _save(fig, "fig6_precision_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — Precision Profile
# ═══════════════════════════════════════════════════════════════════════════

def fig7_precision_profile() -> None:
    """Dual-panel precision: RRMSE + cosine similarity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                    gridspec_kw={"wspace": 0.38,
                                                 "width_ratios": [2, 1]})
    fig.subplots_adjust(top=0.82, bottom=0.16)

    # ── (a) Relative Error ───────────────────────────────────────────────
    metrics_a = ["output\nRRMSE", "$\\partial x$\nRRMSE",
                 "$\\partial w_1$\nnorm rel err", "$\\partial w_2$\nnorm rel err"]
    vals_a = [6.60, 7.48, 0.45, 0.50]
    x = np.arange(len(metrics_a))
    bars = ax1.bar(x, vals_a, 0.52, color=C_FP8, edgecolor="white",
                   linewidth=1.2, zorder=3, alpha=0.85)
    ax1.axhspan(10.0, 14.0, color=C_COST, alpha=0.06, zorder=0)
    ax1.axhline(10.0, color=C_COST, ls="--", lw=1.5, zorder=2,
                label="RRMSE threshold (10%)")

    # value labels above bars
    for b, v in zip(bars, vals_a):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.3,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color=C_SAVE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_a)
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_ylim(0, 14)
    ax1.set_title("(a) Error Metrics", fontweight="bold", pad=8)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ── (b) Cosine Similarity ────────────────────────────────────────────
    metrics_b = ["output", "$\\partial x$"]
    vals_b = [0.998, 0.997]
    x2 = np.arange(len(metrics_b))
    bars2 = ax2.bar(x2, vals_b, 0.42, color=C_ACCENT, edgecolor="white",
                    linewidth=1.2, zorder=3, alpha=0.85)
    ax2.axhline(0.99, color=C_COST, ls="--", lw=1.5, zorder=2,
                label="Threshold (0.99)")
    ax2.axhspan(0.985, 0.99, color=C_COST, alpha=0.06, zorder=0)

    for b, v in zip(bars2, vals_b):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.0004,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=C_SAVE)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_b)
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_ylim(0.985, 1.004)
    ax2.set_title("(b) Cosine Similarity", fontweight="bold", pad=8)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # pass badge — placed below the figure via fig.text
    fig.text(0.5, 0.02,
             "31/31 contract tests PASS  |  3 seeds, subprocess-isolated  |  "
             "Shadow weights BIT-IDENTICAL",
             ha="center", fontsize=9, fontweight="bold", color=C_SAVE,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=C_SAVE,
                       lw=1.0, alpha=0.9))

    fig.suptitle(f"Precision Profile  --  FP8 vs BF16 Ground Truth\n{_SUB}",
                 fontsize=12.5, fontweight="bold", y=0.97)
    _save(fig, "fig7_precision_profile.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — Optimization Design Space
# ═══════════════════════════════════════════════════════════════════════════

def fig8_design_space() -> None:
    """Horizontal bar: shipped optimizations vs dead ends."""
    entries = [
        # (label, delta_mib, status, note)
        ("z_fp8 ctx save\n(was 384M bf16)",           -192, "shipped"),
        ("y1 prequant cache\n(fwd->bwd transfer)",     -96, "shipped"),
        ("z_fp8 early release\n(freed after dgated)",  -198, "shipped"),
        ("Eager cache eviction\n(w1 fused+w2 varlen)", -111, "shipped"),
        ("Deferred bwd cache fill\n(anti-spike)",      -148, "shipped"),
        ("FP8 weight caches\n(4 layouts structural)",  +222, "cost"),
        ("stash_bf16_to_cpu\n(CPU offload+proxy)",       +6, "dead"),
        ("FP8 wgrad\n(dual_quantize path)",               0, "dead"),
        ("bf16 dtype change\n(w.data = fp8)",              0, "dead"),
        ("resize_(0)+proxy\n(storage bounds)",             0, "dead"),
    ]

    labels   = [e[0] for e in entries]
    deltas   = [e[1] for e in entries]
    statuses = [e[2] for e in entries]

    status_color = {"shipped": C_SAVE, "cost": C_COST, "dead": "#9CA3AF"}
    colors = [status_color[s] for s in statuses]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=0.26, right=0.90, top=0.85, bottom=0.08)

    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, deltas, height=0.62, color=colors,
            edgecolor="white", linewidth=1.2, zorder=3, alpha=0.88)

    # value labels — right of positive bars, left of negative bars
    for i, d in enumerate(deltas):
        y = y_pos[i]
        if d != 0:
            x_t = d + (8 if d > 0 else -8)
            ha = "left" if d > 0 else "right"
            ax.text(x_t, y, f"{d:+d} MiB", fontsize=9,
                    fontweight="bold", va="center", ha=ha, color="#1F2937")
        else:
            ax.text(8, y, "N/A (overhead > gain)", fontsize=8,
                    va="center", ha="left", color=C_NEUTRAL, style="italic")

    ax.axvline(0, color="#1F2937", lw=1.0, zorder=2)

    # section separators
    sep1_y = (y_pos[4] + y_pos[5]) / 2
    sep2_y = (y_pos[5] + y_pos[6]) / 2
    ax.axhline(sep1_y, color="#E5E7EB", lw=1.0, zorder=1)
    ax.axhline(sep2_y, color="#E5E7EB", lw=1.0, zorder=1)

    # section labels — use axes transform to avoid clipping
    ax.text(-0.01, 0.72, "SHIPPED", fontsize=8, fontweight="bold",
            color=C_SAVE, va="center", transform=ax.transAxes,
            rotation=90, ha="right")
    ax.text(-0.01, 0.44, "COST", fontsize=8, fontweight="bold",
            color=C_COST, va="center", transform=ax.transAxes,
            rotation=90, ha="right")
    ax.text(-0.01, 0.18, "DEAD ENDS", fontsize=8, fontweight="bold",
            color=C_NEUTRAL, va="center", transform=ax.transAxes,
            rotation=90, ha="right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Memory Impact (MiB)")
    ax.set_xlim(-230, 270)

    handles = [
        Patch(facecolor=C_SAVE,    edgecolor="white", label="Shipped optimization"),
        Patch(facecolor=C_COST,    edgecolor="white", label="Structural cost"),
        Patch(facecolor="#9CA3AF", edgecolor="white", label="Dead end (verified)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    shipped_total = sum(d for d, s in zip(deltas, statuses) if s == "shipped")
    cost_total = sum(d for d, s in zip(deltas, statuses) if s == "cost")
    ax.text(0.97, 0.97,
            f"Shipped: {shipped_total:+d} MiB  |  "
            f"Structural cost: {cost_total:+d} MiB\n"
            f"Session 41: Fwd -122 MiB, Bwd -45 MiB",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            fontweight="bold", color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT, ec="#D1D5DB",
                      lw=1.0))

    ax.set_title(f"Optimization Design Space  --  Shipped vs Dead Ends\n{_SUB}",
                 fontsize=12.5, fontweight="bold", pad=10)
    _save(fig, "fig8_design_space.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 — Buffer Lifecycle Gantt
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _Buf:
    """GPU buffer descriptor: dtype, shape, size, lifetime span, events."""
    name: str
    dtype: str          # BF16 | FP8 | FP32 | INT32 | SCALE
    shape: str
    mib: float
    alive: tuple[int, int]  # (start_phase, end_phase) inclusive 0-based
    events: list            # [(phase_idx, glyph_str), ...]


_DTYPE_CLR = {
    "BF16": C_BF16, "FP8": C_AMBER, "FP32": C_COST,
    "INT32": C_NEUTRAL, "SCALE": C_SCALE,
}


def _bf16_bufs() -> list[_Buf]:
    """BF16 baseline buffer inventory (Ernie shape, 12 tracked tensors)."""
    B = _Buf
    return [
        B("x",           "BF16",  "T*H",    48,   (0, 5), []),
        B("w1",          "BF16",  "E*2I*H", 144,  (1, 5), []),
        B("w2",          "BF16",  "E*H*I",  72,   (2, 3), []),
        B("topk_scores", "FP32",  "T*K",    0.25, (0, 3), []),
        B("gather_idx",  "INT32", "TK",     0.25, (1, 5), []),
        B("z",           "BF16",  "TK*2I",  384,  (1, 3), [(3, "\u2298")]),
        B("y1",          "BF16",  "TK*I",   192,  (1, 3), []),
        B("y2",          "BF16",  "TK*I",   192,  (2, 2), []),
        B("dout",        "BF16",  "T*H",    48,   (3, 5), []),
        B("dz",          "BF16",  "TK*2I",  384,  (3, 5), []),
        B("y1s recomp",  "BF16",  "TK*I",   192,  (3, 3), [(3, "\u21bb")]),
        B("dx",          "BF16",  "T*H",    48,   (5, 5), []),
    ]


def _fp8_bufs() -> list[_Buf]:
    """FP8 frontier buffer inventory (Ernie shape, 20 tracked tensors)."""
    B = _Buf
    return [
        B("x",           "BF16",  "T*H",    48,   (0, 5), []),
        B("x_fp8",       "FP8",   "T*H",    24,   (1, 5), [(1, "\u26a1")]),
        B("x_scales",    "SCALE", "T/128",  0.5,  (1, 5), []),
        B("w1",          "BF16",  "E*2I*H", 144,  (1, 5), []),
        B("w1_fp8",      "FP8",   "E*H*2I", 72,   (1, 5), [(1, "\u2192")]),
        B("w2",          "BF16",  "E*H*I",  72,   (2, 3), []),
        B("w2_fp8",      "FP8",   "E*I*H",  36,   (2, 5), [(2, "\u2192")]),
        B("topk_scores", "FP32",  "T*K",    0.25, (0, 3), []),
        B("gather_idx",  "INT32", "TK",     0.25, (1, 5), []),
        B("z_fp8",       "FP8",   "TK*2I",  192,  (1, 3), [(3, "\u2298")]),
        B("z_scales",    "SCALE", "blk",    4,    (1, 3), []),
        B("y1_fp8",      "FP8",   "TK*I",   96,   (1, 2), [(1, "\u26a1")]),
        B("y1_scales",   "SCALE", "blk",    6,    (1, 2), []),
        B("y2",          "BF16",  "TK*I",   192,  (2, 2), []),
        B("dout",        "BF16",  "T*H",    48,   (3, 5), []),
        B("dout_fp8",    "FP8",   "T*H",    24,   (3, 5), [(3, "\u26a1")]),
        B("dz",          "BF16",  "TK*2I",  384,  (3, 4), [(4, "\u2298")]),
        B("dz_fp8",      "FP8",   "TK*2I",  192,  (3, 5), [(3, "\u26a1")]),
        B("y1s recomp",  "BF16",  "TK*I",   192,  (3, 3), [(3, "\u21bb")]),
        B("dx",          "BF16",  "T*H",    48,   (5, 5), []),
    ]


def _draw_gantt(ax: plt.Axes, bufs: list[_Buf], title: str) -> None:
    """Render per-buffer Gantt: dtype-coloured bars, event glyphs, peak badge."""
    bufs = sorted(bufs, key=lambda b: (b.alive[0], -b.mib))
    n = len(bufs)

    # Glyph label mapping (ASCII fallback for fonts without full Unicode)
    _GLYPH_LABEL = {
        "\u26a1": "Q",   # ⚡ quantize
        "\u2298": "F",   # ⊘ free / resize_(0)
        "\u2192": "C",   # → cache
        "\u21bb": "R",   # ↻ recompute
    }

    for i, buf in enumerate(bufs):
        s, e = buf.alive
        ax.barh(i, e - s + 0.85, left=s - 0.425, height=0.72,
                color=_DTYPE_CLR[buf.dtype], alpha=0.88,
                edgecolor="white", linewidth=0.9, zorder=3)
        # MiB label — right of bar
        lbl = f"{buf.mib:.0f}" if buf.mib >= 1 else f"{buf.mib:.1f}"
        ax.text(e + 0.52, i, f"{lbl} M", va="center", fontsize=7,
                color="#374151", fontweight="bold")
        # Shape annotation — further right, lighter
        ax.text(e + 1.45, i, f"({buf.shape})", va="center", fontsize=6,
                color="#9CA3AF")
        # Event glyphs: white circle marker + letter code
        for ph, sym in buf.events:
            code = _GLYPH_LABEL.get(sym, sym)
            ax.plot(ph, i, "o", color="white", ms=13, mew=0, zorder=5)
            ax.text(ph, i, code, ha="center", va="center",
                    fontsize=7.5, color=_DTYPE_CLR[buf.dtype],
                    fontweight="bold", zorder=6)

    ax.set_yticks(range(n))
    ax.set_yticklabels([b.name for b in bufs], fontsize=8)
    ax.set_xticks(range(N_PH))
    ax.set_xticklabels(PHASES, fontsize=7.5)
    ax.set_xlim(-0.6, N_PH + 2.0)
    ax.set_ylim(n - 0.5, -0.5)

    # fwd / bwd divider
    ax.axvline(2.5, color=C_COST, lw=1.3, ls="--", alpha=0.45, zorder=2)
    ax.text(1.0, -0.85, "FORWARD", fontsize=7.5, fontweight="bold",
            color=C_BF16, ha="center", clip_on=False)
    ax.text(4.0, -0.85, "BACKWARD", fontsize=7.5, fontweight="bold",
            color=C_COST, ha="center", clip_on=False)

    # Peak tracked memory badge
    phase_mem = [sum(b.mib for b in bufs if b.alive[0] <= p <= b.alive[1])
                 for p in range(N_PH)]
    peak_ph = int(np.argmax(phase_mem))
    peak_mib = phase_mem[peak_ph]
    ax.text(0.99, 0.03,
            f"Peak tracked: {peak_mib:.0f} MiB  (phase {peak_ph})",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=C_ACCENT, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F5F3FF",
                      ec=C_ACCENT, lw=0.8, alpha=0.9))

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT,
                           ec="#D1D5DB", lw=0.8))


def fig9_buffer_lifecycle() -> None:
    """Per-buffer lifecycle Gantt: dtype-coloured bars, events, cumulative MiB."""
    bf16, fp8 = _bf16_bufs(), _fp8_bufs()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": [len(bf16), len(fp8)], "hspace": 0.38},
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.13, right=0.91)

    _draw_gantt(ax1, bf16, f"BF16 Baseline  ({len(bf16)} buffers)")
    _draw_gantt(ax2, fp8,  f"FP8 Frontier  ({len(fp8)} buffers)")

    # Shared dtype legend
    patches = [Patch(facecolor=_DTYPE_CLR[d], edgecolor="white", label=d)
               for d in ["BF16", "FP8", "FP32", "INT32", "SCALE"]]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               frameon=True, edgecolor="#D1D5DB",
               bbox_to_anchor=(0.52, 0.022))

    fig.text(0.52, 0.005,
             "Event markers:   Q = quantize    F = free / resize_(0)    "
             "C = cache (transposed FP8)    R = recompute from ctx",
             ha="center", fontsize=8.5, color="#4B5563", style="italic")

    fig.suptitle(
        "Buffer Lifecycle  \u2014  Per-Tensor Lifetime, Dtype & Memory\n"
        f"{_SUB}",
        fontsize=13, fontweight="bold", y=0.97)
    _save(fig, "fig9_buffer_lifecycle.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 — Dtype Transformation Flow
# ═══════════════════════════════════════════════════════════════════════════

def fig10_dtype_flow() -> None:
    """Operator-level pipeline showing FP8 quantization / dequantisation points."""

    stages = [
        ("Router\n& Meta",
         {"x": "BF16", "router_w": "BF16"},
         {"topk_scores": "FP32", "gather_idx": "INT32"},
         "standard routing"),
        ("Quant\n(fwd)",
         {"x": "BF16"},
         {"x_fp8": "FP8", "x_scales": "SCALE"},
         "T-sized, 128-group\nblockscaled quant"),
        ("GemmGated\n(up-proj)",
         {"x_fp8": "FP8", "w1_fp8": "FP8", "A_idx": "INT32"},
         {"z_fp8": "FP8", "z_scales": "SCALE"},
         "zero-mat kernel\nno TK-sized FP8 copy"),
        ("SwiGLU\n+ cache",
         {"z_fp8": "FP8"},
         {"y1_fp8": "FP8", "y1_scales": "SCALE"},
         "prequant cache\nfor bwd transfer"),
        ("GemmDefault\n(down-proj)",
         {"y1_fp8": "FP8", "w2_fp8": "FP8"},
         {"y2": "BF16"},
         "FP8 fwd GEMM\nBF16 accumulator"),
        ("DGated\n(bwd)",
         {"dout": "BF16", "w2_fp8": "FP8"},
         {"dz": "BF16"},
         "quant dout inside\nDGated kernel"),
        ("Quant\n(bwd: dz)",
         {"dz": "BF16"},
         {"dz_fp8": "FP8"},
         "for actgrad GEMM\nthen dz.resize_(0)"),
        ("Wgrad\nGEMMs",
         {"dz": "BF16", "x_fp8": "FP8", "y1_fp8": "FP8"},
         {"dw1": "BF16", "dw2": "BF16"},
         "BF16 wgrad\n(FP8 wgrad = dead end)"),
        ("Actgrad\nGEMM",
         {"dz_fp8": "FP8", "w1": "BF16"},
         {"dx": "BF16"},
         "FP8 x BF16 matmul\nscatter to (T,H)"),
    ]

    n = len(stages)
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.02, right=0.98)

    col_op = 0.08
    col_in_start = 0.18
    col_out_start = 0.55
    col_note = 0.85
    row_h = 1.0 / (n + 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headers
    for cx, label in [(col_op, "Operator"), (col_in_start + 0.05, "Inputs"),
                      (col_out_start + 0.05, "Outputs"), (col_note, "Notes")]:
        ax.text(cx, 1 - 0.3 * row_h, label, fontsize=10, fontweight="bold",
                color="#1F2937", ha="center", va="center",
                transform=ax.transAxes)

    # Header underline (use plot instead of axhline to allow transform)
    hdr_y = 1 - 0.55 * row_h
    ax.plot([0.02, 0.98], [hdr_y, hdr_y], color="#D1D5DB", lw=1.2,
            transform=ax.transAxes, clip_on=False)

    # Forward / backward separator
    fwd_bwd_boundary = 4.5
    sep_y = 1 - (fwd_bwd_boundary + 1) * row_h
    ax.plot([0.02, 0.98], [sep_y, sep_y], color=C_COST, lw=1.5, ls="--",
            alpha=0.6, transform=ax.transAxes, clip_on=False)
    ax.text(0.005, sep_y, "bwd", fontsize=8, fontweight="bold",
            color=C_COST, va="center", transform=ax.transAxes)

    def _tensor_box(x_c: float, y_c: float, name: str, dtype: str) -> None:
        color = _DTYPE_CLR.get(dtype, C_NEUTRAL)
        w, h = 0.09, row_h * 0.6
        rect = mpatches.FancyBboxPatch(
            (x_c - w / 2, y_c - h / 2), w, h,
            boxstyle="round,pad=0.005", facecolor=color, edgecolor="white",
            linewidth=0.8, alpha=0.85, transform=ax.transAxes, zorder=3)
        ax.add_patch(rect)
        ax.text(x_c, y_c + h * 0.12, name, fontsize=7, fontweight="bold",
                color="white", ha="center", va="center",
                transform=ax.transAxes, zorder=4)
        ax.text(x_c, y_c - h * 0.25, dtype, fontsize=5.5, color="white",
                ha="center", va="center", alpha=0.85,
                transform=ax.transAxes, zorder=4)

    for idx, (op_label, inputs, outputs, note) in enumerate(stages):
        y_c = 1 - (idx + 1) * row_h

        ax.text(col_op, y_c, op_label, fontsize=8, fontweight="bold",
                color="#1F2937", ha="center", va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc=C_LIGHT,
                          ec="#D1D5DB", lw=0.7))

        in_names = list(inputs.keys())
        in_span = 0.30
        for j, tname in enumerate(in_names):
            x = col_in_start + (j + 0.5) * in_span / max(len(in_names), 1)
            _tensor_box(x, y_c, tname, inputs[tname])

        ax.annotate("", xy=(col_out_start - 0.02, y_c),
                    xytext=(col_in_start + in_span + 0.01, y_c),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#9CA3AF",
                                    lw=1.2, connectionstyle="arc3,rad=0"))

        out_names = list(outputs.keys())
        out_span = 0.24
        for j, tname in enumerate(out_names):
            x = col_out_start + (j + 0.5) * out_span / max(len(out_names), 1)
            _tensor_box(x, y_c, tname, outputs[tname])

        ax.text(col_note, y_c, note, fontsize=6.5, color="#6B7280",
                ha="center", va="center", style="italic",
                transform=ax.transAxes)

    patches = [Patch(facecolor=_DTYPE_CLR[d], edgecolor="white", label=d)
               for d in ["BF16", "FP8", "FP32", "INT32", "SCALE"]]
    ax.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
              frameon=True, edgecolor="#D1D5DB",
              bbox_to_anchor=(0.5, -0.02), bbox_transform=ax.transAxes)

    fig.suptitle(
        "FP8 Dtype Transformation Flow  \u2014  Operator-Level Pipeline\n"
        f"{_SUB}",
        fontsize=13, fontweight="bold", y=0.96)
    _save(fig, "fig10_dtype_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

FIGURES = [
    ("Executive Summary",         fig1_executive_summary),
    ("Performance Waterfall",     fig2_performance_waterfall),
    ("Memory Lifecycle",          fig3_memory_lifecycle),
    ("Backward Peak Breakdown",   fig4_backward_breakdown),
    ("Kernel-Level Comparison",   fig5_kernel_comparison),
    ("Precision State Matrix",    fig6_precision_flow),
    ("Precision Profile",         fig7_precision_profile),
    ("Optimization Design Space", fig8_design_space),
    ("Buffer Lifecycle Gantt",    fig9_buffer_lifecycle),
    ("Dtype Transformation Flow", fig10_dtype_flow),
]


def generate_all(out_dir: Optional[str] = None) -> None:
    """Generate all publication-quality figures."""
    global ASSETS
    if out_dir:
        ASSETS = pathlib.Path(out_dir)
    ASSETS.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("SonicMoE FP8 Visualization Suite (Session 41)")
    print("=" * 55)
    for name, func in FIGURES:
        print(f"  Generating: {name}")
        func()
    print("=" * 55)
    print(f"All {len(FIGURES)} figures saved to: {ASSETS}/")


if __name__ == "__main__":
    generate_all()
