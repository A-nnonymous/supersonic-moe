"""Session 53 Frontier Visualization Suite — SonicMoE FP8 Blockscaled.

Generates four publication-quality figures from the 27-shape nsys grid
(3T × 3E × 3I, nsys GPU-projection + memory):

  fig11  Kernel Runtime Budget Breakdown  (6-panel waterfall)
  fig12  Peak Memory Scaling              (grouped bars + heatmap)
  fig13  Computation Data Flow            (BF16 vs FP8 side-by-side diagram)
  fig14  Multi-Dimensional Speedup Scaling (T/E/I interaction heatmaps)

Data source:
  reports/grid_session53/session53_grid_full.json
  (merged from per-GPU nsys outputs; no GPU required for visualization)

Usage:
  python -m visualization          # generates all figures
  python visualization/frontier_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import matplotlib.patches as mpatches                        # noqa: E402
import matplotlib.patheffects as pe                          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
import numpy as np                                           # noqa: E402

try:
    import seaborn as sns                                    # noqa: E402
except ImportError:
    sns = None

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
ASSETS  = ROOT / "assets"
GRID_JSON = ROOT / "reports" / "grid_session53" / "session53_grid_full.json"

# ── Palette ───────────────────────────────────────────────────────────────────
C_BF16      = "#2563EB"
C_FP8       = "#EA580C"
C_STASH     = "#7C3AED"
C_SAVE      = "#16A34A"
C_COST      = "#DC2626"
C_NEUTRAL   = "#6B7280"
C_FWD       = "#3B82F6"
C_BWD       = "#F59E0B"
C_QUANT     = "#DC2626"
C_GEMM      = "#2563EB"

# Heatmap diverging palette
CMAP_SPEEDUP = "RdYlGn"
CMAP_MEMORY  = "RdYlBu_r"

_HW = "Blackwell B30Z  ·  PyTorch 2.11  ·  QuACK 0.3.7"

# ── Canonical axis orderings ──────────────────────────────────────────────────
T_VALS = [8192, 16384, 32768]
E_VALS = [8, 32, 128]
I_VALS = [1536, 2048, 3072]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_style() -> None:
    """Publication-quality matplotlib defaults."""
    if sns is not None:
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
        "axes.titlesize":       12,
        "axes.labelsize":       10.5,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8.5,
        "figure.dpi":           200,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "savefig.pad_inches":   0.15,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(str(out))
    plt.close(fig)
    print(f"    -> {out}")


def _shape_key(T: int, I: int, E: int, K: int = 8) -> str:
    return f"T{T}_I{I}_E{E}K{K}"


def _short_label(T: int, E: int, I: int) -> str:
    return f"T={T//1024}k\nE={E}\nI={I}"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

class GridData:
    """Loads and indexes the 27-shape nsys grid."""

    def __init__(self, path: Path = GRID_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Grid JSON not found: {path}")
        raw = json.loads(path.read_text())
        self.metadata = raw.get("metadata", {})
        self.shapes: dict[str, dict] = raw.get("shapes", {})

        # Build lookup indices
        self._by_tei: dict[tuple[int, int, int], dict] = {}
        for key, val in self.shapes.items():
            shape = val.get("shape", {})
            t, e, i = shape.get("T", 0), shape.get("E", 0), shape.get("I", 0)
            self._by_tei[(t, e, i)] = val

    def get(self, T: int, E: int, I: int) -> dict | None:
        return self._by_tei.get((T, E, I))

    def speedup(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("speedup", 1.0) if d else 1.0

    def bf16_us(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("bf16", {}).get("per_iter_us", 0) if d else 0

    def fp8_us(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("fp8", {}).get("per_iter_us", 0) if d else 0

    def mem_bf16_bwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_bf16", {}).get("peak_bwd_mib", 0) if d else 0

    def mem_fp8_bwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_fp8", {}).get("peak_bwd_mib", 0) if d else 0

    def mem_bf16_fwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_bf16", {}).get("peak_fwd_mib", 0) if d else 0

    def mem_fp8_fwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_fp8", {}).get("peak_fwd_mib", 0) if d else 0

    def budget(self, T: int, E: int, I: int) -> dict[str, dict]:
        d = self.get(T, E, I)
        return d.get("budget_breakdown", {}) if d else {}

    def budget_savings(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return abs(d.get("budget_savings_us", 0)) if d else 0

    def budget_overhead(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("budget_overhead_us", 0) if d else 0

    def n_shapes(self) -> int:
        return len(self.shapes)

    def available_combos(self) -> list[tuple[int, int, int]]:
        return sorted(self._by_tei.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Kernel Runtime Budget Breakdown (waterfall + decomposition)
# ═══════════════════════════════════════════════════════════════════════════════

# Unified category ordering for consistent visualization
_SAVE_CATS = ["Wgrad GEMM", "GemmGated (fwd)", "GemmDGated (bwd)"]
_COST_CATS = [
    "GemmGated ZeroMat (fwd)", "GemmDGated ZeroMat (bwd)",
    "Blockscaled Quant", "Dual Quant", "Row Quant", "ISA Scale Gather",
]


def fig11_kernel_runtime_breakdown(data: GridData) -> None:
    """6-panel grid: waterfall budget for 6 representative shapes.

    Shows 2 rows (T=8k, T=32k) × 3 columns (I=1536, 2048, 3072),
    each with a horizontal budget bar decomposing GEMM savings vs quant overhead.
    All panels use E=8 (baseline expert count) to isolate T×I scaling.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.28})

    T_rows = [8192, 32768]
    I_cols = [1536, 2048, 3072]
    E_fixed = 8

    for ri, T in enumerate(T_rows):
        for ci, I in enumerate(I_cols):
            ax = axes[ri, ci]
            bb = data.budget(T, E_fixed, I)
            sp = data.speedup(T, E_fixed, I)
            bf16 = data.bf16_us(T, E_fixed, I)
            fp8 = data.fp8_us(T, E_fixed, I)

            if not bb:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            # Separate SAVE vs COST categories
            saves = []
            costs = []
            for cat, vals in sorted(bb.items(), key=lambda x: x[1]["delta_us"]):
                d = vals["delta_us"]
                if d < -5:
                    saves.append((cat, abs(d)))
                elif d > 5:
                    costs.append((cat, d))

            saves.sort(key=lambda x: -x[1])
            costs.sort(key=lambda x: -x[1])

            # Draw horizontal stacked bars: one for SAVE (green), one for COST (red)
            labels = []
            values = []
            colors = []
            for cat, v in saves:
                short = cat.replace(" (fwd)", "\n(fwd)").replace(" (bwd)", "\n(bwd)")
                labels.append(short)
                values.append(-v)  # negative = saving
                colors.append(C_SAVE)
            for cat, v in costs:
                short = cat.replace(" (fwd)", "\n(fwd)").replace(" (bwd)", "\n(bwd)")
                labels.append(short)
                values.append(v)
                colors.append(C_COST)

            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8,
                           edgecolor="white", linewidth=0.5, height=0.7)

            # Value labels
            for bar, v in zip(bars, values):
                w = bar.get_width()
                ha = "right" if w < 0 else "left"
                offset = -30 if w < 0 else 10
                ax.annotate(f"{v:+.0f}", xy=(w, bar.get_y() + bar.get_height()/2),
                            xytext=(offset, 0), textcoords="offset points",
                            va="center", ha=ha, fontsize=6.5, fontweight="bold",
                            color="#374151")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(0, color="#374151", lw=0.8)
            ax.set_xlabel("Delta (us)", fontsize=8)
            ax.invert_yaxis()

            # Net annotation
            net = sum(values)
            ax.set_title(
                f"T={T//1024}k, I={I}, E={E_fixed}\n"
                f"BF16={bf16:.0f}  FP8={fp8:.0f}  "
                f"Speedup={sp:.3f}x  Net={net:+.0f}",
                fontsize=9, fontweight="bold",
            )

    # Legend
    fig.legend(
        handles=[
            mpatches.Patch(color=C_SAVE, label="GEMM Savings (replaced by FP8)"),
            mpatches.Patch(color=C_COST, label="FP8 Overhead (quant + ZeroMat)"),
        ],
        loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "SonicMoE Session 53 — Kernel Runtime Budget Breakdown\n"
        f"E={E_fixed} (fixed)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig11_kernel_budget_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Peak Memory Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fig12_memory_breakdown(data: GridData) -> None:
    """3-panel memory analysis:
    (a) Peak backward memory grouped bars (9 shapes at fixed E=8)
    (b) Memory overhead heatmap (FP8 peak_bwd / BF16 peak_bwd) across T×E at I=1536
    (c) Absolute peak scaling — all 27 shapes as scatter, colored by I
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             gridspec_kw={"wspace": 0.30})

    # ── (a) Peak Bwd Memory: BF16 vs FP8 (9 shapes, E=8 fixed) ──────────
    ax = axes[0]
    E_fixed = 8
    combos = [(T, I) for T in T_VALS for I in I_VALS]
    labels = [f"T={T//1024}k\nI={I}" for T, I in combos]
    bf16_peaks = [data.mem_bf16_bwd(T, E_fixed, I) for T, I in combos]
    fp8_peaks  = [data.mem_fp8_bwd(T, E_fixed, I) for T, I in combos]

    x = np.arange(len(combos))
    w = 0.35
    b1 = ax.bar(x - w/2, bf16_peaks, w, label="BF16", color=C_BF16, alpha=0.85)
    b2 = ax.bar(x + w/2, fp8_peaks, w, label="FP8", color=C_FP8, alpha=0.85)

    # Delta % labels
    for i, (bf, fp) in enumerate(zip(bf16_peaks, fp8_peaks)):
        if bf > 0 and fp > 0:
            pct = (fp - bf) / bf * 100
            ax.text(x[i] + w/2, fp + 20, f"{pct:+.1f}%",
                    ha="center", fontsize=6, fontweight="bold", color=C_FP8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Peak Backward Memory (MiB)")
    ax.set_title(f"(a) Peak Bwd Memory (E={E_fixed})", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    # ── (b) Memory Overhead Heatmap: T × E at I=1536 ────────────────────
    ax = axes[1]
    I_fixed = 1536
    matrix = np.zeros((len(T_VALS), len(E_VALS)))
    for ti, T in enumerate(T_VALS):
        for ei, E in enumerate(E_VALS):
            bf = data.mem_bf16_bwd(T, E, I_fixed)
            fp = data.mem_fp8_bwd(T, E, I_fixed)
            matrix[ti, ei] = ((fp - bf) / bf * 100) if bf > 0 else 0

    im = ax.imshow(matrix, cmap=CMAP_MEMORY, aspect="auto",
                   vmin=-5, vmax=15)
    for ti in range(len(T_VALS)):
        for ei in range(len(E_VALS)):
            ax.text(ei, ti, f"{matrix[ti, ei]:+.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if abs(matrix[ti, ei]) > 8 else "#374151")

    ax.set_xticks(range(len(E_VALS)))
    ax.set_xticklabels([f"E={e}" for e in E_VALS])
    ax.set_yticks(range(len(T_VALS)))
    ax.set_yticklabels([f"T={t//1024}k" for t in T_VALS])
    ax.set_title(f"(b) FP8 Memory Overhead % (I={I_fixed})", fontweight="bold")
    fig.colorbar(im, ax=ax, label="FP8 - BF16 (%)", shrink=0.8)

    # ── (c) Absolute Peak Scaling (all 27 shapes) ────────────────────────
    ax = axes[2]
    i_colors = {1536: "#3B82F6", 2048: "#10B981", 3072: "#F59E0B"}
    for I_val, color in i_colors.items():
        bf_vals = []
        fp_vals = []
        for T in T_VALS:
            for E in E_VALS:
                bf = data.mem_bf16_bwd(T, E, I_val)
                fp = data.mem_fp8_bwd(T, E, I_val)
                if bf > 0 and fp > 0:
                    bf_vals.append(bf)
                    fp_vals.append(fp)
        ax.scatter(bf_vals, fp_vals, c=color, s=40, alpha=0.8,
                   label=f"I={I_val}", edgecolors="white", linewidth=0.5, zorder=3)

    # y=x reference line
    lim_max = max(
        max(data.mem_bf16_bwd(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS if data.get(T, E, I)),
        max(data.mem_fp8_bwd(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS if data.get(T, E, I)),
    ) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("BF16 Peak Bwd (MiB)")
    ax.set_ylabel("FP8 Peak Bwd (MiB)")
    ax.set_title("(c) BF16 vs FP8 Peak Memory (all 27 shapes)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "SonicMoE Session 53 — Peak Memory Analysis\n"
        f"27 shapes (3T x 3E x 3I)  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    _save(fig, "fig12_memory_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Computation Data Flow (BF16 vs FP8 side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_box(ax, xy, w, h, text, color, fontsize=8, alpha=0.15,
              textcolor=None, bold=False, edgecolor=None):
    x0, y0 = xy
    ec = edgecolor or color
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, alpha=alpha,
        edgecolor=ec, linewidth=1.2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    tc = textcolor or "#1F2937"
    ax.text(x0 + w/2, y0 + h/2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=tc, zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])


def _draw_arrow(ax, xy_from, xy_to, color="#374151", lw=1.2,
                style="-|>", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, color=color, linewidth=lw,
        connectionstyle=connectionstyle, zorder=5, mutation_scale=12,
    )
    ax.add_patch(arrow)


def _draw_annotation(ax, xy, text, fontsize=6.5, color="#6B7280", **kw):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color,
            ha="center", va="center", style="italic", zorder=10, **kw)


def fig13_computation_dataflow(data: GridData) -> None:
    """Side-by-side BF16 vs FP8 zero-materialization data flow.

    Uses T=8192, E=8, I=1536 as the representative shape (Ernie-scale).
    Shows the key architectural difference: FP8 avoids TK-sized materialization
    via quantize_and_pack(T-sized) + ZeroMat gather-on-the-fly.
    """
    _apply_style()
    T, E, I, K = 8192, 8, 1536, 8
    TK = T * K
    H = 3072

    sp = data.speedup(T, E, I)
    bf16 = data.bf16_us(T, E, I)
    fp8_ = data.fp8_us(T, E, I)
    bf16_bwd = data.mem_bf16_bwd(T, E, I)
    fp8_bwd = data.mem_fp8_bwd(T, E, I)

    fig, axes = plt.subplots(1, 2, figsize=(16, 14))

    BOX_W = 3.2
    BOX_H = 0.55
    COL_X = 2.0
    Y_TOP = 12.5
    DY = -1.15

    for panel_idx, (ax, mode, title_color) in enumerate([
        (axes[0], "bf16", C_BF16),
        (axes[1], "fp8",  C_FP8),
    ]):
        ax.set_xlim(-0.3, 7.5)
        ax.set_ylim(-1.5, 14)
        ax.set_axis_off()
        ax.set_aspect("equal")

        is_fp8 = (mode == "fp8")
        main_color = C_FP8 if is_fp8 else C_BF16

        title = "FP8 Frontier (Zero-Materialization)" if is_fp8 else "BF16 Baseline"
        ax.text(COL_X + BOX_W/2, Y_TOP + 1.2, title,
                fontsize=14, fontweight="bold", ha="center", color=main_color)
        ax.text(COL_X + BOX_W/2, Y_TOP + 0.75,
                f"T={T}, H={H}, I={I}, E={E}, K={K}",
                fontsize=8, ha="center", color="#6B7280", style="italic")

        # ── FORWARD PASS ──
        y = Y_TOP
        ax.text(-0.1, y - 2.5, "FORWARD", fontsize=10, fontweight="bold",
                color=main_color, rotation=90, va="center", ha="center")

        # Input
        _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                  f"x  [T x H]  BF16\n{T} x {H} = {T*H*2/1048576:.0f} MiB",
                  main_color, fontsize=8, alpha=0.12, bold=True)

        # Routing
        y += DY
        _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                    (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                    color=main_color)
        _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                  f"TopK Routing\nA_idx [T -> TK]  ({T} -> {TK})",
                  "#F59E0B", fontsize=7.5, alpha=0.12)

        if is_fp8:
            # Quantize (T-sized, not TK!)
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "quantize_and_pack(x)\nx_fp8 [T x H] FP8 + scales",
                      C_QUANT, fontsize=7.5, alpha=0.15, bold=True)
            _draw_annotation(ax, (COL_X + BOX_W + 0.6, y + BOX_H/2),
                             "T-sized!\nNot TK-sized", fontsize=7, color=C_QUANT)

            # GemmGated ZeroMat
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H * 1.2 - 0.02),
                        color=main_color)
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H * 1.2,
                      "GemmGated ZeroMat\nx_fp8[T x H] + A_idx -> z_fp8[T x 2I]\n"
                      "Gathers on-the-fly via A_idx",
                      C_GEMM, fontsize=7.5, alpha=0.15, bold=True, edgecolor=C_FP8)
            _draw_annotation(ax, (COL_X + BOX_W + 0.8, y + BOX_H * 0.6),
                             "No TK x H FP8\nmaterialized!", fontsize=7.5, color=C_SAVE)

            # Epilogue quant output
            y += DY * 1.1
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H * 1.2 + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H * 1.2 + DY + BOX_H - 0.02),
                        color=main_color)
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "z_fp8  [T x 2I]  FP8\nEpilogue quant -- zero alloc",
                      C_FP8, fontsize=7.5, alpha=0.15, bold=True)
            _draw_annotation(ax, (COL_X + BOX_W + 0.6, y + BOX_H/2),
                             "CUTLASS epilogue\nwrites FP8 directly",
                             fontsize=6.5, color=C_SAVE)
        else:
            # BF16: Gather + GemmGated
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      f"Gather x -> x_TK [TK x H] BF16\n{TK} x {H} = {TK*H*2/1048576:.0f} MiB",
                      C_COST, fontsize=7.5, alpha=0.12)

            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "GemmGated (CUTLASS)\nx_TK[TK x H] x W1[H x 2I] -> z[TK x 2I]",
                      C_GEMM, fontsize=7.5, alpha=0.12)

            y += DY
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      f"z  [TK x 2I]  BF16\n{TK} x {2*I} = {TK*2*I*2/1048576:.0f} MiB",
                      C_BF16, fontsize=7.5, alpha=0.12)

        # SwiGLU
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                  "SwiGLU Activation\ny1 = swish(z[:,:I]) * z[:,I:]",
                  "#8B5CF6", fontsize=7.5, alpha=0.12)

        # DownProj
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if is_fp8:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H * 1.1,
                      "GemmDGated ZeroMat\ny1_fp8[T x I] + A_idx -> o[T x H]\n"
                      "Scatter-reduce in kernel",
                      C_GEMM, fontsize=7.5, alpha=0.15, bold=True, edgecolor=C_FP8)
        else:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "GemmDGated (CUTLASS)\ny1[TK x I] x W2[I x H] -> y2[TK x H]",
                      C_GEMM, fontsize=7.5, alpha=0.12)

        # Scatter + Output
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if not is_fp8:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "Scatter-Reduce y2[TK x H] -> o[T x H]",
                      C_NEUTRAL, fontsize=7.5, alpha=0.12)
            _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                        (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                        color=main_color)
            y += DY

        _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                  f"output  [T x H]  BF16\n{T} x {H} = {T*H*2/1048576:.0f} MiB",
                  main_color, fontsize=8, alpha=0.12, bold=True)

        # ── BACKWARD divider ──
        bwd_y = y + DY * 0.8
        ax.plot([0.3, 6.8], [bwd_y + 0.25, bwd_y + 0.25],
                ls="--", lw=0.8, color="#D1D5DB", zorder=0)
        ax.text(COL_X + BOX_W/2, bwd_y + 0.4, "BACKWARD",
                fontsize=9, fontweight="bold", ha="center", color=main_color, alpha=0.7)

        y = bwd_y - 0.2
        if is_fp8:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "dGemmDGated ZeroMat -> dy1_fp8\n"
                      "Wgrad auto-tune: OFF at I=1536",
                      C_FP8, fontsize=7, alpha=0.12)
        else:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "dGemmDGated -> dy1 [TK x I] BF16\n"
                      "dW2 via Wgrad GEMM",
                      C_BF16, fontsize=7, alpha=0.12)

        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if is_fp8:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "dGemmGated ZeroMat -> dx_fp8\n"
                      "dW1 via Wgrad GEMM",
                      C_FP8, fontsize=7, alpha=0.12)
        else:
            _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                      "dGemmGated -> dx [TK x H] BF16\n"
                      "dW1 via Wgrad GEMM",
                      C_BF16, fontsize=7, alpha=0.12)

        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        _draw_box(ax, (COL_X, y), BOX_W, BOX_H,
                  f"dx  [T x H]  BF16\n{T} x {H} = {T*H*2/1048576:.0f} MiB",
                  main_color, fontsize=8, alpha=0.12, bold=True)

        # Peak memory badge
        if is_fp8:
            mem_text = f"Peak Bwd: {fp8_bwd:.0f} MiB   |   {fp8_:.0f} us/iter"
        else:
            mem_text = f"Peak Bwd: {bf16_bwd:.0f} MiB   |   {bf16:.0f} us/iter"
        ax.text(COL_X + BOX_W/2, -1.1, mem_text,
                fontsize=9, fontweight="bold", ha="center", color=main_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=main_color,
                          alpha=0.08, edgecolor=main_color, linewidth=1.0))

    fig.suptitle(
        "SonicMoE Session 53 — Computation Data Flow\n"
        f"BF16 Baseline vs FP8 Zero-Materialization  |  Speedup: {sp:.3f}x",
        fontsize=14, fontweight="bold", y=0.99,
    )
    _save(fig, "fig13_computation_dataflow.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 14 — Multi-Dimensional Speedup Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fig14_speedup_scaling(data: GridData) -> None:
    """4-panel multi-dimensional speedup analysis:
    (a) T×I heatmap at E=8
    (b) T×E heatmap at I=1536
    (c) Speedup vs token count (line plot, one line per I, E=8)
    (d) Savings-vs-Overhead scatter (all 27 shapes)
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    # ── (a) T × I heatmap at E=8 ─────────────────────────────────────────
    ax = axes[0, 0]
    E_fixed = 8
    matrix_a = np.zeros((len(T_VALS), len(I_VALS)))
    for ti, T in enumerate(T_VALS):
        for ii, I in enumerate(I_VALS):
            matrix_a[ti, ii] = data.speedup(T, E_fixed, I)

    im_a = ax.imshow(matrix_a, cmap=CMAP_SPEEDUP, aspect="auto",
                     vmin=1.0, vmax=1.7)
    for ti in range(len(T_VALS)):
        for ii in range(len(I_VALS)):
            v = matrix_a[ti, ii]
            ax.text(ii, ti, f"{v:.3f}x",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if v > 1.4 else "#374151")

    ax.set_xticks(range(len(I_VALS)))
    ax.set_xticklabels([f"I={i}" for i in I_VALS])
    ax.set_yticks(range(len(T_VALS)))
    ax.set_yticklabels([f"T={t//1024}k" for t in T_VALS])
    ax.set_title(f"(a) Speedup: T x I (E={E_fixed})", fontweight="bold")
    fig.colorbar(im_a, ax=ax, label="Speedup (x)", shrink=0.8)

    # ── (b) T × E heatmap at I=1536 ──────────────────────────────────────
    ax = axes[0, 1]
    I_fixed = 1536
    matrix_b = np.zeros((len(T_VALS), len(E_VALS)))
    for ti, T in enumerate(T_VALS):
        for ei, E in enumerate(E_VALS):
            matrix_b[ti, ei] = data.speedup(T, E, I_fixed)

    im_b = ax.imshow(matrix_b, cmap=CMAP_SPEEDUP, aspect="auto",
                     vmin=1.0, vmax=1.7)
    for ti in range(len(T_VALS)):
        for ei in range(len(E_VALS)):
            v = matrix_b[ti, ei]
            ax.text(ei, ti, f"{v:.3f}x",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if v > 1.4 else "#374151")

    ax.set_xticks(range(len(E_VALS)))
    ax.set_xticklabels([f"E={e}" for e in E_VALS])
    ax.set_yticks(range(len(T_VALS)))
    ax.set_yticklabels([f"T={t//1024}k" for t in T_VALS])
    ax.set_title(f"(b) Speedup: T x E (I={I_fixed})", fontweight="bold")
    fig.colorbar(im_b, ax=ax, label="Speedup (x)", shrink=0.8)

    # ── (c) Speedup vs T (line plot, one line per I, E=8) ────────────────
    ax = axes[1, 0]
    i_colors = {1536: "#3B82F6", 2048: "#10B981", 3072: "#F59E0B"}
    for I_val, color in i_colors.items():
        speeds = [data.speedup(T, E_fixed, I_val) for T in T_VALS]
        ax.plot(T_VALS, speeds, "o-", color=color, lw=2, ms=7,
                label=f"I={I_val}", zorder=3)
        for ti, (t, s) in enumerate(zip(T_VALS, speeds)):
            ax.annotate(f"{s:.3f}x", xy=(t, s),
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", ha="center", color=color)

    ax.axhline(1.0, color="#9CA3AF", ls="--", lw=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(T_VALS)
    ax.set_xticklabels([f"{t//1024}k" for t in T_VALS])
    ax.set_xlabel("Token Count (T)")
    ax.set_ylabel("FP8 Speedup (x)")
    ax.set_title(f"(c) Speedup Scaling with T (E={E_fixed})", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0.95, max(data.speedup(T, E_fixed, I) for T in T_VALS for I in I_VALS) * 1.08)

    # ── (d) Savings vs Overhead scatter (all 27 shapes) ──────────────────
    ax = axes[1, 1]
    t_markers = {8192: "o", 16384: "s", 32768: "D"}
    for T, marker in t_markers.items():
        for I_val, color in i_colors.items():
            savings_vals = []
            overhead_vals = []
            for E in E_VALS:
                s = data.budget_savings(T, E, I_val)
                o = data.budget_overhead(T, E, I_val)
                if s > 0 or o > 0:
                    savings_vals.append(s)
                    overhead_vals.append(o)
            if savings_vals:
                lab = f"T={T//1024}k, I={I_val}" if T == 8192 else None
                ax.scatter(savings_vals, overhead_vals, c=color, marker=marker,
                           s=50, alpha=0.8, edgecolors="white", linewidth=0.5,
                           label=lab, zorder=3)

    # y=x line (break-even)
    lim = max(
        max((data.budget_savings(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS
             if data.get(T, E, I)), default=1),
        max((data.budget_overhead(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS
             if data.get(T, E, I)), default=1),
    ) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="break-even")
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color=C_SAVE, zorder=0)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color=C_COST, zorder=0)
    ax.text(lim * 0.7, lim * 0.3, "FP8 wins", fontsize=10, color=C_SAVE,
            fontweight="bold", alpha=0.5)
    ax.text(lim * 0.3, lim * 0.7, "BF16 wins", fontsize=10, color=C_COST,
            fontweight="bold", alpha=0.5)

    ax.set_xlabel("GEMM Savings (us)")
    ax.set_ylabel("FP8 Overhead (us)")
    ax.set_title("(d) Budget: Savings vs Overhead (all 27 shapes)", fontweight="bold")

    # Custom legend with markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6B7280",
               markersize=7, label="T=8k"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#6B7280",
               markersize=7, label="T=16k"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#6B7280",
               markersize=7, label="T=32k"),
        mpatches.Patch(color="#3B82F6", label="I=1536"),
        mpatches.Patch(color="#10B981", label="I=2048"),
        mpatches.Patch(color="#F59E0B", label="I=3072"),
        Line2D([0], [0], ls="--", color="k", lw=0.8, label="break-even"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9,
              fontsize=7, ncol=2)

    fig.suptitle(
        "SonicMoE Session 53 — Multi-Dimensional FP8 Speedup Scaling\n"
        f"27 shapes (3T x 3E x 3I)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig14_speedup_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_frontier() -> None:
    """Generate all Session 53 frontier figures."""
    print("\n  === SonicMoE Session 53 Frontier Visualization ===")
    data = GridData()
    n = data.n_shapes()
    combos = data.available_combos()
    print(f"  Loaded {n} shapes from {GRID_JSON.name}")

    print(f"\n  Generating: fig11 -- Kernel Runtime Budget Breakdown ({n} shapes)")
    fig11_kernel_runtime_breakdown(data)

    print(f"  Generating: fig12 -- Peak Memory Scaling ({n} shapes)")
    fig12_memory_breakdown(data)

    print(f"  Generating: fig13 -- Computation Data Flow (representative shape)")
    fig13_computation_dataflow(data)

    print(f"  Generating: fig14 -- Multi-Dimensional Speedup Scaling ({n} shapes)")
    fig14_speedup_scaling(data)

    print(f"\n  All frontier figures saved to {ASSETS}/\n")


if __name__ == "__main__":
    generate_frontier()
