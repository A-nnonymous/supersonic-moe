#!/usr/bin/env python3
"""
SonicMoE BF16 / FP8 Dataflow & Memory Visualization
=====================================================

Publication-quality figures comparing the BF16 baseline and FP8 zero-
materialization training paths on Blackwell (B200), Ernie MoE shape.

Figures generated
-----------------
  1. Buffer Lifecycle Gantt      — per-buffer lifetime + cumulative memory
  2. Precision State Matrix      — dtype at each tensor × phase cell
  3. CUTLASS Kernel Pipeline     — GemmGatedSm100ZeroMat internal stages
  4. Forward / Backward Dataflow — side-by-side operator-level comparison
  5. Per-Phase Memory Breakdown  — stacked-bar activation / weight / grad

Usage
-----
    python -m visualization                     # package entry-point
    python visualization/sonicmoe_dataflow.py   # direct invocation

Output directory: <repo_root>/assets/
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt                              # noqa: E402
import matplotlib.patches as mpatches                        # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
import numpy as np                                           # noqa: E402
import seaborn as sns                                        # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class MoEShape:
    """Ernie-4.5 MoE configuration."""
    T: int = 8192
    H: int = 3072
    I: int = 1536
    E: int = 8
    K: int = 8

    @property
    def TK(self) -> int:
        return self.T * self.K

    @property
    def TWO_I(self) -> int:
        return 2 * self.I

    @property
    def label(self) -> str:
        return (f"T={self.T}  H={self.H}  I={self.I}  "
                f"E={self.E}  K={self.K}  TK={self.TK}")


SHAPE = MoEShape()

# ── Phase system ──────────────────────────────────────────────────────

PHASES = [
    "Router\n& Meta",
    "UpProj\nFwd",
    "DnProj\nFwd",
    "DnProj\nBwd",
    "UpBwd\n(wgrad)",
    "UpBwd\n(actgrad)",
]
N_PH = len(PHASES)

# ── Academic colour palette ───────────────────────────────────────────

_P = dict(
    blue    = "#3B82F6",
    amber   = "#F59E0B",
    red     = "#EF4444",
    gray    = "#9CA3AF",
    emerald = "#10B981",
    violet  = "#A78BFA",
    rose    = "#FB7185",
    teal    = "#2DD4BF",
    yellow  = "#FBBF24",
    bg_fwd  = "#EFF6FF",
    bg_bwd  = "#FEF2F2",
    bg_hdr  = "#F3F4F6",
    border  = "#D1D5DB",
)

DTYPE_COLOR = {
    "BF16":  _P["blue"],
    "FP8":   _P["amber"],
    "FP32":  _P["red"],
    "INT32": _P["gray"],
    "SCALE": _P["emerald"],
}


def _apply_style() -> None:
    """Academic-quality matplotlib / seaborn style."""
    sns.set_style("white")
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":         9,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "figure.dpi":        200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.15,
    })


# ═══════════════════════════════════════════════════════════════════════
# Shared drawing primitives
# ═══════════════════════════════════════════════════════════════════════

def _rounded_box(
    ax, x: float, y: float, w: float, h: float,
    label: str, color: str, *,
    fs: float = 7, text_color: str = "white",
    alpha: float = 0.92, lw: float = 0.8, zorder: int = 3,
) -> FancyBboxPatch:
    """Draw a labelled rounded box and return the patch."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03", fc=color, ec="white",
        alpha=alpha, lw=lw, zorder=zorder, mutation_scale=0.5,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=fs,
        color=text_color, fontweight="medium", zorder=zorder + 1,
        linespacing=1.15,
    )
    return p


def _flow_arrow(
    ax, x0: float, y0: float, x1: float, y1: float, *,
    color: str = "#6B7280", lw: float = 1.2, style: str = "-|>",
    zorder: int = 2,
) -> FancyArrowPatch:
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=10,
        color=color, lw=lw, zorder=zorder,
    )
    ax.add_patch(a)
    return a


def _titled_box(
    ax, cx: float, cy: float, w: float, h: float,
    title: str, body: str, bg: str, border: str,
    *, title_fs: float = 7.5, body_fs: float = 6.5,
) -> None:
    """Operator box with a coloured title bar and monospace body."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04", fc=bg, ec=border,
        lw=1.0, zorder=3, alpha=0.92,
    ))
    th = h * 0.25
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy + h / 2 - th), w, th,
        boxstyle="round,pad=0.02", fc=border, ec=border,
        lw=0.5, zorder=4, alpha=0.85,
    ))
    ax.text(cx, cy + h / 2 - th / 2, title,
            ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color="white", zorder=5)
    ax.text(cx, cy - th * 0.3, body,
            ha="center", va="center", fontsize=body_fs,
            color="#1F2937", zorder=5, linespacing=1.2,
            fontfamily="monospace")


def _save(fig, name: str) -> None:
    """Save figure to assets/."""
    out = ASSETS / name
    fig.savefig(str(out), dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════════
# Buffer data definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Buffer:
    """A single tensor with lifetime metadata."""
    label: str
    dtype: str          # BF16 | FP8 | FP32 | INT32 | SCALE
    mib: float
    phase_start: int
    phase_end: int
    events: list = field(default_factory=list)
    # events: [(phase, symbol, tooltip), ...]


def _bf16_buffers() -> list[Buffer]:
    return [
        Buffer("x  (T,H)",              "BF16",  48,  0, 5),
        Buffer("w1 (2I,H,E)",           "BF16", 144,  1, 5),
        Buffer("w2 (H,I,E)",            "BF16",  72,  2, 3),
        Buffer("topk_scores (T,K)",     "FP32", 0.25, 0, 3),
        Buffer("gather_idx (TK,)",      "INT32",0.25, 0, 5),
        Buffer("z  (TK,2I)  preact",    "BF16", 384,  1, 3),
        Buffer("y1 (TK,I)   postact",   "BF16", 192,  1, 3),
        Buffer("y2 (TK,H)",             "BF16", 384,  2, 2),
        Buffer("dout (T,H)",            "BF16",  48,  3, 5),
        Buffer("dz (TK,2I)",            "BF16", 384,  3, 5),
        Buffer("y1s (TK,I) recomp",     "BF16", 192,  3, 3,
               [(3, "↻", "recomp via dgated")]),
        Buffer("dx  (TK,H)",            "BF16", 384,  5, 5),
    ]


def _fp8_buffers() -> list[Buffer]:
    return [
        Buffer("x  (T,H)",               "BF16",  48,  0, 5),
        Buffer("w1 (2I,H,E)",            "BF16", 144,  1, 5),
        Buffer("w2 (H,I,E)",             "BF16",  72,  2, 3),
        Buffer("x_fp8  (T,H)",           "FP8",   24,  1, 1,
               [(1, "⚡", "T-sized rowquant")]),
        Buffer("x_scales_tk  ISA",       "SCALE",  4,  1, 1,
               [(1, "⚡", "gather ISA scales")]),
        Buffer("w1_fp8 (E,H,2I) cache",  "FP8",   72,  1, 1),
        Buffer("z_fp8  (TK,2I)",         "FP8",  192,  1, 3,
               [(1, "⊘", "z BF16→resize_(0)")]),
        Buffer("z_scales (TK,96) u8",    "SCALE",  6,  1, 3),
        Buffer("y1_fp8 (TK,I)",          "FP8",   96,  1, 2,
               [(2, "→", "prequant→DnFwd")]),
        Buffer("y1_scales ISA",          "SCALE",  2,  1, 2),
        Buffer("w2_fp8 (E,I,H) cache",   "FP8",   36,  2, 2),
        Buffer("y2 (TK,H)",              "BF16", 384,  2, 2),
        Buffer("topk_scores (T,K)",      "FP32", 0.25, 0, 3),
        Buffer("gather_idx (TK,)",       "INT32",0.25, 0, 5),
        Buffer("dout (T,H)",             "BF16",  48,  3, 5),
        Buffer("dout_fp8 (T,H)",         "FP8",   24,  3, 3,
               [(3, "⚡", "T-sized quant")]),
        Buffer("dz (TK,2I) temp",        "BF16", 384,  3, 4,
               [(4, "⊘", "resize_(0) after wgrad")]),
        Buffer("dz_fp8 (TK,2I)",         "FP8",  192,  3, 5,
               [(4, "→", "prequant→UpBwd(A)")]),
        Buffer("y1s (TK,I) recomp",      "BF16", 192,  3, 3,
               [(3, "↻", "from z_fp8 + e8m0")]),
        Buffer("dx  (TK,H)",             "BF16", 384,  5, 5),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Buffer Lifecycle Gantt
# ═══════════════════════════════════════════════════════════════════════

def _draw_gantt(ax, buffers: list[Buffer], title: str) -> None:
    n = len(buffers)
    for i in range(N_PH):
        bg = _P["bg_fwd"] if i < 3 else _P["bg_bwd"]
        ax.axvspan(i - 0.45, i + 0.45, color=bg, zorder=0)

    for idx, buf in enumerate(buffers):
        y = n - 1 - idx
        color = DTYPE_COLOR[buf.dtype]
        width = (buf.phase_end - buf.phase_start) + 0.7
        ax.barh(y, width, left=buf.phase_start - 0.35, height=0.65,
                color=color, alpha=0.85, ec="white", lw=0.8, zorder=2)
        sz = (f"{buf.mib:.0f} MiB" if buf.mib >= 1
              else f"{buf.mib * 1024:.0f} KiB")
        ax.text((buf.phase_start + buf.phase_end) / 2, y,
                f"{buf.label}\n{sz}", ha="center", va="center",
                fontsize=5.8, fontweight="medium", color="white"
                if buf.dtype not in ("SCALE", "INT32") else "#1F2937",
                zorder=3)
        for ph, sym, _ in buf.events:
            ax.annotate(sym, xy=(ph + 0.38, y + 0.22), fontsize=7,
                        fontweight="bold", color="#1F2937",
                        ha="center", va="bottom", zorder=4)

    # cumulative memory envelope
    mem = np.zeros(N_PH)
    for buf in buffers:
        for p in range(buf.phase_start, buf.phase_end + 1):
            mem[p] += buf.mib
    ax2 = ax.twinx()
    ax2.fill_between(range(N_PH), mem, alpha=0.12, color="#6366F1")
    ax2.plot(range(N_PH), mem, color="#6366F1", lw=1.5, ls="--",
             marker="o", ms=4, zorder=5)
    for i, v in enumerate(mem):
        ax2.text(i, v + 15, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=6.5, color="#4338CA", fontweight="bold")
    ax2.set_ylim(0, mem.max() * 1.35)
    ax2.set_ylabel("Cumulative MiB", fontsize=8, color="#4338CA")
    ax2.tick_params(axis="y", labelsize=7, colors="#4338CA")
    ax2.spines["top"].set_visible(False)

    ax.set_yticks([])
    ax.set_xticks(range(N_PH))
    ax.set_xticklabels(PHASES, fontsize=7.5)
    ax.set_xlim(-0.6, N_PH - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6,
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc=_P["bg_hdr"], ec=_P["border"], lw=0.8))
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def fig1_memory_lifecycle() -> None:
    """Buffer lifetime Gantt chart with cumulative memory envelope."""
    bf16, fp8 = _bf16_buffers(), _fp8_buffers()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(17, 16),
        gridspec_kw={"height_ratios": [len(bf16), len(fp8)], "hspace": 0.32},
    )
    _draw_gantt(ax1, bf16, "BF16 Baseline — Buffer Lifecycle")
    _draw_gantt(ax2, fp8,  "FP8 Frontier  — Buffer Lifecycle")

    fig.suptitle(
        f"SonicMoE Buffer Lifecycle & Peak Memory  ·  Ernie ({SHAPE.label})",
        fontsize=13, fontweight="bold", y=0.995)
    handles = [mpatches.Patch(color=DTYPE_COLOR[k], label=k)
               for k in ("BF16", "FP8", "FP32", "INT32", "SCALE")]
    handles.append(mpatches.Patch(
        fc="white", ec="black", lw=0.6,
        label="⚡quant   ⊘resize_(0)   →cache   ↻recomp"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=8, frameon=True, edgecolor=_P["border"])
    _save(fig, "fig1_memory_lifecycle.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Precision State Matrix
# ═══════════════════════════════════════════════════════════════════════

# Encoding: 0=absent 1=BF16 2=FP8 3=FP32 4=SCALE
_DTYPE_ENC = {0: "", 1: "BF16", 2: "FP8", 3: "FP32", 4: "SCALE"}

_TENSOR_NAMES = [
    "x", "router_w", "w1", "w2", "topk_scores", "gather_idx",
    "z / z_fp8", "y1 / y1_fp8", "y2",
    "dout / dout_fp8", "dz / dz_fp8", "y1s (recomp)", "dx",
]


def _precision_matrices():
    """Build (bf16, fp8) numeric matrices + annotation dicts."""
    N = len(_TENSOR_NAMES)

    #                       Rtr  UpF  DnF  DnB  UpW  UpA
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
    fp8[5]  = [0, 1, 1, 1, 1, 1]   # gather_idx   (INT32, vis as BF16)
    fp8[6]  = [0, 2, 2, 2, 0, 0]   # z_fp8
    fp8[7]  = [0, 2, 2, 0, 0, 0]   # y1_fp8
    fp8[8]  = [0, 0, 1, 0, 0, 0]   # y2           BF16
    fp8[9]  = [0, 0, 0, 1, 1, 1]   # dout         BF16
    fp8[10] = [0, 0, 0, 2, 1, 2]   # dz: FP8→BF16→FP8
    fp8[11] = [0, 0, 0, 1, 0, 0]   # y1s          BF16
    fp8[12] = [0, 0, 0, 0, 0, 1]   # dx           BF16

    bf16_ann = {
        (0, 1): "gather\nvia A_idx",
        (6, 1): "384 MiB\n(TK,2I)",
        (7, 1): "192 MiB\n(TK,I)",
        (6, 3): "used by\ndgated",
        (10, 3): "384 MiB",
        (11, 3): "↻ recomp",
    }
    fp8_ann = {
        (0, 1):  "quant→FP8\nT-sized",
        (6, 1):  "192 MiB z_fp8\n(BF16→freed)",
        (7, 1):  "96 MiB y1_fp8",
        (7, 2):  "→ prequant\ncache",
        (9, 3):  "⚡ quant T-sized",
        (10, 3): "dz FP8 192 MiB",
        (10, 4): "⊘ resize_(0)\ndz BF16",
        (10, 5): "→ dz_fp8\nprequant",
        (11, 3): "↻ from\nz_fp8+e8m0",
    }
    return bf16, fp8, bf16_ann, fp8_ann


def fig2_precision_flow() -> None:
    """Heatmap of tensor precision state across execution phases."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = ["#F9FAFB", _P["blue"], _P["amber"], _P["red"], _P["emerald"]]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    bf16, fp8, bf16_ann, fp8_ann = _precision_matrices()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5),
                                    gridspec_kw={"wspace": 0.25})

    for ax, mat, ann, title in [
        (ax1, bf16, bf16_ann, "BF16 Baseline"),
        (ax2, fp8,  fp8_ann,  "FP8 Frontier"),
    ]:
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, N_PH, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(_TENSOR_NAMES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(range(N_PH))
        ax.set_xticklabels(PHASES, fontsize=7.5)
        ax.set_yticks(range(len(_TENSOR_NAMES)))
        ax.set_yticklabels(_TENSOR_NAMES, fontsize=8)

        for r in range(len(_TENSOR_NAMES)):
            for c in range(N_PH):
                v = int(mat[r, c])
                if v == 0:
                    continue
                fc = "white" if v in (1, 2, 3) else "#1F2937"
                ax.text(c, r, _DTYPE_ENC[v], ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=fc)

        for (r, c_), note in ann.items():
            ax.annotate(
                note, xy=(c_, r), xytext=(c_ + 0.42, r - 0.38),
                fontsize=5, color="#374151", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#9CA3AF"),
                bbox=dict(boxstyle="round,pad=0.15", fc="#FFFBEB",
                          ec="#FCD34D", lw=0.5, alpha=0.9),
            )

        ax.axvline(2.5, color="#DC2626", lw=1.5, ls="--", alpha=0.6)
        ax.text(2.5, -0.9, "← fwd | bwd →", ha="center", fontsize=7,
                color="#DC2626", fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8,
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc=_P["bg_hdr"], ec=_P["border"], lw=0.8))

    handles = [
        mpatches.Patch(color=_P["blue"],  label="BF16"),
        mpatches.Patch(color=_P["amber"], label="FP8 (e4m3fn)"),
        mpatches.Patch(color=_P["red"],   label="FP32"),
        mpatches.Patch(color="#F9FAFB", ec="#D1D5DB", lw=0.8,
                       label="Not present"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=True, edgecolor=_P["border"])
    fig.suptitle(
        f"Tensor Precision State per Execution Phase  ·  Ernie ({SHAPE.label})",
        fontsize=12.5, fontweight="bold", y=1.01)
    _save(fig, "fig2_precision_flow.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — CUTLASS Kernel Pipeline
# ═══════════════════════════════════════════════════════════════════════

def fig3_kernel_pipeline() -> None:
    """GemmGatedSm100ZeroMat kernel: GMEM → SMEM → Regs → GMEM."""
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "GemmGatedSm100ZeroMat  ·  CUTLASS Fused GEMM + SwiGLU + BlockScaled Quant\n"
        f"(FP8 forward, Ernie: M=TK={SHAPE.TK}, N=2I={SHAPE.TWO_I}, K=H={SHAPE.H})",
        fontsize=12, fontweight="bold", pad=12)

    # memory tier backgrounds
    tiers = [
        ((-0.3, 0, 2.1, 7.2), "#F0FDF4", "#86EFAC", "GMEM (HBM)",    0.75),
        ((2.3,  0, 2.3, 7.2), "#EFF6FF", "#93C5FD", "SMEM (96 KiB)", 3.45),
        ((5.0,  0, 2.6, 7.2), "#FFF7ED", "#FDBA74", "Registers / TMEM", 6.3),
        ((8.0,  0, 2.3, 7.2), "#F0FDF4", "#86EFAC", "GMEM (output)", 9.15),
    ]
    tier_text_colors = ["#166534", "#1E40AF", "#9A3412", "#166534"]
    for (rect, fc, ec, label, txt_x), tc in zip(tiers, tier_text_colors):
        ax.add_patch(FancyBboxPatch(
            (rect[0], rect[1]), rect[2], rect[3],
            boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2, zorder=0))
        ax.text(txt_x, 7.0, label, fontsize=10, fontweight="bold",
                ha="center", color=tc)

    # GMEM inputs
    _gmem_in = [
        (5.6, "mA  (T,H)\nFP8 e4m3fn\n24 MiB",       _P["amber"]),
        (4.4, "A_idx (TK,)\nINT32\n0.25 MiB",          _P["gray"]),
        (3.2, "mB  (E,H,2I)\nFP8 e4m3fn\n72 MiB",     _P["amber"]),
        (2.0, "mSFA (TK,H÷32)\nISA-packed\n~4 MiB",   _P["emerald"]),
        (0.8, "mSFB (E,H÷32,2I)\nISA-packed\n~2 MiB", _P["emerald"]),
    ]
    for y, label, col in _gmem_in:
        tc = "white" if col != _P["emerald"] else "#1F2937"
        _rounded_box(ax, 0.0, y, 1.5, 0.7, label, col, fs=6.5,
                     text_color=tc)

    # SMEM ring buffers
    _smem = [
        (5.6, "sAIdx\n(128,) INT32\nring ×2",  _P["gray"]),
        (4.4, "sA  (128,32)\nFP8 tile\nring ×4",  _P["amber"]),
        (3.2, "sB  (128,32)\nFP8 tile\nring ×4",  _P["amber"]),
        (2.0, "sSFA (4,32)\nISA-packed\nring ×4",  _P["emerald"]),
        (0.8, "sSFB (4,32)\nISA-packed\nring ×4",  _P["emerald"]),
    ]
    for y, label, col in _smem:
        tc = "white" if col != _P["emerald"] else "#1F2937"
        _rounded_box(ax, 2.6, y, 1.7, 0.7, label, col, fs=6,
                     text_color=tc)

    # load arrows: GMEM → SMEM
    load_labels = ["cp.async\ngather", "LDG\nprefetch", "TMA", "TMA", "TMA"]
    ys = [5.95, 4.75, 3.55, 2.35, 1.15]
    for y, lbl in zip(ys, load_labels):
        _flow_arrow(ax, 1.55, y, 2.55, y, color="#059669", lw=1.3)
        ax.text(2.05, y + 0.05, lbl, fontsize=5.5, ha="center",
                va="center", color="#065F46", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec="#A7F3D0", lw=0.5, alpha=0.9))

    # WGMMA compute block
    _rounded_box(ax, 5.3, 4.2, 2.0, 1.6,
                 "WGMMA\n(128×128 tile)\n━━━━━━━━━━━━━━\nacc: FP32\n"
                 "K-loop: H÷32 iters\nSFA×SFB descaling\n4 MMA/iter",
                 "#7C3AED", fs=6.2)
    for sy in [4.75, 3.55]:
        _flow_arrow(ax, 4.35, sy, 5.25, 5.0, color="#7C3AED", lw=1.0)
    for sy in [2.35, 1.15]:
        _flow_arrow(ax, 4.35, sy, 5.25, 4.6, color=_P["emerald"], lw=1.0)

    # SwiGLU epilogue
    _rounded_box(ax, 5.3, 2.7, 2.0, 1.1,
                 "SwiGLU Epilogue\n━━━━━━━━━━━━━━\n"
                 "silu = gate·σ(gate)\npostact = silu × up\n"
                 "all in FP32 regs",
                 "#DB2777", fs=6.2)
    _flow_arrow(ax, 6.3, 4.15, 6.3, 3.85, color="#7C3AED", lw=1.5)

    # blockscaled quant epilogue
    _rounded_box(ax, 5.3, 0.9, 2.0, 1.4,
                 "BlockScaled Quant\n━━━━━━━━━━━━━━\n"
                 "amax over 32 elems\n→ E8M0 exponent\n"
                 "→ scale = 2^(254−e8m0)\nz_fp8 = clamp(z·s, ±448)",
                 "#EA580C", fs=6.0)
    _flow_arrow(ax, 6.3, 2.65, 6.3, 2.35, color="#DB2777", lw=1.5)

    # dual output arrows
    _flow_arrow(ax, 7.35, 3.3, 7.95, 3.3, color=_P["blue"], lw=1.5)
    ax.text(7.65, 3.48, "bf16\ncast", fontsize=5.5, ha="center",
            color="#1D4ED8", fontweight="bold")
    _flow_arrow(ax, 7.35, 1.5, 7.95, 1.5, color=_P["amber"], lw=1.5)

    # GMEM outputs
    _gmem_out = [
        (4.8, "mD  (TK,I)\ny1 postact\nBF16  192 MiB",         _P["blue"]),
        (3.4, "mD₂ (TK,2I)\nz preact\nBF16  384 MiB",          _P["blue"]),
        (2.0, "mZ  (TK,2I)\nz_fp8 preact\nFP8  192 MiB",       _P["amber"]),
        (0.8, "mZScale (TK,96)\nE8M0 scales\nUINT8  6 MiB",    _P["emerald"]),
    ]
    for y, label, col in _gmem_out:
        tc = "white" if col != _P["emerald"] else "#1F2937"
        _rounded_box(ax, 8.2, y, 1.9, 0.7 if y < 4.0 else 0.9,
                     label, col, fs=6.5, text_color=tc)

    _flow_arrow(ax, 7.35, 3.1, 8.15, 5.2, color=_P["blue"], lw=1.0)
    _flow_arrow(ax, 7.35, 2.9, 8.15, 3.7, color=_P["blue"], lw=1.0)
    _flow_arrow(ax, 7.35, 1.6, 8.15, 2.35, color=_P["amber"], lw=1.0)
    _flow_arrow(ax, 7.35, 1.2, 8.15, 1.15, color=_P["emerald"], lw=1.0)

    # key insight callout
    ax.annotate(
        "Zero-Materialization:\n"
        "A(T,H) FP8 in GMEM\n"
        "cp.async gathers rows via A_idx\n"
        "→ NO TK-sized FP8 copy!\n"
        "Saves 192 MiB HBM",
        xy=(2.6, 5.95), xytext=(0.0, 0.2),
        fontsize=7, color="#991B1B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2",
                  ec="#FCA5A5", lw=1.0, alpha=0.95),
        arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.2),
    )
    ax.text(3.45, 0.15,
            "Pipeline: 4-stage ring buffers  •  mbarrier sync  •  "
            "1 CTA / SM  •  occupancy = 1",
            fontsize=7, ha="center", color="#4B5563", style="italic")
    _save(fig, "fig3_kernel_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Forward + Backward Dataflow
# ═══════════════════════════════════════════════════════════════════════

# Operator descriptors: (phase_label, title, bf16_body, fp8_body, y, h)
_OPERATORS = [
    # ── Forward ──────────────────────────────────────────────────
    (
        "①", "Router + Metadata",
        ("x(T,H) @ router_w.T → logits(T,E)\n"
         "softmax_topk → scores(T,K) indices(T,K)\n"
         "→ gather_idx(TK,)  expert_offset(E+1,)"),
        None,  # shared — same for both
        18.0, 1.15,
    ),
    (
        "②", "UpProj Fwd",
        ("CUTLASS GemmGatedSm100\n"
         "A=x(T,H) BF16  gather via A_idx\n"
         "B=w1(E,H,2I) BF16  TMA load\n"
         "──────────────────────────\n"
         "Epilogue: SiLU(gate)×up  FP32 regs\n"
         "──────────────────────────\n"
         "→ z (TK,2I) BF16  384M  [ctx save]\n"
         "→ y1(TK,I)  BF16  192M  [ctx save]"),
        ("quant_and_pack(x) → x_fp8(T,H) 24M\n"
         "gather_isa_scales → x_scales_tk\n"
         "precompute_w1 → w1_fp8 72M (cached)\n"
         "──────────────────────────\n"
         "GemmGatedSm100ZeroMat ★\n"
         " cp.async gather A(T) via A_idx\n"
         " NO TK-sized FP8 copy!\n"
         "──────────────────────────\n"
         "quant z→z_fp8 192M  z.resize_(0) ⊘\n"
         "quant y1→y1_fp8 96M y1.resize_(0) ⊘\n"
         "push prequant cache 'fwd'"),
        14.7, 2.8,
    ),
    (
        "③", "DownProj Fwd",
        ("CUTLASS GemmSm100 (standard)\n"
         "A=y1(TK,I) BF16  B=w2(E,I,H) BF16\n"
         "──────────────────────────\n"
         "→ y2(TK,H) BF16  384M\n"
         "──────────────────────────\n"
         "router_fwd: y2 × scores → o(T,H)\n"
         "scatter-reduce via reverse_idx"),
        ("pop prequant 'fwd' → y1_fp8 (0 copy)\n"
         "precompute w2 → w2_fp8 36M (cached)\n"
         "──────────────────────────\n"
         "blockscaled_fp8_gemm_varlen\n"
         "  A=y1_fp8  B=w2_fp8\n"
         "→ y2(TK,H) BF16  384M\n"
         "──────────────────────────\n"
         "router_fwd: same as BF16"),
        10.7, 2.3,
    ),
    # ── Backward ─────────────────────────────────────────────────
    (
        "④", "DownProj Bwd",
        ("router_bwd → dout_exp(TK,H)\n"
         "──────────────────────────\n"
         "gemm_dgated (CUTLASS fused)\n"
         "  A=dout_exp  B=w2  PreAct=z(BF16)\n"
         "  activation=swiglu\n"
         "  colvec_scale=topk_scores\n"
         "──────────────────────────\n"
         "→ dz(TK,2I) BF16  384M\n"
         "→ y1s(TK,I) 192M  ↻recomp from z\n"
         "→ ds (router grad, colvec_reduce)\n"
         "──────────────────────────\n"
         "wgrad: dout.T @ y1s → dw2"),
        ("quant dout → dout_fp8(T,H) 24M ⚡\n"
         "gather_isa_scales → dout_scales_tk\n"
         "──────────────────────────\n"
         "GemmDGatedFP8CLoadSm100ZeroMat\n"
         "  A=dout_fp8  B=w2_fp8\n"
         "  preact=z_fp8 + z_scales (e8m0)\n"
         "  LDG z_fp8 in epilogue → dequant\n"
         "  dSwiGLU in FP32 regs\n"
         "──────────────────────────\n"
         "→ dz 384M  y1s ↻  ds\n"
         "wgrad: dout.T @ y1s → dw2 (BF16)\n"
         "quant dz→dz_fp8 192M cache 'bwd'"),
        5.8, 3.1,
    ),
    (
        "⑤", "UpProj Bwd",
        ("wgrad: gemm\n"
         "  A=x(T,H).T  B=dz(TK,2I)\n"
         "  gather via A_idx, cu_seqlens_k\n"
         "→ dw1(E,2I,H) BF16\n"
         "──────────────────────────\n"
         "actgrad: gemm  (BF16)\n"
         "  A=dz(TK,2I)  B=w1(E,2I,H).T\n"
         "→ dx_expanded(TK,H) BF16\n"
         "──────────────────────────\n"
         "token_reduce: dx(TK,H) → dx(T,H)"),
        ("wgrad (BF16 perf mode):\n"
         "  A=x.T  B=dz(BF16) → dw1\n"
         "  dz.resize_(0)  ⊘ FREE 384M\n"
         "──────────────────────────\n"
         "pop prequant 'bwd' → dz_fp8\n"
         "precompute w1T → w1T_fp8\n"
         "actgrad (FP8):\n"
         "  blockscaled_fp8_gemm_varlen\n"
         "  A=dz_fp8  B=w1T_fp8\n"
         "→ dx_expanded(TK,H) BF16\n"
         "──────────────────────────\n"
         "token_reduce: same as BF16"),
        1.5, 2.8,
    ),
]


def fig4_dataflow() -> None:
    """Dual-column forward+backward dataflow comparison."""
    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.5, 20.5)
    ax.axis("off")

    BX, FX, W = 2.4, 7.8, 4.0   # column centres, box width

    # column headers
    for cx, label, col in [(BX, "BF16 Baseline", _P["blue"]),
                            (FX, "FP8 Frontier",  _P["amber"])]:
        ax.add_patch(FancyBboxPatch(
            (cx - W / 2 - 0.1, 19.6), W + 0.2, 0.7,
            boxstyle="round,pad=0.06", fc=col, ec="white", lw=2, zorder=5))
        ax.text(cx, 19.95, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=6)

    # phase backgrounds
    ax.add_patch(FancyBboxPatch((-0.15, 9.85), 10.5, 9.6,
        boxstyle="round,pad=0.08", fc=_P["bg_fwd"], ec=_P["border"],
        lw=0.8, zorder=0))
    ax.text(5.1, 19.35, "FORWARD", fontsize=10, fontweight="bold",
            ha="center", color="#1E40AF", style="italic")
    ax.add_patch(FancyBboxPatch((-0.15, -0.35), 10.5, 9.8,
        boxstyle="round,pad=0.08", fc=_P["bg_bwd"], ec=_P["border"],
        lw=0.8, zorder=0))
    ax.text(5.1, 9.3, "BACKWARD", fontsize=10, fontweight="bold",
            ha="center", color="#991B1B", style="italic")

    # operator boxes
    for num, name, bf16_body, fp8_body, y, h in _OPERATORS:
        bf16_title = f"{num} {name}"
        is_shared = fp8_body is None
        fp8_title = bf16_title if is_shared else f"{num} {name} (FP8)"

        _titled_box(ax, BX, y, W, h, bf16_title, bf16_body,
                    "#DBEAFE" if y > 9 else "#FEE2E2",
                    "#2563EB" if y > 9 else "#DC2626",
                    title_fs=8, body_fs=6.5)
        _titled_box(ax, FX, y, W, h, fp8_title,
                    bf16_body if is_shared else fp8_body,
                    ("#E0E7FF" if is_shared else "#FFF7ED") if y > 9
                    else "#FFF7ED",
                    ("#6366F1" if is_shared else "#EA580C") if y > 9
                    else "#EA580C",
                    title_fs=8,
                    body_fs=6.3 if not is_shared else 6.5)

    # "SHARED" badge on Router
    ax.text(5.1, 18.15, "SHARED", fontsize=9, fontweight="bold",
            ha="center", color="#4338CA",
            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                      ec="#818CF8", lw=1))

    # inter-phase arrows
    arrow_ys = [(17.35, 16.15), (13.25, 11.95), (9.5, 7.45), (4.15, 2.95)]
    for cx in (BX, FX):
        for y0, y1 in arrow_ys:
            _flow_arrow(ax, cx, y0, cx, y1, color="#9CA3AF", lw=1.5)

    # cross-column annotations
    _annotations = [
        (15.5, (FX - W / 2 - 0.05, 15.1),
         "FP8 saves:\n"
         "z: 384→198 MiB (−48%)\n"
         "y1: 192→98 MiB (−49%)\n"
         "Total: −370 MiB activation",
         "#065F46", "#ECFDF5", "#6EE7B7", "#059669"),
        (12.0, (FX - W / 2 - 0.05, 11.2),
         "prequant cache:\n"
         "y1_fp8 + scales passed\n"
         "across autograd boundary\n"
         "via _PREQUANTIZED_SCALES",
         "#7C2D12", "#FFF7ED", "#FDBA74", "#EA580C"),
        (2.3, (FX - W / 2 - 0.05, 1.8),
         "dz BF16→FP8 handoff:\n"
         "wgrad uses dz BF16\n"
         "then resize_(0) frees 384M\n"
         "actgrad uses dz_fp8 (192M)",
         "#7C2D12", "#FFF7ED", "#FDBA74", "#EA580C"),
    ]
    for txt_y, xy, text, tc, fc, ec, ac in _annotations:
        ax.annotate(
            text, xy=xy, xytext=(5.1, txt_y),
            fontsize=6.8 if txt_y > 10 else 6.5,
            color=tc, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec=ec, lw=1),
            arrowprops=dict(arrowstyle="-|>", color=ac, lw=1.2),
        )

    fig.suptitle(
        f"SonicMoE Forward + Backward Dataflow  ·  Ernie ({SHAPE.label})",
        fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig4_dataflow.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Per-Phase Memory Breakdown
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhaseMemory:
    """Memory breakdown per phase (MiB)."""
    weights:     list[float]
    activations: list[float]
    gradients:   list[float]
    metadata:    list[float] = field(
        default_factory=lambda: [1, 1, 1, 1, 1, 1])


_BF16_MEM = PhaseMemory(
    weights     = [  0, 144, 216, 216, 144, 144],
    activations = [ 48, 624, 1008, 624, 48,  48],
    gradients   = [  0,   0,   0, 624, 432, 432],
)
_FP8_MEM = PhaseMemory(
    weights     = [  0, 216, 108, 144, 144, 144],
    activations = [ 48, 370, 862, 432,  48,  48],
    gradients   = [  0,   0,   0, 648, 240, 432],
)


def fig5_memory_peak() -> None:
    """Stacked bar chart: per-phase memory by category."""
    phases = ["Router\n& Meta", "UpProj\nFwd", "DnProj\nFwd",
              "DnProj\nBwd", "UpBwd\n(W)", "UpBwd\n(A)"]
    x = np.arange(len(phases))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, title, accent, mem in [
        (ax1, "BF16 Baseline", _P["blue"],  _BF16_MEM),
        (ax2, "FP8 Frontier",  _P["amber"], _FP8_MEM),
    ]:
        m, w, a, g = (np.array(mem.metadata), np.array(mem.weights),
                      np.array(mem.activations), np.array(mem.gradients))
        ax.bar(x, m, 0.65, label="Metadata",    color="#D1D5DB")
        ax.bar(x, w, 0.65, bottom=m,
               label="Weights",     color="#93C5FD")
        ax.bar(x, a, 0.65, bottom=m + w,
               label="Activations", color=accent, alpha=0.75)
        ax.bar(x, g, 0.65, bottom=m + w + a,
               label="Gradients",   color="#FCA5A5")

        totals = m + w + a + g
        for i, v in enumerate(totals):
            ax.text(i, v + 20, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc=_P["bg_hdr"], ec=_P["border"], lw=0.8))
        if ax is ax1:
            ax.set_ylabel("Memory (MiB)", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Per-Phase Memory Breakdown  ·  Ernie ({SHAPE.label})",
        fontsize=12, fontweight="bold", y=1.01)
    _save(fig, "fig5_memory_peak.png")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

FIGURES = [
    ("Buffer Lifecycle Gantt",      fig1_memory_lifecycle),
    ("Precision State Matrix",      fig2_precision_flow),
    ("CUTLASS Kernel Pipeline",     fig3_kernel_pipeline),
    ("Forward / Backward Dataflow", fig4_dataflow),
    ("Per-Phase Memory Breakdown",  fig5_memory_peak),
]


def generate_all(out_dir: Optional[str] = None) -> None:
    """Generate all figures."""
    global ASSETS
    if out_dir:
        ASSETS = pathlib.Path(out_dir)
    ASSETS.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("SonicMoE Dataflow Visualization")
    print("=" * 50)
    for name, func in FIGURES:
        func()
    print("=" * 50)
    print(f"All figures saved to: {ASSETS}/")


if __name__ == "__main__":
    generate_all()
