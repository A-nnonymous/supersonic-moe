#!/usr/bin/env python3
"""
SonicMoE Unified Buffer Lifecycle × Scoreboard
================================================

Single composite figure fusing:
  • Twin Gantt timeline (BF16 left | FP8 right) — R/W/L segment colouring
  • Cumulative memory envelope overlaid on each Gantt (twin y-axis)
  • Critical-path DAG flow arrows between major buffers
  • Operator-flow summary table (bottom strip)

Usage::

    python visualization/scoreboard_viz.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SB_PATH = ROOT / "scoreboard.json"

# ─── palette ──────────────────────────────────────────────────────────
_S = {"W": "#D32F2F", "R": "#1565C0", "L": "#CFD8DC"}  # state colours
_C = {  # buffer-class colours (for y-label tinting)
    "data": "#1565C0", "weight": "#2E7D32", "grad": "#C62828",
    "quant": "#E65100", "meta": "#6A1B9A", "aux": "#546E7A",
}
_CO = {"data": 0, "weight": 1, "grad": 2, "quant": 3, "meta": 4, "aux": 5}

# ─── fonts ────────────────────────────────────────────────────────────
F = dict(sup=15, title=12.5, label=10.5, tick=9, ann=8, leg=8.5, tbl=8)


def _load() -> dict:
    if not SB_PATH.exists():
        sys.exit(f"scoreboard.json not found — run  python tools/scoreboard.py")
    return json.loads(SB_PATH.read_text())


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _buf_list(bf16: dict, fp8: dict, min_mib: float = 0.5) -> list[str]:
    pool: dict[str, tuple] = {}
    for m in (bf16, fp8):
        for bn, br in m["bufs"].items():
            if br["mib"] < min_mib:
                continue
            k = (_CO.get(br["class"], 9), -br["mib"], bn)
            if bn not in pool or br["mib"] > -pool[bn][1]:
                pool[bn] = k
    return [n for n, _ in sorted(pool.items(), key=lambda x: x[1])]


def _phase_mem(mode: dict) -> np.ndarray:
    return np.array([ph["alloc_mib"] for ph in mode["board"]])


def _critical_flows(mode: dict, min_mib: float = 48) -> list[tuple]:
    """Return [(src_buf, dst_buf, op_name, phase_idx), ...] for major flows."""
    bufs = mode["bufs"]
    flows = []
    for op in mode["ops"]:
        oname, phase = op["id"], op["phase"]
        reads = [r for r in op["R"] if r in bufs and bufs[r]["mib"] >= min_mib]
        writes = [w for w in op["W"] if w in bufs and bufs[w]["mib"] >= min_mib]
        for r in reads:
            for w in writes:
                if r != w:
                    flows.append((r, w, oname, phase))
    flows.sort(key=lambda f: -bufs.get(f[1], {}).get("mib", 0))
    seen = set()
    unique = []
    for f in flows:
        key = (f[0], f[1], f[3])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:12]


# ═════════════════════════════════════════════════════════════════════════
# Gantt + Scoreboard + Memory Envelope (unified panel)
# ═════════════════════════════════════════════════════════════════════════

def _draw_panel(ax: plt.Axes, mode: dict, buf_list: list[str],
                title: str, show_ylabel: bool, draw_arrows: bool) -> None:
    bufs = mode["bufs"]
    board = mode["board"]
    n_ph = len(board)
    n_buf = len(buf_list)

    BAR_H = 0.68
    SEG_W = 0.90

    # ── Phase background tinting by memory pressure ──
    mem = _phase_mem(mode)
    peak = mem.max()
    for j in range(n_ph):
        intensity = 0.012 + (mem[j] / peak) * 0.04 if peak > 0 else 0.012
        ax.axvspan(j, j + 1, color="#90A4AE", alpha=intensity, zorder=0)

    # ── Operator labels at phase column top ──
    for j, ph in enumerate(board):
        ax.text(j + 0.5, -1.2, ph["name"], ha="center", va="bottom",
                fontsize=F["ann"] - 0.5, color="#555", rotation=0,
                fontweight="bold", fontstyle="italic")

    # ── Gantt bars (R/W/L segments) ──
    for i, bn in enumerate(buf_list):
        y = i
        if bn not in bufs:
            continue
        br = bufs[bn]
        alive = br["alive"]
        mib = br["mib"]

        for j in range(n_ph):
            if j < alive[0] or j > alive[1]:
                continue
            state = board[j]["state"].get(bn, "L")
            color = _S.get(state, _S["L"])
            alpha = 1.0 if state in ("R", "W") else 0.28
            ax.barh(y, SEG_W, left=j + (1 - SEG_W) / 2, height=BAR_H,
                    color=color, alpha=alpha, edgecolor="white",
                    linewidth=0.4, zorder=3)
            # State letter inside active segments
            if state in ("R", "W"):
                ax.text(j + 0.5, y, state, ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=4)

        # MiB label right of bar
        ax.text(alive[1] + 1.08, y,
                f"{mib:.0f} MiB" if mib >= 1 else f"{mib:.1f} MiB",
                va="center", ha="left", fontsize=F["ann"],
                color="#424242", fontfamily="monospace", fontweight="bold")

    # ── DAG flow arrows (critical path only) ──
    if draw_arrows:
        flows = _critical_flows(mode, min_mib=48)
        buf_idx = {bn: i for i, bn in enumerate(buf_list)}
        drawn = set()
        for src, dst, op, phase in flows:
            if src not in buf_idx or dst not in buf_idx:
                continue
            yi = buf_idx[src]
            yj = buf_idx[dst]
            if abs(yi - yj) < 1:
                continue
            key = (src, dst, phase)
            if key in drawn:
                continue
            drawn.add(key)
            x_mid = phase + 0.5
            rad = 0.35 if yj > yi else -0.35
            ax.annotate("",
                        xy=(x_mid + 0.05, yj), xytext=(x_mid - 0.05, yi),
                        arrowprops=dict(
                            arrowstyle="-|>", color="#37474F",
                            connectionstyle=f"arc3,rad={rad}",
                            lw=1.2, alpha=0.35, shrinkA=4, shrinkB=4),
                        zorder=2)

    # ── Memory envelope (secondary y-axis right) ──
    ax2 = ax.twinx()
    xs = np.arange(n_ph) + 0.5
    ax2.step(xs, mem, where="mid", color="#E65100", lw=2.0, alpha=0.6, zorder=5)
    ax2.fill_between(xs, mem, alpha=0.08, color="#E65100", step="mid")
    pk_idx = int(np.argmax(mem))
    ax2.plot(xs[pk_idx], mem[pk_idx], "v", color="#E65100", ms=8, zorder=6)
    # Place peak label at far right to avoid overlapping bars
    ax2.annotate(f"peak {mem[pk_idx]:.0f} MiB",
                 xy=(xs[pk_idx], mem[pk_idx]),
                 xytext=(n_ph + 0.8, mem[pk_idx]),
                 fontsize=F["ann"], fontweight="bold", color="#BF360C",
                 arrowprops=dict(arrowstyle="-", color="#BF360C", lw=0.8, alpha=0.5),
                 va="center")
    ax2.set_ylim(0, peak * 1.35)
    ax2.set_ylabel("Σ Allocated (MiB)", fontsize=F["label"] - 1,
                   color="#E65100", alpha=0.7)
    ax2.tick_params(axis="y", labelsize=F["tick"] - 1, labelcolor="#E65100")

    # FWD / BWD divider
    ax.axvline(3, color="#C62828", lw=1.0, ls="--", alpha=0.25, zorder=1)
    ax.text(1.5, -0.5, "FORWARD", ha="center", fontsize=F["ann"],
            color="#1565C0", fontweight="bold", alpha=0.4)
    ax.text(4.5, -0.5, "BACKWARD", ha="center", fontsize=F["ann"],
            color="#C62828", fontweight="bold", alpha=0.4)

    # ── Y-axis: buffer names coloured by class ──
    ax.set_yticks(range(n_buf))
    if show_ylabel:
        ax.set_yticklabels(buf_list, fontsize=F["tick"], fontfamily="monospace")
        for i, bn in enumerate(buf_list):
            cls = bufs[bn]["class"] if bn in bufs else "aux"
            ax.get_yticklabels()[i].set_color(_C.get(cls, "#333"))
            ax.get_yticklabels()[i].set_fontweight("bold")
    else:
        ax.set_yticklabels([])

    # ── X-axis ──
    ax.set_xticks(np.arange(n_ph) + 0.5)
    ax.set_xticklabels([f"P{j}" for j in range(n_ph)],
                       fontsize=F["tick"], fontweight="bold")
    ax.set_xlim(0, n_ph + 2.2)
    ax.set_ylim(n_buf - 0.5, -1.8)

    # Phase grid
    for j in range(n_ph + 1):
        ax.axvline(j, color="#E0E0E0", lw=0.4, zorder=0)

    # Class separator lines
    prev_cls = None
    for i, bn in enumerate(buf_list):
        cls = bufs.get(bn, {}).get("class", "?")
        if prev_cls and cls != prev_cls:
            ax.axhline(i - 0.5, color="#BDBDBD", lw=0.6, ls=":", zorder=1)
        prev_cls = cls

    ax.set_facecolor("#FAFAFA")
    ax.set_title(title, fontsize=F["title"], fontweight="bold", pad=18)
    ax.tick_params(length=0)


# ═════════════════════════════════════════════════════════════════════════
# Operator-Flow Summary Table (bottom strip)
# ═════════════════════════════════════════════════════════════════════════

def _draw_table(ax: plt.Axes, bf16: dict, fp8: dict) -> None:
    """Render phase-by-phase comparison table as a matplotlib table widget."""
    ax.axis("off")
    n_ph = len(bf16["board"])

    headers = ["Phase", "Operator", "BF16 Reads", "BF16 Writes",
               "FP8 Reads", "FP8 Writes", "BF16\nMiB", "FP8\nMiB", "Δ MiB"]

    def _abbrev(names: list[str], bufs: dict, min_mib: float = 0.5) -> str:
        major = [n for n in names if n in bufs and bufs[n]["mib"] >= min_mib]
        minor = len(names) - len(major)
        s = ", ".join(major[:4])
        if len(major) > 4:
            s += f" +{len(major)-4}"
        if minor > 0:
            s += f" (+{minor})"
        return s or "—"

    rows = []
    for j in range(n_ph):
        bph = bf16["board"][j]
        fph = fp8["board"][j]
        bop = bf16["ops"][j]
        fop = fp8["ops"][j]
        b_mib = bph["alloc_mib"]
        f_mib = fph["alloc_mib"]
        delta = f_mib - b_mib
        d_str = f"{delta:+.0f}" if abs(delta) > 0.5 else "0"

        rows.append([
            f"P{j}", bph["name"],
            _abbrev(bop["R"], bf16["bufs"]),
            _abbrev(bop["W"], bf16["bufs"]),
            _abbrev(fop["R"], fp8["bufs"]),
            _abbrev(fop["W"], fp8["bufs"]),
            f"{b_mib:.0f}", f"{f_mib:.0f}", d_str,
        ])

    tbl = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#E3F2FD"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(F["tbl"] + 0.5)
    tbl.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", fontsize=F["tbl"])
        cell.set_facecolor("#BBDEFB")
        cell.set_edgecolor("#90CAF9")

    # Colour delta column
    for i in range(n_ph):
        cell = tbl[i + 1, 8]
        val = float(rows[i][8].replace("+", "")) if rows[i][8] != "0" else 0
        if val > 10:
            cell.set_facecolor("#FFCDD2")
        elif val < -10:
            cell.set_facecolor("#C8E6C9")

    ax.set_title("Operator-Flow Summary  ·  Phase-by-Phase Read/Write Sets & Memory",
                 fontsize=F["title"], fontweight="bold", pad=6)


# ═════════════════════════════════════════════════════════════════════════
# Composite Figure
# ═════════════════════════════════════════════════════════════════════════

def render() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sb = _load()
    bf16, fp8 = sb["bf16"], sb["fp8"]
    blist = _buf_list(bf16, fp8, min_mib=0.5)

    fig = plt.figure(figsize=(26, 20), dpi=150)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.5, 1.2],
        width_ratios=[1, 1],
        hspace=0.15, wspace=0.06,
        left=0.08, right=0.95, top=0.93, bottom=0.03,
    )

    # ── Row 1: Twin Gantt panels ──
    ax_b = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])

    pk_b = bf16["peak"]["alloc_mib"]
    pk_f = fp8["peak"]["alloc_mib"]
    _draw_panel(ax_b, bf16, blist,
                f"BF16 Baseline  ·  {len(bf16['bufs'])} bufs  ·  "
                f"peak {pk_b:.0f} MiB @ P{bf16['peak']['phase']}",
                show_ylabel=True, draw_arrows=True)
    _draw_panel(ax_f, fp8, blist,
                f"FP8 Frontier  ·  {len(fp8['bufs'])} bufs  ·  "
                f"peak {pk_f:.0f} MiB @ P{fp8['peak']['phase']}",
                show_ylabel=False, draw_arrows=True)

    # Shared legend
    leg = [
        mpatches.Patch(fc=_S["W"], label="W  Write (produce)"),
        mpatches.Patch(fc=_S["R"], label="R  Read (consume)"),
        mpatches.Patch(fc=_S["L"], alpha=0.28, label="Live (idle)"),
    ]
    for cls in ["data", "weight", "grad", "quant", "meta"]:
        leg.append(mpatches.Patch(fc="white", ec=_C[cls], lw=2.5,
                                  label=cls.title()))
    ax_f.legend(handles=leg, fontsize=F["leg"], loc="lower right",
                ncol=2, framealpha=0.92, handlelength=1.4,
                handletextpad=0.4, columnspacing=0.8,
                bbox_to_anchor=(1.0, 0.0))

    # ── Row 2: Operator-flow table (full width) ──
    ax_tbl = fig.add_subplot(gs[1, :])
    _draw_table(ax_tbl, bf16, fp8)

    # ── Suptitle ──
    delta = pk_f - pk_b
    fig.suptitle(
        f"SonicMoE Unified Buffer Lifecycle × Scoreboard\n"
        f"Ernie-shape (T=8192, H=3072, I=1536, E=K=8)  ·  "
        f"BF16 peak {pk_b:.0f} → FP8 peak {pk_f:.0f} MiB "
        f"({delta:+.0f} MiB, {delta/pk_b*100:+.1f}%)",
        fontsize=F["sup"], fontweight="bold", y=0.98,
    )

    out = ASSETS / "scoreboard_unified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")

    # ── Print agent-readable summary table to stdout ──
    _print_table(bf16, fp8)


def _print_table(bf16: dict, fp8: dict) -> None:
    """Print compact comparison table for agent/human consumption."""
    hdr = (f"{'Ph':<4} {'Operator':<18} {'BF16':>6} {'FP8':>6} {'Δ':>6}  "
           f"BF16 R→W  /  FP8 R→W")
    sep = "─" * 105
    print(f"\n{sep}\n{hdr}\n{sep}")
    for j, (bph, fph) in enumerate(zip(bf16["board"], fp8["board"])):
        bop, fop = bf16["ops"][j], fp8["ops"][j]
        bm, fm = bph["alloc_mib"], fph["alloc_mib"]
        d = fm - bm
        def _sig(names, bufs):
            return ",".join(n for n in names if bufs.get(n, {}).get("mib", 0) >= 0.5)
        br = _sig(bop["R"], bf16["bufs"])
        bw = _sig(bop["W"], bf16["bufs"])
        fr = _sig(fop["R"], fp8["bufs"])
        fw = _sig(fop["W"], fp8["bufs"])
        print(f"P{j:<3} {bph['name']:<18} {bm:>5.0f}  {fm:>5.0f}  "
              f"{d:>+5.0f}  {br}→{bw}  /  {fr}→{fw}")
    print(f"{sep}\n")


if __name__ == "__main__":
    render()
