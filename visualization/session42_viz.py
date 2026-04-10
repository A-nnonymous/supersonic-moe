#!/usr/bin/env python3
"""Session 42 Final Visualization: rigorous benchmark data."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PROJECT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── Load data ──
bench = json.load(open(os.path.join(PROJECT, "benchmark_final.json")))
kern = json.load(open(os.path.join(PROJECT, "reports", "nsys_final", "kernel_breakdown.json")))

def mode_stats(mode):
    entries = [r for r in bench["results"] if r["mode"] == mode]
    return {
        "fwd_peak": np.mean([r["mem"]["fwd_peak"] for r in entries]),
        "bwd_peak": np.mean([r["mem"]["bwd_peak"] for r in entries]),
        "base": np.mean([r["mem"]["base"] for r in entries]),
        "timing_ms": np.mean([r["timing"]["trimmed_ms"] for r in entries]),
    }

bf16 = mode_stats("bf16")
fp8 = mode_stats("fp8")
stash = mode_stats("fp8_stash")

# Precision
fp8_prec = [r["precision"] for r in bench["results"] if r["mode"]=="fp8" and r.get("precision")]
PREC_OUT = np.mean([p["out_rrmse"] for p in fp8_prec])
PREC_DX = np.mean([p["dx_rrmse"] for p in fp8_prec])

# nsys GPU Projection
BF16_GPU_US = kern["BF16"]["per_iter_us"]
FP8_GPU_US = kern["FP8"]["per_iter_us"]

# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Executive Summary (3 panels)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("FP8 Frontier — Rigorous Benchmark (idle B200, 3 seeds × 3 repeats)",
             fontsize=13, fontweight="bold", y=0.98)

# Panel 1: Memory
ax = axes[0]
labels = ["BF16", "FP8", "FP8+stash"]
fwd = [bf16["fwd_peak"], fp8["fwd_peak"], stash["fwd_peak"]]
bwd = [bf16["bwd_peak"], fp8["bwd_peak"], stash["bwd_peak"]]
x = np.arange(3); w = 0.32
colors = ["#4472C4", "#ED7D31", "#70AD47"]
b1 = ax.bar(x - w/2, fwd, w, label="Forward peak", color=colors, alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, bwd, w, label="Backward peak", color=colors, alpha=0.55, edgecolor="white", hatch="//")
for bar, v in zip(b1, fwd):
    ax.text(bar.get_x()+bar.get_width()/2, v+15, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
for bar, v in zip(b2, bwd):
    ax.text(bar.get_x()+bar.get_width()/2, v+15, f"{v:.0f}", ha="center", fontsize=8)
ax.set_ylabel("Peak Alloc (MiB)"); ax.set_title("Memory", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
ax.set_ylim(0, max(bwd)*1.2)
# Annotate stash savings
delta_fwd = bf16["fwd_peak"] - stash["fwd_peak"]
ax.annotate(f"−{delta_fwd:.0f} MiB\n(−{delta_fwd/bf16['fwd_peak']*100:.1f}%)",
            xy=(2-w/2, stash["fwd_peak"]), xytext=(1.2, stash["fwd_peak"]+200),
            arrowprops=dict(arrowstyle="->", color="#70AD47", lw=1.5),
            fontsize=9, color="#70AD47", fontweight="bold")

# Panel 2: Performance (nsys GPU Projection)
ax = axes[1]
modes_perf = ["BF16", "FP8"]
vals_perf = [BF16_GPU_US, FP8_GPU_US]
bars = ax.bar(modes_perf, vals_perf, color=["#4472C4", "#ED7D31"], alpha=0.85, width=0.5)
for bar, v in zip(bars, vals_perf):
    ax.text(bar.get_x()+bar.get_width()/2, v+50, f"{v:.0f} µs", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("GPU Projection (µs/iter)"); ax.set_title("Performance (nsys)", fontweight="bold")
ax.set_ylim(0, max(vals_perf)*1.25)
speedup = BF16_GPU_US / FP8_GPU_US
ax.text(0.5, 0.92, f"FP8 = {speedup:.2f}× faster", transform=ax.transAxes,
        ha="center", fontsize=11, fontweight="bold", color="#ED7D31",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ED7D31", alpha=0.8))

# Panel 3: Precision
ax = axes[2]
bars = ax.bar(["Output\nRRMSE", "dx\nRRMSE"], [PREC_OUT, PREC_DX],
              color="#ED7D31", alpha=0.85, width=0.5)
for bar, v in zip(bars, [PREC_OUT, PREC_DX]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
ax.axhline(10, color="red", ls="--", lw=1, label="Gate (10%)")
ax.set_ylabel("RRMSE (%)"); ax.set_title("Precision vs BF16", fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim(0, 14)
ax.text(0.5, 0.92, "3 seeds, subprocess-isolated\nStash = BIT-IDENTICAL to no-stash",
        transform=ax.transAxes, ha="center", fontsize=8, color="gray", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(ASSETS, "session42_executive_summary.png"), dpi=150, bbox_inches="tight")
print("Saved: session42_executive_summary.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Kernel Breakdown (BF16 vs FP8 side by side)
# ═══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Kernel-Level GPU Time Breakdown (nsys, idle B200, 10 iters)", fontsize=13, fontweight="bold")

for ax, label, cmap_base in [(ax1, "BF16", "#4472C4"), (ax2, "FP8", "#ED7D31")]:
    kd = kern[label]
    names = [k["name"][:35] for k in kd["kernels"][:8]]
    vals = [k["per_iter_us"] for k in kd["kernels"][:8]]
    other = kd["per_iter_us"] - sum(vals)
    names.append("Other"); vals.append(other)

    colors = plt.cm.Set2(np.linspace(0, 1, len(vals)))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.7)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        pct = v / kd["per_iter_us"] * 100
        ax.text(v + 10, bar.get_y()+bar.get_height()/2, f"{v:.0f} µs ({pct:.0f}%)",
                va="center", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("µs / iter"); ax.set_title(f"{label} — {kd['per_iter_us']} µs/iter", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(ASSETS, "session42_kernel_breakdown.png"), dpi=150, bbox_inches="tight")
print("Saved: session42_kernel_breakdown.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Memory Waterfall
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Forward Peak Memory Waterfall", fontweight="bold", fontsize=13)

items = [
    ("BF16\nbaseline", bf16["fwd_peak"], "#4472C4"),
    ("FP8 GEMM\nsavings", -(bf16["fwd_peak"]-fp8["fwd_peak"]) if fp8["fwd_peak"]<bf16["fwd_peak"] else (fp8["fwd_peak"]-bf16["fwd_peak"]), "#ED7D31"),
    ("Weight\nstash", -(fp8["fwd_peak"]-stash["fwd_peak"]), "#70AD47"),
    ("FP8+stash\nresult", stash["fwd_peak"], "#70AD47"),
]

running = 0
for i, (label, val, color) in enumerate(items):
    if i == 0:
        ax.bar(i, val, color=color, alpha=0.85, width=0.6); running = val
    elif i == len(items)-1:
        ax.bar(i, val, color=color, alpha=0.85, width=0.6)
    else:
        if val < 0:
            ax.bar(i, -val, bottom=running+val, color=color, alpha=0.85, width=0.6)
            running += val
        else:
            ax.bar(i, val, bottom=running, color=color, alpha=0.85, width=0.6)
            running += val
    txt = f"{val:+.0f}" if i > 0 and i < len(items)-1 else f"{val:.0f}"
    y = val if i==0 else running if i==len(items)-1 else running
    ax.text(i, max(val if i==0 else running, 0)+20, f"{txt} MiB", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items])
ax.set_ylabel("MiB"); ax.set_ylim(0, bf16["fwd_peak"]*1.15)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, "session42_memory_waterfall.png"), dpi=150, bbox_inches="tight")
print("Saved: session42_memory_waterfall.png")
plt.close()

print("\nAll figures generated.")
