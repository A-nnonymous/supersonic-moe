#!/usr/bin/env python3
"""Session 42 visualization: Memory comparison bar chart (BF16 vs FP8 vs FP8+stash)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PROJECT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── Data: Session 41 BF16 (nsys GPU Projection, idle B200) ──
# These are the authoritative BF16 numbers from HANDOFF.md
BF16_FWD_PEAK = 1386  # MiB
BF16_BWD_PEAK = 1412  # MiB

# ── Data: Session 42 FP8 + FP8+stash (idle B200, 3-seed mean) ──
bench_path = os.path.join(PROJECT, "session42_benchmark.json")
if os.path.exists(bench_path):
    with open(bench_path) as f:
        bench = json.load(f)
    fp8_entries = [r for r in bench["results"] if r["mode"] == "fp8"]
    stash_entries = [r for r in bench["results"] if r["mode"] == "fp8_stash"]
    FP8_FWD = round(np.mean([r["memory"]["fwd_peak"] for r in fp8_entries]), 1)
    FP8_BWD = round(np.mean([r["memory"]["bwd_peak"] for r in fp8_entries]), 1)
    STASH_FWD = round(np.mean([r["memory"]["fwd_peak"] for r in stash_entries]), 1)
    STASH_BWD = round(np.mean([r["memory"]["bwd_peak"] for r in stash_entries]), 1)
    FP8_TOTAL_MS = round(np.mean([r["timing"]["total_ms"] for r in fp8_entries]), 2)
    STASH_TOTAL_MS = round(np.mean([r["timing"]["total_ms"] for r in stash_entries]), 2)
    PREC_OUT = round(np.mean([r["precision"]["out_rrmse"] for r in fp8_entries if r["precision"]]), 2)
    PREC_DX = round(np.mean([r["precision"]["dx_rrmse"] for r in fp8_entries if r["precision"]]), 2)
else:
    # Fallback to subprocess-isolated numbers from this session
    FP8_FWD, FP8_BWD = 1391, 1491
    STASH_FWD, STASH_BWD = 1112, 1240
    FP8_TOTAL_MS, STASH_TOTAL_MS = 6.25, 6.07
    PREC_OUT, PREC_DX = 6.60, 7.48

BF16_TOTAL_US = 3840  # nsys GPU Projection (µs)

# ═══════════════════════════════════════════════════════════════════════
# Figure: 3-panel executive summary
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Session 42: FP8 + Weight Stash — Executive Summary", fontsize=14, fontweight="bold", y=0.98)

# ── Panel 1: Memory ──
ax = axes[0]
labels = ["BF16", "FP8", "FP8+stash"]
fwd_vals = [BF16_FWD_PEAK, FP8_FWD, STASH_FWD]
bwd_vals = [BF16_BWD_PEAK, FP8_BWD, STASH_BWD]
x = np.arange(len(labels))
w = 0.32
c_fwd = ["#4472C4", "#ED7D31", "#70AD47"]
c_bwd = ["#4472C4", "#ED7D31", "#70AD47"]
bars_fwd = ax.bar(x - w/2, fwd_vals, w, label="Forward peak", alpha=0.85, color=c_fwd, edgecolor="white")
bars_bwd = ax.bar(x + w/2, bwd_vals, w, label="Backward peak", alpha=0.55, color=c_bwd, edgecolor="white", hatch="//")
for bar, v in zip(bars_fwd, fwd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars_bwd, bwd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Peak Alloc (MiB)")
ax.set_title("Memory", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, max(bwd_vals) * 1.25)
# Annotate savings
ax.annotate(f"-{BF16_FWD_PEAK - STASH_FWD:.0f} MiB\n(-{(BF16_FWD_PEAK-STASH_FWD)/BF16_FWD_PEAK*100:.1f}%)",
            xy=(2 - w/2, STASH_FWD), xytext=(2 - w/2 - 0.5, STASH_FWD + 300),
            arrowprops=dict(arrowstyle="->", color="#70AD47", lw=1.5),
            fontsize=9, color="#70AD47", fontweight="bold")

# ── Panel 2: Latency ──
ax = axes[1]
modes = ["BF16\n(nsys GPU)", "FP8\n(CUDA events)", "FP8+stash\n(CUDA events)"]
times = [BF16_TOTAL_US / 1000, FP8_TOTAL_MS, STASH_TOTAL_MS]
colors = ["#4472C4", "#ED7D31", "#70AD47"]
bars = ax.bar(modes, times, color=colors, alpha=0.85, edgecolor="white", width=0.6)
for bar, v in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Total fwd+bwd (ms)")
ax.set_title("Latency", fontweight="bold")
ax.set_ylim(0, max(times) * 1.3)
ax.text(0.5, 0.95, "Note: BF16 = nsys GPU Projection\nFP8 = CUDA Events (includes Python)",
        transform=ax.transAxes, ha="center", va="top", fontsize=7, color="gray", style="italic")

# ── Panel 3: Precision ──
ax = axes[2]
metrics = ["Output\nRRMSE", "dx\nRRMSE"]
values = [PREC_OUT, PREC_DX]
gate = 10.0
bars = ax.bar(metrics, values, color=["#ED7D31", "#ED7D31"], alpha=0.85, edgecolor="white", width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(gate, color="red", linestyle="--", linewidth=1, label=f"Gate ({gate}%)")
ax.set_ylabel("RRMSE (%)")
ax.set_title("Precision vs BF16", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 15)
ax.text(0.5, 0.95, "3 seeds, subprocess-isolated\nStash = BIT-IDENTICAL to no-stash",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="gray", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(ASSETS, "session42_executive_summary.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# Figure: Memory waterfall (FP8 → stash savings breakdown)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("FP8+Stash Memory Savings Waterfall (Forward Peak)", fontweight="bold", fontsize=13)

# Waterfall items
items = [
    ("BF16\nbaseline", BF16_FWD_PEAK, 0, "#4472C4"),
    ("FP8\nGEMM savings", -(BF16_FWD_PEAK - FP8_FWD) if FP8_FWD < BF16_FWD_PEAK else (FP8_FWD - BF16_FWD_PEAK), 0, "#ED7D31"),
    ("w1+w2\nstash to CPU", -(FP8_FWD - STASH_FWD), 0, "#70AD47"),
    ("FP8+stash\nresult", STASH_FWD, 0, "#70AD47"),
]

running = 0
bars_x = []
bars_h = []
bars_b = []
bars_c = []
labels_text = []
for i, (label, delta, _, color) in enumerate(items):
    if i == 0:  # absolute start
        bars_x.append(i); bars_h.append(delta); bars_b.append(0); bars_c.append(color)
        running = delta
    elif i == len(items) - 1:  # absolute end
        bars_x.append(i); bars_h.append(delta); bars_b.append(0); bars_c.append(color)
    else:
        if delta < 0:  # savings
            bars_x.append(i); bars_h.append(-delta); bars_b.append(running + delta); bars_c.append(color)
            running += delta
        else:
            bars_x.append(i); bars_h.append(delta); bars_b.append(running); bars_c.append(color)
            running += delta
    labels_text.append(label)

ax.bar(bars_x, bars_h, bottom=bars_b, color=bars_c, alpha=0.85, edgecolor="white", width=0.6)
for i, (bx, bh, bb, label) in enumerate(zip(bars_x, bars_h, bars_b, labels_text)):
    val = bb + bh if i == 0 or i == len(items) - 1 else -bh if items[i][1] < 0 else bh
    text = f"{items[i][1]:+.0f}" if i > 0 and i < len(items) - 1 else f"{items[i][1]:.0f}"
    ax.text(bx, bb + bh + 15, f"{text} MiB", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(bars_x)
ax.set_xticklabels(labels_text, fontsize=10)
ax.set_ylabel("Forward Peak Alloc (MiB)")
ax.set_ylim(0, BF16_FWD_PEAK * 1.15)
# Connect bars with lines
for i in range(len(bars_x) - 1):
    y = bars_b[i] + bars_h[i] if i == 0 else bars_b[i+1] + bars_h[i+1] if items[i][1] < 0 else bars_b[i] + bars_h[i]
    if i < len(bars_x) - 2:
        ax.plot([bars_x[i] + 0.3, bars_x[i+1] - 0.3], [bars_b[i] + bars_h[i]] * 2 if i == 0 else [running] * 2,
                color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
out_path2 = os.path.join(ASSETS, "session42_memory_waterfall.png")
fig.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.close()

print("\nDone. Run 'python -m visualization' to regenerate full suite.")
