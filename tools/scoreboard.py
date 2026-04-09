#!/usr/bin/env python3
"""SonicMoE Buffer Scoreboard — Operator-buffer dependency & phase-state analysis.

Reads ``manifest.json`` (from ``tools/introspect.py``), builds a causal
dependency DAG between GPU buffers, generates phase-by-phase scoreboard
snapshots, and outputs a compact ``scoreboard.json`` for downstream agents
(e.g. memory-optimization, scheduling).

Format semantics
----------------
  ops       Operator definitions with read (R) / write (W) buffer sets.
  bufs      Buffer inventory: dtype, shape, MiB, producer op, consumer ops,
            alive phase range, buffer class (data/weight/meta/quant/aux).
  dag       Causal edges: [source_buf, operator, dest_buf] — dest depends on
            source because the operator reads source and writes dest.
  board     Phase-by-phase scoreboard: per-buffer state (R/W/L/·), total
            allocated MiB, list of buffers freeable after this phase.
  peak      Phase with maximum allocated memory + top buffer contributors.
  hints     Actionable optimization opportunities (recompute, early-free,
            dtype downcast, storage release).

Usage
-----
    python tools/scoreboard.py                        # default manifest.json
    python tools/scoreboard.py --manifest path.json   # custom manifest
    python tools/scoreboard.py --viz                  # also generate figure

Output: ``scoreboard.json`` at repo root.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.json"
SCOREBOARD_PATH = ROOT / "scoreboard.json"

# ═══════════════════════════════════════════════════════════════════════
# Phase / Operator definitions
# ═══════════════════════════════════════════════════════════════════════

PHASE_NAMES = [
    "Router & Meta",      # 0
    "UpProj Fwd",         # 1
    "DnProj Fwd",         # 2
    "DnProj Bwd",         # 3
    "UpBwd (wgrad)",      # 4
    "UpBwd (actgrad)",    # 5
]
N_PHASES = 6

# Short aliases for routing metadata buffers (always tiny, < 1 MiB each)
_META_BUFS = [
    "expert_freq_offset", "x_gather_idx",
    "s_scatter_idx", "s_reverse_scatter_idx",
    "num_activated_expert_offset",
]

# ── BF16 operator model ────────────────────────────────────────────────
# Derived from sonicmoe/functional/__init__.py:
#   _UpProjection.forward  (line 691) / backward (line 867)
#   _DownProjection.forward (line 1015) / backward (line 1213)
#   TC_Softmax_Topk_Router_Function (line 658)

_BF16_OPS = [
    {"id": "router",   "phase": 0,
     "R": ["x", "router_w"],
     "W": ["topk_scores", "topk_indices",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"]},
    {"id": "up_fwd",   "phase": 1,
     "R": ["x", "w1",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"],
     "W": ["y1", "z"]},
    {"id": "dn_fwd",   "phase": 2,
     "R": ["y1", "z", "w2", "topk_scores",
            "expert_freq_offset", "x_gather_idx",
            "s_scatter_idx", "s_reverse_scatter_idx"],
     "W": ["output"]},
    {"id": "dn_bwd",   "phase": 3,
     "R": ["dout", "z", "w2", "topk_scores",
            "expert_freq_offset", "x_gather_idx",
            "s_scatter_idx", "s_reverse_scatter_idx"],
     "W": ["dz", "dw2"]},
    {"id": "up_bwd_w", "phase": 4,
     "R": ["dz", "x", "w1",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"],
     "W": ["dw1"]},
    {"id": "up_bwd_a", "phase": 5,
     "R": ["dz", "w1",
            "x_gather_idx", "s_reverse_scatter_idx",
            "num_activated_expert_offset"],
     "W": ["dx"]},
]

# ── FP8 operator model ─────────────────────────────────────────────────
# Key differences from BF16:
#   - up_fwd produces FP8 quantized tensors (fwd_fp8, z_fp8)
#   - dn_fwd reads fwd_fp8 (y1 in FP8) via _PREQUANTIZED_SCALES
#   - dn_bwd produces bwd_fp8 (dout quantized), prequant_bwd_0 (dz bf16 ref)
#   - up_bwd_w reads prequant_bwd_0 instead of dz; frees storage after wgrad
#   - up_bwd_a reads bwd_fp8 (dz in FP8) via _PREQUANTIZED_SCALES

_FP8_OPS = [
    {"id": "router",   "phase": 0,
     "R": ["x", "router_w"],
     "W": ["topk_scores", "topk_indices",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"]},
    {"id": "up_fwd",   "phase": 1,
     "R": ["x", "w1",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"],
     "W": ["y1", "z", "fwd_fp8", "prequant_fwd_0", "z_fp8"]},
    {"id": "dn_fwd",   "phase": 2,
     "R": ["y1", "fwd_fp8", "z_fp8", "w2", "topk_scores",
            "expert_freq_offset", "x_gather_idx",
            "s_scatter_idx", "s_reverse_scatter_idx"],
     "W": ["output"]},
    {"id": "dn_bwd",   "phase": 3,
     "R": ["dout", "z_fp8", "w2", "topk_scores",
            "expert_freq_offset", "x_gather_idx",
            "s_scatter_idx", "s_reverse_scatter_idx"],
     "W": ["dw2", "bwd_fp8", "prequant_bwd_0"]},
    {"id": "up_bwd_w", "phase": 4,
     "R": ["prequant_bwd_0", "x", "w1",
            "expert_freq_offset", "x_gather_idx",
            "s_reverse_scatter_idx", "num_activated_expert_offset"],
     "W": ["dw1"]},
    {"id": "up_bwd_a", "phase": 5,
     "R": ["bwd_fp8", "w1",
            "x_gather_idx", "s_reverse_scatter_idx",
            "num_activated_expert_offset"],
     "W": ["dx"]},
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

_DT_SHORT = {
    "torch.bfloat16": "bf16",
    "torch.float8_e4m3fn": "fp8_e4m3",
    "torch.float32": "fp32",
    "torch.int32": "int32",
    "torch.uint8": "uint8",
}


def _short_dtype(dt: str) -> str:
    return _DT_SHORT.get(dt, dt)


def _classify_buf(name: str, role: str, dtype: str) -> str:
    """Classify buffer into: data | weight | grad | meta | quant | aux."""
    if role == "weight":
        return "weight"
    if role == "index" or name in _META_BUFS:
        return "meta"
    if name.startswith("aux_") or name == "softmax_probs":
        return "aux"
    if role == "scale" or "scale" in name or dtype in ("torch.uint8",):
        return "quant"
    if name.startswith("d") and name not in ("dout",):
        # dz, dw1, dw2, dx are gradients; dout is external gradient (classify as data)
        if name in ("dz", "dw1", "dw2", "dx"):
            return "grad"
    if "fp8" in name or "prequant" in name or "bwd_fp8" in name:
        return "quant"
    return "data"


# ═══════════════════════════════════════════════════════════════════════
# Scoreboard Builder
# ═══════════════════════════════════════════════════════════════════════

def _build_mode_scoreboard(
    mode: str,
    tensors: list[dict],
    shape: dict,
) -> dict:
    """Build complete scoreboard for one mode (bf16 or fp8)."""

    ops_model = _FP8_OPS if mode == "fp8" else _BF16_OPS

    # ── 1. Build buffer inventory from manifest ──────────────────────
    # Index manifest tensors by name
    manifest_bufs: dict[str, dict] = {}
    for t in tensors:
        manifest_bufs[t["name"]] = t

    # Build op → reads/writes index for fast lookup
    op_reads: dict[str, set[str]] = {}
    op_writes: dict[str, set[str]] = {}
    for op in ops_model:
        op_reads[op["id"]] = set(op["R"])
        op_writes[op["id"]] = set(op["W"])

    # Collect all buffer names referenced by any operator
    all_op_bufs = set()
    for op in ops_model:
        all_op_bufs.update(op["R"])
        all_op_bufs.update(op["W"])

    # Build buffer records
    bufs: dict[str, dict] = {}
    for bname in sorted(all_op_bufs):
        mt = manifest_bufs.get(bname)
        if mt:
            dtype = _short_dtype(mt["dtype"])
            shp = mt["shape"]
            mib = round(mt["size_mib"], 2)
            role = mt.get("role", "activation")
        else:
            # Buffer is in static model but not in manifest (e.g., prequant
            # tensors whose lifecycle wasn't fully captured by tensor spy)
            dtype = "bf16"
            shp = []
            mib = 0.0
            role = "activation"

        buf_class = _classify_buf(bname, role, mt["dtype"] if mt else "")

        # Producer: the operator that writes this buffer
        producer = None
        for op in ops_model:
            if bname in op_writes.get(op["id"], set()):
                producer = op["id"]
                break

        # Consumers: all operators that read this buffer
        consumers = []
        for op in ops_model:
            if bname in op_reads.get(op["id"], set()):
                consumers.append(op["id"])

        # Alive range: from static model (more accurate than manifest for
        # prequant-cached tensors whose lifecycle is incompletely tracked)
        phases_active = set()
        for op in ops_model:
            if bname in op_reads[op["id"]] or bname in op_writes[op["id"]]:
                phases_active.add(op["phase"])
        # Also include manifest lifecycle if available
        if mt:
            for p in range(mt["create_phase"], mt["free_phase"] + 1):
                phases_active.add(p)

        alive = [min(phases_active), max(phases_active)] if phases_active else [0, 0]

        # Last consumer
        last_consumer = None
        last_consumer_phase = -1
        for op in ops_model:
            if bname in op_reads.get(op["id"], set()):
                if op["phase"] > last_consumer_phase:
                    last_consumer_phase = op["phase"]
                    last_consumer = op["id"]

        buf_rec = {
            "dtype": dtype,
            "shape": shp,
            "mib": mib,
            "class": buf_class,
            "producer": producer,
            "consumers": consumers,
            "alive": alive,
        }
        if last_consumer:
            buf_rec["last_consumer"] = last_consumer

        bufs[bname] = buf_rec

    # ── 2. Build causal DAG ──────────────────────────────────────────
    # Edge: [src_buf, operator, dst_buf] means dst_buf depends on src_buf
    # because the operator reads src and writes dst.
    dag: list[list[str]] = []
    for op in ops_model:
        for r_buf in op["R"]:
            for w_buf in op["W"]:
                # Only include edges where both buffers are in our inventory
                if r_buf in bufs and w_buf in bufs:
                    dag.append([r_buf, op["id"], w_buf])

    # ── 3. Build phase-by-phase scoreboard ───────────────────────────
    board: list[dict] = []
    for ph in range(N_PHASES):
        op_at_phase = None
        for op in ops_model:
            if op["phase"] == ph:
                op_at_phase = op
                break

        # Buffer states at this phase
        state: dict[str, str] = {}
        for bname, brec in bufs.items():
            a_start, a_end = brec["alive"]
            if not (a_start <= ph <= a_end):
                continue  # not alive → absent (omitted from state)
            if op_at_phase and bname in op_writes.get(op_at_phase["id"], set()):
                state[bname] = "W"
            elif op_at_phase and bname in op_reads.get(op_at_phase["id"], set()):
                state[bname] = "R"
            else:
                state[bname] = "L"  # live but idle

        # Total allocated MiB at this phase
        alloc_mib = sum(bufs[b]["mib"] for b in state)

        # Freeable: buffers whose last consumer is at or before this phase
        freeable = []
        for bname in state:
            brec = bufs[bname]
            lc = brec.get("last_consumer")
            if lc:
                lc_phase = None
                for op in ops_model:
                    if op["id"] == lc:
                        lc_phase = op["phase"]
                        break
                if lc_phase is not None and lc_phase <= ph and brec["alive"][1] > ph:
                    freeable.append(bname)
            elif brec["producer"] is not None and brec["alive"][1] == ph and not brec["consumers"]:
                freeable.append(bname)

        board.append({
            "phase": ph,
            "name": PHASE_NAMES[ph],
            "op": op_at_phase["id"] if op_at_phase else None,
            "state": state,
            "alloc_mib": round(alloc_mib, 2),
            "freeable": freeable,
        })

    # ── 4. Peak memory analysis ──────────────────────────────────────
    peak_ph = max(range(N_PHASES), key=lambda p: board[p]["alloc_mib"])
    peak_mib = board[peak_ph]["alloc_mib"]
    peak_bufs = sorted(
        [(b, bufs[b]["mib"]) for b in board[peak_ph]["state"]],
        key=lambda x: -x[1],
    )
    # Only top contributors (>= 1 MiB or top 8)
    peak_top = [
        [b, round(m, 1)] for b, m in peak_bufs
        if m >= 1.0
    ][:8]

    peak = {
        "phase": peak_ph,
        "name": PHASE_NAMES[peak_ph],
        "alloc_mib": round(peak_mib, 1),
        "top": peak_top,
    }

    # ── 5. Optimization hints ────────────────────────────────────────
    hints: list[dict] = []

    # Recompute candidates: large data buffers alive > 2 phases, with a producer
    for bname, brec in bufs.items():
        if brec["class"] != "data" or brec["producer"] is None:
            continue
        span = brec["alive"][1] - brec["alive"][0]
        if span >= 2 and brec["mib"] >= 48:
            deps = [e[0] for e in dag if e[2] == bname]
            dep_str = ", ".join(sorted(set(deps))[:4]) if deps else "?"
            hints.append({
                "kind": "recompute",
                "buf": bname,
                "save_mib": round(brec["mib"], 1),
                "span": span,
                "note": f"recomputable from {{{dep_str}}}",
            })

    # Early-free candidates: buffers alive past their last consumer
    for bname, brec in bufs.items():
        lc = brec.get("last_consumer")
        if not lc:
            continue
        lc_phase = None
        for op in ops_model:
            if op["id"] == lc:
                lc_phase = op["phase"]
                break
        if lc_phase is not None and brec["alive"][1] > lc_phase:
            waste = brec["alive"][1] - lc_phase
            if waste > 0 and brec["mib"] >= 1.0:
                hints.append({
                    "kind": "early_free",
                    "buf": bname,
                    "current_free_phase": brec["alive"][1],
                    "optimal_free_phase": lc_phase,
                    "save_phases": waste,
                    "mib": round(brec["mib"], 1),
                    "note": f"last consumer is {lc} at phase {lc_phase}",
                })

    # FP8-specific: storage release (dz freed after wgrad)
    if mode == "fp8":
        if "prequant_bwd_0" in bufs:
            hints.append({
                "kind": "storage_release",
                "buf": "prequant_bwd_0",
                "at_phase": 4,
                "save_mib": round(bufs["prequant_bwd_0"]["mib"], 1),
                "note": "dz.untyped_storage().resize_(0) after wgrad; "
                        "actgrad uses bwd_fp8 (FP8 dz) instead",
            })

    # Dtype optimization candidates
    for bname, brec in bufs.items():
        if brec["dtype"] == "bf16" and brec["class"] == "data" and brec["mib"] >= 48:
            fp8_save = round(brec["mib"] / 2, 1)
            hints.append({
                "kind": "dtype_opt",
                "buf": bname,
                "from": "bf16",
                "to": "fp8_e4m3",
                "save_mib": fp8_save,
                "note": f"50% savings if quantized to FP8 ({brec['mib']:.0f} → {fp8_save:.0f} MiB)",
            })

    # Sort hints by save_mib descending
    hints.sort(key=lambda h: -h.get("save_mib", h.get("mib", 0)))

    return {
        "ops": ops_model,
        "bufs": bufs,
        "dag": dag,
        "board": board,
        "peak": peak,
        "hints": hints,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════════

def build_scoreboard(manifest: dict) -> dict:
    """Build complete scoreboard from manifest data."""
    meta = manifest.get("metadata", {})
    shape = meta.get("shape", {"T": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8})

    result = {
        "$schema": "sonicmoe-scoreboard-v1",
        "meta": {
            "shape": shape,
            "device": meta.get("device", "unknown"),
            "torch_version": meta.get("torch_version", "unknown"),
            "quack_version": meta.get("quack_version", "unknown"),
            "note": (
                "Scoreboard format: ops define operator R/W sets; bufs map "
                "each buffer to its producer/consumers/lifecycle; dag encodes "
                "causal edges [src,op,dst]; board gives per-phase buffer states "
                "(R=read, W=write, L=live/idle); peak shows memory bottleneck; "
                "hints list optimization opportunities sorted by impact."
            ),
        },
    }

    for mode in ("bf16", "fp8"):
        mode_data = manifest.get("modes", {}).get(mode)
        if not mode_data:
            continue
        tensors = mode_data.get("tensors", [])
        result[mode] = _build_mode_scoreboard(mode, tensors, shape)

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SonicMoE Buffer Scoreboard Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Reads manifest.json, builds operator-buffer dependency graph
            and phase-by-phase scoreboard, outputs scoreboard.json.

            Example:
              python tools/scoreboard.py
              python tools/scoreboard.py --viz
        """),
    )
    parser.add_argument(
        "--manifest", type=str, default=str(MANIFEST_PATH),
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--output", type=str, default=str(SCOREBOARD_PATH),
        help="Output path for scoreboard.json",
    )
    parser.add_argument(
        "--viz", action="store_true",
        help="Also generate scoreboard visualization",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run 'python tools/introspect.py' first.")
        return 1

    manifest = json.loads(manifest_path.read_text())
    scoreboard = build_scoreboard(manifest)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(scoreboard, indent=2, ensure_ascii=False))
    size_kb = out_path.stat().st_size / 1024
    print(f"Scoreboard written to {out_path} ({size_kb:.1f} KB)")

    for mode in ("bf16", "fp8"):
        if mode in scoreboard:
            sb = scoreboard[mode]
            n_bufs = len(sb["bufs"])
            n_edges = len(sb["dag"])
            n_hints = len(sb["hints"])
            pk = sb["peak"]
            print(f"  [{mode}] {n_bufs} buffers, {n_edges} DAG edges, "
                  f"peak={pk['alloc_mib']:.0f} MiB @ phase {pk['phase']} ({pk['name']}), "
                  f"{n_hints} hints")

    if args.viz:
        try:
            from visualization.scoreboard_viz import generate_scoreboard_figures
            generate_scoreboard_figures(scoreboard)
        except ImportError as e:
            print(f"WARNING: Could not import visualization: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
