"""Package entry-point: ``python -m visualization``."""

# Legacy visualization (requires manifest.json + old data files)
try:
    from visualization.sonicmoe_dataflow import generate_all
    generate_all()
except FileNotFoundError:
    print("  [skip] Legacy dataflow: data files not found")
except Exception as e:
    print(f"  [skip] Legacy dataflow: {e}")

# Scoreboard visualization (requires scoreboard.json from tools/scoreboard.py)
try:
    from visualization.scoreboard_viz import render as render_scoreboard
    print("\n  Generating: Unified Scoreboard")
    render_scoreboard()
except FileNotFoundError:
    print("\n  [skip] Scoreboard: scoreboard.json not found — run tools/scoreboard.py first")
except Exception as e:
    print(f"\n  [skip] Scoreboard: {e}")

# Session 53 Frontier visualization (no GPU needed)
try:
    from visualization.frontier_viz import generate_frontier
    generate_frontier()
except FileNotFoundError as e:
    print(f"\n  [skip] Frontier: data file not found — {e}")
except Exception as e:
    import traceback
    print(f"\n  [skip] Frontier: {e}")
    traceback.print_exc()

# BF16 vs FP8 path-comparison visualization (requires compare-viz report)
try:
    from visualization.path_compare_viz import generate_compare_viz
    print("\n  Generating: BF16 vs FP8 Path Comparison")
    generate_compare_viz()
except FileNotFoundError:
    print("\n  [skip] Path comparison: run `python tools/introspect.py --mode compare-viz` first")
except Exception as e:
    import traceback
    print(f"\n  [skip] Path comparison: {e}")
    traceback.print_exc()
