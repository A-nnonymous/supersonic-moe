"""Package entry-point: ``python -m visualization``."""
from visualization.sonicmoe_dataflow import generate_all
generate_all()

# Scoreboard visualization (requires scoreboard.json from tools/scoreboard.py)
try:
    from visualization.scoreboard_viz import render as render_scoreboard
    print("\n  Generating: Unified Scoreboard")
    render_scoreboard()
except FileNotFoundError:
    print("\n  [skip] Scoreboard: scoreboard.json not found — run tools/scoreboard.py first")
except Exception as e:
    print(f"\n  [skip] Scoreboard: {e}")
