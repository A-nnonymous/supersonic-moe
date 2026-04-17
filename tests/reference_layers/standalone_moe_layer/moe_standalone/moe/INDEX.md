# Directory Index: `/tests/reference_layers/standalone_moe_layer/moe_standalone/moe/`

> Reference MoE layer and gating components.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker for test discovery. | — |
| `deep_ep_moe_layer.py` | DeepEPMOELayer — standalone MoE layer using DeepEP communication. | — |
| `moe_statics.py` | MoEStatics — correction-bias and expert-usage tracking. | — |
| `top2_gate.py` | top2gate. | — |
