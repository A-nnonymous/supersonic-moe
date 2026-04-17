# Directory Index: `/tests/ops/`

> Focused operator and module-level tests, including the newer MoE module suite.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker for test discovery. | untracked in git |
| `conftest.py` | Shared fixtures, precision helpers, gold references, and shape constants for FP8 op tests. | untracked in git |
| `test_colwise_quant.py` | Unit tests for colwise_quantize_and_pack and colwise_quantize_cute. | untracked in git |
| `test_dequant.py` | Unit tests for dequantize_blockscaled_fp8. | untracked in git |
| `test_dual_quant.py` | Unit tests for dual_quantize_varlen (fused row+col quant). | untracked in git |
| `test_fused_zy1_quant.py` | Unit tests for fused_z_save_y1_quant. | untracked in git |
| `test_gemm_dgated.py` | Unit tests for gemm_dgated (bwd): torch ↔ BF16 3-way cross-validation. | untracked in git |
| `test_gemm_gated.py` | Unit tests for gemm_gated (fwd): torch ↔ BF16 ↔ FP8 3-way cross-validation. | untracked in git |
| `test_moe_module.py` | MoE module-level regression suite against a pure-torch reference. | — |
| `test_rowwise_quant.py` | Unit tests for quantize_and_pack_activation (row-wise blockscaled FP8 quant). | untracked in git |
| `test_swiglu.py` | Unit tests for SwiGLU forward/backward: torch ↔ BF16 ↔ FP8 3-way cross-validation. | untracked in git |
| `test_varlen_gemm.py` | Unit tests for blockscaled_fp8_gemm_varlen (down-projection): torch ↔ BF16 ↔ FP8 3-way. | untracked in git |
| `test_weight_quant.py` | Unit tests for quantize_and_pack_weight_iso32 (32x32 isotropic blockscaled). | untracked in git |
| `test_wgrad_gemm.py` | Unit tests for blockscaled_fp8_weight_grad_gemm: torch ↔ BF16 ↔ FP8 3-way. | untracked in git |
