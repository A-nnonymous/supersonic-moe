# Directory Index: `/tests/`

> Repository-level regression, integration, and contract tests.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Stable child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `ops/` | Focused operator and module-level tests, including the newer MoE module suite. | — |
| `reference_layers/` | Reference implementations vendored for compatibility and behavior checks. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker for test discovery. | — |
| `count_cumsum_test.py` | Pytest coverage for count cumsum. | — |
| `fp8_frontier_strict_test.py` | FP8 Frontier Strict Test — no implicit fallback, no skip, fail-loud. | — |
| `fp8_large_project_contract_test.py` | Pytest coverage for fp8 large project contract. | — |
| `fp8_native_params_test.py` | Precision and functional tests for the native FP8 params path. | — |
| `fp8_operator_options.py` | Pytest coverage for fp8 operator options. | — |
| `fp8_protocol_test.py` | Pytest coverage for fp8 protocol. | — |
| `moe_blackwell_test.py` | Pytest coverage for moe blackwell. | — |
| `moe_test.py` | Pytest coverage for moe. | — |
| `run_regression.sh` | Shell helper for run regression. | — |
| `test_blockscaled_fp8_varlen.py` | Test blockscaled_fp8_gemm_varlen against bf16 gold reference. | — |
| `test_commons.py` | Pytest coverage for commons. | — |
| `test_cute_blockscaled.py` | Test for CuTe DSL colwise blockscaled quant with swizzled smem transpose. | — |
| `test_epilogue_quant_fwd_bwd.py` | Minimal forward+backward test with epilogue_quant + fp8 D output. | — |
| `test_epilogue_quant_precision.py` | Full-chain precision test for blockscaled FP8 quantization. | — |
| `test_fp8_d_output.py` | Test: can CUTLASS GemmGatedSm100ZeroMatBlockscaledQuant output fp8 D?. | — |
| `test_rcp_precision.py` | Rigorous precision test: rcp.approx E8M0 vs integer bitops E8M0. | — |
| `test_unaligned_fp8_padded.py` | Smoke test: unaligned FP8 padded forward vs BF16 reference. | — |
