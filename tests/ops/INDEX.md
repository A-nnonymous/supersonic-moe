# Directory Index: `/tests/ops/`

> Focused operator and module-level tests. Two categories:
> 1. **Native torch** (pytest): op-level precision tests, run with `python -m pytest tests/ops/ -q`.
> 2. **Paddle compat** (script): integration tests requiring `eb_venv`, run individually.

## Shared infrastructure

| File | Summary |
| --- | --- |
| `conftest.py` | Pytest fixtures, precision helpers (`rrmse`, `cosine_sim`, `assert_fp8_tolerance`), gold E8M0 references, shape constants, skip markers (`requires_blackwell`, `requires_quack`). |

## FP8 quantization tests (native torch, pytest)

| File | Op under test |
| --- | --- |
| `test_rowwise_quant.py` | `quantize_and_pack_activation` — row-wise blockscaled FP8. |
| `test_colwise_quant.py` | `colwise_quantize_and_pack` (Triton) + `colwise_quantize_cute` (CuTe DSL). |
| `test_dequant.py` | `dequantize_blockscaled_fp8`. |
| `test_dual_quant.py` | `dual_quantize_varlen` — fused row+col in one HBM read. |
| `test_fused_zy1_quant.py` | `fused_z_save_y1_quant` — fused z+y1 quantization. |
| `test_weight_quant.py` | `quantize_and_pack_weight_iso32` — 32x32 isotropic blockscaled weight quant. |

## GEMM / kernel tests (native torch, pytest)

| File | Op under test |
| --- | --- |
| `test_gemm_gated.py` | `gemm_gated` (forward up-projection): torch vs BF16 vs FP8 3-way. |
| `test_gemm_dgated.py` | `gemm_dgated` (backward): torch vs BF16 3-way + determinism. |
| `test_swiglu.py` | SwiGLU forward/backward: torch vs BF16 vs FP8 3-way (6 tests). |
| `test_varlen_gemm.py` | `blockscaled_fp8_gemm_varlen` (down-projection): subprocess-isolated 3-way. |
| `test_wgrad_gemm.py` | `blockscaled_fp8_weight_grad_gemm`: torch vs FP8 vs BF16 3-way. |

## Routing and padding correctness (native torch, pytest)

| File | What it validates |
| --- | --- |
| `test_pad_routing.py` | Axiomatic forward routing: no token dropped, no misdirection, no phantom. |
| `test_pad_gradient_integrity.py` | Axiomatic backward: dz[pad]==0, dw/dx negligible diff from unpadded. |

## MoE module integration (Paddle compat, script)

| File | What it tests | Run command |
| --- | --- | --- |
| `test_moe_module.py` | Full MoE pipeline (permute→gate-up→SwiGLU→down→unpermute) BF16+FP8 vs f32 gold. | `python -m pytest tests/ops/test_moe_module.py` |
| `test_moe_general_routing_fp8.py` | `moe_general_routing_inputs` FP8 fwd+bwd + main_grad accumulation + benchmark. | `$EBVENV/bin/python tests/ops/test_moe_general_routing_fp8.py` |
| `test_sonic_moe_func.py` | `SonicMoEFunc` PyLayer (ERNIE-compat): fwd+bwd, per-expert main_grad, multi-iter. | `$EBVENV/bin/python tests/ops/test_sonic_moe_func.py` |

## Diagnostics (Paddle compat, script)

| File | Purpose |
| --- | --- |
| `test_argsort_sync.py` | Reproducer for Paddle argsort 1D `cudaStreamSynchronize` stall. nsys-profilable. |
