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

## Topk kernel correctness regression (Paddle compat, script + subprocess watchdog)

| File | What it validates | Run command |
| --- | --- | --- |
| `test_mlpnode_correctness_large.py` | **Session 66 regression** for the two topk-metadata kernel bugs (Class A grid-spinwait deadlock, Class B grid-cap silent corruption). 9 cases × 5 tensors (out/dx/ds/dw1/dw2) vs BF16 gold; SEQ up to 16384 (TK=131072); skew/extreme/holes/0-token-expert. Subprocess-per-case + 600s hard timeout. | `CUDA_VISIBLE_DEVICES=7 $EBVENV/bin/python tests/ops/test_mlpnode_correctness_large.py` |
| `test_recompute_z.py` | **Session 67** focused validation of `recompute_z` mode: numeric equivalence (out/dx/ds/dw1/dw2 vs `recompute_z=False` baseline, FP8 path) + forward peak-memory drop. Subprocess-per-config. | `CUDA_VISIBLE_DEVICES=0 $EBVENV/bin/python tests/ops/test_recompute_z.py` |
| `test_recompute_z_optionB.py` | **Session 68** Layer-1 bit-exact test for the experimental Option B non-gated quant-only kernel (uniform round-robin routing only — Option B is KNOWN-BROKEN on non-uniform routing; gated by `SONIC_MOE_FP8_RECOMPUTE_OPT_B=1`). | `CUDA_VISIBLE_DEVICES=0 $EBVENV/bin/python tests/ops/test_recompute_z_optionB.py` |
| `test_mlpnode_multilayer.py` | **Session 71** multi-layer / multi-step correctness: two-layer chained, 4 PP-interleaved schedules (1F1B / FFB-FBB / interleaved / out-of-order), and 3-layer × 4-microbatch × 3-optimizer-step main_grad accumulation. Validates that the per-`MoELayer` weight-bound `_NATIVE_W*_GRAD` plumbing has no global-state collisions across layers / steps / interleavings. | `CUDA_VISIBLE_DEVICES=0 $EBVENV/bin/python -m pytest tests/ops/test_mlpnode_multilayer.py -q` |
| `test_precompute_weight_fp8_warmup.py` | **Session 71** bit-exact + speedup test for `precompute_weight_fp8_warmup` — the fused single-pass Triton pair-quantize that replaces the legacy 4-call sequence (~3.2x faster at H=3072 I=1536 E=8). | `CUDA_VISIBLE_DEVICES=0 $EBVENV/bin/python -m pytest tests/ops/test_precompute_weight_fp8_warmup.py -q` |
| `audit_iso32_numerics.py` | **Session 67** pure-PyTorch quant→dequant audit comparing iso32 (32×32) vs 1×32 blockscaled FP8 weight quant on uniform/heavy-tail/per-row-variance shapes. Confirmed bit-identical aggregate metrics. | `python tests/ops/audit_iso32_numerics.py` |
| `bench_iso32_quant_nsys.py` | **Session 67** NVTX-bracketed perf microbench for 4 weight shapes; pair with `tools/parse_nsys_per_iter.py`. | `nsys profile -o iso32_bench python tests/ops/bench_iso32_quant_nsys.py && python tools/parse_nsys_per_iter.py iso32_bench.sqlite` |

## Profiling / nsys benches

| File | Purpose |
| --- | --- |
| `bench_mlpnode_topk_nsys.py` | Gold-standard mlpnode-only nsys profile: BENCH NVTX, sqlite GPU-projection parser, `--imbalance {none,skew,extreme}`. |
| `bench_coldstart_nsys.py` | Cache-clear → JIT → ITER NVTX per-iter + FLUSH NVTX (matches production `node.step()` semantics). |
| `bench_mlpnode_mem.py` | E=32 fwd+bwd peak memory profile (ERNIE shape sweep including SEQ=16384). |
| `bench_wgrad_epilogue.py` | TMA reduce-add vs fused beta=1.0 wgrad epilogue A/B (Session 65). |
