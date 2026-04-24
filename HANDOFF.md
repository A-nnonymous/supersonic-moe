# HANDOFF — Session 62 (2026-04-24)

## Project Status

SonicMoE FP8 MoE kernel library (Blackwell SM100/SM103) integrated into PaddleFleet/ERNIE.

**Branch**: `session60-ds-fix` (pushed to `myrepo` = A-nnonymous/supersonic-moe)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0), Python 3.12

### What Works

| Capability | Evidence |
|---|---|
| FP8 fwd+bwd (out, dx, ds, dw1, dw2) | `test_cold_start_e2e.py`: 6 shapes x 5 tensors, all cos>0.99 |
| ds gradient -> dispatched_probs | Triton `_build_score_src_idx_kernel` + autograd fancy-index, cos=0.9972 |
| Dynamic seqlen (zero CuTe recompile) | `test_jit_optimization.py`: 4 seqlens, 0 new CuTe compiles |
| Cold start -> production | `test_cold_start_e2e.py`: cache clear -> 42s JIT -> 0.05s steady-state |
| `SonicMoEMlpNode.step()` | flush_native_grads + invalidate_weight_caches |
| Input validation | 18 operator wrappers, dtype/stride/shape checks, zero GPU sync |
| nsys GPU-projection benchmark | 2871 us/iter (3-GPU mean, CV=0.6%), +5.7% vs Session 53 baseline |

### What Doesn't Work / Not Yet Done

| Item | Detail |
|---|---|
| `warmup_jit()` standalone | CUDA topk metadata extension build fails under `torch.utils.cpp_extension.load` in Paddle proxy. Works when called from within Paddle-proxy-enabled script. |
| Multi-card (EP>1) E2E | Requires DeepEP buffer setup; single-card only |
| ERNIE training loop integration | MlpNode interface verified, not plugged into actual ERNIE training |
| Pipeline parallelism overlap | `_PREQUANTIZED_SCALES` module-level dict is unsafe under microbatch overlap |
| wgrad QuACK per-call +21% | bwd_wgrad per-call 322us vs baseline 267us; under investigation |

## Session 62 Changes

### Correctness Fixes
- **`_differentiable_router_scores` CPU-GPU sync eliminated**: Replaced boolean indexing (forced cudaStreamSynchronize) with Triton kernel (17us, zero sync)
- **3 compile_keys fixed**: Removed dynamic token dims from `"varlen"`, `"weight_grad"`, `"weight_grad_fast"`. All compile_keys now contain only static model dims.
- **Paddle/PyTorch compat**: `_inplace_version`, `stream_base.raw_stream`, `_offset()`, dtype comparison
- **warmup_jit fix**: Unpack (y1, z), run under Paddle proxy, 2-iter warmup

### Production Hardening
- **`SonicMoEMlpNode.step()` API**: `flush_native_grads() + invalidate_weight_caches()`
- **Input validation**: 18 operator wrappers via `_validate.py`
- **FAST_PATH dict bounds**: 64-entry eviction
- **Dead code cleanup**: 4 dead functions, 2 dead kernels, legacy flags
- **`ASSUME_ALIGNED=True` default**

### Cache Architecture

**Invariant**: `compile_key` never contains dynamic token dimensions.

| compile_key tag | Dynamic fields |
|---|---|
| `"vk"` | NONE |
| `"vk_accum"` | NONE |
| `"varlen"` | NONE (fixed: a_scales removed) |
| `"weight_grad"` | NONE (fixed: capacity removed) |
| `"weight_grad_fast"` | NONE (fixed) |
| `"zeromat_gated"` | NONE |

**fast_path correctness**: `scheduler_args` = device-static. `varlen_args` = recreated per-call. CuTe kernel = `mark_layout_dynamic`. Different shape -> fast_key miss -> never wrong kernel.

## Critical Constraints

1. **ds return path**: Zero native Paddle autograd nodes between gate output and `_DownProjection.apply()`. Violation -> segfault.
2. **bf16 tensor conversion**: Only `torch.from_dlpack()` preserves bf16 correctly (numpy/as_tensor broken).
3. **`_inplace_version`**: Use `_tensor_version()` helper for Paddle/PyTorch compat.

## Performance (Ernie shape, nsys GPU-projection, 3-GPU mean)

| Category | us/iter | % |
|---|:---:|:---:|
| GEMM (CUTLASS/QuACK) | 2142 | 75% |
| FP8 quantization | 482 | 17% |
| Routing + metadata | 181 | 6% |
| Paddle compat | 66 | 2% |
| **GPU-projection** | **2871** | |
