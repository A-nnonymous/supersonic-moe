# HANDOFF — Session 60→61 (2026-04-22)

## Project Status: What Works, What Doesn't

SonicMoE is an FP8 MoE kernel library (Blackwell SM100/SM103). The `paddle_compat` branch integrates it into PaddleFleet/ERNIE via Paddle's torch-proxy.

**Branch**: `session60-ds-fix` (on `myrepo`, based on `fork/paddle`)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0)
**Python**: 3.12 (system), QuACK at `/root/.../zhangyichen/sonicmoe_for_ernie/quack`

### What Works (Verified)

| Capability | Status | Evidence |
|---|---|---|
| FP8 forward (UpProj + DownProj) | **PASS** | `test_moe_layer.py` Path A & B |
| FP8 backward (dx) | **PASS** | x.grad.norm = 0.022 |
| FP8 backward (ds → gate) | **PASS** (after fix) | gate_w.grad.norm = 0.056 (was 0.0) |
| FP8 backward (dw1, dw2 via main_grad) | **PASS** | 16/16 expert weights non-zero |
| Gate without no_grad() | **PASS** (after rewire) | No segfault, ds flows back |
| Path A vs B precision | **PASS** | output RRMSE=0.0004%, cosine>0.999 |
| SonicMoEMlpNode E2E (production path) | **PASS** | Path B in test_moe_layer.py |

### What Doesn't Work / Not Yet Done

| Item | Status | Detail |
|---|---|---|
| nsys GPU-projection profiling | **BLOCKED** | nsys 2025.1.1 doesn't capture CUDA kernels on this B30Z setup. nsys 2026.2.1 binary exists at `/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys` but hasn't been validated. Need `--trace=cuda` and the 2026 binary. |
| Memory waterfall breakdown | **PARTIAL** | Rough numbers only: pre=1478 MiB, peak=3833 MiB (T=8192, ernie shape). No per-phase breakdown yet. |
| Multi-card (EP>1) E2E test | **NOT DONE** | Requires DeepEP buffer setup; single-card only. |
| ERNIE training loop integration | **NOT DONE** | MlpNode interface verified, but not plugged into actual ERNIE training. |

## Three Bugs Fixed This Session

### Bug 1 (P0): ds silently dropped — router could never train

**File**: `sonicmoe/functional/__init__.py:1540`
**Root cause**: `_DownProjection.forward` checked `topk_scores.stop_gradient` to decide whether to compute ds. But Paddle torch-proxy's `.apply()` resets `stop_gradient=True` on all tensor inputs inside forward (mimicking PyTorch `Function.apply()` detach behavior), and doesn't provide `ctx.needs_input_grad`.
**Effect**: `ctx._topk_scores_needs_grad = False` → ds=None → gate_w.grad=0 → router frozen.
**Fix**: `ctx._topk_scores_needs_grad = True` (always compute ds).
**Verification**: gate_w.grad.norm went from 0.0 to 0.056.

### Bug 2 (P0): segfault when gate runs without no_grad()

**File**: `tests/test_moe_layer.py` (architecture issue)
**Root cause**: The old test used `_convert_routing_map_and_probs` (which calls `paddle.topk`) and `_prepare_sonic_inputs` (which calls `.cast()`) between gate output and `_DownProjection.apply()`. These native Paddle ops create autograd nodes (TopkGradNode, CastGradNode). When the torch-proxy backward returns ds, it flows through these native Paddle nodes, which try to access the gradient tensor's metadata via `paddle::Tensor::type()` — but torch-proxy gradient tensors have incompatible metadata → segfault.
**Fix**: Rewired to match PaddleFleet's pattern — pass gate's `topk_weights`/`topk_indices` directly to `_UpProjection.apply()` + `_DownProjection.apply()` with `is_varlen_K=False`. Routing metadata via `general_routing_router_metadata` (integer-only, no autograd nodes).
**Rule**: Between gate output and `_DownProjection.apply()`, NO native Paddle ops that create autograd nodes on the score tensor path.

### Bug 3 (P1): ds extraction used fragile float32 heuristic

**File**: `sonicmoe/ernie_compat/mlp_node_v2.py` (two locations)
**Old code**: `for g in down_grads[3:]: if g is not None and g.dtype == float32: ds = g; break`
**Fix**: Deterministic `ds_idx = 4 if ctx._has_b2 else 3; ds = down_grads[ds_idx]`

## Critical Architecture Constraint (torch-proxy)

**Golden rule**: On the ds gradient return path (from `_DownProjection.backward` back to gate), there must be **zero** native Paddle autograd nodes. The gate's own TopkGradNode (created during gate's forward) is fine — it operates on normal Paddle tensors. But any Paddle op (topk, cast, boolean indexing, etc.) inserted *between* gate output and `_DownProjection.apply()` will create autograd nodes that receive torch-proxy gradient tensors → segfault.

This is why PaddleFleet passes `topk_scores` [T,K] directly to `_DownProjection.apply()` — no intermediate ops.

## Precision (Path A vs Path B, same weights, T=4096)

| Metric | RRMSE (%) | Cosine |
|---|---|---|
| output | 0.0004 | 1.000000 |
| dx | 0.0027 | 1.000000 |
| dw1 | 0.1656 | 0.999896 |
| dw2 | 0.1656 | 0.999928 |

dw1/dw2 ~0.17% RRMSE comes from `is_varlen_K=False` (Path A) vs `is_varlen_K=True` + route-level padding (Path B). Different wgrad GEMM tile paths in BF16.

## Performance (NOT yet validated by nsys GPU-projection)

The Session 59 numbers in the old HANDOFF (2739 µs frontier vs 3190 µs MlpNode) were measured with a working nsys setup. Current environment's nsys 2025.1.1 can't capture CUDA kernels on B30Z. Use nsys 2026.2.1 at `/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys`.

CUDA-event rough timing (not gold standard, includes CPU overhead):
- Forward: ~2.7 ms (T=8192, H=3072, I=1536, E=8, K=8)
- Backward: ~3.4 ms
- Total: ~6.1 ms/iter

## Memory (T=8192, ernie shape, rough)

- Pre-forward: 1478 MiB
- Peak (fwd+bwd): 3833 MiB
- Δ transient: ~2355 MiB (dominated by y1/z activations + wgrad intermediates)

## Key Information Sources

| What | Where |
|---|---|
| PaddleFleet MoE single-card sonic path | `PaddleFleet/src/paddlefleet/transformer/moe/moe_layer.py:887-950` |
| PaddleFleet _UpProj/_DownProj signatures | `PaddleFleet/src/paddlefleet/ops/sonicmoe/functional/__init__.py` (NOTE: different from local — no `selected_experts` param) |
| Local _DownProjection signature | `sonicmoe/functional/__init__.py:1387` (has extra `selected_experts` + `fp8_protocol` params vs PaddleFleet) |
| Frontier nsys methodology | `tools/introspect.py:1426` (`_NSYS_WORKLOAD_TEMPLATE`) — uses tempfile + subprocess, nsys 2026 |
| Route-level padding design | `sonicmoe/functional/__init__.py:222-296` (`_pad_routing_metadata`) |
| QuACK GEMM path | `/root/.../zhangyichen/sonicmoe_for_ernie/quack/quack/gemm_default_epi.py` |
| Env setup | `.runenv.sh` (PYTHONPATH for quack + sonic-moe + ernie-core) |

## Critical Files

| File | Role | Modified this session? |
|---|---|---|
| `sonicmoe/functional/__init__.py` | `_UpProjection`, `_DownProjection` — **ds fix here** | **YES** |
| `sonicmoe/ernie_compat/mlp_node_v2.py` | `SonicMoEMlpNode` + `_SonicMoEDeepEPFunc` — **ds index fix here** | **YES** |
| `tests/test_moe_layer.py` | Path A (direct .apply) + Path B (MlpNode E2E) | **YES (rewired)** |
| `sonicmoe/ernie_compat/deepep_metadata.py` | DeepEP → SonicMoE metadata (topk CUDA kernel) | No |
| `tests/ops/bench_deepep_topk_nsys.py` | nsys benchmark (needs nsys 2026.2.1) | **YES** |
| `tests/precision_compare_paths.py` | Path A vs B precision comparison | **NEW** |
| `docs/session60_lessons.md` | Engineering lessons #67-#72 | **NEW** |

## Running Tests

```bash
source .runenv.sh

# Path A (gate→MLP, ds verified) + Path B (MlpNode E2E, main_grad verified)
CUDA_VISIBLE_DEVICES=0 python tests/test_moe_layer.py

# Precision comparison (same weights, Path A vs B)
CUDA_VISIBLE_DEVICES=0 python tests/precision_compare_paths.py

# nsys profiling (MUST use nsys 2026.2.1 on B30Z)
/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx -o /tmp/mlpnode_e2e --force-overwrite=true \
  python tests/ops/mlpnode_nsys_worker.py
```

## Next Steps (Recommended)

### P0: nsys GPU-projection with nsys 2026.2.1
The benchmark script (`bench_deepep_topk_nsys.py`) and worker (`mlpnode_nsys_worker.py`) are ready. Just need to validate that nsys 2026.2.1 captures CUDA kernel data on B30Z. Update the `nsys_bin` path in the bench script.

### P1: Memory waterfall breakdown
Add per-phase memory tracking (post-metadata, post-UpProj, post-DownProj, post-backward) to the worker script. Use `paddle.device.cuda.memory_allocated()` at each checkpoint (outside the timed region).

### P2: Multi-card E2E with DeepEP
Test with actual `deep_ep.Buffer.dispatch()` / `.combine()` instead of simulated single-card routing. This is the real production path.

### P3: ERNIE training loop plug-in
`SonicMoEMlpNode` interface matches ERNIE's MlpNode. Need to verify end-to-end in the ERNIE training loop (optimizer step, lr schedule, gradient accumulation).
