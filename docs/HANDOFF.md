# SonicMoE Paddle Compat + Route-Level Padding — Handoff

> **Branch:** `paddle_compat`
> **Date:** 2026-04-16 (Session 56)
> **Upstream:** `native-fp8-exploration` + PR #1 "adapt paddle" merged
> **Status:** functionally complete, 27-shape grid verified under Paddle compat

---

## 1. Project State

### What Works

FP8 blockscaled (E4M3 + E8M0, 1x32) training is **fully functional** under both
PyTorch native and Paddle `enable_compat()` on Blackwell SM100a.

| Capability | Status |
|:-----------|:-------|
| FP8 fwd+bwd (E=8) | PASS, 1.31-1.70x speedup |
| FP8 fwd+bwd (E=32/128, route-level padding) | PASS, 1.27-1.79x speedup |
| Paddle compat (all 27 shapes BF16+FP8) | 54/54 PASS |
| Real training loop (5 iters, cpu_optimizer_step) | PASS, all grads nonzero |
| Axiomatic correctness (every token preserved) | PASS, max_diff=8.73e-11 |

### What Changed in This Session

1. **Route-level padding** — replaces token rounding for FP8 128-alignment.
   `_pad_routing_metadata()` pads routing tensors once; all 8 GEMMs run aligned fast path.
   Zero GEMM code changes. Precision delta = exact zero vs aligned path.

2. **Paddle compat** — merged PR #1 (34 files, pure framework shim). Fixed 5 additional
   compat gaps in test harness (torch.norm, torch.dot, Generator, torch.equal, max API).

3. **4-way pad-audit** — BF16 raw / BF16 round / FP8 pad / FP8 round precision comparison.
   Rounding RRMSE ~60% (routing perturbation), padding RRMSE ~6.5% (pure FP8 quant).

### What Does NOT Work / Known Issues

1. **Multi-iter backward without optimizer step** — `dz.untyped_storage().resize_(0)` in
   DownProj backward frees dz storage. Second backward's UpProj gets freed tensor from
   autograd. Works in real training (optimizer step rebuilds FP8 caches), but test harnesses
   that call fwd+bwd multiple times without optimizer step get zero dw1/dx. **Pre-existing**
   bug, not introduced by padding.

2. **Memory measurement inconsistency** — Paddle compat benchmark uses `moe()` full module
   (router autograd saves extra tensors), baseline uses lower-level API. Memory deltas in
   the grid report reflect this API-level difference, not Paddle overhead.

3. **E>8 FP8 baseline comparison** — grid_session53 E>8 FP8 used token rounding, but
   paddle_compat uses route-level padding. Performance deltas conflate routing strategy
   change with Paddle overhead. Only E=8 FP8 is strictly apples-to-apples.

---

## 2. Performance Data

### PyTorch Native (grid_session53, nsys GPU-projection)

Speedup range: **1.29x - 1.70x**, mean 1.53x. All 27 shapes PASS.
BF16 baseline = official SonicMoE, verified within <1%.

### Paddle Compat (27-shape grid, nsys GPU-projection)

| Metric | BF16 | FP8 |
|:-------|-----:|----:|
| Overhead mean | +2.9% | -0.5% (noise) |
| Overhead median | +3.3% | -0.5% |
| Overhead range | [-2.7%, +8.8%] | [-3.0%, +1.6%] |

BF16 overhead from `paddle.enable_compat()` proxy dispatch. FP8 CUTLASS kernels
bypass the proxy — effectively zero overhead.

### Route-Level Padding vs Token Rounding (E=32)

| Metric | FP8+padding | FP8+rounding |
|:-------|:-----------:|:------------:|
| RRMSE vs BF16 raw | **6.5%** | 60.8% |
| nsys us/iter | 2950 | 2915 |
| Memory (bwd MiB) | 2914 | 2909 |

Padding preserves BF16-identical routing; rounding perturbs routing, destroying 60% accuracy.

---

## 3. Key Architecture Decisions

### Route-Level Padding Design

```
Router -> metadata -> [_pad_routing_metadata] -> padded metadata (128-aligned)
                                                      |
                          _UpProjection.apply (aligned fast path, zero GEMM changes)
                                                      |
                          _DownProjection.apply (aligned fast path, zero GEMM changes)
```

- Padding rows: `x_gather_idx=0` (row 0, data irrelevant), `score=0` (nullifies contribution)
- `s_reverse_scatter_idx` stays (T*K,) — padding rows have no reverse mapping
- `topk_scores` extended with zeros; `ds` shape fixed in DownProj backward return
- x is NOT modified (no sentinel row). `dst_idx` maps real→padded positions.

### FP8 Weight Cache Lifecycle

4 caches (`w1_fused`, `w2_varlen`, `w2_dgated`, `w1T_varlen`) keyed by `(data_ptr, _version)`.
Auto-invalidate at optimizer step. NEVER clear between iterations (costs +360us/iter).

### Paddle Compat Shim

- `paddle.enable_compat()` intercepts `import torch` → routes to Paddle
- `sonicmoe/triton_utils.py`: `wrap_triton_kernel` swaps `sys.modules["torch"]` during Triton JIT
- `_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)` — dtype fallback
- `cutlass.torch` → `cuda.bindings.driver` for stream API
- `torch.library.custom_op` → `_custom_op_or_plain` (skip if already registered)

### Paddle Compat Gaps (fixed in tests, may recur in new code)

| Pattern | Issue | Fix |
|:--------|:------|:----|
| `torch.norm(tensor)` | TypeError: frobenius_norm | Use `tensor.norm()` |
| `torch.dot(a, b)` | Same TypeError | Use `(a * b).sum()` |
| `torch.Generator(device=)` | Unsupported | Use `paddle.seed()` |
| `torch.equal(a, b)` | Returns Paddle tensor, not bool | Use `(a == b).all().item()` |
| `tensor.max(dim=).values` | Paddle max API differs | Use Python loop or `paddle.max` |
| `torch.device("cuda:0")` | ProxyModule not callable | Use string `"cuda"` |
| `torch.cuda.memory_allocated(device)` | Paddle needs int device id | Use `0` not `"cuda"` |

---

## 4. File Map

| File | Role |
|:-----|:-----|
| `sonicmoe/functional/__init__.py` | Core: `_pad_routing_metadata`, `_UpProjection`, `_DownProjection`, `moe_TC_softmax_topk_layer` |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | CUTLASS blockscaled FP8 GEMM, `_get_padding_plan` |
| `sonicmoe/triton_utils.py` | Paddle compat: Triton kernel torch/paddle module swap |
| `sonicmoe/moe.py` | `MoE` module, `setup_cpu_optimizer`, `cpu_optimizer_step` |
| `tests/ops/test_moe_module.py` | 17 tests: gold, bf16, fp8, padding, axiomatic |
| `tests/ops/conftest.py` | Shared fixtures, precision helpers (Paddle-compatible) |
| `tools/introspect.py` | Instrumentation engine: trace/nsys/grid/pad-audit modes |
| `tools/paddle_compat_parallel.py` | 27-shape 8-GPU parallel Paddle compat benchmark |
| `docs/pad_audit_methodology.md` | Route-level padding technical design document |
| `reports/paddle_compat/README.md` | 27-shape Paddle compat benchmark report |
| `reports/paddle_compat/grid_paddle_compat_*.json` | Raw benchmark data |
| `reports/grid_session53/session53_grid_full.json` | PyTorch native baseline (27 shapes) |

---

## 5. Environment

| Env | Python | Framework | Quack | Use |
|:----|:-------|:----------|:------|:----|
| xfer | 3.13 | PyTorch 2.11+cu130 | 0.3.7 (native) | PyTorch native benchmark |
| eb_venv | 3.10 | Paddle 3.3.0.dev | 0.3.7 (Paddle-adapted, zhangyichen) | Paddle compat benchmark |
| dev_b | 3.10 | Paddle 3.4.0.dev | N/A (cutlass mismatch) | NOT usable for quack |

**Critical**: eb_venv quack is at `/root/.../zhangyichen/sonicmoe_for_ernie/quack`.
dev_b cutlass lacks `_convert_single_arg` — cannot load quack.

### Reproduce

```bash
# PyTorch native
source /root/.../panzhaowu/envs/xfer/bin/activate
cd /root/.../panzhaowu/lab/sonic-moe
git checkout native-fp8-exploration
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys

# Paddle compat
source /root/.../zhangyichen/baidu/ernie/erniebot/eb_venv/bin/activate
cd /root/.../panzhaowu/lab/sonic-moe
git checkout paddle_compat
export PYTHONPATH=/root/.../zhangyichen/sonicmoe_for_ernie/quack:$PWD
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_moe_module.py::test_pad_routing_metadata_axiomatic -s

# Full 27-shape Paddle compat grid (8 GPUs, ~30 min)
python tools/paddle_compat_parallel.py
```

---

## 6. Lessons Learned

1. **Never use `dz.untyped_storage().resize_(0)` in autograd backward** without ensuring
   the prequant cache is consumed in the same backward pass. Cross-iter cache references
   to freed storage cause silent zero gradients.

2. **`torch.device()` is not callable under Paddle compat** — always use string `"cuda"`.

3. **FP8 weight caches are performance-critical**. Clearing them between iterations costs
   +360us/iter. They are version-keyed and auto-invalidate at optimizer step.

4. **Token rounding destroys routing semantics**. 60% RRMSE at E=32 is routing perturbation,
   not FP8 quantization. Route-level padding preserves bit-identical routing at +1.2% cost.

5. **nsys GPU-projection is the only trustworthy perf metric**. Wall-clock varies 1.68x
   under GPU contention. CUDA events are reliable for A/B but measure different quantities.

6. **Subprocess isolation is mandatory for FP8 tests**. Process-global `_IS_FP8_ACTIVE` flag
   and CUTLASS JIT caches prevent in-process BF16/FP8 switching.

---

## 7. Next Steps

1. **Fair E>8 FP8 baseline**: Re-run grid_session53 E>8 FP8 with route-level padding
   (instead of rounding) on xfer env to get apples-to-apples Paddle overhead numbers.

2. **Memory audit**: The backward memory gap (PD vs PT) needs investigation — likely router
   autograd tensors. Instrument with `torch.cuda.memory_snapshot()` to identify exact tensors.

3. **Multi-iter backward fix**: The `dz.untyped_storage().resize_(0)` issue makes test
   harnesses fragile. Consider keeping dz alive until UpProj backward consumes the prequant,
   or using a WeakRef guard.

4. **ERNIE-core integration test**: Run the full ERNIE MoE layer (not just SonicMoE expert
   computation) under Paddle compat to validate end-to-end compatibility.

5. **Paddle-native FP8 comparison**: The existing cross_framework_bench has Paddle ERNIE
   FP8 data (kitchen+deep_gemm). Compare against Paddle-compat SonicMoE FP8 to quantify
   the CUTLASS blockscaled advantage under the same framework.
