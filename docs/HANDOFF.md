# SonicMoE FP8 Frontier — Handoff

> **Branch:** `paddle_compat`
> **Date:** 2026-04-22 (Session 59)
> **Upstream:** `native-fp8-exploration` + PR #1 "adapt paddle" merged
> **Status:** Production-ready MlpNode. Gradient chain fixed. Identity path removed. Topk precision verified.

---

## Session 59 — Production consolidation of SonicMoEMlpNode

### Critical fix: gradient chain restored (P0)

`SonicMoEMlpNode.forward()` previously called `x.detach()` before passing
`dispatched_hidden_states` into the inner PyLayer. This **severed the autograd
link** between the MLP and `FusedDispatch.backward`, meaning dx could never
flow through the reverse A2A back to the original token owners.

In ERNIE's training loop the backward chain is:

```
FusedCombine.backward → FusionMoePyLayer.backward (= MlpNode.backward)
                       → FusedDispatch.backward (reverse A2A with dx)
```

With `detach()`, `FusedDispatch.backward` received **zero dx**, silently
dropping hidden-layer gradients for all tokens processed by this MoE layer.

**Fix:** removed `x.detach()` and `router_scores.detach()`.  x now passes
through the PyLayer normally; dx flows back to the caller via Paddle autograd.

### Identity layout path removed

The identity layout path (`dispatched_indices=None`, K=1 pre-sorted tokens)
had an **unfixable dx bug**: when `total_pad_rows > 0` (i.e. any expert's
token count is not a multiple of 128), `_UpProjection.backward` produced
`dx_reduced` indexed by expert-sorted position rather than original token
order.  For T=256, E=4, this meant 75% of tokens received zero or wrong dx.

The bug was masked by `x.detach()` — since dx was never propagated, no test
observed the corruption.  Fixing detach without fixing identity-path dx would
have silently corrupted training.

Since all production callers use the topk dispatch path (with
`dispatched_indices` and `dispatched_probs`), the identity path was removed
entirely rather than undertaking a complex metadata redesign.

`dispatched_indices` and `dispatched_probs` are now **required** arguments.

### Other cleanup

- Removed global `_GRAD_PAD_BUF` / `_GRAD_PAD_BUF_ROWS` (identity-path-only
  mutable module-level state; multi-layer sharing was unsafe under async exec).
- Removed `_cached_tpe` / `_cached_metadata` (identity-path metadata cache).
- Removed unused `deepep_to_sonic_metadata` import.
- Removed `use_topk` parameter and all conditional branches in
  `_SonicMoEDeepEPFunc`.
- Updated docstrings with explicit gradient contract matching ERNIE's
  `FusionMoePyLayer`: dx/ds via autograd, dw via `main_grad`.

### Test updates

- `test_mlpnode_precision.py`: fixed `flush_native_grads()` / `_zero_main_grads()`
  call order (was flush-after-zero → 5× warmup grads leaked into main_grad,
  inflating norm 6×).  Fixed Paddle 3.x `from_dlpack` compat (`.detach()`).
  Removed identity-path test cases.  rrmse assert restored.
- `bench_mlpnode_topk_nsys.py`: new nsys GPU-projection benchmark for topk path.
- `test_mlpnode_audit.py`: needs adaptation to topk interface (not yet updated).

### Files changed

| File | Change |
|------|--------|
| `sonicmoe/ernie_compat/mlp_node_v2.py` | -154/+68 lines: detach removed, identity path removed, gradient contract documented |
| `tests/ops/test_mlpnode_precision.py` | flush order fix, dlpack compat, identity tests removed |
| `tests/ops/bench_mlpnode_topk_nsys.py` | new: nsys GPU-projection benchmark for topk path |

### Lessons learned

64. **`detach()` before PyLayer severs the autograd chain.** In ERNIE's
    dispatch→MLP→combine pipeline, dx must flow from MLP backward through
    `FusedDispatch.backward` for the reverse A2A.  `detach()` silently drops
    this gradient.  The correct pattern (matching ERNIE `MlpNode`) is to run
    the MLP under `@no_grad()` internally but let the outer PyLayer handle
    autograd linkage.

65. **Identity layout metadata is incompatible with route-level padding backward.**
    `deepep_to_sonic_metadata` builds `s_reverse_scatter_idx = arange(TK_padded)`
    and `naept = arange(TK_padded+1)`, creating a 1:1 mapping between positions
    and "virtual tokens".  But `_UpProjection` saves `ctx.T = T_orig` (the real
    token count), so `_token_broadcast_backward` reads `dx_expanded[0..T_orig-1]`
    — which crosses expert-segment boundaries and includes pad rows.  This is
    architectural: fixing it requires redesigning the metadata for T_orig-sized
    scatter, which defeats the purpose of the simple identity layout.

66. **Test warm-up flush order matters.** `flush_native_grads()` must precede
    `_zero_main_grads()`, not follow it.  Otherwise the native buffer (accumulated
    during warmup) gets flushed into the freshly-zeroed `main_grad`, leaking N
    warmup iterations of gradients into the measurement.

---

## 1. Project State

### One-Sentence Summary

Blackwell SM100a blockscaled FP8 (E4M3 + E8M0, 1x32) MoE training is **production-ready**:
1.29-1.70x speedup, 6.5% RRMSE, route-level padding for non-aligned experts, zero explicit
cudaStreamSynchronize in backward, Paddle compat verified.

### What Works

| Capability | Status | Evidence |
|:-----------|:-------|:---------|
| FP8 fwd+bwd (E=8, aligned) | PASS, 1.34-1.70x | grid_session53 27-shape nsys |
| FP8 fwd+bwd (E=32/128, route-level padding) | PASS, 1.29-1.79x | pad_audit E=32 + grid |
| Route-level padding backward correctness | PASS, dz[pad]=exact 0 | test_pad_gradient_integrity 8 axioms |
| Paddle compat (all 27 shapes BF16+FP8) | 54/54 PASS | paddle_compat_parallel grid |
| Real training loop (5 iters, cpu_optimizer_step) | PASS, loss decreasing | pad_audit Section 3.4 |
| FP8+Stash (bf16 weights → CPU) | PASS, -24.5% peak mem | fp8_frontier_strict_test |
| 0-size expert handling | PASS, all paths | test_moe_module 7-empty tests |
| Single-stream backward (no sync overhead) | PASS, 0 STREAM_SYNC | Session 58 nsys sqlite verification |

### Session 58 Changes

1. **Removed multi-stream design** — deleted `_WGRAD_STREAM` (dead code, never used) and
   `_DEQUANT_STREAM` (used in 2 backward paths for z-dequant and x_col quant overlap).
   These produced `cudaStreamSynchronize` events via `wait_stream()`. The overlap provided
   no measurable benefit. All quant/dequant ops now run sequentially on the default stream.
   - Modified: `sonicmoe/functional/__init__.py` (removed 45 lines, added 9)
   - Verified: nsys sqlite shows zero type=3 STREAM_SYNCHRONIZE events. Backward path
     has exactly zero `cudaStreamSynchronize` calls. Test correctness PASSED.

### What Does NOT Work / Known Issues

1. **Multi-iter backward without optimizer step** — `dz.untyped_storage().resize_(0)` in
   DownProj backward frees dz storage. Second backward's UpProj gets freed tensor from
   autograd. Works in real training (optimizer step rebuilds FP8 caches). **Pre-existing.**

2. **Memory measurement inconsistency** — Paddle compat benchmark uses `moe()` full module
   (router autograd saves extra tensors), baseline uses lower-level API. Memory deltas in
   the Paddle compat grid report reflect this API-level difference, not Paddle overhead.

3. **E>8 FP8 baseline comparison** — grid_session53 E>8 FP8 used token rounding, but
   `paddle_compat` branch uses route-level padding. Only E=8 FP8 is strictly apples-to-apples.

---

## 2. Performance Numbers

### Speedup (nsys GPU-projection, B30Z, 27-shape grid)

| Dimension | Range | Mean |
|:----------|:------|:-----|
| Overall | **1.29x - 1.70x** | 1.53x |
| By T: 8k / 16k / 32k | 1.44x / 1.55x / 1.60x | - |
| By I: 1536 / 2048 / 3072 | 1.45x / 1.52x / 1.62x | - |
| By E: 8 / 32 / 128 | 1.56x / 1.53x / 1.49x | - |

Scaling rules: `speedup ~ f(I^2)`, larger T helps, E has minimal impact.

### Memory (+overhead vs BF16, backward peak)

Range: **+4.8% to +10.3%** (absolute +87 to +1563 MiB depending on shape).
Source: FP8 shadow weight caches (~650 MiB at large E) + wgrad quant temporaries.
FP8+Stash: **-24.5% peak** (bf16 master weights offloaded to CPU).

### Precision (FP8 vs BF16, identical routing, RRMSE %)

| Tensor | RRMSE | Cosine | Threshold |
|:-------|:-----:|:------:|:---------:|
| output | 6.52% | 0.9979 | <10%, >0.99 |
| dx | 6.53% | 0.9979 | <10%, >0.99 |
| dw1 | 4.27% | 0.9991 | <10%, >0.99 |
| dw2 | 4.72% | 0.9989 | <10%, >0.99 |

Precision is pure FP8 quantization noise — route-level padding contributes **exact zero** error.

### Route-Level Padding vs Token Rounding (E=32)

| Metric | FP8+padding | FP8+rounding |
|:-------|:-----------:|:------------:|
| RRMSE vs BF16 raw | **6.5%** | 60.8% |
| nsys us/iter | 2950 | 2915 |
| Memory (bwd MiB) | 2914 | 2909 |

Padding is **9x more precise** at +1.2% latency cost.

---

## 3. Core Architecture

### FP8 Frontier Pipeline

```
x(bf16) → quantize_and_pack(T→fp8+ISA_scales)
        → GemmGated_ZeroMat(fp8 GEMM, gathers x via A_idx, zero TK-sized materialization)
          → epilogue: blockscaled quant z to fp8 in registers (optional)
          → SwiGLU: fused forward + y1 fp8 quant + ISA scale pack
        → blockscaled_fp8_gemm_varlen(y1_fp8 @ w2_fp8 → y2_bf16, down-proj)
        → router_forward(scatter-add with scores, T*K → T)
```

Backward (6 dSwiGLU paths):
- **CUTLASS fused** (`gemm_dgated.py`): `colvec_scale=score` multiplies grad BEFORE dSwiGLU → dz[pad]=0
- **5 Triton kernels** (`swiglu_triton.py`): `dy1_s = dy1 * s_val` BEFORE dSwiGLU → dz[pad]=0
- Both paths: `dz[pad]=0 → dw1 zero contrib, dx[0] zero pollution`

### Route-Level Padding

`_pad_routing_metadata()` transforms 5 routing tensors once. All 8 GEMMs see 128-aligned segments.
Zero GEMM code changes. Padding rows: `x_gather_idx=0`, `score=0` (IEEE 754 exact zero).
Full mathematical proof: `docs/pad_audit_methodology.md`

### Key Invariant (Session 57 Discovery)

**dz[pad] is exactly zero.** Session 57 audited all 6 backward paths and confirmed that
score-gating occurs BEFORE dSwiGLU in every path, making `dz[pad]=0` an IEEE 754 guarantee
(`finite * 0.0 = 0.0`). Corrected the erroneous claim in `pad_audit_methodology.md`
Section 2.2 and added `test_pad_gradient_integrity.py` with 8 axioms proving this property.

### Stream Design (Session 58 Cleanup)

**All forward and backward ops run on the default CUDA stream.** Prior sessions (7, 15)
experimented with side-stream overlap (z-dequant || dout-quant). Session 58 removed all
side-stream code because: (1) `_WGRAD_STREAM` was dead code — declared but never used;
(2) `_DEQUANT_STREAM` overlap saved ~47µs theoretical but each `wait_stream()` call produced
a `cudaStreamSynchronize` that cost more; (3) cross-stream blocks prevent caching allocator
reuse. The `stream_id` parameter in CuTe DSL kernels (forward.py/backward.py) passes the
default stream's raw pointer — this is NOT multi-stream and was not changed.

### Weight Caches

4 FP8 caches: `w1_fused`, `w2_varlen`, `w2_dgated`, `w1T_varlen`.
Keyed by `(data_ptr, _version)`. Auto-invalidate at optimizer step.
**NEVER clear between iterations** (costs +360us/iter — Session 53 lesson).

---

## 4. Test Coverage

### Padding Correctness (14 axioms total)

| File | Axioms | What |
|:-----|:------:|:-----|
| `tests/ops/test_pad_routing.py` | 6 | Forward: non-alignment detected, segments padded, tokens preserved, scores exact, reverse_scatter remapped, output near-exact |
| `tests/ops/test_pad_gradient_integrity.py` | 8 | Backward: token conservation, score invariant, forward near-exact, **dz[pad]==0 exact**, dw1/dw2/dx near-exact, dx[0] verified, no misrouting |

### Full Test Suite

| Category | Files | Tests | Key Thresholds |
|:---------|:-----:|:-----:|:---------------|
| Quant kernels | 6 files | ~80 | Byte-exact vs gold E8M0 |
| SwiGLU fwd/bwd | 1 file | ~90 | BF16 atol<1e-2, FP8 RRMSE<10% |
| CUTLASS GEMMs | 4 files | ~12 | 3-way cross-validation |
| MoE module | 1 file | ~59 | BF16 RRMSE<1%, FP8<10% |
| FP8 frontier | 1 file | ~10 | Strict: no skip, no fallback |
| FP8 protocol | 1 file | ~26 | Byte-exact, roundtrip<0.35 |
| Padding axioms | 2 files | 14 | Exact 0.0 where guaranteed |

Run: `python -m pytest tests/ops/ -v --tb=short`

### Known Test Gap

FP8 dgated backward kernel lacks an **isolated** 3-way test (only end-to-end coverage via
`fp8_large_project_contract_test.py`). This is pre-existing and documented.

---

## 5. High-Value Information Sources

| Source | Path | Why It Matters |
|:-------|:-----|:---------------|
| Pad audit design doc | `docs/pad_audit_methodology.md` | Mathematical proof of padding correctness, all backward paths analyzed |
| Session 53 grid data | `reports/grid_session53/session53_grid_full.json` | Raw 27-shape benchmark (nsys + memory + precision) |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | 21 phases (Sessions 1-58), 62 lessons, all dead ends documented |
| ERNIE-core MoE layer | `/root/.../liangshuhao/erniebot_test_speed/third_party/ernie-core/src/ernie_core/models/moe/moe_layer.py` | `MlpNode` (line 1776), `FusionFP8Expert` (line 545), `ExpertsGroupGemmContiguousNode` — integration target |
| ERNIE-core FP8 utils | `ernie_core/models/moe/token_dispatcher/fp8_utils.py` | Paddle's `kitchen_quant` + `deep_gemm` — the comparison point |
| nsys sqlite (Session 58) | `/root/.../panzhaowu/output/nsys_stream_sync/moe_fp8_routing.sqlite` | Verified zero STREAM_SYNCHRONIZE events after fix |
| Official BF16 baseline | `/lab/official/sonic-moe` (env: `official_bf16`) | ONLY valid BF16 reference |
| Paddle quack fork | `/root/.../zhangyichen/sonicmoe_for_ernie/quack` | eb_venv quack with Paddle compat patches |
| pd_run.sh reference | `/root/.../zhangyichen/sonicmoe_for_ernie/supersonic-moe/tests/pd_run.sh` | Reference env config for Paddle tests (eb_venv, SONIC_MOE_FP8_MODE=perf, USE_QUACK_GEMM=1) |

---

## 6. Environment

| Env | Python | Framework | Quack | Use |
|:----|:-------|:----------|:------|:----|
| xfer | 3.13 | PyTorch 2.11+cu130 | 0.3.7 (native) | PyTorch native benchmark |
| eb_venv | 3.10 | Paddle 3.3.0.dev | 0.3.7 (Paddle-adapted) | Paddle compat benchmark |
| dev_b | 3.10 | Paddle 3.4.0.dev | N/A (cutlass mismatch) | NOT usable for quack |

**Critical**: eb_venv quack is at `/root/.../zhangyichen/sonicmoe_for_ernie/quack`.

### Reproduce

```bash
# Padding tests (eb_venv, ~15s)
source /root/.../zhangyichen/baidu/ernie/erniebot/eb_venv/bin/activate
export SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 QUACK_CACHE_DIR=./my_quack_cache
CUDA_VISIBLE_DEVICES=1 python tests/ops/test_pad_gradient_integrity.py
CUDA_VISIBLE_DEVICES=1 python tests/ops/test_pad_routing.py

# FP8 routing test (eb_venv, ~30s — the test used for Session 58 stream sync validation)
CUDA_VISIBLE_DEVICES=1 python tests/ops/test_moe_general_routing_fp8.py

# nsys profiling (eb_venv — do NOT use --capture-range=nvtx with Paddle)
nsys profile --trace=cuda,nvtx --output=/tmp/profile \
  python tests/ops/test_moe_general_routing_fp8.py --bench
nsys export --type=sqlite --output=/tmp/profile.sqlite /tmp/profile.nsys-rep
# Query: SELECT syncType, COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION GROUP BY syncType
# syncType=3 (STREAM_SYNCHRONIZE) should be 0

# PyTorch native nsys benchmark (xfer)
source /root/.../panzhaowu/envs/xfer/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --nsys-shapes 8192,3072,1536,8,8

# Full 27-shape grid (8 GPUs, ~15 min)
python tools/introspect.py --mode grid --gpu 8
```

---

## 7. Lessons Learned (Selected High-Value)

### From Session 58 (stream sync elimination)

61. **Side-stream overlap in backward provides no net benefit.** `_DEQUANT_STREAM` theoretically
    saved ~47µs by overlapping z-dequant with dout-quant, but each `wait_stream()` call produced
    a `cudaStreamSynchronize` (CUPTI type=3), and cross-stream blocks prevent caching allocator
    reuse. Net effect was negative. Always prefer single-stream unless overlap window is >100µs.
    (Note: engineering_log Phase 15 recorded "removed cross-stream overlap" for wgrad, but
    `_DEQUANT_STREAM` survived until Session 58 in two backward paths. Now fully removed.)

62. **nsys `--capture-range=nvtx` does NOT work with Paddle's `nvprof_nvtx_push`.** Paddle's NVTX
    markers use a different domain/API than nsys expects. Use `--capture-range=none` (or omit
    the flag entirely) when profiling Paddle-based tests. Export to sqlite and query
    `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` with `syncType=3` to find stream syncs.

63. **`_WGRAD_STREAM` was dead code.** Declared in Session 32, but `_get_wgrad_stream()` was
    never called anywhere — all wgrad paths used the default stream. A reminder to grep for
    callers before assuming infra is live.

### From Session 57 (doc audit + gradient test)

59. **Score-gating guarantees dz[pad]=0 exactly across ALL 6 backward paths.** The CUTLASS
    `colvec_scale` and Triton `dy1 * s_val` both multiply score into gradient BEFORE dSwiGLU.
    IEEE 754: `finite * 0.0 = 0.0`. This makes route-level padding backward-safe by construction.

60. **GPU matmul tiling depends on problem shape.** Two float64 matmuls with different row counts
    (7 vs 128) produce ULP-level differences (~2e-16) due to different reduction order. Test
    assertions for matmul-derived quantities must use epsilon tolerance, not bit-exact comparison.
    Score-gating zeros (dz[pad]=0) ARE bit-exact because they depend only on multiplication by zero.

### From Earlier Sessions (canonical, not stale)

- **Weight cache invalidation must be version-keyed, not eager** (Lesson 48). Clearing every backward = +360us.
- **Token rounding destroys routing semantics** (Lesson 4). 60% RRMSE is routing perturbation.
- **Subprocess isolation is mandatory for FP8 tests** (Lesson 6). Process-global `_IS_FP8_ACTIVE`.
- **I scaling dominates FP8 ROI** (Lesson 53). GEMM savings ~ O(I^2), quant overhead ~ O(I).
- **num_warps=1 for bandwidth-bound Triton** (Lesson 43). Counter-intuitive: fewer warps = more blocks = better SM utilization.
- **Never resize_(0) tensors aliased in caches** (Lesson 49). Autograd ctx tensors may share storage.

---

## 8. Insights for Next Agent

1. **The FP8 frontier is compute-efficient but memory-heavy.** +5-10% backward memory is fundamental
   (FP8 shadow caches + wgrad quant temps). FP8+Stash (-24.5%) is the solution for memory pressure.

2. **Forward quant overhead is the remaining bottleneck.** At I=1536, FP8 forward is 19% slower
   than BF16 (quant-and-pack on critical path). Epilogue-level forward quantization would eliminate
   this, but requires CUTLASS DSL work.

3. **Route-level padding is mathematically complete.** It handles forward AND backward with exact zero
   error from padding. No further correctness work needed. The only improvement would be reducing
   the number of padding rows for near-aligned segments (minor perf gain).

4. **Paddle compat is a thin shim.** BF16 has +2.9% overhead from proxy dispatch; FP8 has ~0%
   because CUTLASS kernels bypass the proxy. The shim is stable but fragile — new code using
   `torch.device()`, `torch.norm()`, or `torch.Generator()` will break under Paddle.

5. **The test suite is comprehensive but has one gap.** FP8 dgated backward kernel lacks an
   isolated kernel-level test (only end-to-end coverage). Adding a `test_gemm_dgated_fp8` would
   close the last coverage gap.

6. **All backward sync is eliminated.** After Session 58, the backward path has zero explicit
   `cudaStreamSynchronize`. The forward still has 6 framework-internal `stream_wait_event`
   calls (CUPTI type=2, from Paddle/PyTorch internals), which are harmless and unavoidable.
   The remaining D2H syncs in forward are: `_get_cu_seqlens_cpu()` → `.tolist()` (guarded by
   `ASSUME_ALIGNED`), `_auto_capacity()` → `.max().item()`, and `_unpack_grouped_rows()` /
   `blockscaled_fp8_gemm()` → `.item()`.

7. **ERNIE-core MlpNode is the integration target.** `MlpNode` (moe_layer.py:1776) orchestrates
   unzip → expert GEMM → zip. Its GEMM backend (`ExpertsGroupGemmContiguousNode`) uses
   `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`. Replacing this with sonic-moe's
   CUTLASS DSL GEMM is the next major milestone. Key compatibility: both use FP8_ALIGN=128,
   both use 1x32 blockscaled quantization, both have subbatch support.

---

## 9. Next Steps

### 9.1 ERNIE-core MlpNode Integration (Priority 1)

**Goal**: Replace ERNIE-core's `ExpertsGroupGemmContiguousNode` GEMM backend with sonic-moe's
FP8 frontier (`_UpProjection` + `_DownProjection`) for CUTLASS DSL performance on Blackwell.

**ERNIE MlpNode Architecture** (moe_layer.py:1776):
```
dispatch(hidden_states) → (hs_2d_dispatched, dispatched_indices, dispatched_probs)
  → UnZipNode.forward (gather per-expert tokens, tilewise_quant → FP8)
  → ExpertsGroupGemmContiguousNode.forward (FP8 group GEMM: up+gated+SwiGLU+down)
  → ZipNode.forward (scatter-add weighted by probs)
→ combine output
```

**Integration Approach**:
1. **Adapter class** wrapping sonic-moe's `moe_general_routing_inputs()` to accept MlpNode's
   `(hs_2d_dispatched, dispatched_indices, dispatched_probs)` input format.
2. **GEMM substitution**: MlpNode's `ExpertsGroupGemmContiguousNode` → sonic-moe's
   `_UpProjection` (up+gated+SwiGLU) + `_DownProjection` (down-proj, varlen).
3. **Key compatibility points**:
   - Both use FP8_ALIGN=128 for expert segment alignment
   - Both use 1x32 blockscaled E4M3+E8M0 quantization
   - MlpNode's `tokens_per_expert` list maps to sonic-moe's `cu_seqlens` (cumsum of padded counts)
   - MlpNode.unzip (gather) = sonic-moe's `x_gather_idx` scatter
   - MlpNode.zip (scatter-add) = sonic-moe's `token_gather_sum_kernel`
4. **SwiGLU convention**: ERNIE uses split-half (`gate=[:I], up=[I:]`), sonic-moe uses
   interleaved (`gate=even, up=odd`). Use `split_to_interleaved()` / `interleaved_to_split()`
   (already verified bit-exact round-trip in Session 54).
5. **Weight lifecycle shim**: ERNIE stores `fp8_weight_stacked`/`fp8_scale_stacked` on the
   parameter object; sonic-moe uses `_STASHED_FP8_WEIGHTS` dict keyed by `(data_ptr, _version)`.
   Need a bridge to populate sonic-moe's cache from ERNIE's pre-quantized weights.
6. **Prob scaling difference**: ERNIE applies `o2 = swiglu(o1) * probs` between SwiGLU and
   down-proj; sonic-moe applies it after down-proj in `_router_forward`. Mathematically
   equivalent for linear down-proj, but intermediate tensors differ — test both orders.

**MlpNode subbatch support**: MlpNode has `moe_subbatch_token_num_after_dispatch` (fixed-size
subbatch) and `use_auto_subbatch` (memory-aware dynamic subbatch). These map naturally to
sonic-moe's expert-level GEMM splitting in `_UpProjection`/`_DownProjection`.

### 9.2 Fair E>8 FP8 Baseline

Re-run grid_session53 E>8 FP8 with route-level padding (instead of rounding) on xfer env.
Currently E>8 numbers conflate routing strategy with Paddle overhead.

### 9.3 Epilogue Forward Quantization

Move `quantize_and_pack_activation(x)` into the GemmGated epilogue (like z epilogue quant).
Would eliminate ~130us forward overhead — the dominant bottleneck at I=1536.

### 9.4 FP8 Dgated Isolated Test

Add `test_gemm_dgated.py::test_fp8_dgated` with 3-way cross-validation
(torch gold vs BF16 CUTLASS vs FP8 blockscaled). Currently only end-to-end coverage.

### 9.5 Remaining D2H Sync Points in Forward

These `.item()` / `.tolist()` calls cause D2H transfers in the forward path. They are
acceptable if the ops before them are already synchronizing (GEMM completion), but could
be eliminated for fully asynchronous pipelines:
- `_get_cu_seqlens_cpu()` → `.tolist()` (guarded by `SONIC_MOE_FP8_ASSUME_ALIGNED=1`)
- `_auto_capacity()` → `.max().item()`
- `_unpack_grouped_rows()` → `cu_seqlens_m[-1].item()`
- `blockscaled_fp8_gemm()` → `cu_seqlens_m[-1].item()`
- `prepare_sonic_inputs()` (test only) → `pad_counts.max().item()`

---

## 10. File Map

| File | Role |
|:-----|:-----|
| `sonicmoe/functional/__init__.py` | Core: `_pad_routing_metadata`, `_UpProjection`, `_DownProjection` |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | CUTLASS FP8 GEMM, `_get_padding_plan`, ISA scale packing |
| `sonicmoe/quack_utils/gemm_gated.py` | Forward CUTLASS DSL + epilogue blockscaled quant |
| `sonicmoe/quack_utils/gemm_dgated.py` | Backward CUTLASS DSL + FP8 z decompression |
| `sonicmoe/quack_utils/swiglu_triton.py` | 7 SwiGLU Triton variants (fwd/bwd, fused quant) |
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | CuTe DSL colwise FP8 quant (NCU-optimized) |
| `sonicmoe/moe.py` | MoE module, weight caches, cpu_optimizer |
| `sonicmoe/triton_utils.py` | Paddle compat Triton kernel module swap |
| `docs/pad_audit_methodology.md` | Route-level padding design + mathematical proof |
| `tests/ops/test_pad_gradient_integrity.py` | 8 backward correctness axioms |
| `tests/ops/test_pad_routing.py` | 6 forward correctness axioms |
| `tests/ops/test_moe_module.py` | 59 module-level precision tests |
| `tools/introspect.py` | All-in-one profiling: nsys/grid/precision/pad-audit |
| `reports/grid_session53/` | 27-shape benchmark data |
| `reports/paddle_compat/` | Paddle compat 27-shape grid report |
