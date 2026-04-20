# SonicMoE FP8 Frontier — Handoff

> **Branch:** `paddle_compat`
> **Date:** 2026-04-20 (Session 58)
> **Upstream:** `native-fp8-exploration` + PR #1 "adapt paddle" merged
> **Status:** Clean frontier. All tests pass. Multi-stream sync eliminated.

---

## 1. Project State

### One-Sentence Summary

Blackwell SM100a blockscaled FP8 (E4M3 + E8M0, 1x32) MoE training is **production-ready**:
1.29-1.70x speedup, 6.5% RRMSE, route-level padding for non-aligned experts, Paddle compat verified.

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

**dz[pad] is exactly zero.** The original Session 56 HANDOFF did not explicitly address this.
Session 57 audited all 6 backward paths and confirmed that score-gating occurs BEFORE dSwiGLU
in every path, making `dz[pad]=0` an IEEE 754 guarantee (`finite * 0.0 = 0.0`).
Corrected the erroneous claim in `pad_audit_methodology.md` Section 2.2 and added
`test_pad_gradient_integrity.py` with 8 axioms proving this property.

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
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | 19 phases, 59 lessons, all dead ends documented |
| NCU kernel profile | `/tmp/ncu_quant2.ncu-rep` (ephemeral) | 23-kernel metrics (regs, occ%, DRAM%, LD eff) |
| ERNIE-core FP8 | `ernie_core/models/moe/token_dispatcher/fp8_utils.py` | Paddle's `kitchen_quant` + `deep_gemm` — the comparison point |
| Official BF16 baseline | `/lab/official/sonic-moe` (env: `official_bf16`) | ONLY valid BF16 reference |
| Paddle quack fork | `/root/.../zhangyichen/sonicmoe_for_ernie/quack` | eb_venv quack with Paddle compat patches |

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

# PyTorch native nsys benchmark (xfer)
source /root/.../panzhaowu/envs/xfer/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --nsys-shapes 8192,3072,1536,8,8

# Full 27-shape grid (8 GPUs, ~15 min)
python tools/introspect.py --mode grid --gpu 8
```

---

## 7. Lessons Learned (Selected High-Value)

### From Session 58 (stream sync elimination)

61. **Multi-stream design has no perf benefit in sonic-moe.** `_WGRAD_STREAM` (dead code, never
    called) and `_DEQUANT_STREAM` (used in 2 backward paths) added explicit `cudaStreamSynchronize`
    (via `wait_stream()`) without measurable overlap gains. Removing them eliminates all type=3
    STREAM_SYNCHRONIZE events from nsys. Backward path now has zero sync calls.

62. **nsys `--capture-range=nvtx` + Paddle's `nvprof_nvtx_push` don't interoperate.** Use
    `--capture-range=none` when profiling Paddle-based tests with nsys.

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

---

## 9. Next Steps

1. **Fair E>8 FP8 baseline**: Re-run grid_session53 E>8 FP8 with route-level padding (instead of
   rounding) on xfer env. Currently E>8 numbers conflate routing strategy with Paddle overhead.

2. **Epilogue forward quantization**: Move `quantize_and_pack_activation(x)` into the GemmGated
   epilogue (like z epilogue quant). Would eliminate ~130us forward overhead.

3. **FP8 dgated isolated test**: Add `test_gemm_dgated.py::test_fp8_dgated` with 3-way
   cross-validation (torch gold vs BF16 CUTLASS vs FP8 blockscaled).

4. **Memory audit**: Investigate backward memory gap (PD vs PT). Use `torch.cuda.memory_snapshot()`
   to identify exact tensors. The gap is likely router autograd tensors.

5. **ERNIE-core MlpNode integration**: Replace ERNIE-core's `ExpertsGroupGemmContiguousNode`
   GEMM backend with sonic-moe's FP8 frontier (`_UpProjection` + `_DownProjection`) to get
   CUTLASS DSL performance. Integration approach:
   - **Adapter class** wrapping sonic-moe's `moe_general_routing_inputs()` to match MlpNode's
     `(hs_2d_dispatched, dispatched_indices, dispatched_probs)` interface
   - MlpNode.forward calls unzip → sonic-moe FP8 GEMM (up+gated+down) → zip
   - Key mappings: MlpNode.unzip = sonic-moe's `x_gather_idx` scatter, MlpNode.zip = `token_gather_sum`
   - FP8_ALIGN (128) matches sonic-moe's 128-alignment padding
   - MlpNode's subbatch support maps to sonic-moe's expert-level GEMM splitting
   - Weight format: ERNIE uses `fp8_weight_stacked`/`fp8_scale_stacked` per-layer;
     sonic-moe uses `_STASHED_FP8_WEIGHTS` cache. Need a shim to bridge weight lifecycle.

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
