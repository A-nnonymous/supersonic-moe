# HANDOFF — Session 62 (2026-04-24)

> **Single source of truth.** `docs/HANDOFF.md` redirects here. `reports/fp8_upgrade/engineering_log.md` has the full chronological log.

**Branch**: `session60-ds-fix` on `myrepo` (A-nnonymous/supersonic-moe)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0)
**Python**: 3.12 (system), Paddle torch-proxy + QuACK at `/root/.../zhangyichen/sonicmoe_for_ernie/quack`

---

## 1. What Works

| Capability | Evidence | Cosine |
|---|---|:---:|
| FP8 fwd output (out) | `test_cold_start_e2e.py`, 6 shapes | 0.9979 |
| FP8 bwd dx (hidden_states grad) | same | 0.9975 |
| FP8 bwd ds (dispatched_probs grad) | same, Triton `_build_score_src_idx_kernel` | 0.9972 |
| FP8 bwd dw1 (up-gate weight grad via main_grad) | same | 0.9975 |
| FP8 bwd dw2 (down-proj weight grad via main_grad) | same | 0.9972 |
| Dynamic seqlen (zero CuTe recompile) | `test_jit_optimization.py`: 4 seqlens, 0 compiles | N/A |
| Cold start (all cache cleared → production) | `test_cold_start_e2e.py`: 42s JIT → 0.05s steady | N/A |
| `SonicMoEMlpNode.step()` API | flush grads + invalidate caches | N/A |
| Input validation (18 operators) | `_validate.py`, zero GPU sync | N/A |
| nsys GPU-projection | 2871 µs/iter, 3-GPU mean, CV=0.6% | N/A |

## 2. What Doesn't Work

| Item | Detail |
|---|---|
| `warmup_jit()` standalone (no Paddle) | `torch.utils.cpp_extension.load()` incompatible with Paddle proxy kwargs. Works inside Paddle-enabled script. |
| Multi-card EP>1 | Requires DeepEP buffer; single-card only tested |
| ERNIE training loop plug-in | MlpNode interface verified, not yet in actual ERNIE training |
| Pipeline microbatch overlap | `_PREQUANTIZED_SCALES` module-level dict unsafe under overlap |
| wgrad per-call latency | QuACK varlen_k: 322µs vs baseline 267µs (+21%); cause unknown |

## 3. Performance (Ernie shape T=8192 E=8 I=1536 K=8 H=3072)

### GPU-Projection Breakdown (nsys sqlite, BENCH region, 3-GPU mean)

| Category | µs/iter | % of GPU time |
|---|:---:|:---:|
| GEMM fwd upproj (GatedFP8 zeromat) | 451 | 15.7% |
| GEMM bwd upproj act (DGatedFP8) | 401 | 14.0% |
| GEMM bwd wgrad (QuACK varlen_k) × 4 | 1289 | 44.9% |
| FP8 colwise quant+pack × 3 | 234 | 8.1% |
| FP8 dual varlen quant | 153 | 5.3% |
| FP8 row quant+pack × 3 | 80 | 2.8% |
| token_gather_sum (routing) × 2 | 148 | 5.1% |
| CUDA topk metadata + score_src_idx | 34 | 1.2% |
| Paddle framework kernels | 67 | 2.3% |
| **GPU-projection total** | **2871** | |
| **Host Python overhead** (not in GPU-proj) | **~2200** | |
| **CUDA events total** | **~5100** | |

### Comparison with Session 53 Baseline (native PyTorch `MoE` module)

| GEMM | Baseline per-call µs | Current per-call µs | Delta |
|---|:---:|:---:|:---:|
| fwd_upproj (GatedFP8) | 451.9 | 451.1 | -0.2% |
| bwd_upproj (DGatedFP8) | 382.9 | 401.4 | +4.8% |
| bwd_wgrad (QuACK) | 266.6 | 322.3 | +20.9% |

GEMM kernel efficiency is near-identical (fwd -0.2%). The +5.7% total delta (2871 vs 2715) comes from new routing metadata kernels (34µs) and Paddle compat framework kernels (67µs).

### Memory

- Peak backward: ~3400 MiB (Ernie shape, FP8)
- native_grad buffers: ~432 MiB persistent ([E,2I,H] + [E,H,I] fp32)
- FP8 weight caches: ~650 MiB at E=128 (version-keyed, auto-invalidate)

### Precision

| Tensor | Cosine Sim | RRMSE % |
|---|:---:|:---:|
| output | 0.9979 | 6.5 |
| dx | 0.9975 | 7.0 |
| ds | 0.9972 | — |
| dw1 | 0.9975 | 4.7 |
| dw2 | 0.9972 | 4.9 |

Verified across 6 shapes (N=512..16384, K=4..8), all cos > 0.99.

## 4. JIT Cache Design

### Principle: compile_key is static-only

All 7 compile_key patterns contain only static model dimensions (H, I, E, dtype, tile config). **Zero** token-count-dependent values. Dynamic dims handled at runtime via `mark_layout_dynamic`.

| compile_key tag | Cache | Dynamic fields |
|---|---|---|
| `"vk"` | `_COMPILE_CACHE_VK` | NONE |
| `"vk_accum"` | `_COMPILE_CACHE_VK_ACCUM` | NONE |
| `"varlen"` | `_COMPILE_CACHE` | NONE (Session 62 fix: removed a_scales_packed.size(1)) |
| `"weight_grad"` | `_COMPILE_CACHE` | NONE (Session 62 fix: removed capacity) |
| `"weight_grad_fast"` | `_COMPILE_CACHE` | NONE (Session 62 fix) |
| `"zeromat_gated"` | `_zeromat_compile_cache` | NONE |
| (grouped) | `_COMPILE_CACHE` | NONE (capacity from env var) |

### fast_path correctness proof

`_GEMM_FAST_PATH*` dicts cache `(compiled_fn, scheduler_args, epi_args)` keyed by exact problem shape. When fast_key misses, falls through to compile_key lookup.

- **compiled_fn**: safe to reuse — `mark_layout_dynamic` handles any token count at runtime
- **scheduler_args**: safe — contains only `max_active_clusters` + `max_swizzle_size` (device properties)
- **epi_args**: safe — default `EpilogueArguments()`, no shape-dependent fields
- **varlen_args**: safe — recreated per-call from fresh `cu_seqlens_m/k`
- **Conclusion**: fast_path can never serve a wrong kernel. Different shapes → different fast_key → miss → slow path.

### Cache lifecycle

```
Training iteration:
  forward():  compile_key hit → fast_path hit → CuTe kernel launch
  backward(): same
  optimizer.step()
  node.step():
    flush_native_grads()        # _NATIVE_W1/W2_GRAD → per-expert main_grad
    invalidate_weight_caches()  # clear _W_CACHE + FP8 weight caches + topk cache
```

## 5. Critical Constraints

1. **ds gradient path**: Between gate output and `_DownProjection.apply()`, **zero** native Paddle autograd nodes on the score tensor path. Paddle `topk()`, `.cast()`, `amp.decorate` all create autograd nodes that segfault when receiving torch-proxy gradient tensors.

2. **Paddle bf16 tensor conversion**: `tensor.cpu().numpy()` returns `uint16` (wrong). `torch.as_tensor()` returns `float16` (wrong). **Only `torch.from_dlpack()` preserves bf16.**

3. **`_inplace_version` compat**: Paddle = `_inplace_version()` (method), PyTorch = `._version` (attribute). Use `_tensor_version()` helper in `blockscaled_fp8_gemm.py`.

4. **CUDA stream compat**: Paddle = `stream.stream_base.raw_stream`, PyTorch = `stream.cuda_stream`. Use `hasattr` branch. Fixed in `gemm_gated.py`, `gemm_dgated.py`, `gemm_sm100_fp8_zeromat.py`, `blockscaled_fp8_gemm.py`.

5. **`ctx.saved_tensor()` vs `ctx.saved_tensors`**: Paddle PyLayer = method, PyTorch = attribute. Warmup must run under Paddle proxy to use the correct API.

## 6. Lessons Learned (Session 62)

73. **`mark_layout_dynamic` has zero GPU kernel overhead.** nsys GPU-projection A/B test: dynamic 2868µs vs static 2886µs (Δ=-0.6%, noise). The overhead is purely Python-side (~1000µs host-time for CuTe binding calls), invisible to GPU timeline.

74. **CUDA events ≠ GPU kernel time.** CUDA events = 5100µs, GPU-projection = 2871µs. The 2200µs gap is pure Python host overhead (Paddle autograd, CuTe binding, tensor alloc). Never use CUDA events for kernel-level comparison.

75. **nsys `cuda_gpu_kern_sum` is wrong for multi-region profiles.** Without NVTX filtering, warmup kernels inflate the sum. Always use NVTX BENCH range + manual sqlite query.

76. **Paddle bf16 → numpy is silently broken.** `paddle.randn(dtype='bfloat16').cpu().numpy()` returns `uint16` array with raw bit patterns. This produces NaN/garbage when fed to `torch.tensor(numpy_array)`. Only `torch.from_dlpack(paddle_tensor)` works correctly.

77. **compile_key must not contain ISA-packed scale shapes.** `_storage_per_batch(total_M, K)` changes with total_M, so `a_scales_packed.size(1)` is dynamic. Including it caused silent recompilation.

78. **`_auto_capacity` makes tensor shapes dynamic.** `capacity = ceil(max_expert_tokens / 128) * 128` changes with routing. Any compile_key containing `tensor.shape` where that tensor's dim depends on capacity is effectively dynamic.

## 7. High-Value Information Sources

| Source | Path | Why |
|---|---|---|
| Cold-start E2E test | `tests/ops/test_cold_start_e2e.py` | 6-shape × 5-tensor precision + JIT validation |
| JIT optimization test | `tests/ops/test_jit_optimization.py` | Correctness + zero-recompile + memory |
| nsys benchmark | `tests/ops/bench_mlpnode_topk_nsys.py` | GPU-projection timing (use with `nsys --resolve-symbols=false`) |
| nsys sqlite files | `/panzhaowu/output/nsys/s62_*.sqlite` | Session 62 profiles for all shapes |
| Session 53 grid data | `reports/grid_session53/session53_grid_full.json` | 27-shape native PyTorch baseline |
| Pad audit proof | `docs/pad_audit_methodology.md` | Mathematical proof: dz[pad]=0 exactly |
| Session 60 lessons | `docs/session60_lessons.md` | torch-proxy gradient chain bugs (Lessons 67-72) |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Phases 1-22, 78 lessons |
| Environment | `/panzhaowu/env.md` | Machine setup, Paddle pitfalls, perf methodology |

## 8. Insights and Next Steps

### Insights

1. **GEMM is the bottleneck, not quant.** GEMM = 75% of GPU time. FP8 quant = 17%. Paddle compat = 2%. Optimizing quant further has diminishing returns; the next big win is faster GEMM (e.g. CUTLASS 4.x, or reducing wgrad from 4 calls to 2 via tiling).

2. **Host Python overhead is the dominant wall-clock gap.** GPU-projection = 2871µs, CUDA events = 5100µs. The 2200µs Python overhead is 43% of wall-clock but invisible to GPU. Reducing Python dispatch (fewer tensor creations, cached varlen_args) would help.

3. **wgrad +21% per-call regression needs investigation.** QuACK varlen_k: 322µs vs baseline 267µs. Both use `mark_layout_dynamic`. Could be: (a) different tensor stride patterns from Paddle proxy, (b) different cu_seqlens_k distribution, (c) register pressure from beta=1.0 epilogue. Worth an NCU --clock-control=none comparison.

4. **FP8 weight shadow caches dominate memory overhead.** +5-10% backward peak comes from 4 FP8 weight cache copies. FP8+Stash (-24.5%) is the solution but requires CPU↔GPU orchestration.

### Next Steps (Priority Order)

1. **ERNIE training loop integration** — Plug `SonicMoEMlpNode` into actual PaddleFleet `MlpNode` slot. Key: weight convention (split-half ↔ interleaved), prob scaling order, subbatch support.

2. **wgrad +21% root cause** — NCU --clock-control=none A/B with identical shapes on native vs Paddle path. If it's stride-related, fix in tensor permutation; if it's beta=1.0 epilogue, consider separate accumulate kernel.

3. **Multi-card EP>1** — Set up DeepEP buffer, test dispatch→MlpNode→combine pipeline with A2A.

4. **Epilogue forward quantization** — Move x→fp8 quant into GemmGated epilogue to eliminate ~130µs forward overhead.

5. **Pipeline overlap safety** — Replace module-level `_PREQUANTIZED_SCALES` with per-context storage for safe microbatch overlap.
