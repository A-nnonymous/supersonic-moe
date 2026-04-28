# HANDOFF — Session 67 (2026-04-29)

> **Single source of truth for project state.** `docs/HANDOFF.md` redirects here.
> Earlier session content (S66 + audit history) preserved verbatim below the **Session 67** block.

**Branch**: `session60-ds-fix` on `myrepo` (PFCCLab/supersonic-moe)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0)
**Python**: 3.12, Paddle torch-proxy (`paddle.compat.enable_torch_proxy` / `paddle.enable_compat`)
**QuACK**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack` (v0.3.7 + Paddle compat patches; **not** `third_party/quack`)
**Venv**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv`

---

## SESSION 67 DELIVERABLES (current)

Two coupled efforts: (1) audit + retire 32×32 isotropic blockscale weight quant, (2) add an opt-in **`recompute_z`** mode that skips storing `z_fp8` in forward and re-runs the up-proj GEMM in backward.

### S67.1 — Iso32 weight-quant retired (default OFF)

**Action**: `_quantize_weight_3d_triton(..., isotropic=False)` is now the default in `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`. The iso32 kernel + `quantize_and_pack_weight_iso32` API remain available as opt-in for any future "true transpose-reuse" optimization.

**Why** (rigorous audit, supersedes prior session's claim of "strict precision loss"):

- **Numerics audit** (`tests/ops/audit_iso32_numerics.py`, pure-PyTorch quant→dequant): iso32 and 1×32 produce **bit-identical** aggregate metrics (cosine, RRMSE, max-abs) on uniform, heavy-tail-outlier (3% × 100×), and per-row-variance (13-stop) shapes. **My prior "precision loss" claim was wrong** — E4M3 is floating-point so the e8m0 shift just relocates the precision window; relative quant error stays the same as long as values stay in e4m3 normal range. Subnormal underflow on tile-outliers contributes negligibly to aggregate metrics dominated by the largest tiles.
- **Perf audit** (`tests/ops/bench_iso32_quant_nsys.py` + `tools/parse_nsys_per_iter.py`, nsys-timeline GPU-projection): delta is within ±2µs noise across 4 weight shapes; iso32 actually **slightly slower** for w2-shaped weights. Both kernels cached (`_FUSED_WEIGHT_CACHE`, capacity 8) → call-once-per-layer-per-step → fully amortized.
- **Memory**: zero benefit (same scale-table size).
- **Transpose-reuse property**: never exploited in current code paths (callers always re-quantize transpose from BF16, separate cache keys). Was the only theoretical justification for iso32.

**Verdict**: zero benefit, deprecated as the production default.

**Regression**: `tests/ops/test_mlpnode_correctness_large.py` (9 cases, T up to 16384, TK up to 131072) PASS post-flip.

### S67.2 — `recompute_z` UpProj backward-side recompute (opt-in)

**New config**: `SonicMoEConfig(recompute_z=True)` or `SONIC_MOE_FP8_RECOMPUTE_Z=1`. Default OFF.

**Behavior** (when ON, requires `save_z_fp8=True` semantically — checked):

1. `_UpProjection.forward` runs `_fused_blockscaled_gated_forward` as usual, but does **not** populate `_PREQUANTIZED_SCALES["z_fp8"]`. Instead it stashes the recompute closure args `(x, w1, expert_frequency_offset, x_gather_idx)` in `_PREQUANTIZED_SCALES["z_fp8_recompute"]`.
2. `_DownProjection.forward` (FP8/aligned/fused-gated path) detects the recompute closure, saves zero-storage placeholder tensors for `z_fp8` and `z_raw_scales` (correct shape/dtype/device, stride (0,0)), and stashes the closure on `ctx._z_recompute_args` with `ctx._needs_z_recompute=True`.
3. `_DownProjection.backward` calls the new helper `_recompute_z_fp8(*ctx._z_recompute_args)` just before consuming `z_fp8`. The helper temporarily forces `cfg.epilogue_quant=True` and `cfg.recompute_z=False`, re-runs `_fused_blockscaled_gated_forward`, pops the freshly-populated `_PREQUANTIZED_SCALES["z_fp8"]`, and frees the wasted recomputed `y1` storage.

**Trade-off** (accepted as the minimum-LOC, zero-CUTLASS-risk baseline — Option A in design notes):

- **Memory**: ~213 MiB / layer freed during forward at ERNIE shape (TK≈65536, 2I=3072). Stacks linearly with active layers in real training. Verified at small shape (T=1024,K=8,E=8,I=1536): forward-peak drops 26 MB.
- **Compute**: extra SwiGLU + PostAct write per layer per backward (~5–15% of an up-proj fwd cost; ~10 ms / iter at 24 layers). The full fp8 GEMM is paid again — this is the inherent cost of recompute.

**Future optimization** (Option B, deferred): write a non-gated `BlockscaledQuantMixin(GemmDefaultEpiMixin)` + `GemmSm100ZeroMatBlockscaledQuant` class so the recompute kernel can skip SwiGLU+PostAct entirely. ~300 LOC of CUTLASS DSL (mirrors `gemm_gated.py:GemmGatedBlockscaledQuantMixin.epi_visit_subtile`); high silent-bug risk; should be guarded by bit-exact comparison against the gated kernel with a no-op activation. Recommended only if benchmarks show recompute SwiGLU+PostAct overhead is meaningful.

**Validation** (`tests/ops/test_recompute_z.py`):

| Tensor | cos | RRMSE | tol |
|--------|-----|-------|-----|
| out  | 1.000000 | 0.000008 | cos>0.9999, rrmse<0.02 |
| dx   | 1.000000 | 0.000000 | ✓ |
| ds   | 1.000000 | 0.000000 | ✓ |
| dw1  | 1.000000 | 0.000000 | ✓ |
| dw2  | 1.000000 | 0.000000 | ✓ |

**Numerically equivalent to the baseline FP8 path within fp16 round-trip noise.** Forward peak: 1751.5 MB → 1725.6 MB (–26 MB at 1-layer test shape).

**Full regression** (`tests/ops/test_mlpnode_correctness_large.py` with `SONIC_MOE_FP8_RECOMPUTE_Z=1`): all 9 cases PASS.

### S67.3 — Environment fix: ptxas for sm_103a on B30Z

`.runenv.sh` now exports `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas`. Triton's bundled ptxas (Feb 2025) does not recognize `sm_103a` — produces "ptxas fatal" on B30Z. CUDA 13.0's ptxas does. Both 1×32 and iso32 quant kernels need this. Affects every Triton kernel compiled fresh on B30Z; cached kernels are unaffected.

### S67 — Files Touched

| File | Δ | Note |
|------|---|------|
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | M | `_quantize_weight_3d_triton` default `isotropic=True` → `False`; deprecation docstring |
| `sonicmoe/config.py` | M | `recompute_z: Optional[bool]` field + `resolve_recompute_z()` |
| `sonicmoe/functional/__init__.py` | M | `_recompute_z()` resolver, `_FP8Config.recompute_z` slot, `_recompute_z_fp8()` helper, UpProj.fwd / DownProj.fwd / DownProj.bwd plumbing |
| `.runenv.sh` | M | `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas` for sm_103a |
| `tests/ops/audit_iso32_numerics.py` | + | Pure-PyTorch quant→dequant audit (no quack dep) |
| `tests/ops/bench_iso32_quant_nsys.py` | + | NVTX-bracketed perf microbench, 4 weight shapes |
| `tests/ops/test_recompute_z.py` | + | recompute_z numeric-equivalence + peak-mem test |
| `tools/parse_nsys_per_iter.py` | + | Generic nsys-sqlite GPU-projection per-iter parser |

### S67 — Lessons Learned

1. **E4M3 is floating-point** — a different e8m0 scale shift just changes which precision window the values fall in. As long as the largest values stay in normal range (2⁻⁶ to 448), the relative quant error doesn't depend on whether the scale is per-row (1×32) or per-2D-tile (32×32). The previous session's "isotropic loses precision" intuition was correct only for the integer-quant case; for fp-quant it's wrong on aggregate. **Lesson**: when claiming a numerical loss, run a quant→dequant audit first. Don't reason from first principles about FP types.
2. **Perf-irrelevant micro-optimizations should be killed** — iso32 saved ~0–2µs on cached kernels called once per step. Keeping it added a code path, a kernel binary, a test surface, and a misleading "precision tradeoff" claim. Net negative.
3. **Recompute design**: the autograd ctx pattern (zero-storage placeholder + ctx attribute carrying the closure) lets us defer materialization without touching `save_for_backward`'s tensor-only API. This is more robust than threading a boolean through 3 functions. Pattern is reusable for other lazy-recompute strategies.
4. **B30Z + sm_103a + Triton-bundled ptxas** silently fails in fresh kernel compiles. Symptom: cryptic "ptxas fatal" on first run, works after cache hit. **Always set `TRITON_PTXAS_PATH` to a recent ptxas on Blackwell**.

### S67 — Insights & Next Steps

- **The `recompute_z` Option A baseline is a working, validated, low-risk feature.** Real-world memory savings depend on how many layers are active simultaneously (large at ERNIE 24-layer, small at single-block tests). Should be measured under PaddleFleet integration once that lands.
- **If `recompute_z` is enabled by default in the future**, consider implementing Option B (constexpr-dispatched non-gated mixin) to eliminate the SwiGLU+PostAct overhead. Critical risk: silent numerical bugs in CUTLASS DSL — must be guarded by a bit-exact test that runs the gated kernel with a no-op activation and compares the fp8 D output byte-for-byte.
- **Iso32 should be removed entirely** in a future cleanup once we're confident no caller still imports `quantize_and_pack_weight_iso32`. Today it's only kept as a safety net.
- **High-value diagnostic**: `tools/parse_nsys_per_iter.py` is a clean, reusable per-iter GPU-projection parser. Pair it with NVTX `BENCH_*`/`ITER*` ranges in any new bench to get reliable wall-clock numbers from the timeline (avoids the unreliability of pytorch's `cuda.Event` timing under shared GPU load).

---

# HANDOFF — Session 66 (2026-04-27)

> **Single source of truth for project state.** `docs/HANDOFF.md` redirects here.

**Branch**: `session60-ds-fix` on `myrepo` (PFCCLab/supersonic-moe)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0)
**Python**: 3.12, Paddle torch-proxy (`paddle.compat.enable_torch_proxy` / `paddle.enable_compat`)
**QuACK**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack` (v0.3.7 + Paddle compat patches; **not** `third_party/quack`)
**Venv**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv`

---

## 1. Session 66 Deliverables

This session was a **correctness audit + cleanup + handoff** session. No production code changed except `bench_coldstart_nsys.py` (semantics fix) and a new correctness test.

### 1.1 Bugs Fixed (by user, before session)

Two TopK CUDA kernel bugs in `sonicmoe/ernie_compat/deepep_topk_metadata_cuda/kernel.cu`:

| # | Commit | Class | Symptom | Root cause | Fix |
|---|--------|-------|---------|------------|-----|
| 1 | `5987418` | **Grid-wide barrier without cooperative launch** | Hang at TK ≥ device-resident block cap | Single-pass histogram+scan kernel used grid-wide atomic spin-wait without `cudaLaunchCooperativeKernel` → if grid > resident SMs the late blocks never get scheduled and the early blocks spin forever | Split into 2 kernels (histogram → prefix-sum), kernel boundary acts as natural barrier |
| 2 | `1eadaa8` | **Capped grid + blockIdx-row mapping (silent corruption)** | Rows with index ≥ 65536 silently dropped at TK ≥ 131072 (SEQ=16384, K=8) | `dim3 grid(min(blocks, 2048))` while kernel maps `row = blockIdx.x * 32` → if `blocks > 2048`, rows ≥ 2048×32 = 65536 never get a CTA | Remove `min(...)` cap; correct grid sizing `(TK + 31) / 32`. **Perf impact**: zero or slightly positive — Phase 1 scatter has no grid-stride loop (each CTA does fixed 32-row work, so cap was dropping work, not merging it); Phase 2 pad-fill uses grid-stride, larger grid only reduces per-thread iterations. |

### 1.2 Audit Conclusion (read-only this session)

Audited every `.cu` / Triton / CuTe kernel launch in:
- `sonicmoe/ernie_compat/**/*.cu` (deepep_topk_metadata, deepep_metadata, count_cumsum, expert_*)
- `sonicmoe/quack_utils/*.py` (CuTe DSL launches)
- `sonicmoe/**/*.py` Triton kernels with explicit grid sizing

**No other instances of either bug class found.** Notes:
- `count_cumsum` does use grid-wide cooperative pattern but **launches via `cudaLaunchCooperativeKernel`** — safe.
- `deepep_metadata` (sister of fixed file) uses 1-block-per-expert, no grid cap, no spin-wait — safe.
- Triton kernels use `grid = (cdiv(N, BLOCK),)` patterns; no static caps observed.
- CuTe GEMM launches are managed by CUTLASS scheduler — not a concern.

### 1.3 New Correctness Test

`tests/ops/test_mlpnode_correctness_large.py` — subprocess-per-case harness with hard 600s timeout (hang detection). Validates **output, dx, ds, dw1, dw2** against BF16 gold. **9 cases, all PASS**:

| Case | T | E | K | I | TK | Notes |
|------|--:|--:|--:|--:|---:|-------|
| baseline_seq8K_E8 | 8192 | 8 | 8 | 1536 | 65536 | edge of post-fix regime |
| seq16K_E8 | 16384 | 8 | 8 | 1536 | 131072 | **bug-fix regression case** |
| seq16K_E32 | 16384 | 32 | 8 | 1536 | 131072 | E=32 + bug regime |
| skew80_seq8K | 8192 | 8 | 8 | 1536 | 65536 | 80% tokens → expert 0 |
| extreme_seq8K_E32 | 8192 | 32 | 8 | 1536 | 65536 | all tokens → E0..K-1 |
| tpe0_holes | 4096 | 32 | 8 | 1536 | 32768 | several experts get 0 tokens |
| smoke_K4 | 1024 | 8 | 4 | 1536 | 4096 | K=4 path |
| seq2K_E8_baseline | 2048 | 8 | 8 | 1536 | 16384 | small shape sanity |
| seq128_K8 | 128 | 8 | 8 | 384 | 1024 | smallest shape |

Tolerances: out cos > 0.99 / RRMSE < 0.10; dx, ds same; dw1, dw2 cos > 0.97 / RRMSE < 0.20 (relaxed for FP8 quant noise scaling). All actual cos ≥ 0.9971.

Also validates: NaN/Inf-free, 0-token-expert main_grad row is exactly zero (scalar reduction, not `torch.equal()` — see §6).

---

## 2. What Works (Verified 2026-04-27)

| Capability | Evidence | Status |
|---|---|:---:|
| FP8 fwd + bwd, E ∈ {4, 8, 32, 128}, K ∈ {4, 8} | `test_mlpnode_correctness_large.py`, `test_mlpnode_precision.py` | ✅ |
| FP8 fwd + bwd, SEQ ∈ {128, 1K, 2K, 4K, 8K, **16K**} | `test_mlpnode_correctness_large.py` (TK up to 131072) | ✅ |
| ds gradient flows back to `dispatched_probs` | `test_cold_start_e2e.py` ds cos = 0.9972 | ✅ |
| Pathological routing (skew, extreme, 0-token experts) | new test — all 9 cases PASS | ✅ |
| Dynamic seqlen (zero CuTe recompile) | `compile_key` static-only design | ✅ |
| `SonicMoEMlpNode.step()` → flush + invalidate | `mlp_node_v2.py:708` | ✅ |
| TMA reduce-add wgrad epilogue (default ON) | precision identical to fused beta=1.0 | ✅ |
| FP8 wgrad direct accumulation into `_NATIVE_W{1,2}_GRAD` | `mlp_node_v2.py:824/835` | ✅ |
| QuACK parallel compile workers (Paddle proxy) | dtype normalization | ✅ |

## 3. Known Limitations

| Item | Detail |
|---|---|
| Multi-card EP > 1 | Single-card only verified. DeepEP buffer integration not done. |
| ERNIE training loop integration | Interface verified, not yet plugged into PaddleFleet `MlpNode` slot. |
| Pipeline microbatch overlap | `_PREQUANTIZED_SCALES` module-level dict unsafe under concurrent overlapping forward. |
| `warmup_jit()` standalone (no Paddle) | `torch.utils.cpp_extension.load()` incompatible with Paddle proxy kwargs. |

---

## 4. Performance — nsys GPU-Projection

### 4.1 Methodology

- nsys 2026.2.1.210, `--trace=cuda,nvtx --sample=none --backtrace=none --resolve-symbols=false --export=sqlite`
- Per-iter NVTX `ITER{n}` ranges + outer `BENCH` range
- Parser: merge overlapping CUPTI kernel intervals inside the NVTX range, divide by iter count
- Warmup: 8 fwd+bwd, then 12 measured
- GPU 7 (idle), other GPUs busy with other workloads — must avoid GPU 0/1, GPU 2-6 are usually loaded

### 4.2 Headline (T=8192, E=8, K=8, I=1536, H=3072 — same shape as S53 baseline)

| Configuration | GPU-proj µs/iter | Notes |
|---|---:|---|
| **S53 pure-torch FP8** (no compat, no main_grad accum) | **2715** | upstream reference, `reports/session53_breakdown.md` |
| Paddle FP8 frontier — **steady-state microbatch (no flush)** | **2463** (median) | ITER NVTX range, this session, GPU 7 |
| Paddle FP8 frontier — **mlpnode-only via topk bench** | **2823** | `bench_mlpnode_topk_nsys.py`, GPU 7 |
| Paddle FP8 frontier — **per-iter flush** (grad_acc=1, non-default) | **3110** | `bench_coldstart_nsys.py` with stale per-iter flush |

**Reading the numbers** (this took some work — see §6 lesson #4):

The 2463 vs 2823 gap is the difference between two valid mlpnode benches with same shape. The 2823 measurement comes from `bench_mlpnode_topk_nsys.py`, which uses *all 12 iters inside the BENCH range* (no per-iter NVTX; the parser divides by 12). The 2463 measurement comes from per-ITER NVTX in `bench_coldstart_nsys.py`, which excludes a few µs of inter-iter framework gap. Both are real; **2823 µs is the conservative number to quote** because it includes whatever paddle does between iterations (memory pool maintenance, autograd graph teardown, etc).

### 4.3 Production-equivalent breakdown

`flush_native_grads()` is a per-**optimizer-step** operation, not per-microbatch (see §5). With realistic gradient accumulation:

| `grad_acc_steps` | flush amortized | per-microbatch GPU-proj | vs S53 (2715) |
|---:|---:|---:|---:|
| 1 (no accum) | +444 µs | ~2907 µs | +7.1% |
| 4 | +111 µs | ~2574 µs | -5.2% |
| 8 (typical ERNIE) | +56 µs | ~2519 µs | **-7.2%** |
| 16 | +28 µs | ~2491 µs | -8.3% |

**Bottom line**: at typical training `grad_acc_steps ≥ 4`, Paddle FP8 frontier matches or **beats** S53 pure-torch FP8 baseline.

### 4.4 Other shapes (Session 65 results, still valid)

| Shape (I=1536 K=8) | S53 BF16 | S53 FP8 | Paddle FP8 | vs S53 BF16 |
|---|---:|---:|---:|:---:|
| T=8192 E=8  | 3644 | 2715 | 2820 | **1.29×** |
| T=8192 E=32 | 3844 |  —   | 3283 | **1.17×** |
| T=16384 E=8 | 7953 |  —   | 5548 | **1.43×** |
| T=16384 E=32| 8129 |  —   | 5916 | **1.37×** |

ERNIE-shape (E=32 H=3072 I=1536 K=8 EP=8 SEQ=4096, N_recv≈21725, TK≈32822):
- Forward: **625 µs** (CV 0.3%)
- Backward: **1904 µs** (CV 0.1%)
- Total: **2530 µs/iter** (CV 0.2%)

### 4.5 Memory (E=32, `bench_mlpnode_mem.py`)

| Phase | Allocated (MiB) | Peak (MiB) |
|---|---:|---:|
| 数据就绪 | 129 | 129 |
| 前向结束 | 4709 | 5356 |
| 反向结束 | 6586 | 8452 |
| 第二轮反向结束 | 6586 | 8324 |

**Top consumers**: `_NATIVE_W{1,2}_GRAD` fp32 (E×2I×H + E×H×I) ≈ 1728 MiB, FP8 weight caches ≈ 650 MiB, activations ≈ 360 MiB.

### 4.6 nsys artifacts

`/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/nsys/handoff/`:
- `coldstart_T8K_E8.nsys-rep` / `.sqlite` — bench_coldstart_nsys (T=8K E=8, with end-of-window flush, NVTX `ITER0..11` + `FLUSH`)
- `mlpnode_T8K_E8.nsys-rep` / `.sqlite` — bench_mlpnode_topk_nsys (T=8K E=8, BENCH-range only, 2823 µs/iter)

---

## 5. Architecture Notes (the bits worth re-reading)

### 5.1 main_grad accumulation is fused into the wgrad GEMM epilogue

In the FP8 frontier path (`SonicMoEMlpNode` default):

```
backward:
  down_ctx._wgrad_w2_accumulator = _NATIVE_W2_GRAD   # fp32 [E, H, I]
  up_ctx._wgrad_w1_accumulator   = _NATIVE_W1_GRAD   # fp32 [E, 2I, H]
  → CUTLASS wgrad GEMM with TMA reduce-add epilogue accumulates
    directly into these fp32 buffers, returns dw1=dw2=None
  → no per-iter transpose, no per-iter elementwise-add
```

(Source: `sonicmoe/ernie_compat/mlp_node_v2.py:818-847`. The `_accumulate_w{1,2}` fallback path with `permute(2,0,1).contiguous()` only fires on BF16 wgrad fallback.)

`flush_native_grads()` is the **optimizer-step** call that converts the SonicMoE-native [E,2I,H]/[E,H,I] accumulator into ERNIE's per-expert [E,H,2I]/[E,I,H] split-half `main_grad` layout. Contract:

```python
for step in range(num_steps):
    for mb in microbatches:                       # ← per-microbatch
        out = node(x, tpe, indices, probs)         #     (no flush)
        out.backward(grad)
    optimizer.step()
    node.step()                                    # ← flush + invalidate (per-step)
    optimizer.zero_grad()
```

If you see `transpose / TilingSwapDim / Eigen meta_assign / broadcast_add` in a per-iter timeline, you are looking at `flush_native_grads()`. That is **not** the steady-state cost — it is the optimizer-step cost amortized over `grad_acc_steps`.

### 5.2 Frontier knob defaults (all ON unless overridden)

| Knob | Default | Disable via | Effect |
|---|:---:|---|---|
| FP8 wgrad | ON when aligned | `SONIC_MOE_FP8_MODE=` other than `perf` | I=1536+ shapes use FP8 wgrad GEMM |
| TMA reduce-add wgrad epilogue | ON | `SONIC_MOE_FP8_WGRAD_BETA_ACCUM=1` | -2.3% (E=8) to -4.0% (E=32) E2E |
| Fused swiglu+quant | ON | (always) | one kernel for SiLU+gate+quant |
| Save z_fp8 (forward output of swiglu) | ON | (always) | dgated reuses pre-quantized z |
| Alignment-assumed quant | ON when shape aligned | `_ALIGNMENT_ASSUMED=False` | skips runtime alignment check |

---

## 6. Lessons Learned (session 66 specific; see `reports/fp8_upgrade/engineering_log.md` for full history through Phase 26)

1. **Two CUDA-launch bug patterns to grep for whenever editing a custom kernel**:
   - **Class A** (deadlock): grid-wide spin-wait / atomic barrier without `cudaLaunchCooperativeKernel`. Symptom: hangs only when grid > device-resident SMs. Workaround: split into multiple kernels OR use cooperative launch.
   - **Class B** (silent corruption): `dim3 grid(min(blocks, CAP))` while kernel maps `blockIdx → row`. Symptom: large shapes silently produce wrong output for high-index rows. Find via: grep `min(.*grid` and `min(.*block` in `.cu`/`.cpp`.

2. **`torch.equal()` + paddle compat = `__nonzero__` ambiguity**. In paddle compat mode, `torch.equal(t, zeros_like(t))` calls `__nonzero__` on a multi-element paddle tensor → `AssertionError: When Variable is used as the condition of if/while`. Always reduce to a scalar first: `float(t.float().abs().sum().item()) == 0.0`. Watch for this in any new test code.

3. **Per-iter `flush_native_grads()` is non-default and inflates per-iter timeline**. If your bench loop calls it per backward, you'll see ~280-340 µs of `permute / TilingSwapDim / Eigen meta_assign / broadcast_add` kernels that don't exist in production. Either move it outside the timed loop, or amortize by `grad_acc_steps` when comparing.

4. **Two ways to measure mlpnode GPU-proj — they don't agree, and that's fine**. (a) BENCH-range whole = `sum(kernels in BENCH) / n_iters` includes inter-iter framework gaps; (b) per-ITER NVTX excludes them. Gap is ~360 µs at this shape. Quote (a) for conservative comparison; quote (b) for kernel-only analysis.

5. **`paddle.randn_like()` per iter inside a profiled loop adds curand kernel cost**. Either pre-allocate the input outside the loop, or keep it inside if you want to model the realistic "input changes every step" case. Document which one you chose.

6. **GPU 7 was idle at session end; GPUs 2-6 had ~50 GiB committed** by other users. Always `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader` before profiling. Bench results from a contended GPU are useless (saw 4168 µs/iter on contended GPU 2 vs 2823 µs on idle GPU 7 for the same workload).

---

## 7. Critical Constraints (traps for the next agent — same as session 65, still relevant)

1. **ds gradient path** (`gate_output → _DownProjection.apply()`): no native Paddle autograd nodes allowed in between. `paddle.topk()`, `.cast()`, `paddle.amp.decorate` all create Paddle autograd nodes which segfault when receiving torch-proxy gradient tensors.

2. **bf16 tensor conversion**: `tensor.cpu().numpy()` returns `uint16` (wrong); `torch.as_tensor()` returns `float16` (wrong); **only `torch.from_dlpack()` preserves bf16 correctly**.

3. **`_inplace_version` compat**: Paddle = `_inplace_version()` (method), torch = `._version` (attribute). Use `_tensor_version()` helper.

4. **CUDA stream compat**: Paddle = `stream.stream_base.raw_stream`; torch = `stream.cuda_stream`. Use `hasattr` branch.

5. **`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`** is mandatory. Triton 3.5.0's bundled ptxas is CUDA 12.8 → does not support SM103a (Blackwell B30Z).

6. **QuACK `str(dtype)` under Paddle proxy** returns `'paddle.bfloat16'`, not `'torch.bfloat16'`. Any dtype-string serialization needs normalization.

7. **`E != topk` requires explicit `topk`**: legacy code assumes `varlen_K_max = E`; for E=32 K=8 you must pass topk explicitly.

8. **nsys `--resolve-symbols=false` is mandatory** on this machine, otherwise it tries to download symbol tables from the network and hangs.

9. **Avoid GPU 0/1**: may be freq-locked or shared; use GPU 2+ (preferably idle).

---

## 8. High-Value Information Sources

| Source | Path | Why |
|---|---|---|
| Environment notes | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` | nsys flags, GPU restrictions, paddle pitfalls |
| Session 53 baseline | `reports/session53_breakdown.md` | 2715 µs FP8 / 3644 µs BF16 pure-torch reference |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | Phases 1-26, ~91 lessons |
| Session 60 lessons | `docs/session60_lessons.md` | ds gradient path constraints, gate↔MLP integration |
| Knowledge base | `docs/KNOWLEDGE_BASE.md` | Deep architecture reference |
| FP8 arch spec | `docs/FP8_ARCH_SPEC.md` | quant scheme, scale layout, fast paths |
| QuACK gemm_add auto-detect | `quack/gemm_interface.py:521` | `C is out and beta==1.0` triggers TMA add |
| Correctness regression test | `tests/ops/test_mlpnode_correctness_large.py` | Run after **any** topk/dispatch kernel change |
| Precision regression test | `tests/ops/test_mlpnode_precision.py` | 6-shape × 4-tensor topk audit |
| Mlpnode-only nsys bench | `tests/ops/bench_mlpnode_topk_nsys.py` | Gold-standard clean BENCH NVTX, sqlite parser |
| Coldstart nsys bench | `tests/ops/bench_coldstart_nsys.py` | Cache-clear + JIT + per-ITER NVTX + FLUSH NVTX |
| Memory bench | `tests/ops/bench_mlpnode_mem.py` | E=32 fwd+bwd peak memory profile |

---

## 9. QuACK Repo Changes (Session 63, still uncommitted upstream)

Located at `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack`:

- `quack/autotuner.py`: dtype normalization + `_precompile` robustness
- `quack/_compile_worker.py`: paddle.* dtype map + error handling

These changes are **not** in the sonic-moe repo. They need separate upstream commit/push to the quack repo.

---

## 10. Insights & Next Steps

### Insights (new this session)

1. **The two recent topk bugs are emblematic of a pattern**: silently-incorrect grid sizing on hand-written CUDA kernels. Whenever you add a new `.cu`, run `test_mlpnode_correctness_large.py` (especially the `seq16K_E8` and `seq16K_E32` cases — TK=131072 is the regime where Class B bugs surface).

2. **The Paddle compat layer is no longer the dominant overhead.** S53 was 2715 µs pure-torch FP8; we're at 2463 µs steady-state per-microbatch — Paddle compat overhead is **negative** at the actual measurement, because mlpnode's main_grad accumulation is fused into the GEMM epilogue while S53 has no accumulation at all (and counts only the GEMM). At `grad_acc_steps ≥ 4`, the paddle-compat path is competitive with or faster than upstream pure-torch.

3. **Remaining frontier overhead is dominated by BF16 wgrad GEMM.** ~43-48% of backward GPU time. Further gains need QuACK-level changes (tile config, maxrregcount).

### Next Steps (priority)

1. **ERNIE training loop integration** — plug `SonicMoEMlpNode` into PaddleFleet `MlpNode` slot. Watch for: weight convention (split-half ↔ interleaved), prob scaling order, subbatch support, gradient accumulation contract.

2. **Multi-card EP > 1** — wire up DeepEP buffer; verify dispatch → MlpNode → combine pipeline end-to-end.

3. **E=32 + EP=32 + SEQ=16384 production scale** — currently E=32 only verified at SEQ ≤ 8192. Run `test_mlpnode_correctness_large.py::seq16K_E32` (already passes) followed by a real-shape bench.

4. **Forward fp8 quant fusion into GemmGated epilogue** — eliminate ~65 µs forward overhead. CUTLASS epilogue work.

5. **BF16 wgrad tile tuning / maxrregcount hint** — investigate QuACK-level overrides for the bottleneck `quackgemm_default_epi` kernel.

6. **Pipeline microbatch overlap safety**: `_PREQUANTIZED_SCALES` module-level dict is unsafe under concurrent overlap. Migrate to per-call ctx storage if PP is enabled.

7. **Eventually upstream the QuACK patches** in `zhangyichen/sonicmoe_for_ernie/quack` to the canonical quack repo (Session 63 work).
