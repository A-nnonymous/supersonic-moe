# HANDOFF — Session 65 (2026-04-24)

> **Single source of truth.** `docs/HANDOFF.md` redirects here.

**Branch**: `session60-ds-fix` on `fork` (PFCCLab/supersonic-moe)
**Hardware**: NVIDIA B30Z (SM103, CUDA 13.0)
**Python**: 3.12, Paddle torch-proxy (`paddle.enable_compat()`)
**QuACK**: `/root/.../zhangyichen/sonicmoe_for_ernie/quack` (v0.3.7, with Paddle compat patches)

---

## 1. What Works (Verified 2026-04-24)

| Capability | Evidence | Status |
|---|---|:---:|
| FP8 fwd + bwd (E=8, topk=8) | `test_mlpnode_precision.py`, `bench_mlpnode_mem.py` | PASS |
| FP8 fwd + bwd (E=32, topk=8, has -1 entries) | `bench_mlpnode_mem.py` E=32 with real EP routing | PASS |
| FP8 fwd + bwd (E=128, topk=8) | `/tmp/test_all_e_seq.py` | PASS |
| Precision: all cos > 0.99 (6 shapes incl. E=32) | `test_mlpnode_precision.py` 6 shapes × 4 tensors | PASS |
| ds gradient flows back to dispatched_probs | `test_cold_start_e2e.py` ds cos=0.9972 | PASS |
| Dynamic seqlen (zero CuTe recompile) | compile_key static-only design | PASS |
| `SonicMoEMlpNode.step()` API | flush grads + invalidate caches | PASS |
| QuACK parallel compile workers (Paddle proxy) | dtype normalization fix in autotuner | PASS |
| TMA reduce-add wgrad epilogue | 2-4% E2E improvement, bitwise deterministic | PASS |

## 2. What Doesn't Work / Known Issues

| Item | Detail |
|---|---|
| `warmup_jit()` standalone (no Paddle) | `torch.utils.cpp_extension.load()` incompatible with Paddle proxy kwargs |
| Multi-card EP>1 | Requires DeepEP buffer; single-card only tested |
| ERNIE training loop plug-in | MlpNode interface verified, not yet in actual ERNIE training |
| Pipeline microbatch overlap | `_PREQUANTIZED_SCALES` module-level dict unsafe under overlap |
| `test_cold_start_e2e.py` K=4 timing | K=4 JIT takes >5s (8.3s), triggers timing FAIL — precision is correct |

## 3. Performance — GPU-Projection (nsys sqlite)

### Session 65: TMA Reduce-Add wgrad epilogue (default)

| Shape (I=1536, K=8) | S53 BF16 (µs) | Paddle FP8 (µs) | vs BF16 |
|---|:---:|:---:|:---:|
| T=8192 E=8 | 3644 | **2820** | **1.29x** |
| T=8192 E=32 | 3844 | **3283** | **1.17x** |
| T=16384 E=8 | 7953 | 5548 | **1.43x** |
| T=16384 E=32 | 8129 | 5916 | **1.37x** |

### TMA Reduce-Add vs Fused Beta=1.0 (Session 65 optimization)

| Shape | Baseline beta=1.0 (µs) | TMA add (µs) | Improvement |
|---|:---:|:---:|:---:|
| T=8192 E=8 | 2886 | 2820 | **-65 µs (-2.3%)** |
| T=8192 E=32 | 3420 | 3283 | **-138 µs (-4.0%)** |

BF16 wgrad GEMM kernel (`quackgemm_default_epi`, 4 calls/iter):
- E=8: 321→305 µs/call (-5.0%), E=32: 429→396 µs/call (-7.7%)

**Mechanism**: `add_to_output=True` in QuACK triggers TMA hardware `CopyReduceBulkTensorTileS2GOp(ReductionOp.ADD)` on store. No C tensor load → no `epi_c_stage` → regs 86→50 → SM occupancy ~2-3→~5 blocks/SM.

**Determinism**: Safe because `tile_count_semaphore=None` (no split-K) — each CTA exclusively owns its output tiles. Verified bitwise-identical in isolated benchmark.

**Fallback**: `SONIC_MOE_FP8_WGRAD_BETA_ACCUM=1` reverts to legacy fused beta=1.0 epilogue.

nsys-rep files: `reports/wgrad_tma_add_nsys/*.nsys-rep`

### ERNIE-Shape (E=32, H=3072, I=1536, K=8, EP=8, SEQ=4096)

N_recv ≈ 21725, TK ≈ 32822, TK_padded ≈ 34816

| Phase | GPU-proj (µs) | CV |
|---|:---:|:---:|
| **Forward** | **625** | 0.3% |
| **Backward** | **1904** | 0.1% |
| **Total (fw+bw)** | **2530** | 0.2% |

### Memory (E=32, bench_mlpnode_mem.py)

| Phase | Allocated (MiB) | Peak (MiB) |
|---|:---:|:---:|
| 数据就绪 | 129 | 129 |
| 前向结束 | 4709 | 5356 |
| 反向结束 | 6586 | 8452 |
| 第二轮反向结束 | 6586 | 8324 |

主要显存消耗：
- native_grad buffers (_NATIVE_W1/W2_GRAD): fp32, E×2I×H + E×H×I ≈ 1728 MiB
- FP8 weight caches: ~650 MiB
- 激活 (x + z_fp8 + output): ~360 MiB

### Precision (Session 65, test_mlpnode_precision.py, 6 shapes)

| Shape | out cos | dx cos | dw1 cos | dw2 cos | RRMSE (max) |
|---|:---:|:---:|:---:|:---:|:---:|
| N=128 K=4 E=4 I=384 | 0.9979 | 0.9975 | 0.9975 | 0.9972 | 7.5% |
| N=128 K=8 E=8 I=384 | 0.9979 | 0.9975 | 0.9975 | 0.9971 | 7.6% |
| N=512 K=4 E=8 I=1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 | 7.5% |
| N=512 K=8 E=8 I=1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 | 7.5% |
| N=1024 K=8 E=8 I=1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 | 7.5% |
| N=256 K=8 E=32 I=1536 | 0.9979 | 0.9975 | 0.9975 | 0.9971 | 7.6% |

ds gradient (test_cold_start_e2e.py): cos=0.9972 across all 6 shapes.

## 4. Session 65 Changes (TMA Reduce-Add Optimization)

### What changed

1. **New function `_run_cutlass_blockscaled_gemm_varlen_k_tma_add()`** in `blockscaled_fp8_gemm.py`:
   - Uses `add_to_output=True` + `C=GemmTensorInfo(None)` instead of `beta=1.0` + loaded C tensor
   - Separate caches: `_COMPILE_CACHE_VK_TMA_ADD`, `_GEMM_FAST_PATH_VK_TMA_ADD`
   - Registers: ~50 (vs 86 for fused beta=1.0)

2. **Replaced all 4 wgrad accumulate calls** in `functional/__init__.py`:
   - FP8 w1 wgrad (~line 1248): `_accumulate` → `_tma_add`
   - FP8 w2 wgrad (~line 1900): `_accumulate` → `_tma_add`
   - BF16 w1 wgrad (~line 1290): `gemm(beta=1.0)` → `gemm_add(C=accum, out=accum, beta=1.0)`
   - BF16 w2 wgrad (~line 1948): same

3. **Env flag `SONIC_MOE_FP8_WGRAD_BETA_ACCUM`** (default off): reverts to legacy fused epilogue

4. **New benchmark** `tests/ops/bench_wgrad_epilogue.py`: isolated A/B comparison, 4 production shapes

### QuACK `gemm_add()` auto-detection mechanism
When `C is out` (identity check) and `beta==1.0` and `cu_seqlens_m is None`, `gemm_add()` automatically uses `add_to_output=True` with `C=None`. This is the BF16 wgrad path. The FP8 path uses `_tma_add()` directly.

## 5. Bugs Fixed (Sessions 63-64, prior to this session)

### Bug 1: `_build_score_src_idx_kernel` PTXASError
- E=32 crash from `tl.min(vector, axis=0)` on SM103a (Triton bundled ptxas=CUDA 12.8)
- Fix: scalar selection sort with WORK_ptr scratch buffer

### Bug 2: `varlen_K_max = E` instead of `topk`
- E=32,topk=8 crash (MAX_K=32 vs correct 8). Masked when E==topk==8
- Fix: `varlen_K_max=(K if K is not None else E)` + `_topk` class var

### Bug 3: QuACK `_compile_worker.py` Paddle dtype
- `'paddle.bfloat16'` in `_dtype_map` → KeyError → BrokenPipeError
- Fix: dtype normalization + paddle.* entries + robustness hardening

## 6. Critical Constraints (给下一个 agent 的陷阱警告)

1. **ds gradient path**: gate output → `_DownProjection.apply()` 之间，**不能有** native Paddle autograd 节点。Paddle `topk()`, `.cast()`, `amp.decorate` 都会创建 autograd 节点，收到 torch-proxy gradient tensor 时会 segfault。

2. **Paddle bf16 tensor 转换**: `tensor.cpu().numpy()` 返回 `uint16`（错误）。`torch.as_tensor()` 返回 `float16`（错误）。**只有 `torch.from_dlpack()` 正确保留 bf16。**

3. **`_inplace_version` 兼容**: Paddle = `_inplace_version()` (method), PyTorch = `._version` (attribute)。用 `_tensor_version()` helper。

4. **CUDA stream 兼容**: Paddle = `stream.stream_base.raw_stream`, PyTorch = `stream.cuda_stream`。用 `hasattr` 分支。

5. **`TRITON_PTXAS_PATH`**: 必须设为 `/usr/local/cuda/bin/ptxas`。Triton 3.5.0 bundled ptxas 是 CUDA 12.8，不支持 SM103a (Blackwell)。

6. **QuACK `str(dtype)` under Paddle proxy**: 返回 `'paddle.bfloat16'` 不是 `'torch.bfloat16'`。任何序列化 dtype 字符串的代码都要做 normalization。

7. **`E != topk` 时必须显式传 topk**: 旧代码假设 E==topk。当 E=32, topk=8 时，`varlen_K_max` 必须用 topk 而非 E。

8. **nsys `--resolve-symbols=false`**: 必须加此 flag，否则 nsys 会尝试从网络下载符号表而 hang。参见 `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md`。

9. **不要使用 GPU 0-1**: 可能被 freq-locked 或被其他进程占用。测试/profiling 始终使用 GPU 2+。

## 7. High-Value Information Sources

| Source | Path | Why |
|---|---|---|
| 环境教训 | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` | nsys flags, GPU 限制, Paddle 兼容性 pitfalls |
| 精度测试 | `tests/ops/test_mlpnode_precision.py` | 6-shape × 4-tensor topk precision audit (Session 65 updated) |
| 冷启动测试 | `tests/ops/test_cold_start_e2e.py` | 6-shape × 5-tensor (incl. ds) + timing + memory |
| wgrad epilogue bench | `tests/ops/bench_wgrad_epilogue.py` | Isolated TMA add vs fused beta A/B comparison |
| nsys profiling脚本 | `tests/ops/bench_mlpnode_topk_nsys.py` | Clean MLP-node-only nsys profile with NVTX BENCH range |
| nsys 结果 | `reports/wgrad_tma_add_nsys/RESULTS.json` | Session 65 TMA optimization A/B GPU-projection data |
| QuACK gemm_add auto-detect | `quack/gemm_interface.py:521` | `gemm_add()` auto-detects `add_to_output` when `C is out` |
| Engineering log | `reports/fp8_upgrade/engineering_log.md` | 86 lessons across 25 phases (Sessions 1-65) |
| QuACK autotuner | `quack/autotuner.py` | Session 63 修复了 Paddle dtype compat + robustness |

## 8. QuACK 仓库改动 (Session 63)

**仓库位置**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack`

改动文件:
- `quack/autotuner.py`: dtype normalization + _precompile robustness
- `quack/_compile_worker.py`: paddle.* dtype map + error handling

**注意**: 这些改动在 quack 仓库中，不在 sonic-moe 仓库中。需要单独 commit/push 到 quack 仓库。

## 9. Insights & Next Steps

### Insights

1. **TMA reduce-add is free performance.** No precision change, no functional change, just lower register pressure from not loading C tensor. 2-4% E2E improvement with zero downside.

2. **BF16 wgrad GEMM is the remaining bottleneck.** After TMA optimization, the `quackgemm_default_epi` kernel still takes 43-48% of total backward GPU time. Further optimization would require QuACK-level changes (maxrregcount hints, tile size tuning).

3. **vs S53 native PyTorch gap reduced.** T=8192 E=8 went from 1.06x slower to 1.04x slower vs S53 FP8. T=8192 E=32 went from 1.15x to 1.12x. The remaining gap is the inherent cost of fp32 main_grad accumulation (S53 has no accumulation).

4. **FP8 wgrad kernel (`_tma_add`) shows 6-13% speedup in isolation.** But it's only 2 calls/iter and fast (~600µs total), so E2E contribution is small vs the 4 BF16 wgrad calls (~1200-1600µs).

5. **Host Python overhead 不影响 GPU pipeline**: CUDA events ~5.4ms/iter 但 GPU-projection 只有 2.8ms。GPU 利用率约 52%（单 MLP layer），在完整 transformer 中会更高。

### Next Steps (Priority)

1. **ERNIE training loop 集成** — 将 `SonicMoEMlpNode` 接入 PaddleFleet MlpNode 插槽。关键: weight convention (split-half ↔ interleaved), prob scaling order, subbatch support。

2. **E=32 生产规模验证** — 当前 E=32 用 SEQ=4096 验证通过。需要用完整 SEQ=16384+EP=32 规模验证显存和延迟。

3. **Epilogue forward quantization** — 将 x→fp8 量化融入 GemmGated epilogue，消除 ~65µs forward overhead。需要 CUTLASS epilogue modification。

4. **Multi-card EP>1** — 接入 DeepEP buffer，验证 dispatch→MlpNode→combine pipeline。

5. **BF16 wgrad tile tuning** — 当前 BF16 wgrad GEMM 用 default tile config。调查 QuACK 是否支持 tile shape override 或 maxrregcount hint。
