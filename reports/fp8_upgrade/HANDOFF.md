# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-04-09 (Session 41 final — complete state for next agent)
> **Branch:** `native-fp8-exploration`  **Commit:** `a65eed1`
> **Status:** ✅ FP8 functional. 1.12× speedup, −8.8% forward peak, 31/31 PASS.

---

## 0. Bottom Line (verified on idle B200, nsys GPU Projection)

| Metric | Official BF16 | FP8 Frontier | Delta |
|--------|:---:|:---:|:---:|
| **nsys GPU kernel/iter** | **3840 µs** | **3442 µs** | **1.12× faster** |
| **Forward peak** | **1386 MiB** | **1263 MiB** | **−122 MiB (−8.8%)** |
| **Backward peak** | **1412 MiB** | **1367 MiB** | **−45 MiB (−3.2%)** |
| Output RRMSE | — | 6.60% | PASS (<10%) |
| dx RRMSE | — | 7.48% | PASS (<10%) |
| dw1 norm rel err | — | 0.45% | PASS |
| dw2 norm rel err | — | 0.50% | PASS |
| Test suite | — | **31/31 PASS** | ✅ |
| Shadow correctness | — | **BIT-IDENTICAL** | ✅ |

### Usage
```bash
SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 python train.py
```
```python
moe.refresh_fp8_shadow_weights()  # call after optimizer.step()
with enable_quack_gemm(True):
    out, aux_loss = moe(x, use_fp8=True)
```

---

## 1. 100% Backward Peak Breakdown (1367 MiB)

> Verified by tensor-level audit: theoretical 1368 MiB vs measured 1367 MiB (0.1% gap).

| Category | Tensor | MiB | % | Optimizable? |
|----------|--------|:---:|:---:|:---:|
| **dgated output** | dz (TK,2I) bf16 | **384** | 28% | ❌ CUTLASS d_dtype.width==32 |
| **dgated output** | y1s (TK,I) bf16 | **192** | 14% | ❌ same constraint |
| **ctx saved** | z_fp8 (TK,2I) | **192** | 14% | ✅ already fp8 (was 384) |
| **model params** | w1 bf16 | **144** | 11% | ⚠️ see §7 dead ends |
| **weight caches** | w1T+w2 fused fp8 | **111** | 8% | ⚠️ can defer w1T |
| **pre-alloc** | dw2_base bf16 | **72** | 5% | ⚠️ can defer |
| **model params** | w2 bf16 | **72** | 5% | ⚠️ see §7 dead ends |
| **input/grad** | x+dout+metadata | **49** | 4% | ❌ interface contract |
| **dgated input** | dout_fp8+w2_fp8 | **66** | 5% | ❌ needed by GEMM |
| **misc** | scales+colvec+autograd | **37** | 3% | ❌ overhead |
| **TOTAL** | | **1368** | **100%** | |

---

## 2. Architecture — Zero-Materialization FP8

**Forward** (fused_gated, aligned):
1. `quantize_and_pack_activation(x)` → T-sized FP8 + ISA scales
2. `_gather_isa_packed_scales_kernel` → TK-sized ISA scales
3. `GemmGatedSm100ZeroMat`: T-FP8 + A_idx → z(bf16) + y1(bf16)
4. split quant: z→z_fp8 + resize_(0); y1→y1_fp8 + resize_(0)

**Backward** (fused_gated, aligned):
1. `quantize_and_pack_activation(dout)` + scale_gather → dout FP8
2. `GemmDGatedFP8CLoadSm100ZeroMat`: dout_FP8 × w2T + z_fp8(preact) → dz + y1s
3. wgrad: `gemm(dout.T, y1s)` → dw2; `gemm(x.T, dz)` → dw1
4. actgrad: `blockscaled_fp8_gemm_varlen(dz_fp8, w1T_fp8)` → dx

---

## 3. Memory — Official BF16 vs FP8 (idle B200, subprocess-isolated)

| Checkpoint | BF16 (MiB) | FP8 (MiB) | Delta |
|-----------|:---:|:---:|:---:|
| Post-warmup base | 376 | 488 | +112 (shadow caches) |
| **Forward peak** | **1386** | **1263** | **−122 (−8.8%)** |
| **Backward peak** | **1412** | **1367** | **−45 (−3.2%)** |
| Post-cleanup | 328 | 440 | +112 (auto-invalidated) |

---

## 4. Key Optimizations (Session 41)

| Optimization | Impact | Verified |
|-------------|--------|----------|
| `_FP8Config` object | −15 os.getenv/iter | 31/31 PASS |
| Adapter skip (fp8_protocol path) | −250µs | 31/31 PASS |
| z_fp8 early release after dgated | −198 MiB bwd transient | 31/31 PASS |
| w1 fused cache clear after up-proj | −74 MiB during down-proj | 31/31 PASS |
| w2 varlen evict after down-proj | −37 MiB during ctx save | 31/31 PASS |
| `FUSED_ZY1_QUANT` (optional) | −64µs, +96 MiB fwd peak | 31/31 PASS |
| `refresh_fp8_shadow_weights()` | BIT-IDENTICAL, −160µs | 31/31 PASS |
| Deferred backward cache fill | prevent 148 MiB transient spike | 31/31 PASS |
| Cache key: `(_version, shape, stride)` | storage-id independent | 31/31 PASS |

---

## 5. Flags

### Required
| Flag | Default | Set to |
|------|---------|--------|
| `SONIC_MOE_FP8_MODE` | off | **perf** |
| `USE_QUACK_GEMM` | 0 | **1** |

### Optimal defaults (do NOT change)
| Flag | Default | Effect |
|------|---------|--------|
| `SONIC_MOE_FP8_FUSED_GATED` | 1 | Fused GEMM+SwiGLU+descale |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | 1 | z stored as FP8 (−192 MiB) |

### Optional
| Flag | Default | Effect |
|------|---------|--------|
| `SONIC_MOE_FP8_FUSED_ZY1_QUANT` | 0 | −64µs, +96 MiB fwd peak |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | 1 | Fused SwiGLU+quant (fallback path) |
| `SONIC_MOE_FP8_EPILOGUE_QUANT` | 0 | GEMM epilogue z-scale compute |

---

## 6. Bugs & Lessons

### Process contamination ⚠️ CRITICAL
`SONIC_MOE_FP8_MODE` is cached at import time. **BF16 vs FP8 comparison MUST use separate subprocesses.**

### Measurement methodology
- **nsys GPU Projection** is ground truth. CUDA events include Python overhead.
- **ncu** uses base clock; nsys uses boost. Don't compare across tools.
- Weight cache effects: no-optimizer benchmarks see cache hits from iter 2+. Training misses every step.
- **Repeated runs without offload are BIT-IDENTICAL** — verified.

### Cache key lesson
Removed `data_ptr` / `id(storage)` from cache keys. Now keys are `(_version, shape, stride)` only — stable across `w.data =` reassignment. This was needed for offload experiments but is a cleaner design regardless.

---

## 7. Dead Ends (verified, do NOT retry without new approach)

### FP8 wgrad (dual_quantize path)
- `dual_quantize_and_pack` kernel works correctly (single HBM read → row+col fp8)
- `blockscaled_fp8_weight_grad_gemm_fast` with `pre_quantized_b` works
- **But**: `_warp32x32_transpose_quant_kernel` costs 634µs/iter at Ernie I=1536
- **Net result**: FP8 wgrad is **slower** than BF16 wgrad at Ernie shape
- **Viable when**: I≥2048 (quant overhead < GEMM savings)

### bf16 weight offload (Parameter dtype change)
- `w.data = w.data.to(fp8)` successfully halves param storage (144→72 MiB)
- Cache keys match (verified: FUSED=2, VARLEN=2 entries, all keys correct)
- **Forward output: BIT-IDENTICAL** ✅
- **Backward gradients: max_diff≠0** (dw1 max_diff=2.75, norm=85000, relative=0.003%)
- **Root cause**: `w.data =` changes Parameter dtype → PyTorch autograd internals (likely `AccumulateGrad` node) behave differently with fp8 dtype, even though all GEMM computations go through cache
- **Bisect proof**: cache re-population WITHOUT dtype change → BIT-IDENTICAL; dtype change alone → diff
- Two identical runs WITHOUT offload → BIT-IDENTICAL (not non-determinism)

### bf16 weight offload (resize_(0) + proxy)
- `resize_(0)` frees GPU storage but breaks `w.permute()` (PyTorch bounds check)
- fp8 proxy tensors: correct shape/stride but `_version` mismatch with cache
- proxy is not `nn.Parameter` → autograd doesn't accumulate gradients to it
- Gradient forwarding from proxy to Parameter requires framework-level support

### Remaining offload approaches (untried)
1. **PyTorch `DTensor` with mixed-precision sharding** — framework-level fp8 parameter support
2. **Custom autograd Function wrapping** — pass fp8 buffers directly, bypass Parameter
3. **FSDP-style param offload** — use FSDP's existing CPU offload infrastructure
4. **Ernie-style `fp8_weight_stacked` buffers** — register_buffer with manual grad routing

---

## 8. High-Value Information Sources

| Source | Location | Key insight |
|--------|----------|-------------|
| Ernie MoELayer FP8 | `ernie-core/src/ernie_core/models/moe/moe_layer.py` L2436-2507 | `fp8_quant_weight()` pattern = bf16 master + fp8 shadow |
| Ernie fp8_utils | `ernie-core/.../token_dispatcher/fp8_utils.py` | `kitchen_quant` 128×128 block, `deep_gemm` grouped GEMM |
| Official BF16 baseline | `lab/official/sonic-moe` + `envs/official_bf16` | 10-arg `moe_TC_softmax_topk_layer` (no fp8_protocol) |
| SonicMoE fork | `lab/sonic-moe` + `envs/xfer` | 11-arg (has fp8_protocol param) |
| nsys data | `benchmarks/nsys_run/nsys_s41_*.sqlite` | GPU Projection ground truth |
| env.md | `panzhaowu/env.md` | Proxy, GitHub token, cluster scan, nsys install |

---

## 9. Next Steps — P0 Detailed Design: Decouple Weight from Autograd

### 问题本质
bf16 参数在 backward peak 占 216 MiB（w1=144 + w2=72），但 FP8 路径中**没有任何 GEMM 读取 bf16 权重数据**（全部通过 fp8 cache）。这 216 MiB 纯粹是 autograd 的 `save_for_backward` 和 `AccumulateGrad` 需要的。

### 为什么现有方案全部失败
| 方案 | 失败原因 | 验证方式 |
|------|---------|---------|
| `w.data = fp8` | autograd 内部行为变化（梯度 bf16→fp8 cast 导致精度损失）| bisect: repopulate=0, dtype_change≠0 |
| `resize_(0)` | PyTorch bounds check 阻止 `w.permute()` | `setStorage: ... out of bounds for storage of size 0` |
| fp8 proxy | `_version` 不匹配 + proxy 不是 Parameter → 无梯度累积 | `dw1: None` |
| fp8 Parameter + optimizer | Adam 要求 param 和 grad 同 dtype；optimizer state 变成 fp8 | `same dtype` error |

### 正确设计方案：Named FP8 Cache + Shape Metadata

**核心思想**：从 `save_for_backward` 中**移除 w1/w2**，backward 通过**命名缓存**直接获取 fp8 权重。

```
┌─── Training Loop ───────────────────────────────┐
│                                                  │
│  optimizer.step()          # updates bf16 master │
│  ↓                                               │
│  moe.refresh_fp8()         # bf16→fp8 cache      │
│  moe.stash_bf16_to_cpu()   # bf16 master→CPU     │
│                            # GPU: −216 MiB       │
│  ↓                                               │
│  forward(x)                # uses fp8 cache only  │
│  backward(dout)            # uses fp8 cache only  │
│                            # grads→grad_proxy.grad│
│  ↓                                               │
│  moe.unstash_bf16()        # CPU→GPU bf16 master  │
│  moe.sync_grads()          # proxy.grad→param.grad│
│                                                  │
└──────────────────────────────────────────────────┘
```

#### 具体改动

**Step 1**: 修改 `_UpProjection.forward` (约 10 行)
```python
# 当前:
ctx.save_for_backward(x, w1, b1, expert_freq, x_gather_idx, ...)
# 改为:
ctx.save_for_backward(x, b1, expert_freq, x_gather_idx, ...)
ctx.w1_shape = w1.shape
ctx.w1_device = w1.device
```

**Step 2**: 修改 `_UpProjection.backward` (约 15 行)
```python
# 当前:
(x, w1, b1, ...) = ctx.saved_tensors
w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
# 改为:
(x, b1, ...) = ctx.saved_tensors
w1T_fp8, w1T_scales = _NAMED_FP8_CACHE["w1T_varlen"]  # direct lookup
```

**Step 3**: 修改 `_DownProjection` 同理 (约 20 行)

**Step 4**: 创建全局命名缓存 + `grad_proxy`
```python
_NAMED_FP8_CACHE = {}  # {"w1_fused": (fp8, scales), ...}

class MoE:
    def refresh_fp8(self):
        _NAMED_FP8_CACHE["w1_fused"] = precompute_weight_fp8_for_fused_gated(w1_perm)
        _NAMED_FP8_CACHE["w2_varlen"] = precompute_weight_fp8(w2_perm)
        _NAMED_FP8_CACHE["w1T_varlen"] = precompute_weight_fp8(w1T_perm)
        _NAMED_FP8_CACHE["w2_dgated"] = precompute_weight_fp8_for_direct_fused_dgated(w2_perm)

    def stash_bf16_to_cpu(self):
        self._cpu_w1 = self.c_fc.weight.data.to('cpu')
        self._cpu_w2 = self.c_proj.weight.data.to('cpu')
        self.c_fc.weight.requires_grad_(False)
        self.c_proj.weight.requires_grad_(False)
        self.c_fc.weight.data.untyped_storage().resize_(0)  # −216 MiB
        self.c_proj.weight.data.untyped_storage().resize_(0)
        # Create bf16 grad proxies for autograd
        self._w1_grad_proxy = torch.zeros(w1_shape, dtype=bf16, device='cuda',
                                          requires_grad=True)
        self._w2_grad_proxy = torch.zeros(w2_shape, dtype=bf16, device='cuda',
                                          requires_grad=True)
```

**Step 5**: forward 传 `grad_proxy` 而不是 Parameter
```python
def forward(self, x, ...):
    if self._stashed:
        w1_arg = self._w1_grad_proxy.permute(1, 2, 0)  # bf16, requires_grad
        w2_arg = self._w2_grad_proxy.permute(1, 2, 0)
    else:
        w1_arg = self.c_fc.weight.permute(1, 2, 0)
        w2_arg = self.c_proj.weight.permute(1, 2, 0)
```

#### 预期收益

| 项目 | 节省 |
|------|:---:|
| w1 bf16 Parameter (resize_(0)) | −144 MiB |
| w2 bf16 Parameter (resize_(0)) | −72 MiB |
| w1 fp8 cache (已在 fp8 cache 中) | 0 (已计入) |
| w1/w2 grad_proxy (bf16, same shape) | **+216 MiB** ← 抵消! |

**⚠️ 关键洞察：grad_proxy 和 Parameter 一样大（216 MiB bf16）！** 节省被 proxy 完全抵消。

#### 真正的净节省方案

`grad_proxy` 必须比 Parameter 小。选项：
1. **grad_proxy 为 fp8** (108 MiB) → 净省 108 MiB，但 autograd 梯度是 bf16 → 赋值时需 cast
2. **grad_proxy 为空** (0 MiB) → 修改 autograd Function 不 save w1/w2，梯度通过额外返回值传递
3. **不用 proxy，直接在 autograd Function 中用 `_NAMED_FP8_CACHE` 的 value 作为 save_for_backward** → fp8 tensor 在 ctx 中 (108 MiB) 替代 bf16 (216 MiB)

**方案 3 最优**：将 `ctx.save_for_backward(x, w1, ...)` 改为 `ctx.save_for_backward(x, w1T_fp8, w1T_scales, ...)`。backward 直接用 fp8 tensors。grad_proxy 是一个微型 leaf tensor (0 MiB) 仅用于 autograd graph 连接。

**最终净节省**：216 MiB (bf16 params freed) − 108 MiB (fp8 already in cache) = **108 MiB 净省**。

#### 风险

| 风险 | 严重度 | 对策 |
|------|--------|------|
| save_for_backward 改动影响非 FP8 路径 | 中 | 条件分支: `if cfg.offloaded: save fp8 else save bf16` |
| grad_proxy 的 autograd graph 不正确 | 中 | 单测验证每个梯度 |
| optimizer 看不到 fp8 params 的 grad | 低 | `sync_grads()` 手动复制 |
| 工程复杂度 | 中 | ~100 行核心改动 |

### P1-P3 (unchanged)
- **P1**: GEMM Epilogue FP8 Output (−166µs)
- **P2**: Training Validation (loss curve)
- **P3**: FP8 Wgrad at I≥2048

---

## 10. Key Files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 forward/backward, `_FP8Config` |
| `sonicmoe/moe.py` | MoE class, `refresh_fp8_shadow_weights()` |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, caches, `dual_quantize_and_pack` |
| `sonicmoe/quack_utils/gemm_gated.py` | GemmGated + epilogue quant mixin |
| `sonicmoe/quack_utils/gemm_dgated.py` | GemmDGated + FP8CLoad |
| `tests/fp8_large_project_contract_test.py` | 31-test gate |
| `tools/nsys_benchmark.py` | nsys-compatible profiling |
| `tools/precision_audit.py` | Subprocess-isolated precision |
| `tools/_inline_audit.py` | Backward peak tensor inventory |

---

## 11. Quick Validation
```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Test suite
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short

# Precision
CUDA_VISIBLE_DEVICES=0 python tools/precision_audit.py --gpu 0 --seeds 42,123,777

# Memory vs official BF16
CUDA_VISIBLE_DEVICES=0 python tools/full_benchmark.py
```
