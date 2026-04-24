# HANDOFF — Session 63 (2026-04-24)

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
| Precision: all cos > 0.99 | `test_mlpnode_precision.py` 4 shapes × 5 tensors | PASS |
| ds gradient flows back to dispatched_probs | Triton `_build_score_src_idx_kernel` | PASS |
| Dynamic seqlen (zero CuTe recompile) | compile_key static-only design | PASS |
| `SonicMoEMlpNode.step()` API | flush grads + invalidate caches | PASS |
| QuACK parallel compile workers (Paddle proxy) | dtype normalization fix in autotuner | PASS |

## 2. What Doesn't Work / Known Issues

| Item | Detail |
|---|---|
| `warmup_jit()` standalone (no Paddle) | `torch.utils.cpp_extension.load()` incompatible with Paddle proxy kwargs |
| Multi-card EP>1 | Requires DeepEP buffer; single-card only tested |
| ERNIE training loop plug-in | MlpNode interface verified, not yet in actual ERNIE training |
| Pipeline microbatch overlap | `_PREQUANTIZED_SCALES` module-level dict unsafe under overlap |

## 3. Performance — GPU-Projection (nsys sqlite, verified 2026-04-24)

### ERNIE-Shape (E=32, H=3072, I=1536, K=8, EP=8, SEQ=4096)

N_recv ≈ 21725, TK ≈ 32822, TK_padded ≈ 34816

| Phase | GPU-proj (µs) | CV |
|---|:---:|:---:|
| **Forward** | **625** | 0.3% |
| **Backward** | **1904** | 0.1% |
| **Total (fw+bw)** | **2530** | 0.2% |

### Forward Kernel Breakdown (625 µs, 34 kernels)

| Kernel | µs | % |
|---|:---:|:---:|
| CUTLASS FP8 ZeroMat GEMM (UpProj) | 275 | 44.1% |
| CUTLASS GEMM (DownProj) | 131 | 21.0% |
| FP8 quantize_and_pack | 65 | 10.4% |
| token_gather_sum (router scatter) | 57 | 9.1% |
| _build_score_src_idx (topk sort) | 30 | 4.7% |
| histogram_and_prefix (metadata) | 22 | 3.5% |

### Backward Kernel Breakdown (1903 µs, 17 kernels)

| Kernel | µs | % |
|---|:---:|:---:|
| CUTLASS GEMM wgrad | 814 | 42.8% |
| CUTLASS varlen accumulate (wgrad acc) | 664 | 34.9% |
| CUTLASS FP8 ZeroMat GEMM (actgrad) | 253 | 13.3% |
| FP8 quantize_and_pack | 91 | 4.8% |
| token_gather_sum (scatter bw) | 55 | 2.9% |

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

### Precision (test_mlpnode_precision.py, FP8 vs BF16 gold)

| Shape | out cos | dx cos | dw1 cos | dw2 cos |
|---|:---:|:---:|:---:|:---:|
| N=128 K=4 E=4 I=384 | >0.99 | >0.99 | >0.98 | >0.98 |
| N=128 K=8 E=8 I=384 | >0.99 | >0.99 | >0.98 | >0.98 |
| N=512 K=4 E=8 I=1536 | >0.99 | >0.99 | >0.98 | >0.98 |
| N=512 K=8 E=8 I=1536 | >0.99 | >0.99 | >0.98 | >0.98 |

## 4. Session 63 Bugs Fixed (3 bugs)

### Bug 1: `_build_score_src_idx_kernel` PTXASError

- **症状**: E=32 crash, `tl.min(vector, axis=0)` PTXASError on SM103a
- **根因**: Triton 3.5.0 bundled ptxas 是 CUDA 12.8，不支持 SM103a。`tl.min` 向量 reduce 生成了 SM103a 不支持的 PTX 指令
- **修复**: 用纯标量操作重写（WORK_ptr scratch buffer + `tl.static_range` selection sort）
- **文件**: `mlp_node_v2.py:438-474`

### Bug 2: `varlen_K_max = E` instead of `topk`

- **症状**: E=8,topk=8 正常（巧合 E==topk），E=32,topk=8 crash（MAX_K=32 vs 正确值 8）
- **根因**: `_DownProjection.forward` 把 `E` 当 `varlen_K_max` 传给 `token_gather_sum_kernel`，该值作为 `tl.constexpr MAX_K` 参与 autotune key。E=32 导致编译了错误的 kernel 特化
- **修复**: `varlen_K_max=(K if K is not None else E)`；topk 通过 `_SonicMoEDeepEPFunc._topk` 类变量传递
- **文件**: `mlp_node_v2.py:690,768,793` + `functional/__init__.py:1345,1494`
- **为什么之前没暴露**: 所有测试和生产配置都用 E==topk==8

### Bug 3: QuACK `_compile_worker.py` Paddle dtype 不兼容 — BrokenPipe 根因

- **症状**: 冷启动 autotuning 时 `BrokenPipeError` crash backward
- **根因**: Paddle proxy 下 `str(tensor.dtype)` 返回 `'paddle.bfloat16'` 而非 `'torch.bfloat16'`。autotuner `_precompile` 序列化 tensor metadata → worker 的 `_dtype_map` 只有 `'torch.*'` → KeyError → worker crash → BrokenPipeError
- **修复** (两层防御):
  1. `autotuner.py`: `_normalize_dtype_str()` 在序列化时将 `'paddle.*'` 转为 `'torch.*'`
  2. `_compile_worker.py`: `_dtype_map` 增加 `'paddle.*'` 条目
  3. `_precompile` 整体包 try/except (best-effort, 失败 graceful fallback)
  4. `_send`/`_recv` 加 BrokenPipe/OSError catch
- **文件**: quack repo `quack/autotuner.py`, `quack/_compile_worker.py`
- **影响**: 修复后并行编译正常工作（`Pre-compiling 36 configs with 8 workers` 45s 完成），不再 fallback 到单线程

## 5. JIT Cache Design

### compile_key 原则: 只含静态模型维度

所有 compile_key 只含 (H, I, E, dtype, tile config)。**不含** TK、total_M、capacity 等动态值。动态维度通过 `mark_layout_dynamic` 运行时处理。

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

## 6. Critical Constraints (给下一个 agent 的陷阱警告)

1. **ds gradient path**: gate output → `_DownProjection.apply()` 之间，**不能有** native Paddle autograd 节点。Paddle `topk()`, `.cast()`, `amp.decorate` 都会创建 autograd 节点，收到 torch-proxy gradient tensor 时会 segfault。

2. **Paddle bf16 tensor 转换**: `tensor.cpu().numpy()` 返回 `uint16`（错误）。`torch.as_tensor()` 返回 `float16`（错误）。**只有 `torch.from_dlpack()` 正确保留 bf16。**

3. **`_inplace_version` 兼容**: Paddle = `_inplace_version()` (method), PyTorch = `._version` (attribute)。用 `_tensor_version()` helper。

4. **CUDA stream 兼容**: Paddle = `stream.stream_base.raw_stream`, PyTorch = `stream.cuda_stream`。用 `hasattr` 分支。

5. **`TRITON_PTXAS_PATH`**: 必须设为 `/usr/local/cuda/bin/ptxas`。Triton 3.5.0 bundled ptxas 是 CUDA 12.8，不支持 SM103a (Blackwell)。

6. **QuACK `str(dtype)` under Paddle proxy**: 返回 `'paddle.bfloat16'` 不是 `'torch.bfloat16'`。任何序列化 dtype 字符串的代码都要做 normalization。

7. **`E != topk` 时必须显式传 topk**: 旧代码假设 E==topk。当 E=32, topk=8 时，`varlen_K_max` 必须用 topk 而非 E。

## 7. Lessons Learned (Session 63, appended)

79. **`str(dtype)` under Paddle proxy returns `'paddle.bfloat16'` not `'torch.bfloat16'`**. This breaks any code that maps dtype strings to torch dtype objects. Always normalize with `s.replace('paddle.', 'torch.')`.

80. **Hidden semantic coupling: `E == topk` assumption.** When all tests use E=8 topk=8, a bug where `E` is used instead of `topk` goes undetected. Always test with E >> topk (e.g. E=32 topk=8) to catch these.

81. **Triton `tl.min(vector, axis=0)` generates PTX that old ptxas can't handle.** On SM103a with Triton's bundled CUDA 12.8 ptxas, vector reduction intrinsics fail silently during PTX assembly. Scalar loops are universally compatible.

82. **QuACK autotuner `_precompile` subprocess workers crash silently.** Worker crash → parent BrokenPipeError on `_send` → entire backward dies. Always wrap `_precompile` in try/except since it's a pure optimization.

83. **bench_mlpnode_mem.py `make_inputs` 极慢 (~30s for SEQ=7168).** 纯 Python 循环 + numpy 操作 57344 次。看起来像 hang。用 SEQ=512 做冒烟测试。

## 8. High-Value Information Sources

| Source | Path | Why |
|---|---|---|
| 精度测试 | `tests/ops/test_mlpnode_precision.py` | 4-shape × 5-tensor topk precision audit |
| 显存基准 | `tests/ops/bench_mlpnode_mem.py` | E=32 full fwd+bwd memory profile (迁移自 liangshuhao) |
| nsys 解析脚本 | `/tmp/parse_gpu_proj.py` | 从 sqlite 解析 GPU-projection (merged overlapping kernels) |
| QuACK autotuner | `quack/autotuner.py` | Session 63 修复了 Paddle dtype compat + robustness |
| QuACK compile worker | `quack/_compile_worker.py` | Session 63 修复了 dtype map + error handling |

## 9. QuACK 仓库改动 (Session 63)

**仓库位置**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack`

改动文件:
- `quack/autotuner.py`: dtype normalization + _precompile robustness (try/except, _send/_recv error handling, worker shutdown timeout)
- `quack/_compile_worker.py`: paddle.* dtype map + error handling for _send/_recv/import

**注意**: 这些改动在 quack 仓库中，不在 sonic-moe 仓库中。需要单独 commit/push 到 quack 仓库，或告知 quack 维护者合入。

## 10. Insights & Next Steps

### Insights

1. **GEMM 占 GPU 时间 78%**。Forward: CUTLASS GEMM 65%, 量化 10%. Backward: wgrad+accumulate 78%, actgrad 13%. 优化量化内核的收益已接近天花板，下一个大优化点是 wgrad GEMM（占 backward 78%）。

2. **wgrad accumulate (varlen_k) 是 backward 最大单项开销**。664µs (34.9%) 来自 CUTLASS varlen accumulate。这是一个将 varlen wgrad 累加到 native buffer 的操作，可能可以通过 CUTLASS 4.x 的 split-K 或 tiling 优化。

3. **Host Python overhead 不影响 GPU pipeline**。NVTX wall-clock ~6.7ms/iter 但 GPU-projection 只有 2.5ms。GPU 利用率约 37%（单 MLP layer），在完整 transformer 中会更高。

4. **并行 autotuning 对冷启动很关键**。36 configs × 8 workers = 45s 并行编译。单线程需要 ~360s (8x)。确保 Paddle 环境下 worker 能正常工作。

### Next Steps (Priority)

1. **ERNIE training loop 集成** — 将 `SonicMoEMlpNode` 接入 PaddleFleet MlpNode 插槽。关键: weight convention (split-half ↔ interleaved), prob scaling order, subbatch support。

2. **E=32 生产规模验证** — 当前 E=32 用 SEQ=4096 验证通过。需要用完整 SEQ=16384+EP=32 规模验证显存和延迟。

3. **wgrad accumulate 优化** — backward 34.9% 来自 varlen accumulate。考虑 split-K 或 stream-K 替代。

4. **Epilogue forward quantization** — 将 x→fp8 量化融入 GemmGated epilogue，消除 ~65µs forward overhead。

5. **Multi-card EP>1** — 接入 DeepEP buffer，验证 dispatch→MlpNode→combine pipeline。
