# FP8 Engineering Log

This log records concrete code changes plus their immediate validation and performance numbers.

## 2026-03-24 - Blackwell pre-SwiGLU 融合量化前向接线

### 指标注释（先看这个）

- 精度基线：官方 `bf16` 路径。
- 显存基线：官方 `bf16` 路径。
- 性能基线 1：本轮改动前的旧 `fp8_protocol blackwell` 小 shape 路径（同机、同命令、同共享环境）。
- 性能基线 2：官方 `bf16` 小 shape 路径（同机、同命令、同共享环境）。
- 重要说明：当前 8 张 Blackwell 卡几乎全部 `99%~100%` 利用率，因此本条目的性能数据只用于**同噪声环境下的前后对照**，不作为最终绝对性能验收。绝对性能仍需等待空闲卡后用主 shape 复测。

### 改动

- 在 `sonicmoe/functional/fp8_cutely_fused.py` 中新增真正的 pre-SwiGLU 高性能路径：
  - 输入直接消费 `_UpProjection` 返回的 `z`（pre-SwiGLU）。
  - 前向量化改为 `cutify.fused_weighted_swiglu_act_quant_best(...)`。
  - 反量化改为 `cutify.fused_act_dequant_best(...)`。
- 在 `sonicmoe/functional/__init__.py` 中将 Blackwell + QuACK 的 `fp8_protocol` 默认前向路径切到：

```text
z(pre-SwiGLU) -> fused_weighted_swiglu_act_quant_best -> fused_act_dequant_best -> y1
```

- 对 `I % 128 != 0` 的尾块进行了 pre-SwiGLU 对齐填充，保证像 `2880` 这样的宽度也能走融合路径。
- 为了恢复 CUDA graph capture 可用性，没有直接在融合 kernel 内走 `ue8m0` 打包返回；而是先保留 kernel 产出的 float32 dequant scale，再在 SonicMoE 侧编码为 `e8m0`。

### 收益来源说明

- 前向收益来源：
  - 不再走旧的 post-SwiGLU torch reference `quantize + dequantize`。
  - 改为直接复用现有 Cute/CUDA 储备，把 `SwiGLU + blockwise quant` 合到一个 pre-SwiGLU kernel 里。
- 端到端收益来源：
  - 主要来自前向边界开销下降。
  - 本轮**没有**改 backward 主 kernel，因此 bwd 改善只是前向边界更轻带来的连带收益，不应误判为 backward kernel 已优化。
- 显存收益来源：
  - 这一轮的主要目标是吞吐而不是进一步降显存，所以显存几乎不变。

### 正确性验证

命令：

```bash
USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py -k 'preact_cutely_fused_path_matches_reference_boundary or boundary_keeps_finite_forward_backward or blackwell_fp8_protocol_runtime_and_reference_quant'
USE_QUACK_GEMM=1 python -m pytest -q tests/moe_blackwell_test.py
```

结果：

```text
3 passed, 3 deselected
1 passed
```

### 精度数据

命令：

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

结果（新路径）：

- output RMSE vs bf16：`0.00131936`
- loss RMSE vs bf16：`0.00000015`

对照（旧路径）：

- output RMSE vs bf16：`0.00002498`
- loss RMSE vs bf16：`0.00000000`

解释：

- 精度有可见回退，来源是 pre-SwiGLU fused path 与旧 post-SwiGLU reference path 在量化前激活舍入位置不同。
- 目前 loss RMSE 仍非常小，说明训练目标量级没有发散。
- 后续如果要继续压低 RMSE，需要继续对齐：
  - pre-SwiGLU 激活布局与 reference 的数值路径；
  - `pow2/e8m0` scale 的编码细节；
  - optional prob 融合后的真实语义位置。

### 显存数据

同一条命令输出：

- bf16 peak memory：`380.25 MiB`
- 新 fp8 peak memory：`134.44 MiB`

对照（旧路径）：

- 旧 fp8 peak memory：`134.38 MiB`

解释：

- 相对官方 bf16，当前 fp8 仍少用 `245.81 MiB`，约 `64.64%`。
- 相对旧 fp8 路径，本轮显存几乎不变（`+0.06 MiB`），这符合预期，因为本轮主要替换的是量化/反量化算子实现，而不是缓存结构。

### 性能数据

同机、同 shape（`1024,512,512,32,4`）对照：

- 官方 bf16：
  - Fwd inference：`0.176 ms`
  - Fwd+Bwd：`3.162 ms`
  - Bwd：`2.987 ms`
- 旧 fp8 路径（改动前）：
  - Fwd inference：`0.392 ms`
  - Fwd+Bwd：`3.972 ms`
  - Bwd：`3.580 ms`
- 新 fp8 路径（本轮）：
  - Fwd inference：`0.229 ms`
  - Fwd+Bwd：`3.697 ms`
  - Bwd：`3.468 ms`

收益：

- 相对旧 fp8：
  - Fwd inference 提升 `41.58%`
  - Fwd+Bwd 提升 `6.92%`
  - Bwd 提升 `3.13%`
- 相对官方 bf16：
  - Fwd inference 仍慢 `30.11%`
  - Fwd+Bwd 仍慢 `16.92%`

解释：

- 这说明“先把 torch-side boundary 换成已有的 Cute/CUDA 融合算子”这一步是有效的，尤其是前向收益很直接。
- 但它也说明仅靠替换量化/反量化实现还不够；要继续逼近甚至超过 bf16，下一步必须继续往前推进，把：
  - `prob/topk_scores`
  - 更少的中间张量
  - 真正的 backward 融合
  继续吃进主线。

### 兼容性修复

- 初始版本在 benchmark 的 CUDA graph capture 中失败，根因是 `cutify` 的 `ue8m0` 打包辅助逻辑在 capture 期间触发了不允许的操作。
- 已修复为：
  - kernel 内先输出 float32 dequant scale；
  - SonicMoE 侧再编码成 `e8m0`；
  - benchmark 现已恢复可运行。

### 下一步

- 把 `topk_scores/prob` 的语义真正前移到融合 epilogue，而不是继续留在 router 后处理。
- 结合 Paddle 的：
  - `fp8_quant_blockwise_kernel.cu`
  - `fused_stack_transpose_quant_kernel.cu`
  - `fused_transpose_split_quant_kernel.cu`
  评估是否需要在 `operator-incubator` 再孵化一个更贴近 SonicMoE 合同的新 Cute quant kernel。
- 开始准备 paired backward kernel，把当前“前向已融合、反向未融合”的状态继续推进。

## 2026-03-24 - Blackwell FP8 functional boundary wiring

### Change

- added a functional-boundary `fp8_protocol` argument to:
  - `sonicmoe/moe.py::MoE.forward`
  - `sonicmoe/functional/__init__.py::moe_TC_softmax_topk_layer`
- added `apply_activation_fp8_protocol(...)` in `sonicmoe/functional/fp8_reference.py`
- the current boundary implementation quantizes/dequantizes the up-projection activation between `_UpProjection` and `_DownProjection`
- backward uses a straight-through estimator so the training path stays usable while the fused kernel does not exist yet
- `1x128` tail blocks are now padded internally and sliced back to original width, so shapes like `2880` are legal

### Correctness validation

Command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Result:

```text
18 passed, 91 skipped
```

### Performance regression

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test
```

Baseline before boundary wiring:

- Fwd inference: `24.164 ms`, `539.9 TFLOPS`
- Fwd training: `23.549 ms`
- Fwd+Bwd: `72.987 ms`, `536.2 TFLOPS`
- Bwd: `48.823 ms`, `534.4 TFLOPS`

Baseline after boundary wiring, protocol disabled:

- Fwd inference: `23.026 ms`, `566.6 TFLOPS`
- Fwd training: `23.478 ms`
- Fwd+Bwd: `75.443 ms`, `518.8 TFLOPS`
- Bwd: `52.417 ms`, `497.8 TFLOPS`

Interpretation:

- the default path did not regress in forward
- the end-to-end training path is a little slower than the earlier baseline, so future changes must keep checking this command

### Protocol-enabled performance

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell
```

Current numbers:

- Fwd inference: `63.715 ms`, `204.8 TFLOPS`
- Fwd training: `64.160 ms`
- Fwd+Bwd: `119.803 ms`, `326.7 TFLOPS`
- Bwd: `56.087 ms`, `465.2 TFLOPS`

Interpretation:

- the current FP8 boundary path is **correctness scaffolding**, not a performance win
- the large forward slowdown is expected because quant/dequant is still implemented as separate torch-side reference ops
- the next fused-kernel milestone must eliminate this overhead by folding quantization into the up-projection epilogue

### Next action

- replace the torch-side `apply_activation_fp8_protocol(...)` boundary path with a fused up-projection epilogue implementation
- keep using the same benchmark command above before and after every important performance-facing change

## 2026-03-24 - Parallel Blackwell regression entry

### Change

- installed `pytest-xdist` into `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer`
- added:
  - `make test-blackwell-full`
  - `make test-blackwell-parallel PYTEST_WORKERS=2`

### Measurement

Serial command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Serial runtime:

```text
18 passed, 91 skipped in 187.28s
```

Parallel command:

```bash
make test-blackwell-parallel PYTEST_WORKERS=2
```

Parallel runtime:

```text
18 passed, 91 skipped in 168.14s
real 168.41
```

### Conclusion

- `xdist` with `2` workers is a real win for the current Blackwell-targeted regression subset on this machine
- keep the parallel target opt-in rather than default, because these tests are still GPU-heavy and a higher worker count may oversubscribe the device

## 2026-03-24 - Boundary memory optimization

### Change

- removed the full-width float32 activation copy from `quantize_activation_blockwise(...)`
- dequantization now writes directly to the requested output dtype instead of materializing a full float32 activation first
- kept the same protocol semantics: `e4m3` activations, `e8m0` scales, `1x128` granularity, tail padding for non-divisible widths

### Correctness validation

Command:

```bash
python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py
```

Result:

```text
18 passed, 91 skipped
```

### Precision delta

Measured against the no-protocol baseline on the Blackwell training shape:

- max abs diff: `0.0013427734375`
- mean abs diff: `0.00018952522077597678`

The optimization did not change these error numbers relative to the previous boundary implementation.

### Memory delta

Single-run peak memory on `T=32768, H=2880, I=2880, E=64, K=8`:

Before optimization:

- baseline fwd peak: `9611.98 MiB`
- baseline e2e peak: `11826.48 MiB`
- fp8 boundary fwd peak: `15312.85 MiB`
- fp8 boundary e2e peak: `15312.85 MiB`

After optimization:

- baseline fwd peak: `9611.98 MiB`
- baseline e2e peak: `11826.48 MiB`
- fp8 boundary fwd peak: `13017.85 MiB`
- fp8 boundary e2e peak: `13017.85 MiB`

Interpretation:

- fp8 boundary peak memory dropped by `2295.00 MiB` (~`15.0%`) on both forward and end-to-end peak
- baseline memory stayed unchanged

### Performance delta

Benchmark command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 32768,2880,2880,64,8 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell
```

Before optimization, fp8 boundary:

- Fwd inference: `63.715 ms`
- Fwd+Bwd: `119.803 ms`

After optimization, fp8 boundary:

- Fwd inference: `56.065 ms`
- Fwd+Bwd: `109.117 ms`

Interpretation:

- fp8 boundary forward improved by `7.650 ms` (~`12.0%`)
- fp8 boundary end-to-end improved by `10.686 ms` (~`8.9%`)
- the path is still much slower than the bf16 baseline, so the next real win still depends on a fused up-proj epilogue

## 2026-03-24 - Metric harness and multi-GPU shard prep

### Change

- added `--report_fp8_metrics` to `benchmarks/moe-cute.py`
- added `make test-blackwell-multigpu BLACKWELL_TEST_GPUS=...`
- added `tools/run_blackwell_test_shards.py`
- added `--dry-run` to the shard launcher so command routing can be validated even when all 8 GPUs are busy
- added an env-gated adapter landing point in `sonicmoe/functional/fp8_cutely_fused.py`
- threaded the adapter behind `SONIC_MOE_FP8_CUTELY_FUSED`

### Validation

Dry-run command:

```bash
python tools/run_blackwell_test_shards.py --gpus 0,1,2 --dry-run
```

Result:

```text
[blackwell-shard] gpu=0 tests=tests/fp8_protocol_test.py
[blackwell-shard] gpu=1 tests=tests/moe_blackwell_test.py
[blackwell-shard] gpu=2 tests=tests/moe_test.py
```

Metric probe command:

```bash
USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek 1024,512,512,32,4 --dtype BFloat16 --activation swiglu --skip_test --fp8_protocol blackwell --report_fp8_metrics
```

Result:

```text
FP8 metrics vs bf16 baseline output_rmse=0.00002498, loss_rmse=0.00000000, bf16_peak_mib=380.25, fp8_peak_mib=134.38
PASS
```

### Interpretation

- the benchmark harness now emits the bf16-vs-fp8 metrics required by the current reporting policy
- the shard launcher is safe to invoke on a saturated machine in dry-run mode before selecting idle GPUs
- the new adapter shim keeps default behavior unchanged while fixing the code landing point for the real fused epilogue
- fused-op analysis confirmed that the incubator quant kernel consumes pre-SwiGLU `(T, 2H)` activations, so a direct swap at the current post-SwiGLU boundary would be semantically wrong
