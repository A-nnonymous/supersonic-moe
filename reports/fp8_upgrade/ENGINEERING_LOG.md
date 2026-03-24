# FP8 Engineering Log

This log records concrete code changes plus their immediate validation and performance numbers.

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
