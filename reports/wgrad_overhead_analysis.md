# Session 64: Wgrad GEMM Overhead Root Cause Analysis

## Summary

Paddle FP8 比 S53 native PyTorch FP8 慢 6-15%（取决于 shape）。通过 3 次独立 nsys 试验 × 4 shapes（GPU 2-7，全部 idle），逐 kernel 分解后发现：

**100% 的 overhead 来自 Wgrad GEMM（实际是 fused BF16 accumulate GEMM）。**

## 数据（3 trials × 11 iters/trial = 33 datapoints/shape）

| Shape | Paddle FP8 (µs) | S53 FP8 (µs) | Wgrad Δ (µs) | Total Δ (µs) | Wgrad 占比 |
|---|:---:|:---:|:---:|:---:|:---:|
| T=8192 E=8 | 2897±20 | 2715 | +206 | +182 | 113% |
| T=8192 E=32 | 3369±10 | 2922 | +498 | +447 | 111% |
| T=16384 E=8 | 5574±77 | 5227 | +342 | +347 | 99% |
| T=16384 E=32 | 5930±40 | 5432 | +563 | +498 | 113% |

> Wgrad 占比 > 100% 是因为其他类别 Paddle 反而更快（-24 到 -65µs），抵消了部分 overhead。

## 根因：fused accumulate epilogue 增加 register 压力

每次迭代有 4 个 Wgrad GEMM 调用（同名 `quackgemm_default_epiGemmDefaultSm100`）：

### Per-call 对比 (T8192_E8)

| GEMM Variant | Grid | S53 regs | S53 µs | Paddle regs | Paddle µs | Delta |
|---|---|:---:|:---:|:---:|:---:|:---:|
| FP8 compute #1 | (6312,1,1) | 54 | ~196 | 54 | ~222 | +26µs |
| FP8 compute #2 | (6312,1,1) | 54 | ~196 | 54 | ~222 | +26µs |
| BF16 accumulate (dw1) | (288,1,8) | **50** | 325 | **86** | 449 | **+124µs** |
| BF16 accumulate (dw2) | (144,1,8) | **50** | 174 | **86** | 226 | **+52µs** |

### 原因

S53 **没有** main_grad 累加逻辑。S53 的 4 个 wgrad GEMM 全是 `D = A@B`（simple epilogue, regs=50）。

Paddle 有 main_grad 累加：2 个 accumulate GEMM 使用 `D = A@B + 1.0*C`（fused epilogue），epilogue 需要：
- 从 HBM 加载 C（fp32 accumulator tensor）
- 执行 `beta * C + GEMM_result`
- 写回 D

这导致编译器分配 86 regs/thread（vs 50），occupancy 从 ~5 blocks/SM 降到 ~2-3 blocks/SM，latency hiding 能力大幅下降。

### 为什么 GemmGated/GemmDGated 不受影响

| Kernel | S53 config | Paddle config | Delta |
|---|---|---|---|
| GemmGated ZeroMat (fwd) | grid(6312,1,1) block(384) regs=168 | 相同 | <2% |
| GemmDGated ZeroMat (bwd) | grid(3156,1,1) block(384) regs=168 | 相同 | <4% |

这两个 kernel 不涉及 wgrad accumulate，代码路径完全相同，所以性能一致。

## Category-level Breakdown (T8192_E32, worst case 15.3% overhead)

| Category | Paddle(µs) | S53 FP8(µs) | Delta(µs) | Notes |
|---|:---:|:---:|:---:|---|
| Wgrad GEMM | 1697.6 | 1208.4 | **+489.2** | fused epilogue regs=86 |
| GemmGated ZeroMat (fwd) | 488.2 | 497.4 | -9.2 | identical |
| GemmDGated ZeroMat (bwd) | 426.3 | 420.6 | +5.7 | identical |
| Blockscaled Quant | 247.9 | 240.8 | +7.1 | |
| Dual Quant | 167.8 | 156.2 | +11.6 | |
| Token Gather | 146.5 | 147.6 | -1.1 | identical |
| Row Quant | 77.8 | 79.4 | -1.6 | identical |
| Score Src Idx | 25.5 | 0.0 | +25.5 | Paddle-specific topk kernel |
| Router Metadata | 18.6 | 0.0 | +18.6 | Paddle-specific routing |
| Paddle Framework | 14.4 | 0.0 | +14.4 | phi:: kernels |
| Elementwise Ops | 31.3 | 80.5 | -49.2 | S53 has more torch ops |
| Other | 10.3 | 56.2 | -45.9 | S53 misc overhead |

## 结论

1. **这不是 Paddle framework 的 overhead。** 非 GEMM 类别加总，Paddle 反而比 S53 更快（-24 到 -65µs）。
2. **100% overhead 来自 fused wgrad accumulate epilogue 的 register 压力。** regs=50→86 导致 30-38% per-call 降速。
3. **这是功能差异，不是 bug。** S53 不做 main_grad 累加，Paddle 做。在实际训练循环中 S53 也需要某种方式累加 grad，差距会缩小。
4. **优化方向：** 降低 fused epilogue register 压力（如 shared memory staging），或用 separate accumulate kernel 替代 fused epilogue（如果 separate add 比 fused GEMM overhead 更低）。

## Methodology

- 3 independent trials per shape, each on a different idle GPU (2-7)
- 5 warmup + 12 measured iterations, skip first measured iteration
- GPU-projection: merged overlapping kernel intervals in NVTX ranges
- S53 baseline: `/root/.../reports/grid_session53/session53_grid_full.json` + raw sqlite
- Kernel classification: regex pattern matching on demangled kernel names
- Register count: from `CUPTI_ACTIVITY_KIND_KERNEL.registersPerThread` in nsys sqlite
