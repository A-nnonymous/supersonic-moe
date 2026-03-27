# FP8 Blockscaled Upgrade — Status & Handoff

> **Last updated: 2026-03-27**

---

## 1. 现状概述

SonicMoE 全链路 blockscaled FP8 已完成 **8/8 GEMM 算子切换至 blockscaled 1x32**：
- Forward (2 GEMM): blockscaled 1x32 ✅
- Backward act-grad (2 GEMM): blockscaled 1x32 ✅
- Backward weight-grad (2 GEMM): **已改为 `blockscaled_fp8_weight_grad_gemm`** ✅ (本次 session 完成)

**核心路径**: Token rounding routing（生产模式）保证 128-aligned segments，blockscaled FP8 零 padding 开销。

**当前代码状态**: 已修改未验证。Weight-grad blockscaled 代码已写入 `sonicmoe/functional/__init__.py`，但由于集群 GPU 资源全满（16 节点全部 100% 占用），benchmark 和 RMSE 验证未完成。

---

## 2. 代码架构

### GEMM 算子状态 (全部 8/8 blockscaled)

| 算子 | 精度 | 实现 | 验证状态 |
|------|------|------|----------|
| up-proj forward | blockscaled 1x32 | fused `gemm_gated` + a_scales/b_scales | ✅ 已验证 |
| down-proj forward | blockscaled 1x32 | `blockscaled_fp8_gemm_varlen` | ✅ 已验证 |
| down-proj act-grad | blockscaled 1x32 | separate `blockscaled_fp8_gemm_varlen` + SwiGLU bwd | ✅ 已验证 |
| up-proj act-grad | blockscaled 1x32 | `blockscaled_fp8_gemm_varlen` | ✅ 已验证 |
| down-proj weight-grad | **blockscaled 1x32** | `blockscaled_fp8_weight_grad_gemm` | ⚠️ 待验证 |
| up-proj weight-grad | **blockscaled 1x32** | `blockscaled_fp8_weight_grad_gemm` | ⚠️ 待验证 |

### 本次代码变更详情 (functional/__init__.py)

**变更 1: down-proj weight-grad (line ~919-928)**
```python
# 旧代码 (per-tensor):
dout_fp8_wg = dout.to(torch.float8_e4m3fn)
y1s_fp8_wg = y1s.to(torch.float8_e4m3fn)
gemm(dout_fp8_wg.T, y1s_fp8_wg, out=dw2.permute(2, 0, 1), ...)

# 新代码 (blockscaled):
dout_gathered = dout[x_gather_idx]  # (TK, H)
blockscaled_fp8_weight_grad_gemm(
    dout_gathered, y1s, expert_frequency_offset,
    out=dw2.permute(2, 0, 1),
)
```

**变更 2: up-proj weight-grad (line ~609-618)**
```python
# 旧代码 (per-tensor):
dz_for_wgrad = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)
x_fp8_wg = x.to(torch.float8_e4m3fn)
dz_fp8_wg = dz_for_wgrad.to(torch.float8_e4m3fn)
gemm(x_fp8_wg.T, dz_fp8_wg, out=dw1.permute(2, 1, 0), ...)

# 新代码 (blockscaled):
dz_for_wgrad = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)
x_gathered_wg = x[x_gather_idx]  # (TK, H)
blockscaled_fp8_weight_grad_gemm(
    x_gathered_wg, dz_for_wgrad, expert_frequency_offset,
    out=dw1.permute(2, 1, 0),
)
```

**变更 3: 消除冗余 dout_expanded (节省 512 MiB)**
- 旧 blockscaled path 中 `dout_expanded = dout[x_gather_idx]` 创建了 (65536, 4096) bf16 中间张量，仅用于 weight-grad 的 `.to(fp8)`
- 现已替换为 `dout_gathered = dout[x_gather_idx]` 直接传入 `blockscaled_fp8_weight_grad_gemm`，逻辑等价但避免了额外复制

### Fused Kernels

| Kernel | 功能 | 代码位置 |
|--------|------|----------|
| `gather_quantize_and_pack_kernel` | gather + bf16→fp8 + ISA scale pack | `blockscaled_fp8_gemm.py:889` |
| `quantize_and_pack_kernel` | bf16→fp8 + ISA scale pack | `blockscaled_fp8_gemm.py` |
| `swiglu_forward_quant_triton` | SwiGLU + blockscaled quant (fwd) | `swiglu_triton.py` |
| `swiglu_backward_quant_triton` | dSwiGLU + score + blockscaled quant (bwd) | `swiglu_triton.py` |

### 数据流关键机制

- **`_PREQUANTIZED_SCALES`**: Dict-based FP8 tensor 复用机制
  - `"fwd"`: `swiglu_forward_quant_triton` 输出的 y1_fp8 + scales → `_DownProjection.forward` 复用（跳过重复 quantize）
  - `"bwd"`: `swiglu_backward_quant_triton` 输出的 dz_fp8 + scales → `_UpProjection.backward` act-grad 复用
  - 每次复用通过 `is` identity check 验证安全性

### 已修复 Bug

- **QuACK varlen alignment**: `colvec_scale=s.float()` — fp32 指针始终 32-bit aligned（BF16 指针在 domain_offset 后可能 16-bit aligned，触发 async copy 崩溃）

---

## 3. 精度数据 (前 6/8 GEMM 的验证结果)

| 测试 | 结果 | 条件 |
|------|------|------|
| Contract tests 8/8 | PASSED (5% rtol/atol) | `SONIC_MOE_FP8_MODE=perf`, 不含 large_shape |
| Official moe_blackwell_test | PASSED | 标准测试 |
| Fused gemm_gated D RelRMSE | 3.75% | shape: 65536×4096×2048, E=128 |
| Fused gemm_gated PostAct RelRMSE | 5.29% | 同上 |

**Weight-grad blockscaled 精度**: ⚠️ 待 RMSE 验证。验证脚本已就绪 (`tools/rmse_verification.py`)。

---

## 4. 性能数据

> **全部性能数据待采集** — 本次 session 集群 16 节点全部 100% GPU 占用，无法运行 benchmark。

**Benchmark 工具**: `tools/final_benchmark.py`
**RMSE 工具**: `tools/rmse_verification.py` (比较 FP8 vs BF16 的 output/dx/d_scores/dw1/dw2)
**Shape**: T=8192, H=4096, I=1024, E=128, K=8 (token rounding routing)

---

## 5. 已发现的关键问题 (HIGH VALUE)

### 问题 1: up-proj weight-grad 的 dz 有损转换 (BUG — 影响 dw1 精度)

**位置**: `_UpProjection.backward` line 611

当 fused swiglu quant 开启时 (默认)，`_DownProjection.backward` 将 `dz` 设为 `dz_fp8`（blockscaled fp8 tensor，line 910）。在 `_UpProjection.backward` 中：

```python
dz_for_wgrad = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)  # line 611
```

`dz` 是 `float8_e4m3fn`，所以执行 `.to(torch.bfloat16)`。**这是错误的**：

- blockscaled fp8 tensor 的实际值 = `fp8_raw * 2^(e8m0_scale_per_block)`
- `.to(bf16)` 只转换 `fp8_raw` 部分，**丢失了 block scale factor**
- 后续 `blockscaled_fp8_weight_grad_gemm` 接收到的是**缺少 scale 的错误 bf16 值**
- 最终 dw1 weight gradient 的量级可能偏差很大

**修复方向**:
1. 在 `_DownProjection.backward` 中将 `dz` 正确反量化回 bf16（需要 block scale 信息）
2. 或让 `blockscaled_fp8_weight_grad_gemm` 接受 pre-quantized fp8 + scales 输入，跳过内部量化
3. 或不在 backward boundary 替换 `dz` 为 fp8 — 保持 bf16 传递

**注意**: 这个 bug 在旧的 per-tensor weight-grad 路径中也存在（`.to(bf16)` 再 `.to(fp8)`），但因为 per-tensor fp8 没有 block scale 分离（scale 融入值中），影响更小。

### 问题 2: quantize 开销是 FP8 性能退化主因

**完整 E2E quantize 操作清单** (blockscaled path, separate backward):

| # | 阶段 | 操作 | 形状 | 类型 |
|---|------|------|------|------|
| 1 | FWD up-proj | `quantize_and_pack_activation(x_gathered)` | (TK, 4096) | blockscaled Triton |
| 2 | FWD up-proj | `swiglu_forward_quant_triton(z)` | (TK, 2048)→(TK, 1024) | fused Triton |
| 3 | FWD down-proj | SKIPPED via `_PREQUANTIZED_SCALES["fwd"]` | — | — |
| 4 | BWD down-proj | `gather_quantize_and_pack_activation(dout, idx)` | (T, 4096)→(TK, 4096) | blockscaled Triton |
| 5 | BWD down-proj | `swiglu_backward_quant_triton(dy1, z, s)` | (TK, 1024)+(TK, 2048) | fused Triton |
| 6 | BWD up-proj act-grad | SKIPPED via `_PREQUANTIZED_SCALES["bwd"]` | — | — |
| 7 | BWD down-proj wgrad | `blockscaled_fp8_weight_grad_gemm` 内部量化 x2 | (TK, H) + (TK, I) | 内部 |
| 8 | BWD up-proj wgrad | `blockscaled_fp8_weight_grad_gemm` 内部量化 x2 | (TK, H) + (TK, 2I) | 内部 |

**BF16 对比**: BF16 用 fused `gemm_dgated` 单 kernel 完成 GEMM+dSwiGLU+score，零量化开销。FP8 多了 4 次 blockscaled quantize（#1,#2,#4,#5）+ weight-grad GEMM 内部的 4 次量化。

### 问题 3: weight-grad GEMM 内部有不必要的 pack/unpack

`blockscaled_fp8_weight_grad_gemm` 内部流程：
1. `_pack_grouped_rows`: 将 flat tokens 按 expert 分组 → (E, capacity, dim)
2. `.transpose(1,2).contiguous()`: → (E, dim, capacity)
3. `quantize_activation_blockscaled_fast`: bf16 → fp8 + e8m0 scales
4. `pack_blockscaled_1x32_scales`: ISA layout packing
5. CUTLASS batched GEMM

其中 step 1 的 `_pack_grouped_rows` 和 gather 操作有冗余：
- down-proj wgrad 中 `dout_gathered = dout[x_gather_idx]` 已经做了 gather，然后 `_pack_grouped_rows` 再做一次 expert 分组
- 可以考虑直接将 already-gathered 数据 reshape + pad 为 grouped 格式，跳过 Triton scatter

---

## 6. 优先级与下一步

| 优先级 | 任务 | 状态 | 说明 |
|--------|------|------|------|
| **P0** | 修复 dz 有损转换 bug | 🔴 TODO | 影响 dw1 精度，必须在验证前修复 |
| **P0** | RMSE 验证 + benchmark | 🔴 TODO | 等 GPU 空闲后运行 `tools/rmse_verification.py` |
| P1 | 消除 backward quantize overhead | IN PROGRESS | 已有分析，需实现 |
| P2 | weight-grad 复用 act-grad fp8 数据 | PENDING | 消除重复量化 |
| P3 | 显存优化 — 统一 weight cache | PENDING | |

**下一个 agent 应做的第一件事**:
1. 修复 `_UpProjection.backward` line 611 的 dz 有损转换 bug
2. 在 GPU 空闲时运行 `python tools/rmse_verification.py` 验证精度
3. 运行 `python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"` 验证 8/8 pass
4. 运行 `python tools/final_benchmark.py` 采集性能数据（BF16 / FP8 两种模式）

---

## 7. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass excluding large_shape)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# BF16 E2E benchmark (token rounding)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=off \
  python tools/final_benchmark.py

# FP8 E2E benchmark (token rounding, fused gated fwd + separate bwd)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_FUSED_GATED=1 \
  python tools/final_benchmark.py

# RMSE verification (per-step FP8 vs BF16)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python tools/rmse_verification.py

# 查找集群空闲 GPU
python tools/cluster_idle_launch.py scan
# 在远程节点运行（注意：scan 只报告真正 idle 的 GPU，低利用率但有内存占用的不算 idle）
python tools/cluster_idle_launch.py launch --command "..." --workdir "$(pwd)"
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `sonicmoe/functional/__init__.py` | 核心 forward/backward + FP8 调度 (`_UpProjection`, `_DownProjection`) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | blockscaled GEMM + `blockscaled_fp8_weight_grad_gemm` + fused quantize kernels |
| `sonicmoe/quack_utils/swiglu_triton.py` | fused SwiGLU+quantize Triton kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | fused GEMM+SwiGLU + blockscaled (forward) |
| `sonicmoe/quack_utils/gemm_dgated.py` | fused GEMM+dSwiGLU + blockscaled (backward) |
| `sonicmoe/quack_utils/gemm_interface.py` | QuACK GEMM interface wrappers |
| `tools/rmse_verification.py` | Per-step RMSE 验证工具 |
| `tools/final_benchmark.py` | E2E 性能 benchmark |
| `tools/cluster_idle_launch.py` | 集群 GPU 扫描 + 远程启动 |
| `reports/fp8_upgrade/BLOCKSCALED_ALIGNMENT.md` | 128-alignment 硬约束分析 |

### 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `off`/`perf`/`mem` | FP8 模式 (`off`=BF16, `perf`=FP8) |
| `SONIC_MOE_FP8_FUSED_GATED` | `0`/`1` | 启用 fused gemm_gated forward |

---

## 8. 教训与陷阱

1. **128-row alignment 是硬约束**: blockscaled FP8 在 Blackwell 上要求每个 expert segment M-dim 是 128 的倍数。Token rounding routing 保证了这一点，vanilla top-K 需要 padding（会引入巨大性能退化，之前误报的 4x 退化就是这个原因）。
2. **QuACK varlen alignment bug**: BF16 指针偏移后可能不满足 32-bit alignment，必须用 `colvec_scale=s.float()` 强制 fp32 指针。
3. **blockscaled fp8 的 `.to(bf16)` 是有损操作**: 不会反应用 block scale factor，只做 raw fp8→bf16 数值转换。正确的反量化需要乘以 per-block e8m0 scale。
4. **`_PREQUANTIZED_SCALES` identity check**: 用 Python `is` 检查确保 tensor 没被复制或替换，安全复用 pre-quantized 数据。
5. **集群 GPU 扫描的 "idle" 定义**: `cluster_idle_launch.py` 用利用率 < 10% + 内存 < 5000 MiB 判断 idle。有些 GPU 0% 利用率但占了 70 GiB 内存（reserved），不算 idle。
6. **SSH 远程执行需要 cd 到项目目录**: 直接 `ssh node python tools/xxx.py` 会在 `/root` 下执行，找不到文件。必须 `ssh node "cd $WORKDIR && python tools/xxx.py"`。
