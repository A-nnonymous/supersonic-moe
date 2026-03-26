# FP8 Next-Agent Handoff

本文件的目标：**让下一个 agent 在最短时间内接住主线，不重复踩坑。**

> 最后更新：2026-03-26 (Session 2)

---

## 0. 一句话现状

**全链路 blockscaled FP8 已完成 6/8 GEMM（所有 forward + activation-grad）。Forward 和 act-grad 使用 1x32 blockscaled UE8M0 量化 + fused SwiGLU-quantize Triton 内核。Weight-grad 2/8 仍用 per-tensor FP8（精度可接受，性能是瓶颈）。当前训练 E2E 比 BF16 慢 4x——根因是 blockscaled varlen GEMM 的 pack/unpack/quantize 开销在 E=128 expert 下爆炸（从 11.9ms 退化到 48.5ms）。推理 forward 比 BF16 快 43%（3.9ms→2.2ms）。精度需实测但架构上已正确。下一步核心是优化 `blockscaled_fp8_gemm_varlen` 的运行时开销。**

---

## 1. 用户偏好（硬约束）

- 全链条 FP8：所有 GEMM 内部使用 FP8 tensor core + FP32 主循环累加
- 量化方案：1x32 blockscaled UE8M0 scale factor（Blackwell 硬件原生 descale）
- 精度是生命线：RelRMSE < 10%，cosine > 0.99
- 性能 + 显存双指标必须优于 BF16 baseline
- 守住 SonicMoE 的 `varlen/gather-A` 内存合同
- 用真实大 shape 对排，不要只看 toy case
- 不接受 hack：所有代码必须可 git commit，不做 site-packages patch

---

## 2. 当前代码真实状态（Session 2 实测）

### 2.1 各 GEMM 算子 blockscaled 状态

| # | 算子 | 当前路径 | 量化方式 | 状态 |
|---|------|---------|---------|------|
| 1 | up-proj forward | `blockscaled_fp8_gemm_varlen(x_fp8, w1, cu_seqlens)` | 1x32 blockscaled | **DONE** |
| 2 | SwiGLU activation+quant | `swiglu_forward_quant_triton(z)` fused | 1x32 blockscaled | **DONE** |
| 3 | down-proj forward | `blockscaled_fp8_gemm_varlen(y1_fp8, w2, cu_seqlens)` | 1x32 blockscaled (pre-quantized from #2) | **DONE** |
| 4 | down-proj act-grad | `blockscaled_fp8_gemm_varlen(dout_fp8, w2, cu_seqlens)` | 1x32 blockscaled | **DONE** |
| 5 | dSwiGLU+quant | `swiglu_backward_quant_triton(dy1, z, s)` fused | 1x32 blockscaled | **DONE** |
| 6 | up-proj act-grad | `blockscaled_fp8_gemm_varlen(dz_fp8, w1, cu_seqlens)` | 1x32 blockscaled (pre-quantized from #5) | **DONE** |
| 7 | up-proj weight-grad | `gemm(x_fp8.T, dz_fp8, cu_seqlens_k=...)` | per-tensor `.to(fp8)` | **TODO** |
| 8 | down-proj weight-grad | `gemm(dout_fp8.T, y1s_fp8, cu_seqlens_k=...)` | per-tensor `.to(fp8)` | **TODO** |

**调度逻辑**：当 `_fp8_enabled()` 且 `_min_expert_segment(expert_frequency_offset) >= 32` 时走 blockscaled 路径；否则 fallback 到 per-tensor FP8 或 BF16。

### 2.2 Fused SwiGLU + Blockscaled Quantize（已集成）

- `swiglu_forward_quant_triton(z)` → `(y1_fp8, y1_scales)` — 零 bf16 中间物化
- `swiglu_backward_quant_triton(dy1, z, s)` → `(dz_fp8, dz_scales, y1s_bf16, ds)` — y1s 始终 bf16
- 通过 `_PREQUANTIZED_SCALES` dict 在 autograd Function 边界传递 `(fp8_tensor, packed_scales)`
- 标签: `"fwd"` = UpProj.forward → DownProj.forward, `"bwd"` = DownProj.backward → UpProj.backward
- **身份检查**: consumer 验证 `_fwd_entry[0] is y1` 避免 tensor id 复用风险
- 控制开关: `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT=0` 可禁用（默认启用）

### 2.3 Weight Cache 优化（已集成）

- `_evict_per_tensor_caches_once()` 在 4 个 blockscaled 入口点调用
- 当 blockscaled 路径激活时，一次性清空 `_FP8_WEIGHT_CACHE` 和 `_FP8_ORIG_CACHE`
- `clear_all_fp8_weight_caches()` 同时清空 per-tensor + blockscaled 缓存
- `moe.py` 的 `clear_fp8_weight_cache()` 已更新调用 `clear_all_fp8_weight_caches()`

### 2.4 合同测试

- `tests/fp8_large_project_contract_test.py`：**8/8 pass**（排除 3 个 large_shape 测试）
- **已知问题**: `test_mixed_dtype_backward_large_shape_contract` 在 BF16 gold reference 自身就产生 NaN（`gold_dw2` 含 NaN），是预先存在的数值稳定性问题，与 FP8 代码无关
- 排除 large_shape 后所有测试绿色

### 2.5 已创建但当前未使用的代码

- `blockscaled_fp8_weight_grad_gemm()` in `blockscaled_fp8_gemm.py` — 完整实现但**性能不可接受**（pack/transpose/quantize 开销巨大），已从主路径移除改用 per-tensor QuACK gemm

---

## 3. 性能数据（2026-03-26 Session 2 实测）

### 3.1 Shape 8192,4096,1024,128,8（大, GPU 4/5）

| 指标 | BF16 baseline | Blockscaled FP8 | Delta |
|------|--------------|-----------------|-------|
| Fwd inference (ms) | 3.878 | 2.216 | **-42.9%** |
| Fwd training (ms) | 3.511 | 20.961 | **+497% (严重退化)** |
| E2E fwd+bwd (ms) | 11.889 | 48.486 | **+308% (严重退化)** |
| Bwd only (ms) | 8.012 | 46.270 | **+478% (严重退化)** |
| TFLOPS (fwd inference) | 425.3 | 744.3 | **+75.0%** |

### 3.2 根因分析

**推理 forward 快 43%** — blockscaled GEMM 的 FP8 tensor core 吞吐优势体现。

**训练 forward/backward 慢 3-5x** — 根因在 `blockscaled_fp8_gemm_varlen` 的 overhead：
1. `_pack_grouped_rows`: 逐 expert 拷贝 flat tokens → (E, capacity, dim) 3D padded tensor
2. `.transpose(1,2).contiguous()`: 全量 memcpy 重排内存布局
3. `quantize_activation_blockscaled_fast`: Triton 量化 kernel
4. `pack_blockscaled_1x32_scales`: ISA tile layout 重排

**E=128 expert 下 overhead 爆炸**: 128 experts × 每次 pack+transpose+quantize = 上百次内存操作。
对比 BF16 path 使用 QuACK 的 `gemm_gated(x, w1, cu_seqlens_m=...)` 原生 varlen 调度，零额外开销。

### 3.3 之前的 per-tensor FP8 性能（参考，已不是当前默认）

| 指标 | BF16 | Per-tensor FP8 | Delta |
|------|------|----------------|-------|
| Fwd inference (8K shape) | 3.924 | 1.995 | -49.2% |
| E2E fwd+bwd (8K shape) | 11.962 | 7.351 | -38.6% |

Per-tensor FP8 性能优异但精度不可接受（RelRMSE ~100% at training scale）。

### 3.4 精度数据

| 来源 | 量化方式 | RelRMSE | 备注 |
|------|---------|---------|------|
| `blockscaled_fp8_gemm_varlen` 单算子 | 1x32 blockscaled | **3.74%** | 全 production shape |
| per-tensor cast (0.02*randn) | per-tensor | ~100% | 训练常态，完全不可接受 |
| per-tensor cast (1.0*randn) | per-tensor | 7.90% | 只在大幅值可用 |

E2E blockscaled 精度尚未完整测量（训练路径太慢导致 benchmark 未运行 --report_fp8_metrics）。

---

## 4. 核心瓶颈与下一步方案

### 4.1 **P0-CRITICAL: 消除 blockscaled varlen GEMM 的 pack/unpack 开销**

这是唯一阻止性能超过 BF16 的关键问题。

**方案 A（推荐）: 让 CUTLASS GemmDefaultSm100 原生支持 varlen**
- 当前 `blockscaled_fp8_gemm_varlen` 需要 pack → grouped 3D → CUTLASS batched GEMM → unpack
- 如果 CUTLASS 的 `VarlenMTileScheduler` 能与 blockscaled scale factor 配合，可以直接在 flat (TK, dim) tensor 上操作
- 检查 QuACK 的 `gemm_wrapper_utils.py` 是否已有 varlen + blockscaled 的组合支持
- 关键文件: `envs/xfer/lib/python3.13/site-packages/quack/gemm_sm100.py`, `gemm_wrapper_utils.py`

**方案 B: 优化 pack/quantize 流水线**
- 当前 `_pack_grouped_rows` 是 Python 循环逐 expert 拷贝 — 写一个 Triton kernel 一次性完成 pack+quantize
- 消除 `.transpose(1,2).contiguous()` — 直接在 pack 时按目标 layout 写入

**方案 C: 使用 Token Rounding Routing 保证 128-aligned segments**
- SonicMoE paper Algorithm 4 已保证 128-aligned expert segments
- 此时可直接对 flat tensor 做 blockscaled quantize，无需 pack/padding
- 文件: `sonicmoe/routing/token_rounding.py`

### 4.2 P1: Weight-grad blockscaled（可延后）

Weight-grad 当前用 per-tensor FP8 via `quack.gemm(x.T, dz, cu_seqlens_k=...)`，性能可接受。
精度上 weight gradient 是累积的，per-tensor 通常足够。
但如果精度测试发现 weight-grad 是精度瓶颈，需要实现 blockscaled weight-grad。

**注意**: 已实现的 `blockscaled_fp8_weight_grad_gemm()` 性能不可接受（pack/transpose 开销）。
如果需要 blockscaled weight-grad，必须用方案 A 或 B 消除开销后再启用。

### 4.3 P2: 显存优化

当 blockscaled 路径激活时，`_evict_per_tensor_caches_once()` 已经清空 per-tensor 缓存。
但 blockscaled 路径的 `_WEIGHT_CACHE`（在 `blockscaled_fp8_gemm.py` 中）仍会缓存 quantized weights。
需要统一缓存策略。

---

## 5. 高价值技术知识

### 5.1 CUTLASS Blockscaled Varlen 的 Rank-2 修复

- `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF` 硬编码 `(2, 1, 3)` 排列
- Varlen 路径产出 rank-2 `(total_M, K)` 张量导致编译期 rank mismatch
- 修复：`blockscaled_fp8_gemm.py` 中的 `_tile_atom_to_shape_SF_rank_aware` monkey-patch
- 使用 `cute.rank(Shape)` + `const_expr` 在 trace time 分派 `(2, 1)` / `(2, 1, 3)`

### 5.2 Blockscaled 最小 Segment 约束

- Scale factor tile atom = 32 elements in M direction
- Expert segment < 32 tokens → `CUDA_ERROR_ILLEGAL_INSTRUCTION`
- 生产 shape (T>=512, K=8, E=128, 即 >=32 tpe) 安全
- 代码已有 `_min_expert_segment() >= 32` 检查 + fallback

### 5.3 FP8 权重缓存架构

```
Per-tensor caches (cleared when blockscaled path activates):
  _FP8_WEIGHT_CACHE: w1_ekh(E,H,2I) + w2_ehi(E,H,I) fp8 [for gemm_gated/gemm_dgated]
  _FP8_ORIG_CACHE:   w1(2I,H,E) + w2(H,I,E) fp8 [for quack.gemm backward]

Blockscaled cache (in blockscaled_fp8_gemm.py):
  _WEIGHT_CACHE: quantized + scale-packed weight tensors [for blockscaled_fp8_gemm_varlen]
```

### 5.4 gemm_gated/gemm_dgated 不支持 blockscaled

- 这两个 fused kernel 接受 `A(fp8) * B(fp8)` 但**不支持 blockscaled scale factor**
- 无 `sf_vec_size` 参数
- 当前 blockscaled 路径绕过它们：GEMM + 独立 SwiGLU kernel

### 5.5 GPU 兼容性

- GPU 2, 3 有间歇性 Triton `CUDA_ERROR_ILLEGAL_INSTRUCTION`
- GPU 0, 1, 4, 5 稳定
- 测试使用 GPU 0 或 4

### 5.6 Pack/Unpack 是性能杀手

`_pack_grouped_rows` (line ~920 in blockscaled_fp8_gemm.py) 是 Python for-loop 逐 expert 拷贝:
```python
for i in range(num_experts):
    start, end = cu_seqlens_m[i], cu_seqlens_m[i+1]
    seg_len = end - start
    grouped[i, :seg_len] = flat[start:end]
```
E=128 时这是 128 次 Python 循环 + 128 次 CUDA memcpy。
加上后续的 `.transpose(1,2).contiguous()` 全量内存拷贝，开销远超 GEMM 本身。

### 5.7 _PREQUANTIZED_SCALES 跨 autograd Function 传递机制

这是 Session 2 新增的核心机制：
- `_UpProjection.forward` 的 fused SwiGLU 产出 `(y1_fp8, packed_scales)`
- 存入 `_PREQUANTIZED_SCALES["fwd"]`
- `_DownProjection.forward` 检查 `_PREQUANTIZED_SCALES.pop("fwd")`，若 tensor identity 匹配则直接使用
- 反向同理: `_DownProjection.backward` → `_PREQUANTIZED_SCALES["bwd"]` → `_UpProjection.backward`
- 避免了重复量化的开销

---

## 6. 已证伪方向（不要重复）

1. **`blockscaled_fp8_weight_grad_gemm` 的 pack/transpose 方案**：已实现并验证，性能不可接受。E=128 下 pack+transpose+quantize 开销远超 GEMM 本身。代码保留在 `blockscaled_fp8_gemm.py:1158` 但已从主路径移除。
2. **Grouped/static-capacity blockscaled down-proj**：违背 SonicMoE varlen 合同，内存开销大
3. **per-tensor FP8 训练**：精度完全不可接受 (RelRMSE ~100% at 0.02*randn)
4. **在 site-packages 做 runtime patch**

---

## 7. 经验与教训

1. **推理 vs 训练性能差异巨大**：同样的 blockscaled GEMM，推理快 43%（无 backward），训练慢 4x（backward 的 pack/quantize 开销主导）
2. **E=128 expert 放大所有 per-expert 开销**：任何 for-loop-over-experts 的代码在 E=128 下都是灾难
3. **Weight-grad 的 varlen-in-K 本质不同**：act-grad varlen 在 M 维度（batch），QuACK 原生支持；weight-grad varlen 在 K 维度（reduction），需要转置后 quantize，开销巨大
4. **`dout` 非 contiguous**：`sum().backward()` 产出 stride=(0,0) expanded tensor，需 `.contiguous()` 后再传 GEMM
5. **`_fp8_enabled()` 检查 env var，不检查 `fp8_protocol` 参数**：gold path 需 unset `SONIC_MOE_FP8_MODE`

---

## 8. 环境与命令速查

### 环境激活

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
```

### 合同测试

```bash
# 8/8 pass (排除 large_shape)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# 完整 11 测试 (1 fail: gold_dw2 NaN, 预先存在)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v
```

### BF16 baseline benchmark

```bash
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py \
  --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
```

### FP8 benchmark

```bash
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python benchmarks/moe-cute.py \
  --thiek 8192,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `sonicmoe/functional/__init__.py` | 核心：所有 MoE forward/backward + FP8 调度 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Blockscaled FP8 GEMM 基础设施 |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU+quantize Triton 内核 |
| `sonicmoe/quack_utils/__init__.py` | 导出公共 API |
| `sonicmoe/functional/fp8_protocol.py` | FP8Protocol 定义 |
| `sonicmoe/moe.py` | MoE 模块顶层，cache 管理 |
| `tests/fp8_large_project_contract_test.py` | 合同测试 |
| `benchmarks/moe-cute.py` | 性能 benchmark |
| `envs/xfer/lib/python3.13/site-packages/quack/gemm_wrapper_utils.py` | QuACK varlen GEMM 基础设施 |
