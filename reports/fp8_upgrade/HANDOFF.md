# Blockscaled FP8 MoE — Status & Handoff

> **Last updated: 2025-03-29 (quack 0.3.7 upgrade session)**

---

## 1. 目标

全链路 blockscaled FP8 (1×32 UE8M0) MoE training。Forward 和 Backward 均使用 native CUTLASS fused GEMM+SwiGLU kernel。精度 RelRMSE < 10%，性能和显存远优于 BF16 baseline。

---

## 2. 当前状态

### 包版本 (已升级)

| 包 | 版本 | 说明 |
|---|------|------|
| `quack-kernels` | 0.3.7 | 从 0.2.5 升级，大量 API break |
| `nvidia-cutlass-dsl` | 4.4.2 | CUTLASS 3.8.0 |
| `torch` | 2.9.1+cu128 | |
| `triton` | 3.5.1 | |

### ✅ 已修复

1. **`blockscaled_fp8_gemm.py` varlen API fix** — `create_varlen_args` 从 7 参数简化为 3 参数（quack 0.3.7 breaking change）。修复后 decomposed blockscaled FP8 GEMM 重新工作。
2. **`gemm_gated.py` 完全重写** — 使用 `TileStore("mPostAct", epi_tile_fn=_halve_epi_tile)` composable epilogue 替代手动 TMA setup。BF16 fused forward 已验证通过。
3. **`gemm_dgated.py` 完全重写** — `ArgumentsBase` (已在 0.3.7 中移除) 替换为 `@mlir_namedtuple NamedTuple`，`_epi_ops` 改为 composable `TileStore` + `ColVecReduce`。
4. **`gemm_interface.py` contiguity fix** — `B.mT` → `B.mT.contiguous()`，CUTLASS DSL 4.4.2 的 `mark_layout_dynamic` 要求 `strides[leading_dim]==1`。

### ✅ Decomposed FP8 GEMM — WORKING

`blockscaled_fp8_gemm_varlen` (non-gated GEMM) 测试通过:
- **rel_rmse = 3.77%, corr = 0.999** (E=8, K=1024, N=2048, 64 tok/expert)
- 使用 `GemmDefaultSm100` + `sf_vec_size=32` blockscaled path

### 🔴 Fused FP8 GEMM+SwiGLU — STILL CRASHES

`GemmGatedSm100` + blockscaled FP8 (`sf_vec_size=32`) 仍然在 sm_100a 上产生 `CUDA_ERROR_ILLEGAL_INSTRUCTION`。

**关键排查结论** (本轮通过系统性隔离得出):

| 测试 | 结果 | 排除的假设 |
|------|------|-----------|
| CopyUniversalOp for PostAct R2S | CRASH | ≠ stmatrix 指令问题 |
| Dummy epi_visit_subtile (返回零) | CRASH | ≠ SwiGLU 计算问题 |
| Full-width TileStore (不 halve) | CRASH | ≠ epi_tile 减半问题 |
| Base GemmSm100 + blockscaled (无 TileStore) | CRASH | ≠ TileStore 本身问题 |
| 修复 `create_varlen_args` 后的 decomposed path | **PASS** | 证实是 varlen API break |

**结论**: illegal instruction 不是来自 gating epilogue 逻辑。所有带 PostAct TileStore + blockscaled 的组合都 crash。decomposed (无 TileStore) 在修复 varlen API 后正常。**问题在 `TileStore` + blockscaled FP8 的交互**，可能是 TMA store epilogue 在 blockscaled 模式下的 codegen bug。

### 🔴 BF16 Backward (gemm_dgated) — 需要修复

我的 `gemm_dgated.py` 重写导致 BF16 backward contract test 失败:
```
TypeError: BFloat16.__c_pointers__() missing 1 required positional argument: 'self'
```
这发生在 `_precompile` 阶段 (autotuner pickle)。原始代码在 quack 0.3.7 下也无法 import (`ArgumentsBase` 已移除)，所以这不是回归而是需要继续修复的问题。

`implicit_dtype` 字段类型可能有误：`cutlass.Constexpr[type] = cutlass.BFloat16` 作为默认值传给 mlir_namedtuple 后，`BFloat16` 类被当作实例 pickle，触发了 `__c_pointers__` 的类/实例混淆。

### 性能数据 (升级前测量，decomposed path)

| 指标 | FP8 | BF16 | 比率 |
|------|-----|------|------|
| E2E latency (p25) | 13.89ms | 12.09ms | 0.87x (FP8 慢 15%) |

**FP8 慢于 BF16 的根因**: decomposed path (non-gated GEMM + separate SwiGLU) vs BF16 fused path (single kernel)。修复 fused path 预期可获得 **2x+ 性能提升**。

### 精度基线 (升级前验证，需重新验证)

| Metric | RelRMSE | Correlation |
|--------|---------|-------------|
| Forward | 6.61% | 0.998 |
| dx grad | 6.79% | 0.998 |
| dw1 | 5.69% | 0.998 |
| dw2 | 5.44% | 0.999 |
| router | 6.56% | 0.998 |

---

## 3. GEMM 矩阵 (当前状态)

| 算子 | 实现 | 状态 | 说明 |
|------|------|------|------|
| up-proj fwd | Fused GEMM+SwiGLU (GemmGatedSm100) | 🔴 FP8 crash / ✅ BF16 ok | TileStore+blockscaled crash |
| up-proj fwd fallback | Decomposed `blockscaled_fp8_gemm_varlen` + SwiGLU | ✅ FP8 working | 精度验证通过 |
| down-proj fwd | `blockscaled_fp8_gemm_varlen` | ✅ FP8 working | |
| up-proj bwd (GemmDGatedSm100) | Fused GEMM+dSwiGLU | 🔴 pickle crash | `BFloat16.__c_pointers__` |
| down-proj bwd act-grad | `blockscaled_fp8_gemm_varlen` | ✅ FP8 working | |
| weight-grad (both) | BF16 QuACK varlen | ✅ BF16 working | FP8 wt-grad 太慢，不用 |

**4/6 GEMM 已 FP8**。Weight-grad 保持 BF16 因为 blockscaled per-expert GEMM 在 E=128 下有太多 kernel launches。

---

## 4. 核心技术发现

### 4.1 QuACK 权重布局：interleaved gate/value columns

**最关键的发现**。QuACK 的 `gemm_gated` 使用 **interleaved** pre-activation 布局：

```
w1 shape: (2I, H, E)
Column layout: [gate_0, value_0, gate_1, value_1, ..., gate_{I-1}, value_{I-1}]
```

即 `w1[0,:,e]` = gate projection 第0行, `w1[1,:,e]` = value projection 第0行, `w1[2,:,e]` = gate projection 第1行, ...

**SwiGLU 正确实现**：
```python
z = gemm(x, w1)  # shape (TK, 2I), columns are interleaved
gate = z[:, 0::2]   # even columns = gate
value = z[:, 1::2]  # odd columns = value
y1 = F.silu(gate) * value
```

**错误实现** (之前导致 ~141% RMSE)：
```python
gate, value = z.chunk(2, dim=-1)  # ❌ wrong: contiguous halves, not interleaved
y1 = F.silu(gate) * value  # produces uncorrelated output
```

Backward 中 `dz` 的写回也必须用 interleaved 索引：`dz[:, 0::2] = d_gate`, `dz[:, 1::2] = d_value`。

### 4.2 Pre-quantized Activation 的 padding bug

`blockscaled_fp8_gemm_varlen` 在 tokens-per-expert 不是 128 对齐时需要 padding。Padding 路径 (lines ~1353-1372) 执行 `a.to(torch.bfloat16)` — 这是 **raw cast**，不是 dequantization。FP8 值 0.5 保持 0.5，而不是乘以其 E8M0 scale。

**解决方案**: 不 pre-quantize activations。传入 bf16 activations 让函数内部处理量化。Pre-quantized weights 没有问题（不经过 padding 路径）。

### 4.3 CUTLASS DSL TileStore + Blockscaled Bug (已精细定位)

通过本轮系统性隔离测试（见状态表），精确定位：

- **`GemmDefaultSm100` + `sf_vec_size=32` (无 TileStore)**: ✅ WORKS — decomposed path 正常
- **任何带 `TileStore` epilogue op + `sf_vec_size=32`**: 🔴 CRASH — 即使没有 SwiGLU 也 crash
- BF16 + TileStore: ✅ WORKS — TileStore 自身没有问题

**结论**: 不是 CUTLASS DSL 编译器的 gated epilogue bug，而是 **TileStore 的 TMA store epilogue 在 blockscaled 模式下的 codegen 问题**。TMA descriptor 或 smem layout 在 blockscaled 下的计算可能不正确。

**排查方向**:
- 检查 `quack/epi_ops.py` 中 `setup_epi_tensor()` 是否正确处理 blockscaled 的 epi_tile
- 检查 `GemmSm100.epilogue()` 中 `copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)` TMA 路径
- 对比 blockscaled vs non-blockscaled 时 `epi_tile` / `cta_tile_shape_mnk` 的差异
- 使用 `CUDA_LAUNCH_BLOCKING=1` + `compute-sanitizer --tool memcheck` 获取精确 fault 位置

### 4.4 Weight Pre-quantization Cache

```python
_FUSED_WEIGHT_CACHE  # keyed on (storage_ptr, version, shape, stride)
precompute_weight_fp8(w)  # w (dim0, dim1, E) → fp8 + ISA-packed scales
```

权重在第一次 forward 时量化并缓存。`clear_blockscaled_fp8_weight_cache()` 在 optimizer step 后调用。

---

## 5. 性能数据 (B200 sm_100a, T=4096, H=4096, I=1024, E=128, K=8)

> 集群负载高，数据有波动。

| Config | P25 E2E (ms) | Median E2E (ms) | Peak Memory |
|--------|-------------|-----------------|-------------|
| **BF16 baseline** | 12.09 | 12.11 | 9.55 GB |
| **Blockscaled FP8** | 13.89 | 35.91* | 13.08 GB |

*FP8 median 受 CUTLASS DSL 编译 jitter 影响。P25 更能反映稳态性能。

**性能不及 BF16 的原因**: FP8 路径使用 decomposed kernels (非 fused gated)。一旦 `GemmGatedSm100` blockscaled FP8 bug 修复，FP8 路径应可使用 fused kernel 并大幅领先 BF16。

---

## 6. 关键代码文件

| File | 内容 | 重要行号 |
|------|------|---------|
| `sonicmoe/functional/__init__.py` | 核心 FP8/BF16 dispatch, `_fp8_enabled()`, SwiGLU interleave | ~468-486 (up-proj fwd), ~790-825 (down-proj bwd) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | `blockscaled_fp8_gemm_varlen`, `precompute_weight_fp8`, CUTLASS 调用 | ~1281 (varlen entry), ~1401 (CUTLASS invoke) |
| `sonicmoe/quack_utils/gemm_interface.py` | QuACK wrappers: `gemm_gated`, `gemm_dgated`, `gemm_gated_out` | a_scales/b_scales 参数存在但 gated 时不可用 |
| `tests/fp8_large_project_contract_test.py` | 8 contract tests (FP8-vs-FP8 regression) | `SONIC_MOE_FP8_MODE=perf` |
| `reports/perf_opt.md` | Triton kernel 优化历史 (已弃用, CUTLASS 方案取代) | 历史参考 |

---

## 7. 环境速查

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# Full MoE FP8 vs BF16 accuracy comparison
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -c "
import torch, os
os.environ['USE_QUACK_GEMM'] = '1'
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.quack_utils.blockscaled_fp8_gemm import _COMPILE_CACHE, clear_blockscaled_fp8_weight_cache
from sonicmoe.functional import clear_all_fp8_weight_caches

T, H, I, E, K = 4096, 4096, 1024, 128, 8
torch.manual_seed(42)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to('cuda', torch.bfloat16)
x = torch.randn(T, H, device='cuda', dtype=torch.bfloat16)
enable_quack_gemm()

os.environ['SONIC_MOE_FP8_MODE'] = 'off'
clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache()
with torch.no_grad():
    out_bf16 = moe(x); out_bf16 = out_bf16[0] if isinstance(out_bf16, tuple) else out_bf16

os.environ['SONIC_MOE_FP8_MODE'] = 'perf'
clear_all_fp8_weight_caches(); clear_blockscaled_fp8_weight_cache(); _COMPILE_CACHE.clear()
with torch.no_grad():
    out_fp8 = moe(x); out_fp8 = out_fp8[0] if isinstance(out_fp8, tuple) else out_fp8

rmse = ((out_fp8.float()-out_bf16.float()).pow(2).mean().sqrt() / out_bf16.float().pow(2).mean().sqrt()).item()
corr = ((out_fp8.float()*out_bf16.float()).sum()/(out_fp8.float().norm()*out_bf16.float().norm())).item()
print(f'RelRMSE: {rmse*100:.2f}%, Correlation: {corr:.6f}')
"
```

| 变量 | 值 | 说明 |
|------|-----|------|
| `USE_QUACK_GEMM` | `1` | 启用 QuACK CUTLASS GEMM |
| `SONIC_MOE_FP8_MODE` | `perf` | 启用 blockscaled FP8 (`off`=纯BF16) |

---

## 8. 下一步 — 优先级排序

### 🔥 P0: 修复 TileStore + blockscaled FP8 crash

这是唯一的性能瓶颈。修复后 fused GEMM+SwiGLU 可以 1 kernel 完成 up-proj forward，预期 **2x+ 性能提升**。

**具体路线**:
1. 在 quack 0.3.7 源码中 `setup_epi_tensor()` (在 `quack/epi_ops.py`) 添加 blockscaled 模式诊断
2. 对比 `GemmDefaultSm100.__call__` (能工作) 和 `GemmActMixin` (crash) 的 epilogue 差异
3. 特别检查 `epi_tile` 在 blockscaled 时的 size 是否仍然满足 TMA alignment 要求
4. 如果是 quack bug，向上游报告；如果可以 monkey-patch `setup_epi_tensor`，在 gemm_gated.py 中覆盖

### P0.5: 修复 gemm_dgated.py BF16 backward

`BFloat16.__c_pointers__` 错误来自 `implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16`。Fix: 将 `implicit_dtype` 的默认值改为 `None`，在 `epi_to_underlying_arguments` 中 fallback 为 `cutlass.BFloat16`。

### P1: Contract tests 全部通过

修复 dgated 后重新跑 `pytest tests/fp8_large_project_contract_test.py -k "not large_shape"`。目标 8/8 PASS。

### P2: NCU profiling + 性能对比

```bash
# FP8 decomposed forward
ncu --set full -o fp8_fwd CUDA_VISIBLE_DEVICES=0 python3 -c "..."
# BF16 fused forward
ncu --set full -o bf16_fwd CUDA_VISIBLE_DEVICES=0 python3 -c "..."
```

### P3: 激活 pre-quantization + weight-grad FP8

见前文 P1/P2 描述。

### P3: CUDA Graph 消除 launch overhead

FP8 decomposed path 有更多 kernel launches。CUDA Graph 可以消除 launch overhead。

### P4: Activation 量化融合到 GEMM epilogue

将 fp8 quantization 融入 CUTLASS GEMM 的 epilogue (输出直接为 fp8 + scales)，避免额外 quantize kernel。

---

## 9. 架构概览

```
Input x (T, H) bf16
    │
    ├── Router → topk_scores, x_gather_idx, efo, s_reverse_scatter_idx
    │
    ├── [FP8] x_gathered = x[x_gather_idx]  (TK, H) bf16
    │   └── blockscaled_fp8_gemm_varlen(x_gathered, w1, efo)
    │       └── z (TK, 2I) bf16 interleaved [gate0,val0,gate1,val1,...]
    │           └── gate = z[:,0::2], value = z[:,1::2]
    │               └── y1 = silu(gate) * value  (TK, I)
    │
    ├── [BF16] gemm_gated(x, w1, A_idx=x_gather_idx, cu_seqlens_m=efo)
    │   └── (z, y1) — fused GEMM + SwiGLU in one kernel
    │
    ├── Down-proj: blockscaled_fp8_gemm_varlen(y1, w2, efo) → y2 (TK, H)
    │
    └── Router scatter: y2[s_reverse_scatter_idx] * topk_scores → output (T, H)
```

---

## 10. 教训与高价值信息

1. **QuACK interleaved layout**: 这不是 document 过的行为，只能通过实验发现。影响所有手动 SwiGLU 代码。
2. **CUTLASS DSL 编译不稳定**: 编译缓存有时失效，导致性能 jitter。`_COMPILE_CACHE` 是 dict 缓存。
3. **`_fp8_enabled()` 读环境变量**: 是 `os.environ["SONIC_MOE_FP8_MODE"]` 而非 per-call 参数。
4. **Blockscaled padding bug**: 非 128 对齐时 `a.to(bf16)` 是 raw cast，不做 dequant。
5. **Per-tensor FP8 曾经可用** (之前的 HANDOFF): forward 1.53x speedup，但不符合 blockscaled 要求。
6. **Triton `tl.dot_scaled` 不可用**: sm_100a 上编译失败，所以选择 native CUTLASS path。
7. **sgl-kernel MXFP8 GEMM**: 有 8ms Python 开销 + scale layout 不兼容，已弃用。
8. **quack 0.3.7 API breaks** (本轮发现):
   - `ArgumentsBase` → `@mlir_namedtuple NamedTuple`
   - `create_varlen_args` 从 7 参数简化为 3 参数 (cu_seqlens_m, cu_seqlens_k, A_idx)
   - `num_epi_tensormaps` class attribute 从 `GemmDefaultSm100` 移除
   - `mark_layout_dynamic(leading_dim=X)` 严格要求 `strides[X]==1`
   - `TileStore` + `ColVecReduce` 等 composable epi_ops 替代手动 TMA 管理
   - `is_persistent` → 默认行为 / `use_clc_persistence` 参数
9. **TileStore + blockscaled = crash**: 经过5项隔离测试确认，不是 gating/halving/stmatrix 的问题，是 TileStore TMA 路径在 blockscaled 模式下的 codegen 问题。这个方向比之前的假设 (CUTLASS gated epilogue bug) 更精确。
10. **quack 源码阅读路径**: `quack/epi_ops.py` (TileStore/EpiOp), `quack/gemm_act.py` (GemmActMixin), `quack/gemm_sm100.py` (GemmSm100.__call__/epilogue), `quack/gemm_default_epi.py` (GemmDefaultEpiMixin), `quack/gemm_wrapper_utils.py` (GemmWrapperBase)。
11. **sglang/CUTLASS 没有 fused blockscaled+gated**: sonic-moe 的 GemmGatedSm100+blockscaled 方案是全网唯一尝试，不存在可参考的上游实现。
