# Blockscaled FP8 MoE — Status & Handoff

> **Last updated: 2025-07-22**

---

## 1. 目标

全链路 blockscaled FP8 (1×32 UE8M0) MoE training。Forward 和 Backward 均使用 native CUTLASS `GemmDefaultSm100` blockscaled varlen GEMM。精度 RelRMSE < 10%，性能和显存优于 BF16 baseline。

---

## 2. 当前状态

### ✅ 已完成 — 全链路 blockscaled FP8 forward + backward

**精度验证通过** (production shape T=4096, H=4096, I=1024, E=128, K=8):

| Metric | RelRMSE | Correlation | 状态 |
|--------|---------|-------------|------|
| Forward output | 6.61% | 0.9978 | ✅ |
| dx (activation grad) | 6.79% | 0.9977 | ✅ |
| dw1 (up-proj weight) | 5.69% | 0.9984 | ✅ |
| dw2 (down-proj weight) | 5.44% | 0.9985 | ✅ |
| d(router_scores) | 6.56% | 0.9978 | ✅ |

Contract tests: **8/8 PASSED** (`-k "not large_shape"`)

### 🔴 关键性能瓶颈 — 需要修复

FP8 目前比 BF16 **慢** (p25: 13.89ms vs 12.09ms, ~0.87x)。原因：

1. **CUTLASS `GemmGatedSm100` / `GemmDgatedSm100` 与 blockscaled FP8 不兼容** — 在 sm_100a 上编译 fused gated epilogue + `sf_vec_size=32` 时产生 illegal instruction 或全零输出
2. 因此 FP8 被迫使用 **decomposed path**: non-gated `GemmDefaultSm100` + 手动 SwiGLU（多 kernel 启动，多次内存读写）
3. BF16 使用高度优化的 fused `gemm_gated` / `gemm_dgated` (单 kernel)
4. CUTLASS DSL 编译有 sporadic jitter（已缓存但偶尔失效）

---

## 3. GEMM 矩阵 (当前配置)

| 算子 | 实现 | Kernel | 说明 |
|------|------|--------|------|
| up-proj fwd | **Blockscaled FP8** | `blockscaled_fp8_gemm_varlen` (GemmDefaultSm100) | bf16 act + cached fp8 weights |
| down-proj fwd | **Blockscaled FP8** | `blockscaled_fp8_gemm_varlen` (GemmDefaultSm100) | bf16 act + cached fp8 weights |
| down-proj bwd act-grad | **Blockscaled FP8** | `blockscaled_fp8_gemm_varlen` (GemmDefaultSm100) | + 手动 SwiGLU backward |
| down-proj bwd wt-grad | BF16 fused | QuACK `gemm` varlen | FP8 wt-grad 7x slower |
| up-proj bwd act-grad | **Blockscaled FP8** | `blockscaled_fp8_gemm_varlen` (GemmDefaultSm100) | bf16 act + cached fp8 weights |
| up-proj bwd wt-grad | BF16 fused | QuACK `gemm` varlen | FP8 wt-grad 7x slower |

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

### 4.3 CUTLASS DSL Gated Kernel Bug

- `GemmGatedSm100` + `sf_vec_size=32` (blockscaled): illegal instruction at E=128, all-zero at small shapes
- `GemmDgatedSm100` + `sf_vec_size=32`: illegal instruction
- `GemmDefaultSm100` + `sf_vec_size=32`: ✅ works perfectly

根本原因: CUTLASS DSL 编译器在为 sm_100a 生成 fused gated/dgated epilogue + blockscaled FP8 代码时有 bug。Per-tensor FP8 (`sf_vec_size=1`) 下这些 kernel 均正常。

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

## 8. 下一步 — 性能优化路径（优先级排序）

### 🔥 P0: 修复 CUTLASS `GemmGatedSm100` blockscaled FP8 支持

这是性能瓶颈的根本原因。当前 FP8 被迫 decompose 为 non-gated GEMM + separate SwiGLU。修复后可直接使用 fused gated kernel，预期 **2x+ 性能提升** (从 2 kernels → 1 kernel + 消除 intermediate 读写)。

**排查方向**:
- CUTLASS DSL `GemmGatedSm100` 的 epilogue codegen 在 `sf_vec_size=32` 时的 NVVM lowering bug
- 可能需要修改 `quack/gemm_sm100.py` 中的 gated epilogue schedule
- 或在 CUTLASS 上游修复后升级 cutlass-dsl 包

### P1: 激活 pre-quantization (避免 padding bug)

当前 activations 以 bf16 传入 `blockscaled_fp8_gemm_varlen`，函数内部做量化。如果能修复 padding 路径的 dequantization (而非 raw cast)，可以将量化前移并跨 forward/backward 复用。

**具体修改**: `blockscaled_fp8_gemm_varlen` lines ~1353-1372，padding 时需要 `a_fp8 * a_scales` 而非 `a.to(bf16)`.

### P2: Weight-grad FP8

当前 weight-grad 使用 BF16 QuACK varlen GEMM。Blockscaled FP8 weight-grad (`blockscaled_fp8_weight_grad_gemm`) 在 E=128 时有 7 个 kernel launch per expert，比 BF16 慢 ~7x。

**解决方案**: 批量 weight-grad GEMM，或使用 QuACK 的 grouped GEMM 接口 + blockscaled scales。

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
