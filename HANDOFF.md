# HANDOFF — Session 60 (2026-04-20)

## Project Status

SonicMoE is an FP8 MoE kernel library on Blackwell (SM100). The `paddle_compat` branch integrates it into ERNIE-core via `SonicMoEMlpNode` — a drop-in replacement for ERNIE's MlpNode that uses DeepEP's pre-sorted token layout.

**Branch**: `paddle_compat`
**Hardware**: NVIDIA B30Z (SM103, Blackwell)
**Environment**: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/baidu/ernie/erniebot/eb_venv`

## What Was Done This Session (Session 60)

### Core Achievement: FP8 Frontier ↔ DeepEP Integration

Implemented `SonicMoEMlpNode` (`sonicmoe/ernie_compat/mlp_node_v2.py`) that wraps the FP8 frontier kernels for ERNIE's DeepEP training flow. Key design decisions:

1. **Route-level padding** (matching frontier `_pad_routing_metadata`): x is NOT padded. Padding rows use `x_gather_idx=0` with `score=0`. Zero contribution, zero x modification. CUDA kernel (`deepep_metadata_cuda/kernel.cu`) updated accordingly.

2. **Fused wgrad GEMM epilogue accumulation**: `_run_cutlass_blockscaled_gemm_varlen_k_accumulate()` in `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` uses CUTLASS epilogue `D = A@B + 1.0*C` (beta=1) to accumulate wgrad directly into fp32 main_grad buffer inside the GEMM kernel. **Zero extra kernels for main_grad accumulation.**

3. **Metadata caching**: When `tokens_per_expert` is unchanged, all routing tensors are reused (zero GPU cost for frozen routing).

4. **`flush_native_grads()`**: Defers the interleave→split-half layout conversion to optimizer-step time (not per-iter).

### Performance (nsys GPU-projection, T=65536, E=8, I=1536, H=3072)

| Metric | Frontier | MlpNode (fused) | Overhead |
|--------|----------|-----------------|----------|
| GPU-projection/iter | 2739 μs | 3190 μs | +16.5% |
| GemmGated (fwd) | 452 μs | 466 μs | +3% |
| GemmDGated (bwd) | 383 μs | 411 μs | +7% |
| quackgemm wgrad | 1067 μs | 1278 μs | +20% (epilogue cost) |
| TilingSwapDim | 0 | 0 | ELIMINATED |
| MultiPrecisionAdd | 0 | 0 | ELIMINATED |

**Remaining +16.5% overhead breakdown:**
- wgrad GEMM epilogue (fp32 C read/write): +211 μs — inherent cost of fused accumulation
- `_quantize_and_pack`: +183 μs — **fundamental**: DeepEP has T=TK (no fan-out duplication; frontier only quantizes T=8192 rows then GEMM gathers via A_idx, we must quantize all 65536 unique rows)
- `token_gather_sum`: +131 μs — 8x more grid blocks in `is_varlen_K=True` path (65536 output positions vs 8192)

**These are NOT migration overhead** — they are fundamental workload differences between "T unique tokens × K experts" (frontier) vs "TK unique token-expert pairs" (DeepEP).

### Precision

| Metric | Value |
|--------|-------|
| Cosine similarity (FP8 vs BF16 gold) | 0.998 |
| SNR | 23.7 dB |
| Determinism (same input, repeated) | bit-exact (max diff = 0) |
| Accumulation correctness (4-iter / 1-iter norm ratio) | 2.00 (= √4, mathematically exact) |
| main_grad norms (T=65536, 12 iters) | w1: 1.88±0.15, w2: 0.94±0.07 |

### Memory (T=65536, E=8)

- Baseline (after warmup): ~2672 MiB
- Peak (during iter): ~5570 MiB
- Per-iter transient: ~2900 MiB (dominated by y1/z [TK, 2I] activations)

## Critical Files

| File | Role |
|------|------|
| `sonicmoe/ernie_compat/mlp_node_v2.py` | `SonicMoEMlpNode` + `_SonicMoEDeepEPFunc` PyLayer |
| `sonicmoe/ernie_compat/mlp_node.py` | Weight stacking, native-layout grad accumulation, `flush_native_grads()` |
| `sonicmoe/ernie_compat/deepep_metadata.py` | DeepEP → SonicMoE metadata (route-level padding, CUDA V2 kernel) |
| `sonicmoe/ernie_compat/deepep_metadata_cuda/kernel.cu` | Vectorized fill kernel (route-level: padding→row 0) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | `_run_cutlass_blockscaled_gemm_varlen_k_accumulate()` — fused wgrad+accum |
| `sonicmoe/functional/__init__.py` | Modified `_UpProjection.backward` + `_DownProjection.backward` to detect `_wgrad_w1/w2_accumulator` |
| `tests/ops/test_e2e_mlpnode.py` | E2E benchmark (--frontier-compare, --nsys, --parse-sqlite) |
| `tests/ops/test_mlpnode_audit.py` | Rigorous precision/performance/memory audit |

## Key Information Sources

| What | Where |
|------|-------|
| Frontier nsys profile (gold reference) | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/nsys/fp8_T8192_I1536_E8K8_211135.sqlite` |
| Frontier nsys workload template | `tools/introspect.py:1426` (`_NSYS_WORKLOAD_TEMPLATE`) |
| ERNIE MlpNode contract | `ernie-core/src/ernie_core/models/moe/moe_layer.py:1601-2200` |
| Paddle moe_permute_kernel | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/Paddle_B/paddle/phi/kernels/gpu/moe_permute_kernel.cu` |
| QuACK GEMM epilogue | `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack/quack/gemm_default_epi.py` |
| Route-level padding design | `sonicmoe/functional/__init__.py:222-296` (`_pad_routing_metadata`) |
| Zero-materialization design | `sonicmoe/functional/__init__.py:104-218` (`_fused_blockscaled_gated_forward`) |

## Hard-Won Lessons

1. **Frontier的T和DeepEP的T语义不同**: Frontier的T=8192是unique tokens，K=8 experts/token → TK=65536 GEMM行。DeepEP的T=65536是token-expert pairs总数（已展开）。直接用T=8192跑MlpNode测试得到的GEMM时间只有frontier的1/8，容易误以为"计算没有真正执行"。**正确的对照: 用T=65536。**

2. **Paddle compat dtype不兼容**: `paddle.int32 != torch.int32`（不同对象）。`count_cumsum`的assert会失败。已在`sonicmoe/count_cumsum/__init__.py`放宽检查。

3. **Non-contiguous tensor的add_()**: Paddle/PyTorch对non-contiguous lhs/rhs的`add_()`会隐式调用`.contiguous()` → 产生`TilingSwapDim1And2` kernel。解决方案: 确保buffer和operand的物理布局一致，或使用GEMM epilogue融合。

4. **GEMM epilogue beta=1 accumulation**: QuACK的`GemmDefaultSm100.EpilogueArguments(beta=Float32(1.0))` + C=D(同一tensor) 实现了零额外kernel的inplace accumulation。但这会增加GEMM本身的时间(~20%)因为epilogue需要额外读写fp32 C tensor。净效果: 省掉170μs的TilingSwap+MultiPrecisionAdd，增加~210μs的epilogue开销 → **当workload较小时略亏，但对大workload有利（epilogue成本不随M增长而增长）**。

5. **`is_varlen_K=True`的开销**: 使用identity layout (每个position映射到1个expert) 时，`token_gather_sum_kernel`的grid size等于TK（65536 blocks），每个block只处理K=1个element。相比frontier的8192 blocks × K=8，总work相同但launch overhead高8x。潜在优化: 对identity permutation case用简单的`out = y2 * score` element-wise kernel替代。

## Next Steps (Recommended Priority)

### P0: DeepEP FP8 Input Support
DeepEP可以直接提供FP8格式的`recv_x`。如果跳过`quantize_and_pack_activation`（87μs），MlpNode的overhead会从+16.5%降到+10%以内。在`_SonicMoEDeepEPFunc.forward`中添加`if x.dtype == torch.float8_e4m3fn`分支。

### P1: token_gather_sum Identity Shortcut
当检测到`s_reverse_scatter_idx`是identity permutation且score全为1时，`_router_forward`可以直接用`out = y2[:T]`（zero-copy slice），彻底消除138μs/iter的gather-sum kernel。

### P2: Quantize Skip for Large T
当T==TK（无fan-out），`_fused_blockscaled_gated_forward`的scale gather是no-op。可以检测这个case，跳过gather kernel（省~17μs）。

### P3: Correctness — Gold Comparison via frontier
当前`test_mlpnode_audit.py`用BF16手动计算作为gold。理想的gold是直接调用frontier的`moe_general_routing_inputs`。但Paddle compat的`zeros_like` tensor type不兼容问题阻塞了这个路径。需要修复`_DownProjection.forward`内部的tensor创建逻辑。

### P4: dx (input gradient) 传播
当前Paddle PyLayer的`backward`返回dx，但上层`x.grad`为None（Paddle autograd对detached input的grad传播行为不同）。需要验证在真实ERNIE训练循环中dx是否正确传播到上游。

## Running Tests

```bash
source .runenv.sh

# Quick validation
CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py

# Frontier-compare (same x/routing, matches frontier methodology)
CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py --frontier-compare --T 65536

# Full precision/memory/performance audit
CUDA_VISIBLE_DEVICES=0 python tests/ops/test_mlpnode_audit.py --T 1024

# nsys profiling
nsys profile -c cudaProfilerApi --capture-range-end=stop --export=sqlite \
  -o /tmp/mlpnode python tests/ops/test_e2e_mlpnode.py --frontier-compare --nsys --T 65536

# Parse nsys results
python tests/ops/test_e2e_mlpnode.py --parse-sqlite /tmp/mlpnode.sqlite
```
