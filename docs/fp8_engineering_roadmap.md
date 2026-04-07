# SonicMoE FP8 Frontier — 完整工程规划

> Branch: `native-fp8-exploration` @ `b651e17` + z_is_fp8 fix
> Date: 2026-04-07
> Baseline: 14 passed / 0 failed (core MoE), full FP8 suite pending

---

## 当前状态 (Session 35)

| 项目 | 状态 |
|------|------|
| z_is_fp8 条件修复 | ✅ Done, 测试中 |
| Epilogue quant (kernel) | ✅ Validated in git history (0 byte error), 待 re-integrate |
| GemmDGated FP8 PreAct | ❌ Hard constraint (CUTLASS DSL) |
| 3-way 架构对比 | ✅ `docs/fp8_architecture_comparison.md` |

---

## Phase 1: 消除 z standalone quant + forward dequant

### 目标
将 GemmGated epilogue 内的 blockscaled quant 集成到生产路径，
消除 `fused_z_save_y1_quant` 的 standalone kernel launch + HBM round-trip。

### Step 1.1: z_is_fp8 条件修复 ✅ (已完成)
- **文件**: `sonicmoe/functional/__init__.py` L1034-1053
- **变更**: +10 行（prequant 检查 + dtype 放宽 + assert 守护）
- **影响**: 无功能变化，为 Step 1.2 铺路
- **收益**: 0（纯基础设施）

### Step 1.2: Re-integrate Epilogue Quant into GemmGated
- **文件**: `sonicmoe/quack_utils/gemm_gated.py` (+~170 行)
- **来源**: git history commits `74719cd..964ccbd`
- **内容**:
  - DSL 原语: `_f32_as_i32`, `_i32_as_f32`, `_rcp_approx_f32`
  - `BlockscaledScaleStore(EpiOp)`: scale gmem write via EpiOp state
  - `GemmGatedBlockscaledQuantMixin`: epilogue 内 integer+carry E8M0 quant
  - `EpilogueArguments` 扩展 `mZScale` 字段
- **风险**: 低（已在 Session 34 验证，0/1M byte mismatch with Triton/Paddle）
- **收益**:
  - 消除 `fused_z_save_y1_quant` kernel (~20-40µs)
  - z 在 epilogue 中直接以 fp8 输出，无 HBM round-trip

### Step 1.3: Re-integrate into GemmGatedSm100ZeroMat
- **文件**: `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` (+~8 行)
- **内容**: `GemmGatedSm100ZeroMatBlockscaledQuant` variant class
- **条件**: env var `SONIC_MOE_FP8_EPILOGUE_QUANT=1`（opt-in）

### Step 1.4: Forward path 集成
- **文件**: `sonicmoe/functional/__init__.py` (~50 行修改)
- **内容**:
  - `_fused_blockscaled_gated_forward` 新增 epilogue quant path
  - epilogue 直接输出 z_fp8 + z_scales → `_PREQUANTIZED_SCALES["z_fp8"]`
  - z_is_fp8 条件（Step 1.1）自动 pick up → ctx.save(z_fp8, z_scales)
  - 无需 `fused_z_save_y1_quant(z, y1)` for z
- **y1 仍独立 quant**: epilogue quant 只处理 z，y1 的 quant 仍由 `quantize_and_pack_activation` 完成

### Step 1.5: 验证
- 31 frontier tests pass (regression)
- per-tensor RRMSE < 3%
- Memory delta: -20~40µs latency, 0 memory delta (z save 大小不变，只是 quant 路径变了)

### Phase 1 收益汇总

| 指标 | Before (frontier) | After (epilogue quant) | Delta |
|------|-------------------|----------------------|-------|
| z quant kernels | 1 (fused_z_save_y1_quant) | 0 (epilogue 内) | **-1 kernel** |
| z HBM round-trip | BF16 write → read → FP8 write | FP8 direct write | **-1 round-trip** |
| Latency (fwd) | baseline | -20~40µs | **-20~40µs** |
| Memory | baseline | same | 0 |

---

## Phase 2: y1 prequant 优化

### 目标
当前 y1 在 GemmGated 输出后由 standalone `quantize_and_pack_activation` 量化。
探索是否可以在 epilogue 中同时输出 y1_fp8+scales。

### Step 2.1: 评估 Epilogue 同时输出 z_fp8 + y1_fp8
- **难点**: epilogue 已经输出 z(preact), y1(postact), z_scales 三个 tensor
  - 再加 y1_fp8 + y1_scales 会把 epilogue 变成 5-output
  - CUTLASS DSL 的 EpiOp 框架能否支持需要验证
- **替代方案**: fused Triton kernel `quantize_and_pack_activation(y1)` 已经很快 (~15µs)
  - 考虑到 y1 刚从 epilogue 写出，L2 还热，standalone quant 开销已经很小
- **结论**: ROI 需要 profiling 确认。如果 epilogue 5-output 导致 register pressure 过高反而更慢。

### Step 2.2: Stream overlap y1_quant ‖ z_scale_pack
- 如果 y1 和 z 的 scale packing 可以在不同 stream 上并行
- 前提: 两者写不同 memory region
- **预期收益**: ~5-10µs (minor)

### Phase 2 收益汇总

| 指标 | Delta |
|------|-------|
| Latency | -5~15µs (marginal) |
| Memory | 0 |
| 工程复杂度 | 中等 |
| **优先级** | **低** — ROI 不确定，Phase 3 更值得投入 |

---

## Phase 3: Backward 优化

### 目标
当前 backward 瓶颈：z_fp8 dequant (~124µs) + dout_quant+gather (~83+28µs)。
两者已在不同 stream 并行，但 z_dequant 仍是 critical path。

### Step 3.1: GemmDGated FP8 PreAct（长期攻坚）
- **目标**: 消除 backward z_dequant kernel
- **方法**: 修改 GemmDGated 的 C tensor 加载，支持 FP8 input + register dequant
- **文件**: `sonicmoe/quack_utils/gemm_dgated.py` (~100-200 行)
- **难点**:
  - L126: `assert args.implicit_dtype.width == 16` — hard constraint
  - L141-144: `recast_tensor(tRS_rC, implicit_dtype)` — bf16x2 packing trick
  - 需要在 C 加载后、bf16x2 packing 前插入 fp8→bf16 dequant
  - Scale 需要以某种方式传入 epilogue（新 EpiOp 或 C tensor 旁路）
- **预期收益**: **-124µs backward latency**
- **优先级**: 高（最大单点收益），但工程难度也最高

### Step 3.2: FP8 Weight Grad (可选)
- **目标**: wgrad `gemm(x^T, dz)` 用 FP8 blockscaled GEMM
- **前提**: x 和 dz 都已有 fp8+scales（x from forward quant, dz from dSwiGLU output）
- **文件**: `sonicmoe/functional/__init__.py` backward 中的 wgrad 路径
- **风险**: 精度影响大（wgrad 精度直接影响收敛）
- **预期收益**: ~2x wgrad TFLOPS（FP8 vs BF16 mainloop）
- **优先级**: 低 — 需要训练 loss 对齐验证

### Step 3.3: Backward dSwiGLU 从 FP8 z 直接计算
- **目标**: `swiglu_backward_from_fp8_triton` 已存在，直接用 z_fp8+scales 做 dSwiGLU
- **前提**: GemmDGated 不再需要 bf16 z（即 Step 3.1 完成后）
- **文件**: backward path 中替换 `dequantize_blockscaled_fp8 + GemmDGated` 为新路径
- **预期收益**: 消除 dequant kernel，但需要 GemmDGated FP8 PreAct 先就绪

### Phase 3 收益汇总

| 子步骤 | Latency Delta | 依赖 | 优先级 |
|--------|--------------|------|--------|
| 3.1 GemmDGated FP8 PreAct | **-124µs** | 无 | ⭐⭐⭐ |
| 3.2 FP8 wgrad | -50~100µs (估) | 精度验证 | ⭐ |
| 3.3 dSwiGLU from FP8 | -20µs | Step 3.1 | ⭐⭐ |

---

## Phase 4: 系统级优化 (长期)

### Step 4.1: z Recompute 开关
- 参考 DeepEP `recompute_moe_gate_up`
- Forward 不存 z，backward 重算 `z = GemmGated(x, w1)`
- **收益**: 完全消除 z activation memory (~213MB FP8 / ~384MB BF16)
- **代价**: +1 forward GEMM (~200µs)
- **适用场景**: 极端 memory 受限（超大 batch / 超大模型）
- **优先级**: 低（z FP8 save 已经很好）

### Step 4.2: EP 通信集成
- SonicMoE 当前是单节点 MoE，无 EP 通信
- 如需扩展到多节点，参考 DeepEP 的 `FusedDispatchAsync` 模式
- FP8 dispatch quant 可复用现有 blockscaled quant 基础设施
- **优先级**: 取决于产品路线图

### Step 4.3: 全 Native FP8 Params
- 预量化 x + 持久 FP8 权重 buffer + FP8 optimizer
- 完全不同的训练范式（类似 MSAMP）
- **优先级**: v2.0 远期

---

## 总收益预期

### 端到端 Latency (Forward + Backward, Ernie shape)

| 阶段 | Forward Delta | Backward Delta | 累计 |
|------|-------------|---------------|------|
| Baseline (frontier) | 0 | 0 | 0 |
| Phase 1 (epilogue quant) | **-20~40µs** | 0 | -20~40µs |
| Phase 3.1 (GemmDGated FP8) | 0 | **-124µs** | -144~164µs |
| Phase 3.2 (FP8 wgrad) | 0 | **-50~100µs** | -194~264µs |

### Memory (Ernie shape: T=8192, K=8, E=8, H=3072, I=1536)

| 阶段 | Delta vs BF16 |
|------|---------------|
| Frontier (已有) | **-171MB** (z FP8 save) |
| Phase 1 | 同上 (0 additional) |
| Phase 4.1 (z recompute) | **-384MB** (完全消除 z) |

### 工程投入估算

| Phase | 代码量 | 风险 | 预期收益 | ROI |
|-------|-------|------|---------|-----|
| Phase 1 | ~230 行 (re-integrate) | 低 | -20~40µs | ⭐⭐⭐ |
| Phase 2 | ~50 行 | 中 | -5~15µs | ⭐ |
| Phase 3.1 | ~200 行 | 高 | -124µs | ⭐⭐⭐ |
| Phase 3.2 | ~30 行 | 高(精度) | -50~100µs | ⭐⭐ |
| Phase 4 | ~100 行 | 低 | memory only | ⭐ |

---

## 推荐执行顺序

```
Now:     Phase 1.1 ✅ → 1.2 → 1.3 → 1.4 → 1.5 (validate)
Next:    Phase 3.1 (GemmDGated FP8 PreAct — 最大单点收益)
Then:    Phase 3.2 (FP8 wgrad — 需训练验证)
Defer:   Phase 2 (marginal ROI), Phase 4 (product-driven)
```
