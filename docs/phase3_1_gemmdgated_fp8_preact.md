# Phase 3.1: GemmDGated FP8 PreAct — 深度技术分析

> 最高单点 ROI 优化：消除 backward z dequant (~124µs)
> Date: 2026-04-07

---

## 1. 全局位置：当前数据流中 3.1 的切入点

### 1.1 Forward 数据流 (FP8 Frontier)

```
x(T,H) ─bf16─→ [quantize_and_pack] ─fp8─→ [scale_gather] ─isa_packed─→
                                                                        ↓
         ┌──────────────────────────────────────────────────────────────┐
         │ GemmGated (CUTLASS Epilogue)                                │
         │                                                             │
         │  FP8 Mainloop: x_fp8 × w1_fp8 → f32 accum                 │
         │         ↓                                                   │
         │  Epilogue: bias → tRS_rD (f32 z registers)                 │
         │         ├─→ SwiGLU(tRS_rD) → tRS_rPostAct → y1 (bf16)     │
         │         └─→ [Phase 1: epilogue quant] → z_fp8 + scales ◄── │  ← Phase 1 切入
         └──────────────────────────────────────────────────────────────┘
                    ↓                              ↓
              z_fp8 (TK,2I)                  y1 (TK,I)
              + z_scales (TK,2I/32)          ↓
              saved to ctx                [quantize] → y1_fp8
                                              ↓
                                     [blockscaled_fp8_gemm_varlen]
                                          y1_fp8 × w2_fp8 → y2 (bf16)
```

### 1.2 Backward 数据流 — 3.1 切入点

```
dout(T,H) ─→ [router_backward] ─→ dout(TK,H) ─bf16─→
                                                       ↓
    ┌─── z dequant (~124µs) ◄── Phase 3.1 消除目标 ──────────────┐
    │                                                             │
    │  z_fp8 + z_scales  ─→ [dequantize_blockscaled_fp8]         │
    │                         ↓                                   │
    │                     z_bf16 (TK, 2I) ← 384MB HBM write!     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────────────────────┐
         │ GemmDGated (CUTLASS Epilogue)                          │
         │                                                        │
         │  BF16 Mainloop: dout × w2^T → f32 accum (= D)        │
         │         ↓                                              │
         │  Epilogue:                                             │
         │    C = PreAct = z_bf16 (loaded from gmem as f32 view) │  ← 3.1 改这里
         │         ↓                                              │
         │    recast(C_f32, bf16) → bf16x2 pairs                 │
         │    bf16x2 → f32x2 → dSwiGLU(x_f32, y_f32, D)        │
         │         ↓                                              │
         │    dx_out (TK, 2I)  +  postact_out (TK, I)           │
         └────────────────────────────────────────────────────────┘
                              ↓                     ↓
                          dz (bf16)            dSwiGLU_out
                              ↓
                 [gemm(x^T, dz)] → dw1
```

### 1.3 3.1 的目标

```
BEFORE (当前):
  z_fp8 + z_scales ──→ [dequant kernel: 124µs] ──→ z_bf16 ──→ GemmDGated(C=z_bf16)

AFTER (Phase 3.1):
  z_fp8 + z_scales ──→ GemmDGated(C=z_fp8, c_scales=z_scales)
                       ↓ (kernel内部)
                       load C_fp8 → register dequant → bf16x2 → dSwiGLU
```

**消除的开销**:
- `dequantize_blockscaled_fp8` kernel launch (~5µs dispatch)
- z_bf16 HBM 写入 (~384MB @ Ernie shape → ~119µs @ 3.2TB/s B200 bandwidth)
- z_bf16 从 GemmDGated C tensor 读回 (~119µs，但与 mainloop 重叠)
- 总计: **~124µs latency + 384MB 临时内存**

---

## 2. GemmDGated 的 C Tensor 处理机制

### 2.1 bf16x2 packing trick

GemmDGated 将 z (shape `(TK, 2I)`, dtype bf16) 作为 C tensor 传入。
但 CUTLASS 的 C tensor 走 TMA 路径，要求 32-bit element。所以：

```python
# gemm_dgated.py L284-289 — Python 层面
assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
# 将 (TK, 2I) bf16 重解释为 (TK, I) f32
PreAct = PreAct.view(torch.float32)
```

这个 `view(float32)` 不做任何数据拷贝，只改了 metadata：
- 原始: `(TK, 2I)` bf16, stride `(2I, 1)`
- view后: `(TK, I)` f32, stride `(I, 1)` — 每个 f32 元素包含 2 个 bf16

### 2.2 Epilogue 中的解包

```python
# gemm_dgated.py L142-144 — CUTLASS DSL epilogue
tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)  # f32 → bf16x2 view
tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))      # bf16 → f32 widen
```

1. `recast_tensor(tRS_rC, bf16)`: 将 f32 寄存器重解释为 bf16 对 — 零开销
2. `load().to(Float32)`: bf16 → f32 类型提升 — 一条 CVT 指令
3. 结果 `tRS_rXY_f32x2`: 2N 个 f32 值 = z 的 gate 和 up 分量

### 2.3 dSwiGLU 计算

```python
# SM100 path (L174-183)
(dgate, dup, postact) = act_bwd_fn(
    (z_gate_0, z_gate_1),      # tRS_rXY_f32x2[4i], [4i+2]
    (z_up_0, z_up_1),          # tRS_rXY_f32x2[4i+1], [4i+3]
    (dout_0, dout_1),          # tRS_rD_scaled[2i], [2i+1]
)
```

`act_bwd_fn` = `dswiglu` — 需要 f32 精度的 gate 和 up 值来计算：
- `dgate = dout * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))`
- `dup = dout * silu(gate)`
- `postact = dout * silu(gate)` (用于 wgrad)

### 2.4 约束链

```
assert PreAct.element_size() == 2     ← Python: 只接受 bf16/fp16
    ↓
PreAct.view(torch.float32)            ← Python: bf16x2 → f32 view
    ↓
C tensor 以 f32 dtype 进入 CUTLASS   ← CUTLASS: TMA loads f32
    ↓
assert args.implicit_dtype.width==16  ← DSL: recast 需要 16-bit
    ↓
recast_tensor(tRS_rC, bf16)           ← DSL: f32 → bf16x2 view
    ↓
load().to(Float32)                    ← DSL: bf16 → f32 widen
    ↓
dSwiGLU(gate_f32, up_f32, dout_f32)  ← DSL: 需要 f32 精度
```

---

## 3. FP8 PreAct 方案分析

### 3.1 方案 A: Register-Level Dequant (推荐)

**思路**: C tensor 仍以 f32 view 进入（保持 TMA 路径不变），但 f32 中不再装 bf16x2，而是装 fp8x4 + 外挂 scales。

```
Memory layout:
  z_fp8: (TK, 2I) fp8e4m3 = (TK, 2I/4) int32 = (TK, I/2) f32
  z_scales: (TK, 2I/32) uint8

TMA load:
  C tensor = z_fp8.view(float32)  → (TK, I/2) f32
  每个 f32 = 4 个 fp8 bytes

Epilogue:
  tRS_rC: (I/2) f32 registers
  ↓
  recast as fp8x4 → 2I fp8 values
  ↓
  dequant: fp8_val * scale → f32 (per 32-element group)
  ↓
  dSwiGLU(gate_f32, up_f32, dout_f32)
```

**关键变更**:
1. Python: `PreAct.view(float32)` 改为 `z_fp8.view(float32)` — N 维度缩小 4x
2. DSL `epi_to_underlying_arguments`: 接受 `implicit_dtype = fp8e4m3` (width=8)
3. DSL `epi_visit_subtile`: recast f32→fp8x4，然后 fp8→f32 dequant（带 scale）
4. Scale tensor 通过新 EpiOp 传入

**难点**:
- C tensor 的 shape 从 `(TK, I)` f32 变成 `(TK, I/2)` f32 → tile_N 不匹配
- GemmDGated 的 mainloop output D 是 `(TK, I)` f32（2N 个 bf16 == N 个 f32）
- C tensor 的 tile 必须与 D tensor 的 tile 对齐 → shape 不同无法对齐

**这是根本性冲突**: D 和 C 必须是相同 shape。如果 C 是 fp8x4 packed，其 shape 只有 D 的 1/2。

### 3.2 方案 B: 保持 C shape，fp8 数据 zero-pad 到 bf16 宽度

```
Memory:
  z_fp8_padded: (TK, 2I) uint16 — 每个 uint16 的低 8 bit = fp8, 高 8 bit = 0
  这是一个 2x 的内存浪费

TMA load:
  C = z_fp8_padded.view(float32) → (TK, I) f32
  每个 f32 = 2 个 padded-fp8-as-uint16

Epilogue:
  recast_tensor(tRS_rC, uint16) → 提取低 8 bit → fp8
  dequant with scale → f32
  dSwiGLU(...)
```

**问题**: 2x 内存浪费（跟 bf16 一样大），失去 FP8 的内存优势。

### 3.3 方案 C: 分离 mainloop 和 epilogue 的 C 路径 (最可行)

```
修改 GemmDGatedMixin:

1. C tensor 不走 TMA/standard epilogue C load
2. 新增一个 EpiOp: "BlockscaledC_FP8Load"
   - 在 begin() 中: 计算 tile 的绝对坐标
   - 在 begin_loop() 中: 传递 subtile 坐标
   - epi_visit_subtile 负责:
     a. 从 gmem 直接 LDG 加载 fp8 bytes (不走 TMA)
     b. 从 gmem LDG 加载 scale bytes
     c. register dequant: fp8 * (2^scale_exp) → f32
     d. 构造 tRS_rXY_f32x2（与现有 dSwiGLU 接口对齐）

3. epi_visit_subtile 的修改:
   if const_expr(self.fp8_preact):
       # C 已经在 EpiOp 中加载并 dequant
       tRS_rXY_f32x2 = epi_loop_tensors["BlockscaledC_FP8Load"]
   else:
       # 现有 bf16x2 路径
       tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
       ...

4. Python: C=None, z_fp8+scales 通过 EpiOp 传入 EpilogueArguments
```

**优势**:
- 不需要 D 和 C shape 一致（C 通过 EpiOp 独立加载）
- FP8 + scales 的总内存 ~55% of bf16
- dequant 在 register 中完成，与 dSwiGLU 融合
- 不需要修改 CUTLASS mainloop 或 TMA

**难点**:
- EpiOp 需要做 gmem→register 的 LDG（不走 TMA，可能慢于 TMA）
- 但 C tensor 的 bandwidth 需求 << D tensor（读 1 次 vs mainloop 的持续读写）
- Scale tensor 是小数据量（2I/32 per row ≈ 192 bytes @ I=3072）

### 3.4 方案 D: 不改 GemmDGated，优化 dequant kernel 本身

```
当前: dequantize_blockscaled_fp8 是 Triton kernel，~124µs
可能优化:
  1. 用 CUTLASS/CUDA kernel 替代 Triton → maybe 80µs?
  2. 与 dout_quant 在同一 stream 并行 → 隐藏延迟
     (当前已经在并行: z_dequant ‖ dout_quant+gather)
  3. z_dequant 输出直接写到 C tensor 位置（避免额外 alloc）

实际收益有限：124µs 中大部分是 HBM bandwidth (~384MB write)
```

---

## 4. 三方对比: Backward 中 PreAct 的处理

### 4.1 数据流对比

| 步骤 | DeepEP | SonicMoE BF16 | SonicMoE FP8 (当前) | SonicMoE FP8 (3.1) |
|------|--------|---------------|--------------------|--------------------|
| z 存储 | BF16 or recompute | BF16 (384MB) | FP8+scales (213MB) | FP8+scales (213MB) |
| z→backward | 直接用 | 直接用 | dequant→BF16 (124µs) | **kernel内 dequant** |
| dSwiGLU | 独立 kernel | GemmDGated fusion | GemmDGated fusion | GemmDGated fusion |
| z 读取次数 | 1 (独立kernel) | 1 (TMA) | 2 (dequant write + TMA read) | **1 (EpiOp LDG)** |
| z bandwidth | 384MB read | 384MB read | 384MB write + 384MB read | **213MB read** |

### 4.2 性能对比

| 指标 | DeepEP | SonicMoE BF16 | SonicMoE FP8 (当前) | SonicMoE FP8 (3.1) |
|------|--------|---------------|--------------------|--------------------|
| z dequant | 0 (bf16) | 0 (bf16) | ~124µs | **0** |
| z HBM traffic | 384MB | 384MB | 768MB (R+W) | **213MB** |
| z memory | 384MB | 384MB | 213MB (saved) | **213MB (saved)** |
| dSwiGLU fusion | ✗ | ✓ | ✓ | ✓ |

### 4.3 架构对比

```
DeepEP Backward:
  z(BF16) ──────────────────→ dSwiGLU_kernel(z, dout) → dx, postact
                                                        ↓
                                                  GroupGEMM(dx, w)

SonicMoE BF16 Backward:
  z(BF16) ──TMA──→ GemmDGated epilogue: recast→dSwiGLU → dx, postact
                   (single fused kernel)

SonicMoE FP8 Current:
  z_fp8 ──→ [dequant kernel] ──→ z_bf16 ──TMA──→ GemmDGated epilogue
                  124µs              384MB write       same as BF16

SonicMoE FP8 Phase 3.1 (方案C):
  z_fp8 ──LDG──→ GemmDGated epilogue: EpiOp dequant→dSwiGLU → dx, postact
  z_scales ─LDG──↗    (single fused kernel, no standalone dequant)
```

---

## 5. 推荐方案: C (方案 C 详细设计)

### 5.1 新增 EpiOp: `BlockscaledPreActLoad`

```python
class BlockscaledPreActLoad(EpiOp):
    """EpiOp: loads FP8 PreAct + scales from gmem, dequants in register.

    Replaces the standard C tensor TMA load for GemmDGated when PreAct is FP8.

    begin(): captures (z_fp8_tensor, z_scales_tensor, tidx, tile_coord, varlen_mgr)
    begin_loop(): computes absolute coordinates for current subtile
    epi_visit_subtile result: dequanted f32x2 values (gate, up pairs)
    """
```

### 5.2 修改 `GemmDGatedMixin.epi_visit_subtile`

```python
@cute.jit
def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
    if const_expr(self.fp8_preact):
        # FP8 PreAct path: C is None, dequanted values come from EpiOp
        preact_dequant_info = epi_loop_tensors["BlockscaledPreActLoad"]
        z_fp8_ptr, z_scale_ptr, m_abs, n_abs = preact_dequant_info

        # Manual LDG: load 2I fp8 bytes for this thread's M-row
        # Each thread owns 1 M-row (SM100 Ld32x32bOp mapping)
        num_elements = cute.size(tRS_rD) * 2  # 2I elements (gate + up)

        # Load fp8 bytes
        fp8_vals = [cute.arch.ldg_u8(z_fp8_ptr, m_abs, n_abs + i)
                    for i in range(num_elements)]

        # Load scale (1 byte per 32-element group)
        n_groups = num_elements // 32
        scales = [cute.arch.ldg_u8(z_scale_ptr, m_abs, n_abs // 32 + g)
                  for g in range(n_groups)]

        # Dequant: fp8 * 2^(scale - 127) → f32
        tRS_rXY_f32x2 = cute.make_rmem_tensor(...)
        for g in range(n_groups):
            dequant_scale = _i32_as_f32(Int32(scales[g]) << Int32(23))
            for j in range(32):
                idx = g * 32 + j
                tRS_rXY_f32x2[idx] = fp8_to_f32(fp8_vals[idx]) * dequant_scale

    else:
        # Standard bf16x2 path (unchanged)
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

    # Rest of dSwiGLU computation unchanged
    ...
```

### 5.3 Python 层修改

```python
# gemm_dgated.py wrapper
def gemm_dgated(..., z_fp8=None, z_scales=None):
    if z_fp8 is not None:
        # FP8 PreAct path: C=None, pass fp8+scales via EpilogueArguments
        epi_args = GemmDGatedFP8PreAct.EpilogueArguments(
            mPostAct=...,
            act_bwd_fn=...,
            mPreActFP8=z_fp8_cute,
            mPreActScales=z_scales_cute,
        )
    else:
        # Standard bf16 path
        PreAct = PreAct.view(torch.float32)
        ...
```

### 5.4 关键实现挑战

| 挑战 | 难度 | 解决方案 |
|------|------|---------|
| LDG vs TMA 性能 | 中 | C 只读 1 次，bandwidth 需求低 (~213MB)，LDG 足够 |
| fp8→f32 转换 | 低 | CUTLASS DSL 有 `cutlass.Float8E4M3FN` 类型支持 |
| scale 传入 epilogue | 低 | 通过 EpiOp，已有 `BlockscaledScaleStore` 参考 |
| C=None 时 TMA 路径 | 中 | GemmDGated 的 C tensor 走 standard epilogue C 路径，需要绕过 |
| register 布局对齐 | 高 | SM100 的 MMA 输出和 epilogue 寄存器映射需要精确匹配 |
| 与 colvec_scale/reduce 交互 | 中 | 需要确保 dout scaling 不受影响 |

---

## 6. 预估收益

### 6.1 Latency

```
当前 backward (FP8 frontier, Ernie shape):
  z_dequant:     ~124µs  ← Phase 3.1 消除
  dout_quant:     ~83µs  (parallel with z_dequant)
  scale_gather:   ~28µs  (parallel with z_dequant)
  GemmDGated:    ~200µs
  wgrad gemm:    ~150µs  (parallel with GemmDGated actgrad)
  ─────────────────────
  Critical path: max(124, 83+28) + max(200, 150) ≈ 124 + 200 = 324µs

Phase 3.1 after:
  dout_quant:     ~83µs
  scale_gather:   ~28µs
  GemmDGated(fp8): ~210µs (slightly slower due to EpiOp LDG, ~5% overhead)
  wgrad gemm:    ~150µs  (parallel)
  ─────────────────────
  Critical path: (83+28) + max(210, 150) ≈ 111 + 210 = 321µs

Wait... 这样看 latency 节省不明显？
```

**重新分析**：z_dequant 和 dout_quant 是并行的。当前的 critical path 取决于哪个更慢。

```
当前并行结构:
  Stream 0 (default): dout_quant(83µs) + scale_gather(28µs) = 111µs
  Stream 1 (dequant):  z_dequant(124µs)
  Sync point: max(111, 124) = 124µs
  Then: GemmDGated(200µs)
  Total bwd: 124 + 200 = 324µs

Phase 3.1:
  Stream 0: dout_quant(83µs) + scale_gather(28µs) = 111µs
  No z_dequant needed!
  Then: GemmDGated_fp8(210µs)
  Total bwd: 111 + 210 = 321µs

Delta: 324 - 321 = 3µs ???
```

**等等 — 这里的分析有误**。z_dequant 的 124µs 包含了 HBM write（384MB），
但 Phase 3.1 的 GemmDGated_fp8 需要读 z_fp8（213MB）而不是 z_bf16（384MB）。
所以 GemmDGated_fp8 的 C tensor 读取量是当前的 55%，epilogue 应该更快：

```
GemmDGated C tensor bandwidth:
  当前:     384MB bf16 read (TMA, 与 mainloop 重叠)
  Phase 3.1: 213MB fp8 read (LDG, 在 epilogue 中)

由于 C 读取与 mainloop 重叠（TMA 预取），当前的 200µs 已经包含了 C 读取。
Phase 3.1 用 LDG 替代 TMA 读取，C 数据量减少但延迟可能增加（LDG 不如 TMA 高效）。
```

### 6.2 真正的收益在 Memory

```
当前: z_dequant 需要额外分配 z_bf16 (384MB) 临时 buffer
Phase 3.1: z_fp8 直接传入 GemmDGated，无临时 buffer

Peak memory 节省: -384MB @ Ernie shape
```

### 6.3 修正后的收益预估

| 指标 | 当前 (FP8 Frontier) | Phase 3.1 | Delta |
|------|--------------------|-----------| ------|
| z dequant kernel | 124µs (stream overlap) | 0 | **-13µs critical path** |
| z_bf16 temp buffer | 384MB | 0 | **-384MB peak memory** |
| GemmDGated C bandwidth | 384MB (TMA) | 213MB (LDG) | **-171MB bandwidth** |
| Total bwd critical path | ~324µs | ~311µs | **-13µs** |

> 注意: latency 节省看似小（13µs），因为 z_dequant 已经与 dout_quant 并行。
> **但 memory 节省是显著的**: -384MB 临时 buffer。
> 在大 batch 或多 pipeline 场景下，这 384MB 可能是 OOM 的关键差距。

### 6.4 如果 Phase 3.1 + z_dequant stream 去掉后

去掉 z_dequant 意味着 `_get_dequant_stream()` 不再需要，简化了 stream 管理。
backward 的 stream 拓扑从 3 stream（default + dequant + wgrad）简化为 2 stream（default + wgrad）。

---

## 7. 实施建议

### 7.1 优先级重评估

Phase 3.1 的 **主要收益是 memory (-384MB)**，不是 latency。
如果当前场景不受 memory 限制，Phase 3.1 的 ROI 低于最初预期。

但如果将 Phase 3.1 与 Phase 1 组合：
- Phase 1: z epilogue quant → 消除 fwd z standalone quant (-20~40µs)
- Phase 3.1: GemmDGated fp8 PreAct → 消除 bwd z_bf16 temp buffer (-384MB)
- **组合效果**: z 全程 FP8，从 forward epilogue → ctx → backward epilogue，
  **从未以 bf16 存在于 HBM**。这是 SonicMoE FP8 的终极形态。

### 7.2 建议的实施顺序

```
1. 先完成 Phase 1 验证 (当前)
2. 研究 EpiOp LDG 的 CUTLASS DSL 可行性
   — 关键实验: 在 epi_visit_subtile 中用 LDG 读 gmem 是否允许
   — 参考: quack/epi_ops.py 的 ColVecLoad (已经做 gmem→register)
3. 实现 BlockscaledPreActLoad EpiOp (方案 C)
4. 修改 GemmDGatedMixin 添加 fp8_preact 条件分支
5. Python 层 threading
6. 端到端验证: 精度 + memory + performance
```

### 7.3 fallback 方案

如果方案 C 的 EpiOp LDG 在 CUTLASS DSL 中不可行（DSL 限制），
可以退而求其次:

- **方案 D (优化 dequant kernel)**: 将 Triton dequant → CUDA/CUTLASS，
  直接写入 GemmDGated 需要的 f32 view 位置，省去一次 alloc+copy
- **方案 E (z recompute)**: 不存 z，backward 重算 `z = GemmGated(x, w1)`
  消除所有 z 相关的 memory 和 bandwidth
  代价: +200µs fwd GEMM (但 memory 彻底消除)

---

## 8. 深度研究发现: N-Dimension Ratio Problem (核心阻塞因素)

> 以下来自对 CUTLASS DSL epilogue 全链路的深度逆向分析

### 8.1 bf16x2 Packing 的 N 维度影响

当前 GemmDGated 的 C tensor 处理链中，bf16→f32 view 将 N 维度缩小 2x：

```
逻辑维度:  z = (TK, 2I) bf16     -- 2I 个 bf16 元素
物理维度:  C = (TK, I) f32       -- I 个 f32 元素 (每个装 2 个 bf16)
比例:      N_logical = 2 * N_physical
```

这个 2:1 比例渗透到 epilogue 的所有层面：
- `cta_tile_shape_postact_mn` = `(tile_M, tile_N)`
- `epi_tile` = `(epi_M, epi_N)` — 每个 epi subtile 处理的 N 元素数
- TMA descriptor 的 tile 坐标
- PostAct/D output store 的 tile 映射

### 8.2 FP8x4 Packing 的 N 维度冲突

如果改用 fp8→f32 view，比例变为 4:1：

```
逻辑维度:  z_fp8 = (TK, 2I) fp8  -- 2I 个 fp8 元素
物理维度:  C = (TK, I/2) f32     -- I/2 个 f32 元素 (每个装 4 个 fp8)
比例:      N_logical = 4 * N_physical
```

**这导致 C 的 tile shape 与 D 的 tile shape 不一致**。CUTLASS epilogue 假设 C 和 D
具有相同的 physical N dimension（它们在同一个 epilogue tile 循环中被并行处理）。

### 8.3 Register-Level Dequant + Repack 方案 (最有前途)

```
目标: 在 epilogue 寄存器中将 fp8x4 → bf16x2 in f32，恢复与 bf16 路径相同的格式

步骤:
  1. TMA load: C = z_fp8.view(f32) → (TK, I/2) f32, tile = (tile_M, tile_N/2)
  2. smem→register: tRS_rC 持有 I/2 个 f32 值（每个 = 4 个 fp8 byte）
  3. [新增] Register dequant:
     for each f32 register:
       extract 4 fp8 bytes via bitcast
       convert each fp8 → f32 (hardware CVT)
       multiply by dequant_scale (from EpiOp-loaded UE8M0 scale)
       convert f32 pair → bf16 pair
       pack bf16x2 → f32 (inverse of unpack2x16_as_2xf32)
  4. 此时 tRS_rC_repacked 持有 I 个 f32 值（每个 = 2 个 bf16）
     ——与 bf16 路径完全相同的格式！
  5. 继续 recast_tensor + .to(Float32) + dSwiGLU (不变)
```

**核心问题**: 步骤 1-2 的 tile shape 是 `(tile_M, tile_N/2)`，但步骤 4-5 期望
`(tile_M, tile_N)`。epilogue loop 的迭代次数、PostAct store、D store 都基于
`tile_N`。如果 C 的 physical N 只有一半，epilogue loop 无法对齐。

### 8.4 可行性评估矩阵 (更新)

| 方案 | 可行性 | Register 变更 | TMA/smem 变更 | tile 对齐问题 | 代码量 |
|------|--------|-------------|--------------|--------------|-------|
| **A: fp8 view-as-f32** | 中 | fp8x4 解包 + dequant + bf16x2 repack | 最小 | **严重**: N/4 vs N/2 | ~200 LOC |
| **B: Native fp8 TMA** | 难 | 同 A | 需要 8-bit TMA + smem copy | **严重**: 同 A | ~300 LOC |
| **C: EpiOp LDG** | 中 | 加 LDG + dequant | 新 EpiOp | **绕过**: C 不走 tile 对齐 | ~150 LOC |
| **D: 优化 dequant kernel** | 易 | 无 | 无 | 无 | ~30 LOC |
| **E: 双 epilogue tile pass** | 中 | 同 A | C 用 half-size tile | 需要 2x epi loop | ~250 LOC |
| **F: C=None + EpiOp gmem load** | **最佳** | dequant in register | 新 EpiOp (参考 ColVecLoad) | **完全绕过** | ~200 LOC |

### 8.5 方案 F 详细设计 (推荐)

**核心思路**: C tensor 设为 `None`，完全不走 TMA/standard C load 路径。
FP8 z 和 scales 通过 **两个新 EpiOp** 从 gmem 直接加载到 register。

```
GemmDGated epilogue flow (方案 F):

  Mainloop: dout × w2^T → tRS_rD (f32 accum) ← 不变

  Epilogue:
    tRS_rC = None (no standard C load)  ← C tensor 不参与 TMA

    EpiOp "FP8PreActLoad":
      begin(): capture (z_fp8_ptr, z_scales_ptr, tile_coords, tidx)
      begin_loop(): compute (m_abs, n_abs) for current subtile

    epi_visit_subtile():
      // 从 EpiOp 获取 dequant 后的 f32 值
      preact_info = epi_loop_tensors["FP8PreActLoad"]
      z_fp8_base, z_scales_base, m_abs, n_abs = preact_info

      // 直接 LDG 加载 fp8 bytes + scales
      for each element in subtile:
        fp8_byte = z_fp8_base[m_abs, n_abs + i]   // 1 byte LDG
        scale = z_scales_base[m_abs, (n_abs + i) // 32]  // shared per 32
        f32_val = fp8_to_f32(fp8_byte) * exp2(scale - 127 + 8)

      // 构造 tRS_rXY_f32x2 (与 bf16 路径相同接口)
      tRS_rXY_f32x2 = ... // gate/up f32 pairs

      // dSwiGLU (不变)
      act_bwd_fn(tRS_rXY_f32x2[gate], tRS_rXY_f32x2[up], tRS_rD_scaled)
```

**方案 F 的优势**:
1. **完全绕过 tile 对齐问题**: C=None，不走 TMA，不需要 C 和 D 同 shape
2. **参考已有模式**: `ColVecLoad` EpiOp 已经做 gmem→register 的 LDG 读取
3. **FP8 读取量小**: 每个 thread 读 ~64 bytes (32 fp8 elements) + 2 bytes (scales)
4. **精度保证**: dequant 在 f32 register 中完成，精度等于 standalone dequant

**方案 F 的风险**:
1. **LDG vs TMA 性能**: LDG 的延迟比 TMA 高，但 C 只读 1 次且数据量小
2. **C=None 时 epilogue 是否跳过 C 路径**: 需要确认 CUTLASS DSL 当 C=None 时
   `tRS_rC` 是否为 None（从代码看 `epi_visit_subtile` 已处理 `tRS_rC=None` 的情况）
3. **SM100 的 register 布局**: LDG 加载的数据在 register 中的排布可能与
   TMA→smem→register 路径不同，需要适配

### 8.6 grouped_gemm.py 中的参考实现

研究发现 `grouped_gemm.py` 有显式的 bf16x2 pack/unpack:

```python
# grouped_gemm.py L691-716
def unpack2x16_as_2xf32(self, a: Float32, dtype):
    """从 1 个 f32 (装 2 个 bf16) 中提取 2 个 f32"""
    vec_f32x1 = vector.from_elements(T.vector(1, T.f32()), (a.ir_value(),))
    vec_f16x2 = vector.bitcast(T.vector(2, vec_dst_type), vec_f32x1)
    res0 = Float32(vector.extract(vec_f16x2, ..., position=[0]))
    res1 = Float32(vector.extract(vec_f16x2, ..., position=[1]))
    return res0, res1

def pack2x16_as_f32(self, a: Float32, b: Float32, dtype):
    """将 2 个 f32→bf16 后打包为 1 个 f32"""
    ...
```

这为方案 F 的 "dequant → f32 → bf16 → bf16x2 in f32" 提供了精确的 DSL API 参考。

---

## 9. 最终路线图

```
Phase 1 (当前, ✅ in progress):
  1.1 z_is_fp8 条件修复 ✅
  1.2 Epilogue quant re-integrate ✅
  1.3 ZeroMat variant ✅
  1.4 Forward path wiring ✅
  1.5 Validation (running)

Phase 3.1 (下一步):
  实施方案 F: C=None + FP8PreActLoad EpiOp

  Step 1: 确认 C=None 时 epilogue 行为
  Step 2: 实现 FP8PreActLoad EpiOp
  Step 3: 修改 GemmDGatedMixin.epi_visit_subtile
  Step 4: Python 层 threading (gemm_dgated wrapper)
  Step 5: 集成到 _DownProjection.backward
  Step 6: 端到端验证

  预期收益:
    - Memory: -384MB (消除 z_bf16 temp buffer)
    - Bandwidth: -555MB (768→213MB z HBM traffic)
    - Latency: -13µs (消除 z_dequant critical path)
    - 代码简化: 去掉 dequant stream + dequant kernel
```
