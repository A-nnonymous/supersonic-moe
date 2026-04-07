# FP8 全链路最优解法 — 数据驱动的优化路线图

> 基于 nsys 实测数据 (T=8192, E=8, K=8, H=3072, I=1536, B200 idle node)
> Branch: `native-fp8-exploration` @ `4beed75`

---

## 1. 现状：nsys 实测瓶颈分析

### 1.1 Backward Kernel 时间分布

```
Kernel                            Time(µs)    %     Precision   Status
─────────────────────────────────────────────────────────────────────────
wgrad up-proj  gemm(x.T, dz)       3,490   66.2%    BF16       ← #1 瓶颈
GemmDGated     dout×w2^T+dSwiGLU     486    9.2%    FP8 BS     ✓ 已优化
actgrad        fp8_gemm(dz, w1^T)    430    8.2%    FP8 BS     ✓ 已优化
wgrad down-proj gemm(dout.T, y1)     387    7.3%    BF16       ← #2 瓶颈
dout quant+gather                    173    3.3%    Triton     stream overlap
z dequant                            130    2.5%    Triton     stream overlap
token_gather_sum                      66    1.2%    Triton     已优化
other                                ~92    1.8%    various    —
─────────────────────────────────────────────────────────────────────────
TOTAL                              5,254   100.0%
```

### 1.2 关键发现

| 发现 | 影响 |
|------|------|
| **wgrad BF16 = 73.5% of backward** | 这是唯一值得大力优化的瓶颈 |
| actgrad 已是 FP8 (8.2%) | 无需优化，已经是理论最优 |
| GemmDGated 已是 FP8 (9.2%) | mainloop 已优化，epilogue 是固定开销 |
| z dequant 只有 2.5% | Phase 3.1 的延迟收益微小 |
| dout quant 已与 z dequant 并行 | stream overlap 已在用 |

### 1.3 Backward 显存分析 (peak = 1306 MiB temp)

```
张量                      大小(MB)   生命周期                   可优化?
────────────────────────────────────────────────────────────────────────
z_bf16 (dequant temp)       384     dequant → GemmDGated end   ✓ Phase 3.1
dz (TK, 2I) bf16            384     GemmDGated → wgrad end     可 fp8 化
dx_expanded (TK, H) bf16    384     actgrad output             必要
dw1 (2I, H, E) bf16         144     wgrad output               必要
dw2 (H, I, E) bf16           72     wgrad output               必要
dout_fp8+scales              198     quant output               必要
其他中间                     ~140     various                    —
────────────────────────────────────────────────────────────────────────
Total temp                 ~1306     MiB
```

---

## 2. 理论最优全链路 FP8 设计

### 2.1 Forward 最优路径 (已基本实现)

```
x(T,H) bf16
  → quantize_and_pack(x) → x_fp8(T,H) + x_scales        [Triton, ~30µs]
  → scale_gather(x_scales, A_idx) → x_scales_tk           [Triton, ~8µs]
  → GemmGatedSm100ZeroMat(x_fp8, w1_fp8, SwiGLU)
       → z(TK,2I) bf16 [D output]                         [CUTLASS, ~800µs]
       → y1(TK,I) bf16 [PostAct output]
  → fused_z_save_y1_quant(z, y1)
       → z_fp8+z_scales [ctx save, 198MB]                 [Triton, ~50µs]
       → y1_fp8+y1_scales [prequant cache]
  → blockscaled_fp8_gemm(y1_fp8, w2_fp8)
       → y2(TK,H) bf16                                    [CUTLASS, ~300µs]
  → router_weighted_scatter → out(T,H) bf16

Forward total: ~1200µs, ctx memory: ~204MB (z_fp8 + z_scales)
```

### 2.2 Backward 当前路径 vs. 最优路径

```
                        当前路径                        最优路径
                        ────────                        ────────
Stream 0 (default):
  dout quant+gather     173µs  FP8                     173µs  FP8 (不变)
Stream 1 (dequant):
  z dequant             130µs  Triton                  0µs   (消除*)
Sync:                   max(173,130)=173µs             173µs

GemmDGated              486µs  FP8 BS                  486µs  FP8 BS (不变**)
  → dz(TK,2I) bf16                                    → dz(TK,2I) bf16
  → postact(TK,I)                                     → postact(TK,I)

dz quant               0µs    (不需要)                 ~80µs  新增 (for FP8 wgrad)

wgrad down-proj:
  gemm(dout.T, y1)     387µs  BF16                    ~190µs FP8 BS wgrad ★

Stream 0 (default):    actgrad                          actgrad
  fp8_gemm(dz,w1^T)    430µs  FP8 BS                  430µs  FP8 BS (不变)
Stream 1 (wgrad):      wgrad up-proj                    wgrad up-proj
  gemm(x.T, dz)        3490µs BF16                    ~1200µs FP8 BS wgrad ★★★

TOTAL                  ~5254µs                         ~2559µs (-51%!)

*  Phase 3.1: 需要 GemmDGated fp8 PreAct 支持
** 如果 Phase 3.1 实现，GemmDGated 直接读 fp8 z，节省 384MB
```

### 2.3 最优路径的显存改善

```
                        当前               最优
                        ────               ────
z_bf16 temp             384 MB             0 MB (Phase 3.1)
dz bf16                 384 MB             384 MB (不变)
dz_fp8+scales           0 MB               ~198 MB (FP8 wgrad 需要)
dw wgrad output         216 MB             216 MB (不变)
────────────────────────────────────────────────
Peak delta              —                  -186 MB net
```

---

## 3. 优化项按 ROI 排序

### 3.1 ⭐⭐⭐ FP8 Wgrad Up-Proj (最高优先级)

**目标**: `gemm(x.T, dz)` BF16 → `blockscaled_fp8_gemm(x_fp8, dz_fp8)`

**收益**: 3490µs → ~1200µs = **-2290µs (-66%)**

**前置条件**:
- x_fp8 + x_scales: forward 中已有，但需要保存到 ctx（或 backward 重新 quant）
- dz_fp8 + dz_scales: GemmDGated 输出 bf16 dz，需要新增 quant kernel

**实现要点**:
```python
# _UpProjection.backward 当前:
gemm(x.T, dz, out=dw1_base.permute(0, 2, 1), cu_seqlens_k=..., A_idx=...)

# 改为:
dz_fp8, dz_scales = quantize_and_pack_activation(dz)       # ~80µs
x_fp8, x_scales = cached_or_requant(x)                     # ~30µs or 0 (cached)
blockscaled_fp8_weight_grad_gemm(x_fp8, dz_fp8, x_scales, dz_scales, out=dw1)
```

**风险**: wgrad 精度直接影响训练收敛。需要对比 FP8 vs BF16 wgrad 的 weight gradient RRMSE。

**关键函数**: `blockscaled_fp8_weight_grad_gemm` 已存在于 `quack_utils/__init__.py`!

### 3.2 ⭐⭐ FP8 Wgrad Down-Proj

**目标**: `gemm(dout.T, y1)` BF16 → `blockscaled_fp8_gemm(dout_fp8, y1_fp8)`

**收益**: 387µs → ~190µs = **-197µs (-51%)**

**前置条件**:
- dout_fp8 + dout_scales: 已有（actgrad 路径已 quant）
- y1_fp8 + y1_scales: 已有（forward prequant cache "bwd" 传递）

**实现**: 类似 up-proj wgrad，更简单因为两个输入已有 fp8。

### 3.3 ⭐ Phase 3.1: z Dequant 消除

**目标**: 消除 384MB z_bf16 temp buffer + 130µs dequant kernel

**收益**: -130µs latency + **-384MB peak memory**

**实现路径** (两个选项):

**Option A: 独立 Triton dequant-to-f32-view kernel**
- 写一个 Triton kernel: 读 z_fp8 + scales → 输出 (TK, I) f32 (bf16x2 packed)
- 替代当前的 `dequant → view(f32)` 两步
- 仍需 384MB 输出 buffer（不节省显存）
- 节省: 1 kernel launch（dequant + view 合并）

**Option B: GemmDGated 内 fp8 PreAct EpiOp** (理想但复杂)
- 需要在 GemmDGated epilogue 中直接加载 fp8 z + dequant
- 完全消除 384MB temp buffer
- 需要深度修改 CUTLASS DSL epilogue（N-dimension 问题）
- 备选: copy quack 的 GemmDGated 代码到本仓库，独立修改

**Option C: z recompute** (简单但 costly)
- backward 重算 z = GemmGated(x, w1)（~800µs）
- 完全消除 z 存储（-192MB ctx - 384MB temp = -576MB）
- 代价: +800µs latency

### 3.4 理论最优 vs. 工程可行性矩阵

| 优化项 | 收益(µs) | 收益(MB) | 难度 | 风险 | ROI |
|--------|---------|---------|------|------|-----|
| FP8 wgrad up-proj | **-2290** | 0 | 低 | 精度 | ⭐⭐⭐⭐ |
| FP8 wgrad down-proj | -197 | 0 | 低 | 精度 | ⭐⭐⭐ |
| dz quant (新增) | +80 | +198 | 低 | 无 | 前置 |
| Phase 3.1 Option B | -130 | **-384** | 极高 | 高 | ⭐ |
| Phase 3.1 Option C | +670 | **-576** | 低 | 无 | ⭐ (mem only) |

---

## 4. 推荐实施路径

```
Phase A: FP8 Wgrad (最高 ROI，预期 -2487µs = -47% backward)
  Step A.1: dz quantize — GemmDGated 输出 dz 后立即量化
  Step A.2: FP8 wgrad down-proj — dout_fp8 × y1_fp8 → dw2
  Step A.3: FP8 wgrad up-proj — x_fp8 × dz_fp8 → dw1
  Step A.4: Precision validation — 比较 FP8 vs BF16 wgrad RRMSE
  Step A.5: Full regression + benchmark

Phase B: 显存优化 (Phase 3.1，预期 -384MB peak)
  Step B.1: 评估 Option A (Triton dequant-to-f32view) 的 ROI
  Step B.2: 评估 Option C (z recompute) 在极端 memory 场景的价值
  Step B.3: 如果 Phase A 验证 FP8 wgrad 精度 OK，
            考虑直接存 dz 为 fp8 → 减少 384MB dz bf16

Phase C: 端到端 (最终目标)
  全链路 FP8 通路 benchmark:
    Forward: FP8 mainloop (已有) + fused quant (已有)
    Backward: FP8 wgrad (Phase A) + FP8 actgrad (已有) + stream overlap
  对比 BF16 baseline 的总收益
```

---

## 5. 全链路理论上限

### 5.1 Latency 理论上限 (Ernie shape)

```
                    当前        理论最优     收益
Forward:            ~1200µs     ~1200µs     0 (已接近最优)
Backward:           ~5254µs     ~2559µs     -2695µs (-51%)
────────────────────────────────────────────────────
Total fwd+bwd:      ~6454µs     ~3759µs     -2695µs (-42%)
```

### 5.2 Memory 理论上限 (Ernie shape)

```
                    当前        理论最优     收益
ctx (z_fp8+scales): 198MB       198MB       0
Backward peak temp: 1306MB      ~922MB      -384MB (-29%)
────────────────────────────────────────────────────
Total peak:         ~2105MB     ~1721MB     -384MB (-18%)
```

### 5.3 达到理论上限的条件

1. ✅ Forward FP8 mainloop — 已实现
2. ✅ Forward fused quant — 已实现
3. ✅ Backward actgrad FP8 — 已实现
4. ✅ Backward GemmDGated FP8 — 已实现
5. ❌ **Backward wgrad FP8 — Phase A** (最大收益点)
6. ❌ **Backward z dequant 消除 — Phase B** (显存收益点)
7. ❌ **Backward dz 存储 FP8 化 — Phase C** (额外显存收益)

---

## 6. 修正后的优化路径 (基于 nsys + 效率分析 + cross-node 验证)

### 6.1 效率分析揭示的真相

| Kernel | Measured | Theory | Efficiency | 瓶颈 |
|--------|---------|--------|-----------|------|
| wgrad up-proj `gemm(x.T, dz, A_idx)` | 3490µs | 733µs | **16%** | A_idx 不规则访存 |
| wgrad down-proj `gemm(dout.T, y1)` | 387µs | 367µs | **71%** | 接近峰值 |
| GemmDGated FP8 | 486µs | 183µs | 38% | epilogue (dSwiGLU) |
| actgrad FP8 | 430µs | 367µs | 85% | 接近峰值 |

**关键发现：wgrad up-proj 的 84% 时间浪费在 A_idx gather 的不规则内存访问上，
不是算力不足。FP8 mainloop 只能加速 compute-bound 部分 (~16%)，无法改善
memory-bound 部分 (~84%)。**

### 6.2 修正后的收益预估

| 优化项 | 原始预估 | 修正预估 | 差异原因 |
|--------|---------|---------|---------|
| FP8 wgrad up-proj | -2290µs | **~-560µs** | memory-bound，FP8 只加速 compute 部分 |
| FP8 wgrad down-proj | -197µs | **~-190µs** | compute-bound，接近 2x |
| wgrad up-proj A_idx 优化 | 0 | **~-2000µs** | 消除 A_idx gather 的根本方案 |
| Phase 3.1 z dequant | -130µs/-384MB | -130µs/-384MB | 不变 |

### 6.3 修正后的最优路径

```
Priority 1: wgrad up-proj A_idx gather 优化 (最大收益 ~2000µs = 38% backward)
  方案 A: 预先 gather x → x_gathered(TK, H)，消除 A_idx
          代价: +384MB 临时 buffer
          收益: wgrad up-proj 从 3490µs → ~1000µs
  方案 B: 改变 wgrad 计算顺序，避免 gather
  方案 C: per-expert loop + cuBLAS batch GEMM

Priority 2: FP8 wgrad down-proj (~190µs saving)
  已接近峰值 (71%)，FP8 可再提升到 ~85%
  前置条件: dout_fp8 + y1_fp8 已有

Priority 3: Phase 3.1 z dequant 消除 (130µs + 384MB saving)
  显存收益为主

Priority 4: FP8 wgrad up-proj (~560µs saving)
  在 A_idx 优化之后才有意义
```

### 6.4 修正后的理论上限

```
                    当前        修正最优      收益
wgrad up (A_idx opt) 3490µs     ~1000µs      -2490µs
wgrad down (FP8)      387µs      ~190µs       -197µs
GemmDGated            486µs       486µs       0
actgrad               430µs       430µs       0
z dequant             130µs         0µs       -130µs + 384MB
other                 331µs       331µs       0
─────────────────────────────────────────────────
TOTAL                5254µs      2437µs       -2817µs (-54%)
```
