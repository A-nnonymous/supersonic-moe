# Plan: 全链路 FP8 参数 (native_fp8_params) 分支

## 假设

在此假设下，整个 MoE 全链路参数都是 FP8：
1. **权重 w1, w2 直接存储为 FP8** — `nn.Parameter(dtype=float8_e4m3fn)`，附带 ISA-packed scales（训练时由 optimizer 维护）
2. **MoE 从前方通信算子接收到的 input x 就是 FP8** — token_gather 时无需量化
3. **只有部分必要的激活值是 bf16**（z for backward PreAct、GEMM output、梯度聚合）

## 消除的开销（对比当前 FP8 frontier 3690µs/iter）

| 当前 kernel | µs/iter | 全链路下 | 说明 |
|------------|---------|----------|------|
| `quantize_and_pack(x)` | ~60 | **消除** | x 已是 FP8 |
| `gather_isa_packed_scales(x)` | ~27 | **消除** | x scales 从 T gather 到 TK，变为直接 gather FP8 数据 |
| `precompute_weight_fp8_for_fused_gated(w1)` | ~0 (cached) | **消除** | w1 本身就是 FP8+scales |
| `precompute_weight_fp8(w2)` | ~0 (cached) | **消除** | w2 本身就是 FP8+scales |
| `precompute_weight_fp8_for_direct_fused_dgated(w2)` | ~0 (cached) | **消除** | 同上 |
| `precompute_weight_fp8(w1^T)` | ~0 (cached) | **消除** | 同上 |
| `quantize_and_pack(dout)` | ~60 | **保留** | dout 从 down-proj 输出是 bf16 |
| `quantize_and_pack(dz)` | ~60 | **保留** | dz 是 bf16 |
| `fused_z_save_y1_quant` | 168 | **简化** | z 和 y1 是 GEMM bf16 输出，仍需 quant。但 y1 可以直接用 FP8 output 省掉 |
| `dequant_blockscaled_fp8(z)` | 130 | **保留** | CUTLASS PreAct 约束不变 |
| Weight cache memory (~650 MiB) | — | **消除** | 无 cache |
| Weight quant/cache 首次开销 | ~100-500µs first iter | **消除** | 无 quant |

### 更大的收益：GemmGated 可以直接输出 FP8 y1

CUTLASS `GemmGatedSm100` 的 `PostAct` dtype 支持 FP8（`assert self.postact_dtype.width in {8, 16}`）。如果让 GemmGated 直接输出 `y1` 为 FP8：
- **消除 `fused_z_save_y1_quant` 的 y1 部分** (~84µs)
- y1 直接 FP8 → down-proj GEMM，**零量化开销**
- z 仍需单独 quant（FP8 save for backward），但更简单

### 关键问题：wgrad 是否可以变 FP8

当前 wgrad 是 BF16 varlen GEMM：`gemm(dout.T, y1s)` 和 `gemm(x.T, dz_bf16)`。
- `dout^T × y1s` → dw2：如果 dout 已有 FP8、y1s 有 FP8，可以用 FP8 GEMM
- `x^T × dz` → dw1：如果 x 已是 FP8、dz 有 FP8，可以用 FP8 GEMM
- 但 wgrad FP8 在之前被验证为 net-negative（colwise quant SM contention）
- 不同点：在全链路假设下，x 和 weight 已经是 FP8，**不需要 colwise quant**！
- 需要验证 `GemmDefaultSm100` 是否支持 A=FP8, B=FP8 的 wgrad 配置

## 实施方案

### Step 1: 创建分支 `native-fp8-params`

从 `fork-main-sync` 创建新分支。

### Step 2: 修改 MoE weight 存储

在 `sonicmoe/moe.py` 中：
- 添加 `native_fp8` 参数
- 当 `native_fp8=True` 时，w1/w2 存储为 `float8_e4m3fn`
- 附带 `w1_scales`/`w2_scales` 作为 buffer（ISA-packed E8M0）
- 提供 `from_bf16_weights(w1_bf16, w2_bf16)` 初始化方法

### Step 3: 修改 forward 路径 (`_UpProjection.forward`)

当 `native_fp8=True` 时：
```
x_fp8(T,H) fp8  [直接收到]
  |-- gather_isa_packed_scales(x_scales, idx) → x_scales_tk   [x 自带 scales]
  |-- w1 已是 fp8, w1_scales 已有
  V
GemmGatedSm100ZeroMat(x_fp8, w1_fp8, A_idx, x_scales_tk, w1_scales)
  → z(TK,2I) bf16, y1(TK,I) fp8  [PostAct 直接输出 fp8]
  |-- z 仅需单独 quant for backward save (single kernel, 不含 y1 部分)
```

**消除**: x quant, weight quant, y1 quant (全部), x scale gather (改为 x 自带)

### Step 4: 修改 forward 路径 (`_DownProjection.forward`)

```
y1_fp8(TK,I) fp8 + y1_scales  [来自 GemmGated PostAct 输出]
  |-- w2 已是 fp8, w2_scales 已有
  V
blockscaled_fp8_gemm_varlen(y1_fp8, w2_fp8, y1_scales, w2_scales)
  → y2(TK,H) bf16
```

**消除**: y1 quant, w2 quant

### Step 5: 修改 backward 路径 (`_DownProjection.backward`)

```
dout(T,H) bf16  [从 router 来的梯度仍是 bf16]
  |-- quantize_and_pack(dout) → dout_fp8 + scales  [保留]
  |-- gather_scales T→TK [保留]
  |-- z dequant on side stream [保留]
  V
GemmDGatedSm100ZeroMat(dout_fp8, w2_fp8, z_bf16, ...)
  → dz(TK,2I) bf16, y1s(TK,I) bf16, ds
  |-- quantize_and_pack(dz) → dz_fp8  [保留, for actgrad]
  |-- wgrad: gemm(dout^T, y1s) → dw2  [探索 FP8 wgrad]
```

### Step 6: 修改 backward 路径 (`_UpProjection.backward`)

```
dz_fp8(TK,2I) + scales  [来自 down-proj bwd pre-quant]
  |-- w1^T 已是 fp8 (transpose view)
  V
blockscaled_fp8_gemm_varlen(dz_fp8, w1T_fp8, ...)
  → dx_expanded(TK,H) bf16
  |-- wgrad: gemm(x^T, dz_bf16) → dw1  [探索 FP8 wgrad]
```

### Step 7: PostAct FP8 输出 — GemmGated 改造

修改 `gemm_gated()` 调用，设置 `postact_dtype=torch.float8_e4m3fn`。
需要验证：
- CUTLASS `GemmGatedSm100ZeroMat` 是否支持 FP8 PostAct output
- PostAct FP8 输出是否自带 blockscaled scales
- 如果不自带 scales，是否需要在 epilogue 后额外 quant（可能消除不了 y1 quant）

### Step 8: Benchmark 脚本

新建 `tools/profile_native_fp8.py`：
- 模拟全链路 FP8 输入：x 预先 quant 为 FP8，weights 预先 quant
- 跳过 x quant、weight quant，直接进 GEMM
- nsys profile 对比 frontier vs native_fp8

### Step 9: 精度测试

新建 `tests/fp8_native_params_test.py`：
- BF16 gold standard 对比
- Forward RRMSE + correlation 验证
- Backward gradient RRMSE 验证
- 覆盖 Ernie production shape

### Step 10: 显存测试

对比三种配置：Official BF16 / FP8 frontier / Native FP8 params 的 peak memory。

## 预期收益估算

基于 Session 33 (3690µs/iter) 数据：

| 消除项 | 预估节省 µs |
|--------|-----------|
| x quant (quantize_and_pack) | 60 |
| x scale gather | 27 |
| fused_z_save_y1_quant → 仅 z quant | ~84 (y1 部分) |
| Weight cache 首次开销 | ~0 (已 cached) |
| **GemmGated PostAct FP8 (如果可行)** | **~84** |
| **合计** | **~170µs** |

预估 native FP8: ~3520µs/iter → **1.12× vs BF16 (3932µs)** (从 1.066× 提升到 1.12×)

如果 FP8 wgrad 可行（之前不可行因为需要 colwise quant，但现在 x 已是 FP8 不需要）：
- GemmDefault BF16 ×4 (1916µs) → 部分变 FP8，可能再省 200-400µs
- 预估: ~3100-3300µs/iter → **1.19-1.27× vs BF16**

显存方面：
- 消除 weight cache: -650 MiB
- x 本身 FP8 vs bf16: -50% input memory
- 预估: FP8 native peak < BF16 peak

## 风险

1. **GemmGated PostAct FP8 输出不带 scales** — CUTLASS 可能只支持直接 cast to FP8 without blockscaled E8M0。需要实测。
2. **FP8 wgrad 精度** — weight gradient 用 FP8 可能精度不足。需要精度测试。
3. **MoE weight 存储为 FP8** — 需要模拟 optimizer 更新 FP8 weight + scales 的逻辑。本次 benchmark 可以用 BF16 weight → quant 一次来初始化。
4. **x 的 scales 格式** — 前方通信算子给出的 x 可能不是 ISA-packed scales，需要一次 gather 或 repack。

## 文件变更清单

1. **新建** `sonicmoe/functional/native_fp8.py` — 全链路 FP8 forward/backward 函数
2. **修改** `sonicmoe/functional/__init__.py` — 添加 `use_native_fp8` 路径分支
3. **修改** `sonicmoe/moe.py` — FP8 weight 存储 + scales buffer
4. **新建** `tools/profile_native_fp8.py` — benchmark 脚本
5. **新建** `tests/fp8_native_params_test.py` — 精度测试
6. **修改** `tools/_profiling_runner.sh` — 添加 `nsys_native_fp8` 模式
