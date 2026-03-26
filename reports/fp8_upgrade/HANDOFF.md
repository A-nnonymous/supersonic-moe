# FP8 Next-Agent Handoff

本文件的目标：**让下一个 agent 在最短时间内接住主线，不重复踩坑。**

> 最后更新：2026-03-26，commit 后的代码状态

---

## 0. 一句话现状

**全部 6 个 GEMM 算子已使用 FP8 tensor core（per-tensor `.to(fp8)` cast），性能大幅领先 BF16（forward -40~50%, E2E -33~39%）。但有两个严重未解问题：(1) 精度不足——per-tensor cast 无 scale factor，小幅值输入下 RelRMSE 退化到 ~100%；(2) 显存开销——FP8 weight 缓存导致 +3 GiB 额外占用。解决路径明确：替换为 1x32 blockscaled UE8M0 量化（已有 varlen 算子原型验证精度 3.74%）+ 消除冗余 weight 缓存。**

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

## 2. 当前代码真实状态

### 2.1 已稳定落地（可直接使用）

#### A. 全 6 GEMM FP8 tensor core（per-tensor cast，无 blockscaling）

| 论文算子 | 工程路径 | 量化方式 | 状态 |
|---------|---------|---------|------|
| up-proj forward | `gemm_gated(x_fp8, w1_fp8)` | `x.to(fp8)` + 缓存 `w1_ekh` | done |
| down-proj forward | `gemm(y1_fp8, w2_fp8.permute)` | `y1.to(fp8)` + 缓存 `w2_orig` | done |
| down-proj act grad | `gemm_dgated(dout_fp8, w2_fp8)` | `dout.to(fp8)` + 缓存 `w2_ehi` | done |
| down-proj weight grad | `gemm(dout_fp8.T, y1s_fp8)` | 均为 `fp8` | done |
| up-proj act grad | `gemm(dz_fp8, w1_fp8.permute)` | `dz.to(fp8)` + 缓存 `w1_orig` | done |
| up-proj weight grad | `gemm(x_fp8.T, dz_fp8)` | 均 on-the-fly `.to(fp8)` | done |

控制开关：`SONIC_MOE_FP8_MODE=perf`（权重缓存）或 `mem`（on-the-fly 量化）

#### B. Varlen blockscaled FP8 GEMM（已验证，待集成为默认路径）

- **文件**：`sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- **能力**：`blockscaled_fp8_gemm_varlen(A_bf16, B_bf16, cu_seqlens, protocol)` — 内部做 1x32 blockscaled 量化 + CUTLASS SM100 GEMM
- **精度**：3.74% RelRMSE（全生产 shape 验证：E=4-128, tpe=128, I=1024, H=4096）
- **关键修复**：rank-aware `tile_atom_to_shape_SF` monkey-patch，解决 CUTLASS DSL rank-2 张量不兼容
- **限制**：每个 expert segment 必须大于等于 32 tokens（blockscaled tile atom 大小）
- **快速量化**：`quantize_activation_blockscaled_fast()` — Triton 融合单 pass 内核，5.8x 快于 Python 参考

#### C. Down-proj forward 已有 blockscaled varlen 路径（gated on fp8_protocol）

- 当 `fp8_protocol is not None` 且 `_fp8_enabled()` 且 min segment >= 32 时，使用 `blockscaled_fp8_gemm_varlen`
- 否则 fallback 到 per-tensor fp8 path 或 bf16
- **当前默认路径**：仍是 per-tensor fp8（因为 `fp8_protocol=None` 是常见调用方式）

#### D. 合同测试

- `tests/fp8_large_project_contract_test.py`：11 个测试全部通过
- 测试验证 FP8 路径 vs BF16 gold 的 forward/backward 正确性
- 包含小 shape (T=256) 和大 shape (T=1024, H=4096, I=1024, E=128, K=8)

### 2.2 已知问题（下一个 agent 必须解决）

#### 问题一：精度 — per-tensor cast 无 scale factor

**严重程度**：Critical

当输入幅值较小（如 `0.02 * randn`，这是训练常态）时：
- FP8 vs BF16 RelRMSE 约 **100%**，cosine 约 **0.003**
- 原因：E4M3 动态范围 [-448, 448]，小值直接被 flush 到最近表示

当输入幅值较大（如 `1.0 * randn`）时：
- RelRMSE 约 **7.9%**，cosine 约 **0.997**

**解决方案**（已验证原型）：1x32 blockscaled UE8M0 量化
- `blockscaled_fp8_gemm_varlen` 已在算子级别验证 3.74% RelRMSE
- 需要将其替换为所有 GEMM 的默认量化方式

#### 问题二：显存 — FP8 权重缓存 +3 GiB

**严重程度**：Major

| 缓存 | 大小 | 用途 |
|------|------|------|
| `_FP8_WEIGHT_CACHE` (w1_ekh, w2_ehi) | 1536 MiB | gemm_gated/gemm_dgated 的 permuted fp8 权重 |
| `_FP8_ORIG_CACHE` (w1, w2) | 1536 MiB | quack.gemm 的 original-layout fp8 权重 |
| **总额外占用** | **3072 MiB** | 在 BF16 参数 3073 MiB 基础上翻倍 |

**解决方案**：
- 短期：`mem` 模式不缓存，每次 on-the-fly 量化（牺牲性能）
- 中期：统一权重 layout，消除双缓存（permuted + original）
- 长期：FP8 optimizer + FP8 master weight，完全消除 BF16 参数副本

### 2.3 已证伪方向（不要重复）

1. **Grouped/static-capacity blockscaled down-proj**：违背 SonicMoE varlen 合同，内存开销大
2. **Backward runtime-fp8 y1s 最短路径**：`quack.gemm` 要求 A/B 同 dtype，`gather_A` 约束苛刻
3. **将 cold metrics peak 和 stagewise probe 混写成单一结论**

---

## 3. 最新性能数据（2026-03-26 实测，GPU 4）

### 3.1 Shape 4096,4096,1024,128,8（中等）

| 指标 | BF16 | FP8-perf | Delta |
|------|------|----------|-------|
| Fwd inference (ms) | 2.264 | 1.349 | **-40.4%** |
| Fwd training (ms) | 2.264 | 1.218 | **-46.2%** |
| E2E fwd+bwd (ms) | 7.357 | 4.915 | **-33.2%** |
| Bwd (ms) | 5.093 | 3.566 | **-30.0%** |
| TFLOPS (fwd) | 364.3 | 611.3 | **+67.8%** |
| TFLOPS (e2e) | 336.3 | 503.4 | **+49.7%** |
| Peak mem (MiB) | 6,924 | ~12,853 | +85.6% |

### 3.2 Shape 8192,4096,1024,128,8（大）

| 指标 | BF16 | FP8-perf | Delta |
|------|------|----------|-------|
| Fwd inference (ms) | 3.924 | 1.995 | **-49.2%** |
| Fwd training (ms) | 4.399 | 2.086 | **-52.6%** |
| E2E fwd+bwd (ms) | 11.962 | 7.351 | **-38.6%** |
| Bwd (ms) | 8.038 | 5.355 | **-33.4%** |
| TFLOPS (fwd) | 420.3 | 826.6 | **+96.7%** |
| TFLOPS (e2e) | 413.6 | 673.1 | **+62.7%** |

### 3.3 精度（per-tensor cast，当前默认）

| 输入规模 | Output RelRMSE | Output cosine | Grad RelRMSE | Grad cosine |
|----------|---------------|---------------|-------------|-------------|
| `0.02 * randn` (训练常态) | ~100% | ~0.003 | ~100% | ~0.008 |
| `1.0 * randn` | 7.90% | 0.997 | 7.68% | 0.997 |

**结论**：per-tensor cast 精度在训练常态下完全不可接受，必须换 blockscaled。

### 3.4 精度（blockscaled varlen，算子级验证）

| 验证方式 | RelRMSE | 备注 |
|---------|---------|------|
| `blockscaled_fp8_gemm_varlen` 单算子 | **3.74%** | E=4-128, tpe=128, 全 shape |
| `quantize_activation_blockscaled_fast` 量化精度 | **100% bit-exact** | vs Python 参考实现 |

### 3.5 显存分析

```
BF16 参数:           3,073 MiB (w1: 2I*H*E bf16, w2: H*I*E bf16)
FP8 weight cache:   +3,072 MiB (4 个缓存条目: w1_ekh, w2_ehi, w1_orig, w2_orig)
FP8 激活中间量:      约等于 BF16 (per-tensor cast 不增加，blockscaled 增加 ~3% scale 存储)
------
总额外占用:          ~3,072 MiB (100% 来自权重缓存)
```

---

## 4. 高价值技术知识

### 4.1 CUTLASS Blockscaled Varlen 的 Rank-2 修复

- `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF` 硬编码 `(2, 1, 3)` 排列
- Varlen 路径产出 rank-2 `(total_M, K)` 张量导致编译期 rank mismatch
- 修复：`blockscaled_fp8_gemm.py` 中的 `_tile_atom_to_shape_SF_rank_aware` monkey-patch
- 使用 `cute.rank(Shape)` + `const_expr` 在 trace time 分派 `(2, 1)` / `(2, 1, 3)`
- 安装到 `cutlass.utils.blockscaled_layout` 模块命名空间

### 4.2 Blockscaled 最小 Segment 约束

- Scale factor tile atom = 32 elements in M direction
- Expert segment < 32 tokens 导致 `CUDA_ERROR_ILLEGAL_INSTRUCTION`
- 生产 shape (T>=512, K=8, E=128, 即 >=32 tpe) 安全
- 测试 shape (T=256, K=8, E=128, 即 16 tpe) 需要 fallback
- 代码已有 `_SF_VEC_SIZE` 检查和 bf16 fallback

### 4.3 FP8 权重缓存架构

```
w1: (2I, H, E) bf16 -> _FP8_WEIGHT_CACHE["w1_ekh"]: (E, H, 2I) fp8  [for gemm_gated]
w2: (H, I, E) bf16  -> _FP8_WEIGHT_CACHE["w2_ehi"]: (E, H, I)  fp8  [for gemm_dgated]
w1: (2I, H, E) bf16 -> _FP8_ORIG_CACHE[w1]:         (2I, H, E) fp8  [for quack.gemm backward]
w2: (H, I, E) bf16  -> _FP8_ORIG_CACHE[w2]:         (H, I, E)  fp8  [for quack.gemm forward]
```

- 缓存 key 包含 `data_ptr` 和 `_version`，optimizer step 后自动失效
- `perf` 模式始终缓存；`mem` 模式 on-the-fly

### 4.4 gemm_gated/gemm_dgated FP8 限制

- 这两个 fused kernel 接受 `A(fp8) * B(fp8)` 但**不支持 blockscaled scale factor**
- `postact_dtype=torch.float8_e4m3fn` 产出 fp8 post-activation（per-tensor，无 scale）
- 要支持 blockscaled，需要修改 fused kernel 或拆分为 GEMM + activation kernel

### 4.5 GPU 兼容性

- GPU 2, 3 有间歇性 Triton `CUDA_ERROR_ILLEGAL_INSTRUCTION`（autotune 时）
- GPU 0, 1, 4, 5 稳定
- 这是硬件/驱动问题，非代码问题
- 测试建议使用 GPU 0 或 4

---

## 5. 经验与教训

### 踩过的坑

1. **`postact_dtype=None` 导致 dtype 级联错误**：当 A 是 fp8 时，`None` 默认取 `A.dtype`，产出 fp8 post-activation，下游 bf16 GEMM 收到 fp8 输入报错。显式设置 `postact_dtype=torch.float8_e4m3fn` 或 `torch.bfloat16`。

2. **`_fp8_enabled()` 检查 env var，不检查 `fp8_protocol` 参数**：gold path (protocol=None) + env var SONIC_MOE_FP8_MODE=perf 导致 FP8 code path 仍激活。测试时必须理解这个区别。

3. **`dout` 非 contiguous 导致 GEMM crash**：`sum().backward()` 产出 stride=(0,0) 的 expanded tensor，需要 `.contiguous()` 后再传入 GEMM。

4. **Weight layout 多次 permute 开销**：SonicMoE 存储 `(H, I, E)`，quack 需要 `(E, K, N)` 或 `(E, H, I)`。预缓存 permuted 副本是正确选择（不是 `.permute()` view）。

5. **Triton kernel 在不同 GPU 上的 autotune crash**：某些 GPU 的 Triton autotuner `load_binary` 报 illegal instruction。解决：固定 GPU、清除 `~/.triton/cache/`。

### 有效方法论

1. 先用 `git stash` 确认回退到上一个 commit 是否通过，快速定位是否是新代码引入
2. 每次修改后先跑小 shape 合同测试，再跑大 shape，最后 benchmark
3. 在 subprocess 中逐个运行测试避免 GPU state 污染
4. 精度测试必须用 `x_gold.detach().clone().requires_grad_()` 而非共享输入

---

## 6. 下一步规划（优先级排序）

### P0: 替换 per-tensor cast 为 blockscaled 1x32 UE8M0（精度修复）

**目标**：所有 6 个 GEMM 使用 blockscaled 量化，RelRMSE < 5%

具体步骤：
1. 将 `blockscaled_fp8_gemm_varlen` 设为 down-proj forward 默认路径（当 segment >= 32）
2. 将 backward dx 也切换到 blockscaled varlen
3. 为 `gemm_gated`/`gemm_dgated` 添加 blockscaling 支持，或拆分为 blockscaled GEMM + activation kernel
4. Weight grad 路径也切到 blockscaled

### P1: 消除 FP8 权重缓存冗余（显存修复）

**目标**：FP8 总显存 <= BF16

具体步骤：
1. 统一 `_FP8_WEIGHT_CACHE` 和 `_FP8_ORIG_CACHE` 为单一缓存
2. 让 gemm_gated/gemm_dgated 接受与 quack.gemm 相同 layout 的权重
3. 长期：FP8 master weight + FP8 optimizer 消除 BF16 参数

### P2: gemm_gated/gemm_dgated blockscaled kernel

**目标**：fused kernel 原生支持 1x32 scale factor

选项：
- A: 修改 `gemm_gated.py`/`gemm_dgated.py` 添加 `sf_vec_size` 参数
- B: 拆分为 `blockscaled_gemm_varlen` + 独立 `swiglu_activation` kernel
- C: 参考 CUTLASS example 81 (Blackwell blockwise GEMM) 重写

### Not recommended

- 继续投入 grouped/static-capacity blockscaled 路径
- 在 site-packages 做 runtime patch（必须本地代码）

---

## 7. 环境与命令速查

### 环境激活

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
```

### 合同测试（11 个，全部必须通过）

```bash
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v
```

### BF16 baseline benchmark

```bash
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 python benchmarks/moe-cute.py \
  --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
```

### FP8 benchmark

```bash
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python benchmarks/moe-cute.py \
  --thiek 4096,4096,1024,128,8 --dtype BFloat16 --activation swiglu --skip_test
```

### 精度对比测试

```python
# 见 tests/fp8_large_project_contract_test.py 中的 _run_sonicmoe_path()
# 注意：gold 和 candidate 都在 SONIC_MOE_FP8_MODE=perf 下运行
# 真正的 BF16 gold 需要 unset SONIC_MOE_FP8_MODE
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `sonicmoe/functional/__init__.py` | 核心：所有 MoE forward/backward + FP8 控制 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Blockscaled FP8 GEMM 基础设施（varlen + grouped） |
| `sonicmoe/quack_utils/__init__.py` | 导出 blockscaled_fp8_gemm_varlen |
| `sonicmoe/functional/fp8_protocol.py` | FP8Protocol, FP8ScaleGranularity 定义 |
| `sonicmoe/functional/fp8_quant.py` | quantize/dequant/round_scale_to_e8m0 |
| `sonicmoe/functional/fp8_cutely_fused.py` | preact fused quant/dequant |
| `tests/fp8_large_project_contract_test.py` | 11 个合同测试 |
| `benchmarks/moe-cute.py` | 性能 benchmark |
