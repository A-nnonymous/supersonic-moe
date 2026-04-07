# Session 35 Handoff — FP8 Full-Chain Optimization

> Branch: `native-fp8-exploration` @ `ce9ef4c` (35+ commits ahead of `b651e17`)
> Date: 2026-04-07

---

## 1. 已完成并验证的工作

### Phase 1: Epilogue Blockscaled Quant
- **z_is_fp8 条件修复** (`c8d27c8`): 14/14 tests pass
- **Epilogue quant 完整实现** (`28f30d1`): 0 RRMSE across all tensors
  - `@mlir_namedtuple` EpilogueArguments (不是普通 NamedTuple!)
  - `BlockscaledScaleStore(EpiOp)`: `param_fields` 返回单 tuple field, `smem_bytes=0`
  - Integer+carry E8M0 算法 in DSL
  - Scale write 用 identity tensor 坐标映射 (99.57% match)
- **但性能不优于 standalone**: +66µs (Triton fused quant 已够快)
- **根因**: autograd 需要 bf16 z → D output 必须 bf16 → epilogue 不减少 HBM traffic

### Phase 3.1: GemmDGated FP8 PreAct — 两种实现

#### A. LDG-based (完整但慢)
- `GemmDGatedFP8PreActMixin` + `FP8PreActLoad(EpiOp)` + identity tensor
- **精度通过**: dx RRMSE=0.027, cosine=0.9996
- **但 2.9x 慢于 BF16+dequant** (LDG vs TMA)
- **显存节省真实**: backward temp 1032 MiB (was 1306 → -274 MiB)

#### B. TMA-based (进行中, 最后一英里)
- `GemmDGatedFP8CLoadMixin`: fp8 C tensor 通过 TMA → smem → register
- **已通过的阶段**:
  - ✅ fp8 C tensor validation + CuTe tensor creation
  - ✅ `make_smem_layout_epi(Float8E4M3FN, ...)` — 需要整数 epi_tile `(128, 64)`
  - ✅ SM100 `epilog_smem_load_and_partition` — 创建 fp8 copy atoms
  - ✅ `epi_to_underlying_arguments` — 跳过 bf16 c_dtype assertions
- **当前阻塞**: TMA atom 创建的 smem/CTA-V-map shape equivalence
  - smem layout (swizzled) shape ≠ CTA V-map (identity tile) shape
  - 需要从 smem layout 的 coalesced shape 推导 CTA V-map

---

## 2. 性能测量方法论 (关键教训!)

### ⚠️ nsys wall-clock ≠ isolated kernel time
- nsys `cuda_gpu_kern_sum` 在 multi-stream 并行时报叠加 wall-clock
- **错误案例**: wgrad 被 nsys 报为 3490µs, 实际 isolated 只有 772µs (71% eff)
- **正确流程**: isolated benchmark → 理论效率 → nsys timeline gap 分析

### ⚠️ FP8 不总是更快
- `blockscaled_fp8_weight_grad_gemm_fast`: 2-5.6x 慢于 BF16 A_idx GEMM
- 根因: pack/quant/transpose overhead > 2x compute gain at Ernie shape
- 所有 backward kernel 在 70-85% BF16 efficiency — 系统已接近最优

### ⚠️ cross-node 验证
- 2+ idle 节点并行跑, CV<3% 才可信
- `nvidia-smi --id=GPU_ID` 检查 util=0%

---

## 3. Backward Kernel Breakdown (isolated, idle B200)

```
wgrad up-proj (A_idx):  772µs  71% BF16 eff
actgrad FP8:            430µs  85% eff
GemmDGated FP8:         486µs
wgrad down-proj:        387µs  71% eff
z dequant:              125µs  bandwidth-limited
dout quant:             114µs  Triton
token scatter:           66µs

nsys timeline backward: 3150µs total
  kernel time: 2622µs (83%)
  gap overhead: 528µs (17%)
  largest gap: 315µs (stream sync before GemmDGated)
```

---

## 4. Phase 3.1 TMA-based FP8 C Load — 技术细节

### 架构
```
Standard:  C = z_bf16.view(f32) → TMA(f32) → smem(f32) → reg(f32) → recast(bf16x2) → dSwiGLU
FP8 TMA:   C = z_fp8            → TMA(fp8) → smem(fp8) → reg(fp8) → cvt(f32) → dequant → dSwiGLU
```

### Shape 关系
```
D tensor: (TK, I) f32    — 每个 f32 = 2 bf16 (gate+up packed)
C tensor (bf16): (TK, I) f32    — same shape as D
C tensor (fp8):  (TK, 2I) fp8   — 2x N, 0.5x bytes
```

### 关键 overrides
1. **`_setup_attributes`**: 用整数 `(m_int, n_int * 2)` 调用 `make_smem_layout_epi(Float8E4M3FN, ...)`
   - `n_int` 从 `self.epi_tile` 提取 (SM100 上是 CuTe Layout, 用 `cute.size()`)
   - 存储 `self._fp8_c_tile_mn = (m_int, n_int * 2)` 给 TMA 用
2. **`_make_tma_epi_atoms_and_tensors`**: 用 `self._fp8_c_tile_mn` 作为 CTA V-map tile
   - ⚠️ 当前问题: smem layout 的 swizzled shape 与 integer tile 不匹配
   - `make_tiled_tma_atom` 要求 smem layout top-level shape == CTA V-map shape
3. **`epilog_smem_load_and_partition`**: doubled register layout (f32 recast to bf16 → 2N)
   - `tRS_rC` 有 2N fp8 elements (对应 N f32 = 2N logical values)
4. **`epi_visit_subtile`**: `tRS_rC.load().to(Float32)` + scale multiply + dSwiGLU

### 当前阻塞点
```
TMA atom creation: make_tiled_tma_atom(op, tensor_d, epi_smem_layout, d_cta_v_layout)

Expected: smem_layout top-level shape == d_cta_v_layout shape
Got:      smem = S<2,4,3> o 0 o ((8,16),(16,1)):((16,128),(1,0))  [swizzled]
          v-map = (128,32):(1@0,1@1)  [identity from integer tile]

The swizzle changes the physical shape. Need to derive v-map from the
COALESCED smem shape, or use the standard epi_tile (Layout type) directly
with the fp8 tensor's composition.
```

### 下一步攻坚方向
1. **从 smem layout 推导正确的 CTA V-map**: `cute.coalesce(epi_smem_layout)` 获取物理 shape
2. **或**: 直接用标准 `epi_tile` (CuTe Layout) 做 composition, 但传 fp8 C tensor shape
3. **或**: 研究 `_make_tma_epi_atoms_and_tensors` 在标准 f32 路径中如何工作,
   复制其 smem/v-map 推导逻辑, 仅改 element type

---

## 5. 需要避开的 Pitfall

| Pitfall | 描述 |
|---------|------|
| `@mlir_namedtuple` vs `NamedTuple` | EpilogueArguments 必须用 `@mlir_namedtuple` 装饰, 否则 quack 框架无法正确传递 |
| `EpiOp.param_fields` | 必须返回单个 `(self.name, object, None)`, 不能多字段 (框架用 `getattr(params, op.name)`) |
| `Float32(fp8_scalar)` | 标量 fp8→f32 转换触发 `nvgpu.cvt_fpext` 的 32-bit aligned vector 要求 — 用 `tensor.load().to(Float32)` 代替 |
| `const_expr(not tuple)` | DSL tuple 不支持 `not` 操作 — 用 `const_expr(x is not None)` |
| `epi_tile` on SM100 | 是 CuTe Layout 对象, 不是 int — 不能直接算术运算, 用 `cute.size()` 提取 |
| `_make_tma_epi_atoms_and_tensors` | 父类是 `@staticmethod`, override 需改为实例方法才能访问 self 属性 |
| `validate_and_prepare_tensors` | C shape 必须与 D 匹配 — fp8 C 传 None, 手动设置 `tensor_infos["C"]` |
| nsys wall-clock | 不等于单 kernel 真实耗时 — 必须用 isolated benchmark |
| FP8 wgrad overhead | `blockscaled_fp8_weight_grad_gemm_fast` 在 Ernie shape 比 BF16 慢 2-5.6x |

---

## 6. 文件位置

| 文件 | 内容 |
|------|------|
| `sonicmoe/quack_utils/gemm_gated.py` | Epilogue quant (BlockscaledScaleStore, GemmGatedBlockscaledQuantMixin) |
| `sonicmoe/quack_utils/gemm_dgated.py` | Phase 3.1 (FP8PreActLoad, GemmDGatedFP8PreActMixin, GemmDGatedFP8CLoadMixin) |
| `sonicmoe/quack_utils/gemm_interface.py` | z_scale_out threading |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | ZeroMat variants |
| `sonicmoe/functional/__init__.py` | z_is_fp8 fix, _use_epilogue_quant, backward integration |
| `docs/fp8_architecture_comparison.md` | 3-way comparison (DeepEP, SonicMoE BF16, FP8) |
| `docs/fp8_full_chain_optimal.md` | Full-chain optimal design + corrected priorities |
| `docs/phase3_1_gemmdgated_fp8_preact.md` | Phase 3.1 deep technical analysis |
| `docs/session35_report.md` | Session 35 complete report |
| `tests/test_phase1_precision.py` | Epilogue quant precision test |
| `tests/bench_dgated_fp8_preact.py` | Phase 3.1 isolated benchmark |
| `tests/rigorous_benchmark.py` | Cross-node validated benchmark |

---

## 7. Git Log (key commits)

```
ce9ef4c  Phase 3.1 TMA — close to solution, smem/TMA shape equivalence
b9a4d66  Phase 3.1 TMA — iterating on smem layout + TMA tile compat
f17b51e  Phase 3.1 TMA-based — fp8 C tensor passes through pipeline
7577113  Phase 3.1 isolated benchmark — FP8 PreAct 2.9x slower
830a3b0  Phase 3.1 PRECISION PASS — identity tensor + vectorized fp8→f32
31dad22  Phase 3.1 kernel compiles and RUNS with vectorized fp8→f32
4beed75  nsys backward profiling — wgrad BF16 is 73.5% (corrected later)
7a53612  benchmark correction — A_idx wgrad 774µs not 3490µs (71% efficient)
28f30d1  epilogue blockscaled quant — full forward+backward PASS, 0 RRMSE
c8d27c8  z_is_fp8 condition fix — accept prequant cache
b651e17  baseline (fork-main-sync) — 31/31 tests pass
```
