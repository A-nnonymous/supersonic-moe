# FP8 Engineering Log

> **Primary reference: [HANDOFF.md](HANDOFF.md)**. This file records historical development context, variable mappings, and lessons learned. For current status, performance data, and next steps, see HANDOFF.md.

> **2026-03-27 Session 3 update**: Token rounding routing eliminates all padding overhead. FP8 training E2E **8.5% faster than BF16** (10.88ms vs 11.89ms). Previous "4x regression" was a benchmark artifact (vanilla top-K routing). Fused quantize+ISA-pack Triton kernel integrated. Contract tests 8/8 pass.

---

## 0. Variable Name Mapping

| Engineering var | Paper variable / meaning | Baseline dtype | Notes |
| --- | --- | --- | --- |
| `x` | layer input `X` | bf16 | stable |
| `router_w` | router weight | bf16 | stable |
| `topk_scores` | selected routing scores `S_{t,e}` | fp32 | forward agg + backward router grad depend on this |
| `topk_indices` | activated expert ids | int32 | metadata |
| `x_gather_idx` | `Gather(X, π_{:,e})` gather map | int32 | varlen GEMM metadata |
| `expert_frequency_offset` | per-expert prefix-sum token count | int32 | cu_seqlens_m for varlen |
| `w1` | up-proj weight `W1_e` | bf16 | blockscaled fp8 cached version in `_WEIGHT_CACHE` |
| `z` | up-proj pre-activation `H_e = X_e W1_e` | bf16 | input to fused SwiGLU+quant |
| `y1` | activated intermediate `A_e = SwiGLU(H_e)` | bf16 / fp8 | fused SwiGLU outputs fp8 directly |
| `w2` | down-proj weight `W2_e` | bf16 | blockscaled fp8 cached |
| `y2` | per-expert down-proj output | bf16 | |
| `o` | final aggregated output `O` | bf16 | |
| `dout` | output gradient `dO` | bf16 | `.contiguous()` needed after `sum().backward()` |
| `dz` | gradient of `z` | bf16 / fp8 | fused dSwiGLU outputs fp8 directly |
| `y1s` | router-weighted `S·A_e` for `dW2` | bf16 | fp8 probe works, but weight-grad consumer not ready |
| `dw1 / dw2` | weight gradients | fp32 accumulation | stable |

---

## 1. Milestone Summary

### Infrastructure (Milestones 1.1-1.5)
FP8 protocol skeleton (`fp8_protocol.py`, `fp8_quant.py`), preact fused boundary, `y1` reuse from QuACK `gemm_gated`, benchmark infrastructure (`--report_fp8_metrics`, `--report_stage_memory`), pre-allocated output buffer contracts.

### Inference Optimizations (Milestones 1.6-1.14)
Removed autograd wrappers from inference path (peak memory -154 MiB), `D=None` gather-A inference (true z elimination), FP8 `y1s/postact_out` in local `gemm_dgated`, generic runtime-FP8 cute tensor conversion, cluster idle scan/launch tooling (`tools/cluster_idle_launch.py`), removed unused `s_scatter_idx` save, minor optimizations batch, contract test stubs for 3 large projects.

### Sprint: Full-Pipeline FP8 (Session 2, Milestone 6.3)
Three-way parallel agent sprint: blockscaled varlen kernel (P1), forward pipeline (P2), backward benchmark (P3). All 6 forward + act-grad GEMMs switched to blockscaled 1x32 UE8M0. Fused SwiGLU+quantize Triton kernels integrated. Weight cache with version-aware invalidation.

### Session 3: Token Rounding + Fused Pack
Identified that "4x training regression" was vanilla top-K benchmark artifact. Token rounding routing guarantees 128-aligned expert segments → zero padding overhead. Implemented fused quantize+ISA-scale-pack Triton kernel. **Result: FP8 training E2E 8.5% faster than BF16.**

---

## 2. Experiment Failures (Do Not Repeat)

### 2.1 Grouped/static-capacity blockscaled down-proj
Optimized pack+quant, eliminated intermediate buffers. **Wall**: grouped_out, static capacity, and grouped→router-aggregation transition layer violate SonicMoE varlen memory contract.

### 2.2 Dummy postact buffer (`SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER=1`)
Emit fp8 dummy postact, reconstruct bf16 y1 from z. **Result**: slower than baseline. Default off.

### 2.3 Backward runtime-fp8 y1s
Local `gemm_dgated` supports fp8 y1s. But downstream `_down_projection_backward_weight` blocked by: (1) `convert_torch_tensor_to_cute_tensor` doesn't support fp8 dlpack, (2) `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd` assumes sm_90a (Hopper), not sm_100a (Blackwell).

### 2.4 Per-tensor FP8 training
RelRMSE ~100% at training-normal input scale (0.02*randn). E4M3 range [-448, 448] flushes small values. **Completely unacceptable.**

### 2.5 `blockscaled_fp8_weight_grad_gemm()`
Fully implemented. Pack/transpose/quantize overhead at E=128 dwarfs GEMM itself. Removed from main path.

### 2.6 Flat varlen-M blockscaled monkey-patch (Milestone 1.10)
CUTLASS `tile_atom_to_shape_SF` requires rank-3 `(2,1,3)`. Even temporary rank-lift fails at `cute.local_tile` coordinate/profile mismatch. **Upstream design limitation** → solved by rank-aware monkey-patch.

---

## 3. Lessons Learned

### 3.1 Best Practices
1. Always ask "does it respect SonicMoE's varlen memory contract?" before "is it more aggressive fp8?"
2. All performance claims must include bf16 baseline
3. Large shape results need: metrics cold run, stagewise probe, theoretical accounting
4. Fix benchmark contracts before discussing performance wins
5. Write "why this is NOT the mainline" clearly — more important than "this path works"

### 3.2 Anti-Patterns
1. Treating grouped/static-capacity as the default evolution direction
2. Mistaking toy case improvements for real mainline progress
3. Ignoring SonicMoE's reuse and scheduling, only focusing on dtype
4. Treating one raw peak measurement as final ground truth
5. Assuming numerical correctness means "close to endgame"

---

## 4. Remaining Gaps

### 4.1 Completed by Sprint + Session 3
- ~~varlen FP8 postact + scales~~ → blockscaled_fp8_gemm_varlen with pre-quantized path
- ~~gather-A preserving down-proj fp8 mainloop~~ → blockscaled path bypasses gather-A
- ~~training 4x regression~~ → was benchmark artifact; token rounding eliminates padding

### 4.2 Still Open
1. `backward mixed-dtype / scaled GEMM` for weight-grad blockscaled
2. `persistent static FP8 weight storage` (optimizer-aware)
3. `fully cudagraph-compatible FP8 path`
4. BF16 token rounding backward: CUTLASS DSL alignment bug in `gemm_dgated` compile
5. Fused SwiGLU kernel scale output → direct ISA layout (currently uses `pack_blockscaled_1x32_scales`)

### 4.3 Avoid
- Do not re-invest into grouped/static-capacity mainline-ization (unless explicitly for ablation)
