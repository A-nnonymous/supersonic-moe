# SonicMoE FP8 Agent Handoff Context

This file provides cold-start context for any agent continuing the blockscaled FP8 optimization work.

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Full project status, insights, and next steps
2. `README.md` § "FP8 Blockscaled Status" — Quick overview
3. `sonicmoe/quack_utils/gemm_gated.py` — Fused GEMM+SwiGLU (rewritten for quack 0.3.7)
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — Core blockscaled FP8 infrastructure
5. `sonicmoe/functional/__init__.py` — MoE forward/backward dispatch (lines 460-500 for FP8 forward)

## Current Blocker

**TileStore + blockscaled FP8 = illegal instruction on sm_100a.**

The decomposed path (GemmDefaultSm100, no TileStore) works perfectly after fixing `create_varlen_args`.
Any kernel using `TileStore` epilogue op + `sf_vec_size=32` crashes.
This blocks fused GEMM+SwiGLU which is essential for beating BF16 performance.

## Environment

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
# Packages: quack-kernels==0.3.7, nvidia-cutlass-dsl==4.4.2, torch==2.9.1+cu128
# GPU: 8x B200 (sm_100a), 183GB each, ~50GB free per GPU, all at 100% util
# NCU profiling still possible despite GPU utilization
```

## Key API Facts (quack 0.3.7)

- `GemmWrapperBase.create_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)` — 3 args only
- `GemmWrapperBase.create_scheduler_args(max_active_clusters, ...)` — returns `TileSchedulerOptions`
- `@mlir_namedtuple` replaces `ArgumentsBase` for `EpilogueArguments`
- `_epi_ops` tuple defines composable epilogue: `TileStore`, `ColVecReduce`, `Scalar`, `RowVecLoad`, etc.
- `mark_layout_dynamic(leading_dim=X)` requires `strides[X]==1` — use `.contiguous()` when needed

## Immediate Priority

1. Fix `TileStore` + blockscaled interaction (P0)
2. Fix `gemm_dgated.py` `BFloat16.__c_pointers__` pickle error (P0.5)
3. Contract tests 8/8 pass (P1)
4. NCU profiling FP8 vs BF16 (P2)