# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For detailed handoff context and debugging history, read `agent.md` and `reports/fp8_upgrade/HANDOFF.md`.

## Scope

- Full-chain blockscaled FP8 (1×32 UE8M0) MoE training on Blackwell (sm_100a)
- Goal: fused GEMM+SwiGLU kernel with blockscaled FP8, performance far exceeding BF16 baseline
- Key files: `sonicmoe/quack_utils/gemm_gated.py`, `blockscaled_fp8_gemm.py`, `functional/__init__.py`

## Non-Negotiable

- Follow sonic-moe's fusion philosophy: fused operators only, decomposed path is fallback only
- Do not compromise on precision or performance for complexity reasons
- All FP8 paths must maintain <10% RelRMSE, >0.99 correlation vs BF16
- Use native CUTLASS/QuACK GEMM path, not Triton `tl.dot_scaled` (broken on sm_100a)

## Architecture

- QuACK (quack-kernels 0.3.7) wraps CUTLASS DSL (nvidia-cutlass-dsl 4.4.2) for Blackwell SM100
- `GemmGatedSm100` = fused GEMM+SwiGLU with composable epilogue (`TileStore` + `_halve_epi_tile`)
- `GemmDefaultSm100` = plain GEMM (decomposed fallback, currently working with blockscaled)
- Blockscaled FP8 uses ISA-packed E8M0 scales with `sf_vec_size=32`
- Weight layout: interleaved gate/value columns `[g0, v0, g1, v1, ...]`

## Environment

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
# See /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md for cluster details
```

## Validation

```bash
# Contract tests
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# Quick decomposed FP8 GEMM smoke test
CUDA_VISIBLE_DEVICES=0 python -c "
from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_gemm_varlen
from sonicmoe.functional.fp8_protocol import FP8Protocol
import torch
E,K,N = 8,1024,2048; tpe=64; T=E*tpe
a = torch.randn(T,K,device='cuda',dtype=torch.bfloat16)
w = torch.randn(N,K,E,device='cuda',dtype=torch.bfloat16)
cu = torch.arange(0,T+1,tpe,device='cuda',dtype=torch.int32)
out = blockscaled_fp8_gemm_varlen(a,w,cu,protocol=FP8Protocol())
torch.cuda.synchronize(); print('OK', out.shape)
"
```

## Read First

1. `reports/fp8_upgrade/HANDOFF.md` — Complete status and debugging history
2. `agent.md` — Quick handoff context
3. `sonicmoe/quack_utils/gemm_gated.py` — Fused forward kernel
4. `sonicmoe/quack_utils/gemm_dgated.py` — Fused backward kernel
5. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — Core FP8 infrastructure