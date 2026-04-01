# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For the complete project state, read `reports/fp8_upgrade/HANDOFF.md`.

## Current Status (2026-04-01)

- **FP8 forward+backward works** with fused GemmGated+SwiGLU CUTLASS kernel
- Forward still materializes TK-sized gathered activation (via `gather_quantize_and_pack_activation`)
- **#1 next task: eliminate forward materialization** — use T-sized quant + A_idx (matching backward)
- Precision: **12/12** tests pass (RelRMSE <10%, correlation >0.99)
- Ernie shape (T=8192, H=3072, I=1536, E=8, K=8): FP8 only 1.04× faster (gather overhead dominates)
- Old shapes (T=4096, H=4096, I=1024–4096, E=128, K=8): 1.66–2.37× wall-clock faster

## Best training path

```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

Or via env vars:
```bash
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf
```

## Critical Design Context

SonicMoE's core design avoids materializing the gathered TK-sized activation. In BF16, A_idx gathers
data rows inside CUTLASS (no TK copy). In FP8, **data AND ISA-packed scales must be gathered together**
because they are row-correlated. The backward dgated already does this correctly (T-quant + A_idx passes
precision). The forward GemmGated does NOT — when tried, it gives 93.5% RRMSE due to scale metadata
mismatch in the CUTLASS kernel. See HANDOFF.md §8.1 for root cause and fix paths.

## Key files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/functional/utils.py` | `enable_fp8()` context manager |
| `sonicmoe/moe.py` | `MoE.forward(use_fp8=True)` entry point |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated (forward, A_idx broken with scales) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated (backward, A_idx works with scales) |
| `reports/fp8_upgrade/HANDOFF.md` | **Complete project state, all numbers, root causes, next steps** |

## Non-Negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback behavior intact
- Exclude `elementwise_kernel` from BF16 baseline (QuACK 0.3.7 bug)
- Official BF16 env: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16`

## Quick validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape" --tb=short
```

## What NOT to waste time on

- FP8 wgrad at T=4096/E=128 (K_per_expert=256 too small, proven dead-end)
- Eager FP8 weight cache eviction (makes memory worse)
- FP8 down-proj at I=1024 (quant cost > GEMM savings)
- See HANDOFF.md §4 for full dead-end list

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, root causes, measurements, next steps
2. `reports/fp8_upgrade/engineering_log.md` — what happened and why
3. `sonicmoe/functional/__init__.py` — main code
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant + GEMM wrappers
