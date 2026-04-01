# SonicMoE Agent Context

Use this as the quick cold-start summary. For full state, measurements, and lessons, read `reports/fp8_upgrade/HANDOFF.md` first.

## Current Status (2026-04-01)

- **FP8 forward+backward works** with fused GemmGated+SwiGLU CUTLASS kernel
- Forward still materializes TK-sized gathered activation → **#1 optimization target**
- Backward dgated uses T-sized quant + A_idx (no materialization, proven correct)
- Precision: **12/12** tests pass (RelRMSE <10%, correlation >0.99)
- Ernie shape (T=8192, H=3072, I=1536, E=8, K=8): FP8 1.04× faster (barely)
- Old shapes (T=4096, H=4096, E=128, K=8): 1.66–2.37× wall-clock faster
- Memory: FP8 uses 1.38–1.48× more (not yet a win)

## Best training path

```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

## Key files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, all decisions |
| `sonicmoe/functional/utils.py` | `enable_fp8()` context manager |
| `sonicmoe/moe.py` | `MoE.forward(use_fp8=True)` entry point |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, weight caches |
| `reports/fp8_upgrade/HANDOFF.md` | **Complete state, root causes, measurements, next steps** |
| `reports/fp8_upgrade/engineering_log.md` | Cleaned milestone log |

## Critical insight for next agent

**In FP8, data AND ISA-packed scales are row-correlated. A_idx must gather both.**

The backward already does this correctly:
```python
dout_fp8, dout_scales = quantize_and_pack_activation(dout)  # T-sized
gemm_dgated_kernel(dout_fp8, ..., A_idx=x_gather_idx, a_scales=dout_scales)  # ✅ passes
```

The forward currently does NOT — it materializes the TK-sized gathered data:
```python
x_fp8, x_scales = gather_quantize_and_pack_activation(x, x_gather_idx)  # TK-sized, ~96µs
gemm_gated(x_fp8, ..., a_scales=x_scales)  # No A_idx, works but slow
```

Attempted fix (T-quant + A_idx) gave 93.5% RRMSE due to GemmGated kernel not handling
scale indexing with A_idx correctly (unlike GemmDGated which does). See HANDOFF.md §8.1.

## Quick validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape" --tb=short
```

## What NOT to waste time on

- FP8 wgrad at T=4096/E=128 (K_per_expert=256 too small)
- Eager FP8 weight cache eviction (makes memory worse)
- FP8 down-proj at I=1024 (quant cost > GEMM savings)
- See HANDOFF.md §4 for full dead-end list

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, all numbers, root causes, next steps
2. `reports/fp8_upgrade/engineering_log.md` — what happened and why
3. `sonicmoe/functional/__init__.py` — main code
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant + GEMM wrappers
