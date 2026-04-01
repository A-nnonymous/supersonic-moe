# SonicMoE Agent Context

Use this as cold-start context for any new agent working on the SonicMoE FP8 blockscaled optimization.
For the complete project state, read `reports/fp8_upgrade/HANDOFF.md`.

## Current Status (2026-04-01, Session 25)

- **FP8 forward+backward works** with zero-materialization CUTLASS kernels
- No TK-sized FP8 activation materialized (follows SonicMoE's core design)
- **15/15** precision tests pass (RelRMSE <10%, correlation >0.99)
- Ernie shape (T=8192, H=3072, I=1536, E=8, K=8): **1.30× wall-clock**, **1.18× GPU projection**
- I=2048 shape: **2.35× wall-clock**

## Best training path

```python
from sonicmoe import MoE, enable_fp8, enable_quack_gemm
with enable_quack_gemm(True):
    out, loss = moe(x, use_fp8=True)
```

## Critical Design Context

SonicMoE's core design avoids materializing gathered TK-sized activations. In BF16, A_idx
gathers data rows inside CUTLASS (no TK copy). The FP8 path now follows the same principle:
- `quantize_and_pack_activation(x)` → T-sized FP8 + T-sized scales
- `_gather_isa_packed_scales_kernel` → TK-sized ISA-packed scales (~54µs, 1.6% of CUDA time)
- `GemmGatedSm100ZeroMat` kernel: T-FP8 + A_idx + TK-scales (no TK FP8 materialization)

The zero-mat kernels subclass GemmSm100 via MRO and override only `__call__` with `@cute.jit`.
Auto-selected in `gemm_gated.py`/`gemm_dgated.py` when `gather_A + blockscaled`.

## Key files

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Main FP8 logic — forward, backward, ctx state persistence |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | **Zero-mat kernel classes** |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Quant kernels, three-step pipeline, scale gather |
| `sonicmoe/quack_utils/gemm_gated.py` | CUTLASS GemmGated wrapper (auto ZeroMat selection) |
| `sonicmoe/quack_utils/gemm_dgated.py` | CUTLASS GemmDGated wrapper (auto ZeroMat selection) |
| `reports/fp8_upgrade/HANDOFF.md` | **Complete project state, root causes, measurements, next steps** |
| `tests/fp8_large_project_contract_test.py` | 15 precision tests |

## Non-Negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback behavior intact
- Official BF16 env: `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16`

## Quick validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

## What NOT to waste time on

- FP8 wgrad (colwise quant SM contention, proven net-negative at ALL shapes)
- torch.as_strided for fake TK shape (PyTorch storage bounds check)
- Rowwise quant + strided view for wgrad (HW requires contiguous K groups)
- FP8 down-proj at I=1536 (quant ≈ GEMM savings)
- Transpose + rowquant for wgrad (transpose alone 1509µs > colwise 260µs)
- See HANDOFF.md §4 for full dead-end list

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, root causes, measurements, next steps
2. `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` — zero-mat kernel (the key innovation)
3. `sonicmoe/functional/__init__.py` — forward/backward flow + ctx state fix
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant kernels + three-step pipeline
