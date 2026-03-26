# SonicMoE FP8 Reports

> Primary reference: [`reports/fp8_upgrade/HANDOFF.md`](fp8_upgrade/HANDOFF.md)
> Detailed history: [`reports/fp8_upgrade/ENGINEERING_LOG.md`](fp8_upgrade/ENGINEERING_LOG.md)
> CTA-tile alignment analysis: [`reports/fp8_upgrade/BLOCKSCALED_ALIGNMENT.md`](fp8_upgrade/BLOCKSCALED_ALIGNMENT.md)

---

## Current Status (2026-03-26 Session 2)

**6/8 GEMM operators** use blockscaled 1x32 UE8M0 FP8 (forward + activation-grad).
**2/8 weight-grad operators** still use per-tensor FP8 `.to(fp8)`.
Fused SwiGLU+quantize Triton kernels integrated (zero bf16 intermediate).

### Performance (shape 8192,4096,1024,128,8)

| Metric | BF16 | Blockscaled FP8 | Delta |
|--------|------|-----------------|-------|
| Forward inference (ms) | 3.878 | 2.216 | **-42.9%** |
| Training fwd (ms) | 3.511 | 20.961 | +497% (regression) |
| E2E fwd+bwd (ms) | 11.889 | 48.486 | +308% (regression) |

### Critical Issue

Training performance **severely regressed** due to `blockscaled_fp8_gemm_varlen` overhead (pack/unpack/quantize per expert × 128 experts). See HANDOFF.md §4 for solutions.

### Precision

- Blockscaled single-operator RelRMSE: **3.74%** (validated)
- E2E blockscaled precision: **not yet measured** (training too slow for benchmark)

## Priority Roadmap

| Priority | Task |
|----------|------|
| **P0** | Eliminate blockscaled varlen GEMM pack/unpack overhead (training performance) |
| **P1** | Weight-grad blockscaled (if precision needed) |
| **P2** | Memory optimization (unify weight caches) |

## Quick-Start

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass excluding large_shape)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```
