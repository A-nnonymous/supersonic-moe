# SonicMoE FP8 Reports

> Primary reference: [`reports/fp8_upgrade/HANDOFF.md`](fp8_upgrade/HANDOFF.md)
> Historical context: [`reports/fp8_upgrade/ENGINEERING_LOG.md`](fp8_upgrade/ENGINEERING_LOG.md)
> CTA-tile alignment analysis: [`reports/fp8_upgrade/BLOCKSCALED_ALIGNMENT.md`](fp8_upgrade/BLOCKSCALED_ALIGNMENT.md)

---

## Current Status (2026-03-27 Session 3)

**6/8 GEMM operators** use blockscaled 1x32 UE8M0 FP8 (forward + activation-grad).
**2/8 weight-grad operators** still use per-tensor FP8 `.to(fp8)`.
Fused SwiGLU+quantize and fused quantize+ISA-pack Triton kernels integrated.

### Performance (shape 8192,4096,1024,128,8, token rounding nr routing)

| Metric | BF16 (ref) | Blockscaled FP8 | Delta |
|--------|-----------|-----------------|-------|
| Forward inference (ms) | 3.878 | 2.216 | **-42.9%** |
| E2E fwd+bwd (ms) | 11.889 | **10.880** | **-8.5%** |
| Wasted ratio | — | 0.000 | zero padding |
| Contract tests | — | **8/8 PASSED** | ✅ |

### Key Insight

Previously reported "4x training regression" was a **benchmark artifact** — vanilla top-K routing triggers 128-alignment padding. Token rounding routing (production mode) guarantees aligned segments, eliminating all overhead.

## Priority Roadmap

| Priority | Task |
|----------|------|
| **P1** | Further training speedup (fuse SwiGLU scale output to ISA layout) |
| **P2** | Weight-grad blockscaled (if precision needed) |
| **P3** | Memory optimization (unify weight caches) |

## Quick-Start

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

# Contract tests (8/8 pass excluding large_shape)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"

# FP8 token rounding benchmark (production mode)
CUDA_VISIBLE_DEVICES=4 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python benchmarks/moe-token-rounding.py --thiekq 8192,4096,1024,128,8,128 --routing nr --skip_test
```
