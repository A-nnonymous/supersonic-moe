# SonicMoE Agent Context

Cold-start context for agents working on SonicMoE FP8 blockscaled optimization.
**Read `reports/fp8_upgrade/HANDOFF.md` first** — it is the single source of truth.

## Current Status (2026-04-12, Session 51)

- **FP8 forward+backward fully functional** — zero-materialization CUTLASS kernels, no TK-sized FP8 activation materialized
- **34/34 precision tests + 20 subtests PASS** (RRMSE <10%, correlation >0.99)
- **Performance** (CUDA events, same-process, B200 under contention):
  - I=1536 (Ernie): **1.08×** speedup (conservative clean-round; BF16 high-variance under contention)
  - I=2048: **1.22×** speedup (very consistent)
- **Memory**: FP8 saves 4–5% peak; **FP8+Stash saves 21–23% peak**
- **Config**: Pythonic `SonicMoEConfig` API (env vars still work as fallback)

## Best training path (Pythonic Config)

```python
from sonicmoe import MoE, SonicMoEConfig
from sonicmoe.enums import ActivationType

moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
           intermediate_size=1536, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True)
with cfg.activate():
    output, aux_loss = moe(x, use_fp8=True)
```

For maximum memory savings, add weight stash:
```python
optimizer.step()
moe.refresh_fp8_shadow_weights()  # bf16 → FP8 caches
moe.stash_bf16_to_cpu()           # -216 MiB GPU (saves 21-23% peak)
# ... forward + backward with FP8 ...
moe.unstash_bf16()                # restore for optimizer
```

## Critical Design Context

SonicMoE avoids materializing gathered TK-sized activations:
- `quantize_and_pack_activation(x)` → T-sized FP8 + T-sized scales
- `_gather_isa_packed_scales_kernel` → TK-sized ISA-packed scales
- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat`: T-FP8 + A_idx + TK-scales (no TK FP8 copy)

**Epilogue quant** (default ON): GemmGated writes z directly as fp8 in CUTLASS epilogue.

## Key files

| File | Role |
|------|------|
| `sonicmoe/config.py` | `SonicMoEConfig` dataclass + thread-local context manager |
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration, FP8 config, cache management |
| `sonicmoe/moe.py` | MoE class, stash/unstash, refresh_fp8_shadow_weights |
| `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | Zero-mat CUTLASS kernel classes |
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | CuTe DSL colwise quant |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, weight caches |
| `tools/introspect.py` | Profiling harness — nsys GPU-projection, precision, memory |
| `reports/fp8_upgrade/HANDOFF.md` | **Complete state, measurements, lessons, next steps** |
| `tests/fp8_large_project_contract_test.py` | 34 precision tests + 20 subtests |

## Non-Negotiable

- Maintain <10% RelRMSE and >0.99 correlation vs BF16
- Use native CUTLASS / QuACK paths, not Triton `tl.dot_scaled`
- Keep non-aligned routing fallback behavior intact

## Quick validation

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short
```

## Critical lessons (see HANDOFF.md for complete list)

1. **`_fp8_mode()` priority**: Context manager overrides env var (fixed Session 51)
2. **Measurement**: CUDA events same-process > nsys on busy cluster > wall-clock
3. **Quant overhead**: 25–42% of FP8 time, #1 optimization target
4. **QuACK JIT cache**: Must manually clear after CuTe kernel edits
5. **Weight stash**: Dominant memory optimization (21-23% peak savings)

## What NOT to waste time on

See HANDOFF.md §7 for complete dead-end list with evidence.

## Read order

1. `reports/fp8_upgrade/HANDOFF.md` — complete state, all numbers, next steps
2. `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` — zero-mat kernel
3. `sonicmoe/functional/__init__.py` — forward/backward flow
4. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` — quant kernels
5. `sonicmoe/config.py` — Pythonic config API
