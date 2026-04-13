# SonicMoE FP8 Optimization — Handoff

> Branch: `native-fp8-exploration`
> Date: 2026-04-13 (Session 53, final)
> Environment: quack-kernels 0.3.7, torch 2.11.0+cu130, B30Z (Blackwell)
> Python: 3.13 via `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python`

---

## 1. Project Status

FP8 blockscaled training is **1.48-1.89× faster** than BF16 across E=8/32/128
at the Ernie shape (T=8192, H=3072, I=1536, K=8) on B30Z.

| E | BF16 (µs) | FP8 (µs) | Speedup | FP8 Bwd Peak | MemΔ vs BF16 |
|---|:---:|:---:|:---:|:---:|:---:|
| 8 | 4050 | 2739 | **1.48×** | 1547 MiB | +6.0% |
| 32 | 4646 | 3008 | **1.54×** | 3624 MiB | +33.2% |
| 128 | 7906 | 4192 | **1.89×** | 11562 MiB | +44.8% |

vs Official BF16 baseline (3767 µs): **1.38× at E=8**.

Method: nsys GPU-projection (gold standard), 20 measured iters.
Precision: RRMSE < 7%, cosine > 0.997 for all output/gradient tensors.

---

## 2. What Was Changed in Session 53

### Critical Bug Fixes (3 cache-related)

1. **VARLEN weight cache eviction** (`functional/__init__.py` line ~1629):
   DownProjection backward was clearing `_VARLEN_WEIGHT_CACHE` every iteration.
   This forced 3 weight re-quantizations per iter (~300µs at E=8, ~980µs at E=128).
   **Fix**: keep cache alive (version-keyed, auto-invalidates at optimizer step).

2. **FUSED weight cache eviction** (same file, lines ~1629/2249):
   Both `moe_TC_softmax_topk_layer` and `moe_general_routing_inputs` cleared
   `_FUSED_WEIGHT_CACHE` between forward and backward. Same re-quant cost.
   **Fix**: keep cache alive.

3. **Cache corruption via ctx.resize_(0)** (line ~1741):
   DownProjection backward freed `ctx._w2_dgated_fp8.untyped_storage()` after
   using it. But this tensor is the SAME Python object as the `_FUSED_WEIGHT_CACHE`
   entry → cache corrupted → "storage of size 0" crash on next forward.
   **Fix**: don't free ctx tensors that alias cache entries.

   **Cost of all 3 fixes**: +~148 MiB backward peak (FP8 weight caches retained).

### Other Changes

4. **Wgrad threshold = 0**: FP8 wgrad enabled at all I values (was I≥2048).
   After cache fixes, FP8 wgrad is profitable even at I=1536.

5. **Non-aligned FP8 = RuntimeError**: Non-128-aligned expert segments raise
   an explicit error instead of producing silent garbage. Callers must use
   token rounding (official `forward_token_choice_rounding` with Mtile=128).

6. **introspect.py enhancements**:
   - `_resolve_python_bin()`: auto-detect working Python binary
   - `_subprocess_env_for_gpu()`: respect shell-level CUDA_VISIBLE_DEVICES
     (fixes GPU contention in parallel runs)
   - nsys-rep files saved to persistent `/panzhaowu/output/nsys/`
   - `--gpu-metrics-devices=all` for SM utilization
   - Memory measurement paired with each nsys run
   - Token rounding for E>8 in nsys/memory workloads
   - Improved kernel categorization (no more "Other" blob)

7. **colwise_quantize_and_pack**: `torch.empty` instead of `torch.full`
   for tile-aligned shapes (saves ~5µs/call fill kernel).

---

## 3. Architecture: How FP8 Works

### Forward Path (aligned, fused_gated)
```
x(T,H) bf16
  → quantize_and_pack_activation → x_fp8(T,H) + ISA scales
  → _gather_isa_packed_scales T→TK
  → GemmGatedSm100ZeroMat (CUTLASS, A_idx gather, no TK materialization)
  → z(TK,2I) bf16, y1(TK,I) bf16
  → z quantize → z_fp8 (save for backward, -50% z memory)
  → y1 quantize → y1_fp8 + ISA scales (prequant cache for DownProj)
  → blockscaled_fp8_gemm_varlen (y1_fp8 @ w2_fp8) → output(TK,H)
  → token_gather_sum → (T,H)
```

### Backward Path (aligned, wgrad ON)
```
dout(T,H) bf16
  DownProj backward:
    → dout quantize + ISA scale gather
    → GemmDGatedZeroMat (CUTLASS fused: GEMM + dSwiGLU + colvec_reduce)
      → dy1(TK,I), dz(TK,2I), y1s(TK,I), ds
    → dual_quantize_varlen(dz) → row_fp8 + col_fp8 (single HBM read)
    → FP8 wgrad: colwise_quant(y1s) || colwise_quant(dout,gather)
      → _run_cutlass_blockscaled_gemm_varlen_k → dw2
    → prequant dz for UpProj

  UpProj backward:
    → FP8 wgrad: col_fp8(dz) + col_fp8(x,gather) → CUTLASS varlen_k → dw1
    → Free dz bf16 (-384 MiB)
    → FP8 actgrad: dz_fp8 @ w1T_fp8 → dx_expanded
    → token_broadcast_backward → dx(T,H)
```

### Key Design Patterns
- **Zero-materialization**: TK-sized FP8 activation never stored in HBM
- **Stash mode**: bf16 params → CPU, compute on FP8 shadow caches
- **Weight cache**: version-keyed `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE`
  auto-invalidate at optimizer step. NEVER free cached tensor storage.
- **Dual quant**: single HBM read → row + col fp8 (saves ~80µs vs separate)
- **Token rounding**: for E>8, `forward_token_choice_rounding(Mtile=128)` ensures
  128-aligned expert segments

---

## 4. Lessons Learned

### Performance Measurement
- **Only nsys GPU-projection is trustworthy**. CUDA events include CPU overhead.
- **Our branch BF16 ≠ Official BF16**. Our branch has ~5-9% FP8 infra overhead
  even in BF16 mode. Always use official SonicMoE as baseline.
- **3+ repeated measurements required**. Single-run data is unreliable.
- **GPU must be idle** (util=0%). Non-idle measurements can be 2-5× off.
- **Parallel runs need GPU isolation**: `CUDA_VISIBLE_DEVICES` at shell level,
  NOT inside subprocess scripts.

### Cache/Memory
- **Never `resize_(0)` on tensors that alias a cache**. This was the single
  biggest bug — caused "storage of size 0" crashes and silent performance
  regression (cache miss → re-quant every iter).
- **Version-keyed caches are self-cleaning**. `w._version` increments at
  optimizer step → stale entries miss → fresh entries written. No need for
  manual eviction.
- **FUSED cache holds ~74 MiB** (w1 fp8). VARLEN holds ~47 MiB (w2 fp8).
  Total ~121 MiB retained is acceptable for ~1000µs/iter savings.

### FP8 Constraints
- **128-alignment is hardware (SM100 ISA scale tile layout)**, not software.
  Cannot be relaxed. Token rounding is the production solution.
- **E>8 with random routing is always non-aligned**. Must use token rounding.
- **E=8 with T=8192 is coincidentally aligned** (seed=42 init → uniform routing).

---

## 5. File Map

### Modified Production Code
| File | Changes |
|------|---------|
| `sonicmoe/functional/__init__.py` | Cache fixes, wgrad threshold=0, non-aligned error, FUSED/VARLEN preservation |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | colwise tile-aligned torch.empty optimization |
| `tools/introspect.py` | Python resolution, GPU isolation, nsys improvements, token rounding, kernel categorization |

### Reports
| File | Content |
|------|---------|
| `reports/session53_breakdown.md` | **Definitive**: final perf/mem/precision data with budget breakdown |
| `reports/session52_perf_mem_precision_report.md` | Historical: pre-Session-53 data (superseded) |
| `reports/nsys_final/nsys_gpu_projection.json` | Machine-readable nsys data |

### BF16 Baseline
| Path | Purpose |
|------|---------|
| `/root/.../lab/official/sonic-moe` | **Official BF16 code** — the ONLY valid baseline |
| `/root/.../envs/official_bf16` | Its Python environment |
| `/root/.../envs/xfer` | FP8 frontier Python 3.13 environment |
| `/root/.../output/nsys/` | Persistent nsys-rep files for GUI inspection |

---

## 6. Next Steps (Prioritized)

### Tier 1: Production readiness
1. **Token rounding integration into MoE.forward()**: currently only
   `moe_general_routing_inputs` supports it. MoE.forward should auto-round
   for E>8 or expose a `token_rounding=True` flag.
2. **Stash for token rounding path**: `moe_general_routing_inputs` doesn't
   support stash (bf16→CPU). Adding this would save ~216 MiB at E=8 and
   reduce base memory at all E values.
3. **Precision regression test**: add E=32/128 precision checks to
   `fp8_large_project_contract_test.py`.

### Tier 2: Further optimization
4. **Fuse dual quant into dSwiGLU**: currently separate kernel. Fusing saves
   1 HBM read of dz (~80µs). Prototype exists in Session 52 HANDOFF.
5. **Stream overlap**: y1s quant || dz quant on separate streams (~50µs).
6. **Reduce FP8 memory at large E**: lazy weight quantization (quant on first
   use, not pre-compute all 128 experts at refresh_fp8_shadow_weights).

### Tier 3: Research
7. **FP8 padding support**: decompose `gemm_dgated` into separate actgrad +
   SwiGLU backward for non-aligned shapes (eliminates token rounding requirement).
8. **Per-expert weight layout**: store weights as (E, dim, dim) contiguous
   to eliminate permute().contiguous() overhead at large E.

---

## 7. How to Run

```bash
# Activate environment
source /root/.../envs/xfer/bin/activate

# FP8 introspect (nsys GPU-projection, the gold standard)
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --gpu 0 \
  --nsys-iters 20 --nsys-warmup 5 --nsys-shapes 8192,3072,1536,8,8

# Multiple shapes in parallel (one GPU per shape)
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python tools/introspect.py --mode nsys --gpu 0 \
    --nsys-shapes <T,H,I,E,K> 2>&1 > /tmp/nsys_gpu${g}.log &
done; wait

# Official BF16 baseline (separate env)
CUDA_VISIBLE_DEVICES=0 /root/.../envs/official_bf16/bin/python \
  /root/.../lab/official/sonic-moe/benchmarks/moe-cute.py \
  --thiek 8192,3072,1536,8,8 --dtype BFloat16 --activation swiglu --skip_test

# Quick correctness check
CUDA_VISIBLE_DEVICES=0 python -c "
import os; os.environ['USE_QUACK_GEMM']='1'; os.environ['SONIC_MOE_FP8_MODE']='perf'
from sonicmoe import MoE; from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
import torch
m = MoE(8,8,3072,1536,ActivationType.SWIGLU,False,0.02).cuda().bfloat16()
m.refresh_fp8_shadow_weights(); m.stash_bf16_to_cpu()
x = torch.randn(8192,3072,dtype=torch.bfloat16,device='cuda',requires_grad=True)
with enable_quack_gemm(True), enable_fp8(True):
    o,l = m(x, use_fp8=True)
(o.sum()+l).backward()
print(f'OK: output.norm={o.norm():.2f}, dx.norm={x.grad.norm():.2f}')
"
```
