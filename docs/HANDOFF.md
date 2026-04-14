# SonicMoE FP8 Optimization — Handoff

> Branch: `native-fp8-exploration`, commit `e5a28ce`
> Environment: quack-kernels 0.3.7, torch 2.11.0+cu130, B30Z (Blackwell)
> Python: 3.13 via `panzhaowu/envs/xfer/bin/python`

---

## 1. Performance (nsys GPU-Projection, default high-perf path)

| T | E | BF16 µs | FP8 µs | Speedup | FP8 Bwd MiB | MemΔ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 8192 | 8 | 3600 | 2698 | **1.33×** | 1547 | +6% |
| 8192 | 32 | 3735 | 2917 | **1.28×** | 2909 | +8% |
| 8192 | 128 | 5056 | 3909 | **1.29×** | 8700 | +10% |
| 32768 | 8 | 15811 | 10518 | **1.50×** | 5359 | +9% |
| 32768 | 32 | 16718 | 10530 | **1.59×** | 6176 | +7% |
| 32768 | 128 | 18382 | 11798 | **1.56×** | 11669 | +8% |

Precision: all RRMSE < 7%, cosine > 0.997. Full data: `reports/session53_breakdown.md`.

---

## 2. Key Code Changes (Session 53)

| Change | File | Impact |
|--------|------|--------|
| VARLEN weight cache preservation | `functional/__init__.py` | +11pp speedup (1.03→1.14×) |
| FUSED weight cache preservation | same | Eliminates re-quant at large E |
| Cache corruption fix (no resize_(0)) | same | Fixes E>8 crash |
| Wgrad threshold=0 | same | FP8 wgrad at all I values |
| Non-aligned → RuntimeError | same | Callers must token-round |
| B.mT without .contiguous() | `gemm_interface.py` | Aligns BF16 with official (<1%) |
| introspect: GPU isolation | `tools/introspect.py` | Correct parallel measurements |
| introspect: token rounding | same | E>8 nsys/precision support |
| introspect: kernel categorization | same | No more "Other" blob |
| CPU optimizer offload (opt-in) | `moe.py` | -3.4 GB base at E=128 |

---

## 3. Architecture

### Default FP8 Path (max performance)
```
Setup: refresh_fp8_shadow_weights() → stash_bf16_to_cpu()
Each iter: fwd(fp8) → bwd(fp8) → zero_grad
Optimizer: unstash → Adam.step → refresh → re-stash
```
All 4 FP8 weight caches retained (w1_fused, w2_varlen, w2_dgated, w1T_varlen).

### Optional CPU Optimizer (max memory savings)
```
Setup: setup_cpu_optimizer(Adam, lr=1e-3)
Each iter: fwd(fp8) → bwd(fp8) → cpu_optimizer_step()
```
Bf16 master weights + Adam states on CPU. Saves ~3.4 GB base at E=128.

### Token Rounding (E>8)
FP8 requires 128-aligned expert segments. For E>8, use official
`forward_token_choice_rounding(Mtile=128)`. BF16 handles any routing natively.

---

## 4. How to Benchmark

```bash
# nsys GPU-projection (gold standard)
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode nsys --gpu 0 \
  --nsys-iters 12 --nsys-warmup 3 --nsys-shapes 8192,3072,1536,8,8

# Precision audit
CUDA_VISIBLE_DEVICES=0 python tools/introspect.py --mode precision --gpu 0 \
  --nsys-shapes 8192,3072,1536,32,8 --precision-seeds 42,123,777

# Parallel multi-shape
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python tools/introspect.py --mode nsys --gpu 0 \
    --nsys-shapes <T,H,I,E,K> 2>&1 > /tmp/gpu${g}.log &
done; wait
```

**Rules:** GPU must be idle. Each shape in isolated subprocess. BF16 uses
`moe_TC_softmax_topk_layer` directly (matches official). FP8 E>8 uses token rounding.

---

## 5. File Map

| File | Role |
|------|------|
| `sonicmoe/functional/__init__.py` | Core FP8 fwd/bwd orchestration |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, CUTLASS wrappers |
| `sonicmoe/quack_utils/gemm_interface.py` | BF16 GEMM wrapper (B.mT fix) |
| `sonicmoe/moe.py` | MoE module: stash, CPU optimizer, refresh |
| `tools/introspect.py` | nsys/precision/report profiling tool |
| `reports/session53_breakdown.md` | Final performance/memory/precision data |
| `docs/HANDOFF.md` | This file |

---

## 6. Next Steps

1. **FP8 Parameter**: store weights as fp8 directly (saves 50% param memory, eliminates shadow cache duplication)
2. **8-bit Adam**: bitsandbytes-style optimizer to reduce optimizer state memory
3. **Fuse dual quant into dSwiGLU**: save 1 HBM read (~80µs)
4. **Stream overlap**: quant kernels on parallel streams (~50µs)
