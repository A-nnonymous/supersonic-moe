# Blockscaled FP8 MoE — Handoff

> **Last updated:** 2026-03-31
> **Status:** aligned fused FP8 act-grad is landed, but **full-chain FP8 is still not production-ready** because the current FP8 weight-grad path is too slow and still loses on local peak memory / inference.

---

## 0. One-screen summary

If you only read one section, read this one.

- The **old `2930us / 455us gap` story is stale**. It described the pre-fused path before the fused gated/dgated integration and before the FP8 wgrad autograd-copy fix.
- The **current near-baseline training path** in this tree is:
  - `USE_QUACK_GEMM=1`
  - `SONIC_MOE_FP8_MODE=perf`
  - `SONIC_MOE_FP8_ASSUME_ALIGNED=1`
  - `SONIC_MOE_FP8_FUSED_GATED=1`
  - `SONIC_MOE_FP8_WGRAD=0`
- Under the authoritative aligned NSYS harness, that path is now only **~125us behind official BF16**.
- The project is **still not done** because turning on `SONIC_MOE_FP8_WGRAD=1` blows training backward up badly.
- The remaining training blocker is now **FP8 weight-grad**, not forward SwiGLU micro-optimization.
- Current local aligned inference and peak-memory measurements are also still **worse than BF16**, so do **not** claim an inference or memory win yet.
- Current correctness signal is good but not complete:
  - `tests/fp8_large_project_contract_test.py`: **8 passed, 3 deselected** with `SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1`
  - No fresh large-shape gold report for the latest full-chain FP8 wgrad path was produced in this session.

---

## 1. Current code state

### 1.1 What landed in this session

These changes are already in the working tree and were regression-tested:

- `sonicmoe/quack_utils/gemm_interface.py`
  - Blockscaled fused `gemm_gated` / `gemm_dgated` now bypass the unsafe autotuner path and use a safe config path.
- `sonicmoe/functional/__init__.py`
  - Added fused aligned blockscaled helpers for forward / backward.
  - Wired aligned FP8 forward/backward to the fused gated path behind `SONIC_MOE_FP8_FUSED_GATED`.
  - Added robust prequant tensor matching across autograd boundaries.
  - Removed explicit standalone `y1` and `dz` prequant kernels from the fused path after they proved low-value.
  - Fixed the FP8 wgrad autograd-layout issue by writing into base-layout buffers (`dw1_base`, `dw2_base`) before returning grad views.
- `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - Added direct fused-dgated weight cache helper to bypass wrapper-side `B.mT.contiguous()` copy.
  - Extended `blockscaled_fp8_wgrad_varlen_k()` with `b_gather_idx` so `dw1` can be computed in parameter-friendly layout.
- `sonicmoe/quack_utils/__init__.py`
  - Exported the new direct fused-dgated helper.

### 1.2 Practical state of feature flags

- `SONIC_MOE_FP8_FUSED_GATED=1`
  - **Recommended** for current aligned-training experiments.
- `SONIC_MOE_FP8_WGRAD=1`
  - **Not recommended as default**. It is functionally working, but the current kernel path is too slow.
- `SONIC_MOE_FP8_ASSUME_ALIGNED=1`
  - Safe only for the aligned-routing harness / production-style rounded routing.

---

## 2. Source-of-truth measurements

### 2.1 Authoritative training performance

These are the numbers to use in any serious comparison. They come from **NSYS NVTX GPU projection with sync barriers** on the aligned production shape (`T=4096 H=4096 I=1024 E=128 K=8`).

| Path | Forward | Backward | Total | Status |
|------|---------|----------|-------|--------|
| **Official BF16** | `777.3us` | `1697.9us` | `2475.2us` | **Authoritative baseline** |
| **Current fused FP8 + BF16 wgrad** | `812.1us` | `1788.2us` | `2600.3us` | Best current training path in this repo |
| **Current fused FP8 + FP8 wgrad** | `812.0us` | `4838.4us` | `5650.4us` | Full-chain FP8 currently loses badly |

### 2.2 Interpretation

- The **aligned fused act-grad path is mostly solved**:
  - current fused FP8 + BF16 wgrad is only `2600.3us`, about **`125.1us` behind** official BF16.
- The **remaining training blocker is almost entirely wgrad**:
  - enabling FP8 wgrad changes backward from `1788.2us` to `4838.4us`
  - forward is effectively unchanged (`812.1us` vs `812.0us`)
- So the current project frontier is **not** “keep trimming forward SwiGLU first”; it is **replace/redesign the FP8 wgrad path**.

### 2.3 Reproduction commands for authoritative training numbers

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

NSYS=/tmp/nsys-cli-2025.1.1/opt/nvidia/nsight-systems-cli/2025.1.1/target-linux-x64/nsys

# Current fused FP8 + BF16 wgrad
CUDA_VISIBLE_DEVICES=0 SONIC_MOE_FP8_FUSED_GATED=1 \
  "$NSYS" profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --force-overwrite=true -o /tmp/sonic_fp8_current \
  python tools/nsys_profile_comprehensive.py --mode fp8

# Current fused FP8 + FP8 wgrad
CUDA_VISIBLE_DEVICES=0 SONIC_MOE_FP8_FUSED_GATED=1 \
  "$NSYS" profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --force-overwrite=true -o /tmp/sonic_fp8wg_current \
  python tools/nsys_profile_comprehensive.py --mode fp8_wgrad

# Export + compare against official baseline
"$NSYS" export --type=sqlite --output=/tmp/sonic_fp8_current.sqlite /tmp/sonic_fp8_current.nsys-rep
"$NSYS" export --type=sqlite --output=/tmp/sonic_fp8wg_current.sqlite /tmp/sonic_fp8wg_current.nsys-rep
python tools/nsys_full_breakdown.py \
  reports/sonic_official_bf16.sqlite \
  /tmp/sonic_fp8_current.sqlite \
  /tmp/sonic_fp8wg_current.sqlite \
  --labels official_bf16 current_fp8_bf16wgrad current_fp8_wgrad
```

### 2.4 Current local train / infer perf + memory snapshot

These are **local event-based** aligned measurements. Use them for **peak-memory checks and rough local iteration trends**, not as the authoritative cross-branch baseline. On this shared machine the event timings can drift materially; the authoritative training numbers are still the NSYS ones in §2.1.

Measured with `tools/measure_aligned_perf_memory.py`.

#### Training (local event timing, aligned routing)

| Path | Total | Peak memory |
|------|-------|-------------|
| BF16 | `4.894ms` | `7.051 GiB` |
| Current fused FP8 + BF16 wgrad | `3.136ms` | `10.746 GiB` |
| Current fused FP8 + FP8 wgrad | `3.325ms` | `10.808 GiB` |

#### Inference (local event timing, aligned routing, `is_inference_mode=True`)

| Path | Total | Peak memory |
|------|-------|-------------|
| BF16 | `1.019ms` | `7.526 GiB` |
| Current FP8 | `4.638ms` | `9.760 GiB` |

### 2.5 Real takeaway on memory / inference

- **Current FP8 does not yet deliver the intended memory win** in this tree.
- **Current FP8 inference is also slower** in the local aligned harness.
- The likely contributors are persistent FP8 weight caches plus extra FP8/BF16 staging buffers, but this was **not** broken down with a dedicated inference-only NSYS pass in this session.
- Treat this as an **open problem**, not a solved one.

---

## 3. Correctness / precision status

### 3.1 What was rerun in this session

```bash
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

Result: **8 passed, 3 deselected**

### 3.2 What the current evidence supports

- No known correctness regression was introduced by the fused gated integration, the direct fused-dgated weight-cache path, or the FP8 wgrad grad-layout fix.
- Existing contract coverage still supports the usual target:
  - RelRMSE under 10%
  - correlation over 0.99
- Historical FP8 metrics from this repo were in the `~5.3% - 6.6%` RelRMSE range with `0.998` correlation for the aligned FP8 path.

### 3.3 What is still missing

- There is **no fresh production-shape full-chain FP8 wgrad gold report** after the latest grad-layout-copy fix.
- So the honest statement is:
  - **correctness looks healthy under current contract coverage**,
  - but **large-shape full-chain FP8 wgrad precision should be re-characterized before anyone enables it by default**.

---

## 4. High-value facts and where they came from

This is the section to preserve and reuse.

### 4.1 The authoritative baseline is `reports/sonic_official_bf16.sqlite`

- File: `reports/sonic_official_bf16.sqlite`
- Analysis tool: `tools/nsys_full_breakdown.py`
- Why it matters:
  - old comparisons against fork BF16 were misleading because fork BF16 had its own extra overheads
  - all future performance claims should keep official BF16 as the denominator

### 4.2 Fused blockscaled gated / dgated are viable if you bypass the unsafe autotuner path

- Files:
  - `sonicmoe/quack_utils/gemm_interface.py`
  - `sonicmoe/functional/__init__.py`
- What we learned:
  - the fused blockscaled path itself was not fundamentally broken
  - the unstable part was the wrapper autotune/config path
- Net result:
  - aligned fused FP8 act-grad path is now viable and competitive

### 4.3 The old fused backward slowdown was a wrapper copy, not a bad dgated kernel

- Files:
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
  - `sonicmoe/functional/__init__.py`
- Root cause:
  - wrapper-side `B.mT.contiguous()` on fused-dgated weights
- Fix:
  - `precompute_weight_fp8_for_direct_fused_dgated()` + low-level direct `gemm_dgated` call
- Why it matters:
  - this changed the story from “fused backward is broken” to “the wrapper dataflow was broken”

### 4.4 The FP8 wgrad giant autograd copy is gone, but the kernel path is still not good enough

- Files:
  - `sonicmoe/functional/__init__.py`
  - `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
- What was fixed:
  - backward grad now writes into base-layout buffers and returns parameter-compatible views
- Evidence:
  - local dense total dropped from `16.225ms` to `6.810ms`
  - the profiler no longer showed the `[128, 2048, 4096]` `aten::copy_`
- Why this matters:
  - the remaining FP8 wgrad problem is now the kernel/dataflow itself, not autograd bookkeeping

### 4.5 The current `blockscaled_fp8_wgrad_varlen_k()` path is still the blocker

- Observed isolated timings after the copy fix:
  - `dw1 (dz^T x)` ~ `1.44ms`
  - `dw2` ~ `0.98ms`
- A simple config sweep did **not** reveal a robust easy win.
- `blockscaled_fp8_weight_grad_gemm_fast()` / grouped-fast was already shown to be much slower than varlen_k in this project and is **not** the obvious drop-in answer.

### 4.6 The most concrete next direction already exists in-tree

- Files:
  - `sonicmoe/functional/backward.py`
  - `sonicmoe/functional/moe_config.py`
  - `sonicmoe/functional/grouped_gemm.py`
- Key symbols:
  - `HopperWgmma_MoE_Up_proj_WeightGrad_Bwd`
  - `HopperWgmma_MoE_Down_proj_WeightGrad_Bwd`
  - `HopperWgmma_MoE_kernel(..., compute_weight_gradient=True)`
- Why this matters:
  - the official-style specialized weight-grad kernels are already present locally
  - current QuACK FP8 path bypasses them under `use_quack_gemm`
  - this is the strongest concrete direction for the next agent: adapt / extend a specialized weight-grad kernel path for FP8, instead of only patching generic varlen_k wrappers

---

## 5. Invalidated narratives (do not repeat them)

These statements are no longer good guidance.

### 5.1 “The problem is a 455us gap to official BF16”

That was true for the **older split FP8 path** (`2930us`) and is now stale for the current fused act-grad branch.

The current reality is:
- `2600.3us` for fused FP8 + BF16 wgrad
- `5650.4us` for fused FP8 + FP8 wgrad

### 5.2 “The next agent should optimize Triton SwiGLU first”

That is no longer the highest-ROI training frontier.

- Forward/act-grad fusion work reduced the non-wgrad training gap to about `125us`.
- Full-chain FP8 still loses because **wgrad** is bad.
- Forward SwiGLU/inference/memory still matter, but **training priority #1 is now FP8 wgrad redesign**.

### 5.3 “Current FP8 already wins on memory”

False for the current tree.

Current local aligned measurements show FP8 peak memory is **higher**, not lower.

### 5.4 “Current FP8 inference already wins”

False for the current tree.

Current local aligned inference measurement is **slower** than BF16.

---

## 6. Lessons from this session

1. **Use official BF16 as the baseline, always.**
2. **Trust NSYS NVTX GPU projection with sync barriers over local event totals.**
3. **One giant hidden copy can dominate the whole story.** Profile before theorizing.
4. **Do not overfit to stale bottlenecks.** After the fused path and copy fixes, the frontier moved from SwiGLU to wgrad.
5. **Do not claim a memory or inference win without measurements.** Current local numbers still lose.
6. **E2E matters more than isolated per-op wins.** The current FP8 wgrad path is functionally correct and better than before, but still unacceptable in full training.

---

## 7. Recommended frontier for the next agent

### 7.1 Stable starting point

Start from the current near-baseline training path:

```bash
USE_QUACK_GEMM=1 \
SONIC_MOE_FP8_MODE=perf \
SONIC_MOE_FP8_ASSUME_ALIGNED=1 \
SONIC_MOE_FP8_FUSED_GATED=1 \
SONIC_MOE_FP8_WGRAD=0
```

### 7.2 What to do next

1. **Redesign FP8 wgrad**, likely by adapting the specialized grouped weight-grad kernels already present in:
   - `sonicmoe/functional/backward.py`
   - `sonicmoe/functional/moe_config.py`
   - `sonicmoe/functional/grouped_gemm.py`
2. If you continue on `blockscaled_fp8_wgrad_varlen_k()`, only pursue changes that materially alter:
   - physical operand layout
   - quantization reuse / elimination
   - the kernel family itself
3. Do **not** spend the next cycle on small config nudges or tiny Triton launch tweaks unless a new profile proves they matter.
4. Re-run the contract tests and the authoritative NSYS comparison before claiming any win.

### 7.3 Read order for the next agent

1. `reports/fp8_upgrade/HANDOFF.md`
2. `reports/fp8_upgrade/engineering_log.md`
3. `agent.md`
4. `sonicmoe/functional/__init__.py`
5. `sonicmoe/quack_utils/blockscaled_fp8_gemm.py`
6. `sonicmoe/functional/backward.py`
7. `sonicmoe/functional/moe_config.py`
8. `sonicmoe/functional/grouped_gemm.py`

---

## 8. Reproduction commands

### 8.1 Contract tests

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1 \
  python -m pytest tests/fp8_large_project_contract_test.py -v -k "not large_shape"
```

### 8.2 Local aligned perf + memory snapshot

```bash
CUDA_VISIBLE_DEVICES=0 python tools/measure_aligned_perf_memory.py
```

### 8.3 Local aligned BF16 vs FP8 benchmark

```bash
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  SONIC_MOE_FP8_ASSUME_ALIGNED=1 SONIC_MOE_FP8_FUSED_GATED=1 SONIC_MOE_FP8_WGRAD=1 \
  python tools/bench_aligned_e2e.py
```

### 8.4 NSYS authoritative comparison

Use the commands from §2.3.

---

## 9. Bottom line

The fused aligned FP8 act-grad branch is no longer the crisis. The crisis is narrower and cleaner:

> **Full-chain FP8 is now blocked by FP8 weight-grad.**
>
> The next serious win will likely come from a specialized FP8 weight-grad kernel path, not from repeating the old “trim Triton SwiGLU first” loop.
