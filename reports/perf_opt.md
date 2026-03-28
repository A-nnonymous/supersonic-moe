# Blockscaled FP8 Varlen GEMM — Performance Optimization Log

## Target
- **Kernel**: `_blockscaled_fp8_varlen_gemm_kernel` (Triton, tl.dot_scaled, SM100 Blackwell B200)
- **Workload**: E=128 experts, 512 tokens/expert, N=4096 (output), K=1024 (contraction), 1×32 UE8M0 blockscale
- **Goal**: Beat BF16 CUTLASS varlen GEMM baseline significantly
- **Constraint**: Must use genuine 1×32 UE8M0 blockscale (no per-tensor shortcuts)

## Baseline Measurements

| Version | Time (ms) | TFLOPS | vs BF16 | Notes |
|---------|-----------|--------|---------|-------|
| BF16 CUTLASS | 1.054 | 522 | 1.00x | `quack.gemm_interface.gemm` varlen |
| V1 original (E,N,K) | 3.182 | 173 | 0.33x | tl.dot_scaled, 128×128×128, 106 regs |
| V2 maxnreg=64 | 1.995 | 276 | 0.53x | Same tile, forced register cap |
| V3 (E,K,N) layout | 2.161 | 254 | 0.49x | B transposed for coalesced N loads |

## Iteration 0: Diagnosis (ncu)

### ncu profile of V1 (full varlen E=128):
```
Grid: 16,384 blocks, Block: 256 threads
Registers/thread: 106
Theoretical occupancy: 25% (2 blocks/SM)
Achieved occupancy: 24.18%
Compute throughput: 21.83%
DRAM throughput: 5.38%
L1 throughput: 54.38%
Shared memory: 33.28 KB dynamic
```

**Key findings:**
1. Register-bound: 106 regs → only 2 blocks/SM → 25% occupancy
2. Tensor core instructions ARE generated (`tcgen05.mma...mxf8f6f4.block_scale`)
3. Only 4,096 tensor ops vs 142,848 FMA ops — tensor cores starved
4. Low compute throughput (21.8%) despite decent grid (55 waves)
5. L1 at 54% — data movement overhead from scattered B weight loads

### Quantization diagnosis:
- `quantize_activation_raw` uses 12-14 separate CUDA kernels → 1.83ms
- Existing fused Triton kernel `quantize_activation_blockscaled_fast` → 0.37ms (5x faster)
- Output is bitwise identical

---

## Iteration 1: [NEXT — register pressure + occupancy attack]
