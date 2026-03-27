# FP8 Engineering Log

> **Primary reference: [HANDOFF.md](HANDOFF.md)**

---

## Key Finding: FP8 blockscaled MoE training slower than BF16

BF16 E2E 7.76ms vs FP8 best 10.08ms (+30%). Root cause: activation quantize overhead (bf16→fp8+ISA pack) × 4-6 GEMM calls per step. BF16's fused `gemm_dgated` has zero quantize overhead.

FP8 推理快 43% (2.2ms vs 3.9ms) — quantize 只做 1-2 次。

---

## Experiment History

1. **Per-tensor FP8**: RelRMSE ~100%. Unacceptable.
2. **Grouped/static-capacity blockscaled**: Violates varlen contract.
3. **blockscaled_fp8_weight_grad_gemm**: Pack/transpose > GEMM at E=128.
4. **Fused dgated + blockscaled**: Works but 11ms backward (vs BF16 5.3ms) — CUTLASS blockscaled mainloop overhead.
5. **QuACK alignment fix**: `colvec_scale=s.float()` fixes bf16 varlen alignment bug.

---

## Lessons

1. BF16 fused `gemm_dgated` is extremely efficient — single kernel, zero overhead.
2. FP8 advantage only materializes when quantize cost is amortized (inference, or pre-quantized dispatch).
3. `colvec_scale=s.float()` cleanly fixes the QuACK varlen alignment bug (fp32 always 32-bit aligned).
4. `gemm_gated` epilogue is accumulator-dtype agnostic — blockscaled plumbing is pure parameter threading.
