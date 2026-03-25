"""Test blockscaled_fp8_gemm_varlen against bf16 gold reference.

The CUTLASS blockscaled kernel currently requires 3-D tensor shapes for
scale-factor layout computation (tile_atom_to_shape_SF uses hard-coded
order (2,1,3)), but the varlen_m path provides a 2-D activation tensor
(total_M, K).  Until the upstream DSL adds rank-2 support, the function
raises NotImplementedError with a detailed diagnostic.

These tests verify that:
1. The function is importable and validates inputs correctly.
2. The NotImplementedError is raised with the expected diagnostic when
   the kernel compilation is attempted.
3. If the kernel ever gains support, the correctness tests will start
   passing automatically.
"""
import os
os.environ["USE_QUACK_GEMM"] = "1"

import torch
import pytest
from sonicmoe.quack_utils.blockscaled_fp8_gemm import blockscaled_fp8_gemm_varlen
from sonicmoe.functional.fp8_protocol import FP8Protocol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBlockscaledFP8VarlenGEMM:
    def _make_test_data(self, total_tokens=512, H=256, I=128, E=4, device="cuda"):
        """Create test data with expert boundaries."""
        tokens_per_expert = total_tokens // E
        cu_seqlens_m = torch.zeros(E + 1, dtype=torch.int32, device=device)
        for i in range(E):
            cu_seqlens_m[i + 1] = cu_seqlens_m[i] + tokens_per_expert

        a = torch.randn(total_tokens, I, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device=device) * 0.1
        return a, w2, cu_seqlens_m

    def _bf16_gold(self, a, w2, cu_seqlens_m):
        """Compute bf16 gold reference."""
        E = w2.size(2)
        total_M = a.size(0)
        H = w2.size(0)
        out = torch.zeros(total_M, H, dtype=torch.bfloat16, device=a.device)
        for e in range(E):
            start = cu_seqlens_m[e].item()
            end = cu_seqlens_m[e + 1].item()
            if end > start:
                out[start:end] = a[start:end] @ w2[:, :, e].T
        return out

    def test_raises_not_implemented(self):
        """Verify the function raises NotImplementedError with kernel diagnostic."""
        a, w2, cu_seqlens_m = self._make_test_data()
        protocol = FP8Protocol()
        with pytest.raises(NotImplementedError, match="tile_atom_to_shape_SF"):
            blockscaled_fp8_gemm_varlen(a, w2, cu_seqlens_m, protocol=protocol)

    def test_input_validation_2d_activation(self):
        """Verify validation rejects 3D activation."""
        a_3d = torch.randn(4, 128, 128, dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn(256, 128, 4, dtype=torch.bfloat16, device="cuda")
        cu = torch.zeros(5, dtype=torch.int32, device="cuda")
        protocol = FP8Protocol()
        with pytest.raises(ValueError, match="2D"):
            blockscaled_fp8_gemm_varlen(a_3d, w2, cu, protocol=protocol)

    def test_input_validation_weight_shape(self):
        """Verify validation rejects 2D weights."""
        a = torch.randn(512, 128, dtype=torch.bfloat16, device="cuda")
        w2_2d = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
        cu = torch.zeros(5, dtype=torch.int32, device="cuda")
        protocol = FP8Protocol()
        with pytest.raises(ValueError, match="3D"):
            blockscaled_fp8_gemm_varlen(a, w2_2d, cu, protocol=protocol)

    def test_basic_correctness(self):
        """Correctness test — will pass once kernel adds varlen+blockscaled support."""
        a, w2, cu_seqlens_m = self._make_test_data()
        protocol = FP8Protocol()
        try:
            result = blockscaled_fp8_gemm_varlen(a, w2, cu_seqlens_m, protocol=protocol)
        except NotImplementedError:
            pytest.skip("blockscaled+varlen not yet supported by CUTLASS kernel")
        gold = self._bf16_gold(a, w2, cu_seqlens_m)
        rmse = ((result.float() - gold.float()) ** 2).mean().sqrt().item()
        assert rmse < 0.05, f"RMSE too high: {rmse}"
        print(f"PASS: RMSE = {rmse:.6f}")

    def test_uneven_experts(self):
        """Uneven expert test — will pass once kernel adds support."""
        E = 4
        tokens = [100, 50, 200, 162]
        total = sum(tokens)
        H, I = 256, 128
        device = "cuda"

        cu_seqlens_m = torch.zeros(E + 1, dtype=torch.int32, device=device)
        for i, t in enumerate(tokens):
            cu_seqlens_m[i + 1] = cu_seqlens_m[i] + t

        a = torch.randn(total, I, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device=device) * 0.1

        protocol = FP8Protocol()
        try:
            result = blockscaled_fp8_gemm_varlen(a, w2, cu_seqlens_m, protocol=protocol)
        except NotImplementedError:
            pytest.skip("blockscaled+varlen not yet supported by CUTLASS kernel")
        gold = self._bf16_gold(a, w2, cu_seqlens_m)
        rmse = ((result.float() - gold.float()) ** 2).mean().sqrt().item()
        assert rmse < 0.05, f"RMSE too high: {rmse}"
        print(f"PASS (uneven): RMSE = {rmse:.6f}")

    def test_production_shape(self):
        """Production shape test — will pass once kernel adds support."""
        a, w2, cu_seqlens_m = self._make_test_data(
            total_tokens=1024, H=4096, I=1024, E=128
        )
        protocol = FP8Protocol()
        try:
            result = blockscaled_fp8_gemm_varlen(a, w2, cu_seqlens_m, protocol=protocol)
        except NotImplementedError:
            pytest.skip("blockscaled+varlen not yet supported by CUTLASS kernel")
        gold = self._bf16_gold(a, w2, cu_seqlens_m)
        rmse = ((result.float() - gold.float()) ** 2).mean().sqrt().item()
        assert rmse < 0.05, f"RMSE too high: {rmse}"
        print(f"PASS (production): RMSE = {rmse:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
