"""Precision and functional tests for the native FP8 params path.

Tests verify that native FP8 (x arrives as FP8, weights stored as FP8)
produces outputs and gradients within acceptable precision bounds relative
to the frontier FP8 path and BF16 gold standard.

Production shapes: T=8192, H=3072, I=1536, E=8, K=8
Small shapes:      T=256,  H=768,  I=256,  E=128, K=8

NOTE: Run separately from fp8_large_project_contract_test.py to avoid
global state contamination. Use: pytest tests/fp8_native_params_test.py -v
"""
import os
import unittest

import torch

# Must be set before importing sonicmoe
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

from sonicmoe import MoE, enable_fp8, enable_native_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType


_SEED = 42


def _require_blackwell():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise unittest.SkipTest("Blackwell-only (SM100+)")


def _make_moe(E=128, K=8, H=768, I=256):
    return MoE(E, K, H, I, ActivationType.SWIGLU, add_bias=False, std=0.02).cuda().bfloat16()


def _make_sample(T=256, H=768):
    x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
    dout = 0.02 * torch.randn_like(x)
    return x, dout


def _rrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative RMSE: RMSE(a-b) / RMS(b)."""
    a_f, b_f = a.float(), b.float()
    rmse = (a_f - b_f).pow(2).mean().sqrt()
    rms = b_f.pow(2).mean().sqrt()
    return (rmse / rms.clamp(min=1e-8)).item()


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.flatten().float(), b.flatten().float()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    return (a_c @ b_c / (a_c.norm() * b_c.norm()).clamp(min=1e-12)).item()


class TestNativeFP8ContextManager(unittest.TestCase):
    """Test enable_native_fp8 context manager behavior."""

    def test_context_manager_enables_fp8_and_quack(self):
        """enable_native_fp8() also enables fp8 and quack gemm."""
        _require_blackwell()
        from sonicmoe.functional.utils import is_fp8_active, is_using_quack_gemm, is_native_fp8_active
        self.assertFalse(is_native_fp8_active())
        with enable_native_fp8():
            self.assertTrue(is_native_fp8_active())
            self.assertTrue(is_fp8_active())
            self.assertTrue(is_using_quack_gemm())
        self.assertFalse(is_native_fp8_active())

    def test_context_manager_restores_state(self):
        """State is restored even if an exception occurs."""
        _require_blackwell()
        from sonicmoe.functional.utils import is_native_fp8_active
        self.assertFalse(is_native_fp8_active())
        try:
            with enable_native_fp8():
                self.assertTrue(is_native_fp8_active())
                raise ValueError("test")
        except ValueError:
            pass
        self.assertFalse(is_native_fp8_active())


class TestNativeFP8ForwardBackward(unittest.TestCase):
    """End-to-end native FP8 vs BF16 gold and frontier FP8 precision tests."""

    def _run_bf16_gold(self, moe, x, dout):
        x_c = x.detach().clone().requires_grad_()
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True):
            out, _ = moe(x_c)
        out.backward(dout)
        return out.detach(), x_c.grad.detach(), {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}

    def _run_frontier_fp8(self, moe, x, dout):
        x_c = x.detach().clone().requires_grad_()
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True), enable_fp8():
            out, _ = moe(x_c, use_fp8=True)
        out.backward(dout)
        return out.detach(), x_c.grad.detach(), {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}

    def _run_native_fp8(self, moe, x, dout):
        x_c = x.detach().clone().requires_grad_()
        moe.zero_grad(set_to_none=True)
        with enable_native_fp8():
            out, _ = moe(x_c, use_fp8=True)
        out.backward(dout)
        return out.detach(), x_c.grad.detach(), {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}

    def test_native_fp8_forward_no_nan(self):
        """Native FP8 forward produces no NaN/Inf."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        out, dx, _ = self._run_native_fp8(moe, x, dout)
        self.assertFalse(torch.isnan(out).any().item(), "NaN in output")
        self.assertFalse(torch.isinf(out).any().item(), "Inf in output")
        self.assertFalse(torch.isnan(dx).any().item(), "NaN in dx")

    def test_native_fp8_output_shape(self):
        """Output shape matches input shape."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        out, dx, _ = self._run_native_fp8(moe, x, dout)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_native_fp8_vs_bf16_forward_rrmse(self):
        """Native FP8 output RRMSE vs BF16 gold < 15%."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        gold_out, _, _ = self._run_bf16_gold(moe, x, dout)
        native_out, _, _ = self._run_native_fp8(moe, x, dout)
        rrmse = _rrmse(native_out, gold_out)
        print(f"  Native FP8 vs BF16 gold forward RRMSE: {rrmse:.4f}")
        self.assertLess(rrmse, 0.15, f"Forward RRMSE too high: {rrmse:.4f}")

    def test_native_fp8_vs_bf16_forward_correlation(self):
        """Native FP8 output correlation with BF16 gold > 0.95."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        gold_out, _, _ = self._run_bf16_gold(moe, x, dout)
        native_out, _, _ = self._run_native_fp8(moe, x, dout)
        corr = _correlation(native_out, gold_out)
        print(f"  Native FP8 vs BF16 gold forward correlation: {corr:.4f}")
        self.assertGreater(corr, 0.95, f"Forward correlation too low: {corr:.4f}")

    def test_native_fp8_vs_frontier_fp8_forward_rrmse(self):
        """Native FP8 vs frontier FP8: RRMSE < 10% (same quantization path)."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        frontier_out, _, _ = self._run_frontier_fp8(moe, x, dout)
        native_out, _, _ = self._run_native_fp8(moe, x, dout)
        rrmse = _rrmse(native_out, frontier_out)
        print(f"  Native FP8 vs Frontier FP8 forward RRMSE: {rrmse:.4f}")
        self.assertLess(rrmse, 0.10, f"Native vs Frontier RRMSE too high: {rrmse:.4f}")

    def test_native_fp8_backward_dx_rrmse(self):
        """dx RRMSE vs BF16 gold < 20%."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        _, gold_dx, _ = self._run_bf16_gold(moe, x, dout)
        _, native_dx, _ = self._run_native_fp8(moe, x, dout)
        rrmse = _rrmse(native_dx, gold_dx)
        print(f"  Native FP8 vs BF16 backward dx RRMSE: {rrmse:.4f}")
        self.assertLess(rrmse, 0.20, f"dx RRMSE too high: {rrmse:.4f}")

    def test_native_fp8_backward_dw_rrmse(self):
        """Weight gradient RRMSE vs BF16 gold < 20%."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        _, _, gold_grads = self._run_bf16_gold(moe, x, dout)
        _, _, native_grads = self._run_native_fp8(moe, x, dout)
        for name in gold_grads:
            if name in native_grads:
                rrmse = _rrmse(native_grads[name], gold_grads[name])
                print(f"  dw RRMSE ({name}): {rrmse:.4f}")
                self.assertLess(rrmse, 0.20, f"dw RRMSE too high for {name}: {rrmse:.4f}")

    def test_native_fp8_deterministic(self):
        """Two runs with same seed produce identical output."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe()
        x, dout = _make_sample()
        out1, _, _ = self._run_native_fp8(moe, x, dout)

        torch.manual_seed(_SEED)
        moe2 = _make_moe()
        x2, dout2 = _make_sample()
        out2, _, _ = self._run_native_fp8(moe2, x2, dout2)

        torch.testing.assert_close(out1, out2, rtol=0, atol=0)


class TestNativeFP8ProductionShape(unittest.TestCase):
    """Production shape (T=8192, H=3072, I=1536, E=8, K=8) tests."""

    def test_production_shape_forward_backward(self):
        """Native FP8 completes without error at production shape."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe(E=8, K=8, H=3072, I=1536)
        x = (0.02 * torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)
        moe.zero_grad(set_to_none=True)
        with enable_native_fp8():
            out, _ = moe(x, use_fp8=True)
        self.assertFalse(torch.isnan(out).any().item())
        out.backward(dout)
        self.assertFalse(torch.isnan(x.grad).any().item())

    def test_production_shape_rrmse(self):
        """Production shape RRMSE vs BF16 < 15%."""
        _require_blackwell()
        torch.manual_seed(_SEED)
        moe = _make_moe(E=8, K=8, H=3072, I=1536)
        x = (0.02 * torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        # BF16 gold
        x_g = x.detach().clone().requires_grad_()
        moe.zero_grad(set_to_none=True)
        with enable_quack_gemm(True):
            gold, _ = moe(x_g)
        gold_out = gold.detach()

        # Native FP8
        x_n = x.detach().clone().requires_grad_()
        moe.zero_grad(set_to_none=True)
        with enable_native_fp8():
            native, _ = moe(x_n, use_fp8=True)
        native_out = native.detach()

        rrmse = _rrmse(native_out, gold_out)
        print(f"  Production shape RRMSE: {rrmse:.4f}")
        self.assertLess(rrmse, 0.15)


if __name__ == "__main__":
    unittest.main()
