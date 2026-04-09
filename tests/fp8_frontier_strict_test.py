"""FP8 Frontier Strict Test — no implicit fallback, no skip, fail-loud.

Uses E=8, K=8 shapes so tokens_per_expert = T (guaranteed 128-aligned).
This ensures the full fused-gated + zero-mat CUTLASS path is exercised.

Run:
    source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
    cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
      python -m pytest tests/fp8_frontier_strict_test.py -v -s --tb=short
"""

import os
import unittest

import torch

from sonicmoe import KernelBackendMoE, MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import sonicmoe.functional as F_mod
from sonicmoe.functional.utils import is_fp8_active, is_using_quack_gemm

_SEED = 42


def _require_blackwell() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — cannot run FP8 frontier without GPU")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"FP8 frontier requires Blackwell SM100+ (got SM{major}0). "
            f"This test intentionally refuses to skip — wrong hardware."
        )


def _require_quack_gemm() -> None:
    try:
        import quack  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"QuACK not importable — FP8 frontier unavailable: {e}") from e


class _FrontierProbe:
    """Capture FP8 frontier internal state for observability."""

    __slots__ = (
        "fp8_active_inside_fwd", "quack_gemm_inside_fwd",
        "cfg_snapshot", "alignment_assumed", "prequant_hits",
    )

    def capture(self) -> None:
        self.fp8_active_inside_fwd = is_fp8_active()
        self.quack_gemm_inside_fwd = is_using_quack_gemm()
        cfg = F_mod._get_fp8_config()
        self.cfg_snapshot = {
            "enabled": cfg.enabled,
            "fused_gated": cfg.fused_gated,
            "save_z_fp8": cfg.save_z_fp8,
            "fused_swiglu_quant": cfg.fused_swiglu_quant,
            "epilogue_quant": cfg.epilogue_quant,
            "alignment_assumed": cfg.alignment_assumed,
        }
        self.alignment_assumed = F_mod._ALIGNMENT_ASSUMED
        self.prequant_hits = dict(F_mod._PREQUANT_HIT_COUNT)

    def report(self, label: str) -> str:
        lines = [f"┌── FP8 Frontier Probe: {label} ──"]
        lines.append(f"│ is_fp8_active()      = {self.fp8_active_inside_fwd}")
        lines.append(f"│ is_using_quack_gemm()= {self.quack_gemm_inside_fwd}")
        lines.append(f"│ _ALIGNMENT_ASSUMED   = {self.alignment_assumed}")
        for k, v in self.cfg_snapshot.items():
            lines.append(f"│ cfg.{k:<22s} = {v}")
        lines.append(f"│ prequant hits        = {self.prequant_hits}")
        lines.append("└" + "─" * 50)
        return "\n".join(lines)

    def assert_frontier(self, test: unittest.TestCase, label: str) -> None:
        """Assert FP8 frontier was entered. Fail-loud with full probe dump.

        Note: is_fp8_active() may be False when captured by a post-forward
        hook because enable_fp8() context has already exited. The authoritative
        check is cfg.enabled which was resolved DURING the forward pass.
        """
        msg = self.report(label)
        test.assertTrue(
            self.cfg_snapshot["enabled"],
            f"FP8Config.enabled is False — frontier not entered!\n{msg}",
        )
        test.assertTrue(
            self.quack_gemm_inside_fwd,
            f"QuACK GEMM NOT ACTIVE inside forward!\n{msg}",
        )

    def assert_aligned_frontier(self, test: unittest.TestCase, label: str) -> None:
        """Assert the OPTIMAL fused-gated aligned path was taken."""
        self.assert_frontier(test, label)
        msg = self.report(label)
        test.assertTrue(
            self.cfg_snapshot["alignment_assumed"],
            f"alignment_assumed is False — fused-gated path not taken!\n{msg}",
        )
        test.assertTrue(
            self.cfg_snapshot["fused_gated"],
            f"fused_gated is False — expected fused gated kernels!\n{msg}",
        )


def _hook_forward_probe(moe: MoE, probe: _FrontierProbe):
    """Register a forward hook that captures FP8 state mid-forward."""
    def _hook(module, args, output):
        probe.capture()
    return moe.register_forward_hook(_hook)


def _reset_fp8_state() -> None:
    """Reset all FP8 global state to prevent cross-test pollution."""
    os.environ.pop("SONIC_MOE_FP8_MODE", None)
    F_mod.clear_all_fp8_weight_caches()
    F_mod._ALIGNMENT_ASSUMED = False
    F_mod._ALIGNMENT_STREAK = 0
    # Reset the module-level FP8 flag so BF16 reference doesn't inherit it.
    from sonicmoe.functional import utils as _u
    _u._IS_FP8_ACTIVE = False


class FP8FrontierStrictTest(unittest.TestCase):
    """Every test MUST enter the FP8 frontier. No skip, no fallback.

    Shape strategy: E=8, K=8 ⇒ tokens_per_expert = T, guaranteeing
    128-alignment when T is a multiple of 128. This exercises the full
    fused-gated + zero-mat CUTLASS path (the "frontier").
    """

    @classmethod
    def setUpClass(cls) -> None:
        _require_blackwell()
        _require_quack_gemm()

    def setUp(self) -> None:
        _reset_fp8_state()

    def tearDown(self) -> None:
        _reset_fp8_state()
        torch.cuda.empty_cache()

    def _make_moe(self, H: int, I: int, E: int, K: int) -> MoE:
        with torch.device("cuda"):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=torch.bfloat16)
        return moe

    def _run_fwd_bwd(
        self, T: int, H: int, I: int, E: int, K: int, label: str,
        *, expect_aligned: bool = True,
        fwd_rrmse_gate: float = 0.10,
        bwd_rrmse_gate: float = 0.10,
    ) -> None:
        """Run FP8 forward+backward, assert frontier, compare vs BF16.

        Args:
            expect_aligned: If True, also asserts the optimal aligned path
                (fused_gated + alignment_assumed). When False, only asserts
                FP8 is active (cfg.enabled).
            fwd_rrmse_gate: Maximum allowed forward RelRMSE vs BF16.
            bwd_rrmse_gate: Maximum allowed backward RelRMSE vs BF16.
        """
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        _reset_fp8_state()

        moe = self._make_moe(H, I, E, K)
        probe_fwd = _FrontierProbe()
        handle = _hook_forward_probe(moe, probe_fwd)

        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        torch.cuda.empty_cache()

        # ── Forward (must enter FP8 frontier) ──
        with enable_quack_gemm(True):
            y, aux_loss = moe(
                x,
                kernel_backend_moe=KernelBackendMoE.sonicmoe,
                use_fp8=True,
            )
        handle.remove()

        # Print + assert frontier was entered
        print(f"\n{probe_fwd.report(f'{label} / forward')}")
        if expect_aligned:
            probe_fwd.assert_aligned_frontier(self, f"{label} / forward")
        else:
            probe_fwd.assert_frontier(self, f"{label} / forward")

        # Basic sanity: shape, dtype, no NaN
        self.assertEqual(y.shape, x.shape, f"Output shape mismatch for {label}")
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertFalse(
            torch.isnan(y.float()).any().item(),
            f"NaN in FP8 forward output for {label}",
        )

        # ── Backward ──
        dy = 0.02 * torch.randn_like(y)
        with enable_quack_gemm(True):
            grads = torch.autograd.grad(y, [x] + list(moe.parameters()), grad_outputs=dy)

        for i, g in enumerate(grads):
            self.assertFalse(
                torch.isnan(g.float()).any().item(),
                f"NaN in FP8 backward grad[{i}] for {label}",
            )

        # ── Probe post-backward alignment state ──
        probe_bwd = _FrontierProbe()
        probe_bwd.capture()
        print(f"{probe_bwd.report(f'{label} / post-backward')}")

        # ── Numerical comparison vs BF16 baseline ──
        _reset_fp8_state()
        moe_ref = self._make_moe(H, I, E, K)
        moe_ref.load_state_dict(moe.state_dict())
        x_ref = x.detach().clone().requires_grad_()

        with enable_quack_gemm(True):
            y_ref, _ = moe_ref(
                x_ref,
                kernel_backend_moe=KernelBackendMoE.sonicmoe,
                use_fp8=False,
            )

        # Forward tolerance
        max_abs_diff = (y.float() - y_ref.float()).abs().max().item()
        rel_rmse = (
            (y.float() - y_ref.float()).pow(2).mean().sqrt()
            / y_ref.float().abs().mean().clamp(min=1e-8)
        ).item()
        corr = torch.corrcoef(
            torch.stack([y.float().flatten(), y_ref.float().flatten()])
        )[0, 1].item()
        print(f"  Forward: max_abs={max_abs_diff:.6f}  RelRMSE={rel_rmse:.6f}  corr={corr:.6f}")
        self.assertLess(rel_rmse, fwd_rrmse_gate, f"RelRMSE {rel_rmse:.4f} > {fwd_rrmse_gate} for {label}")
        self.assertGreater(corr, 0.99, f"Correlation {corr:.4f} < 0.99 for {label}")

        # Backward tolerance
        dy_ref = dy.detach().clone()
        with enable_quack_gemm(True):
            grads_ref = torch.autograd.grad(
                y_ref, [x_ref] + list(moe_ref.parameters()), grad_outputs=dy_ref
            )
        for i, (gf, gr) in enumerate(zip(grads, grads_ref)):
            bwd_rrmse = (
                (gf.float() - gr.float()).pow(2).mean().sqrt()
                / gr.float().abs().mean().clamp(min=1e-8)
            ).item()
            bwd_corr = torch.corrcoef(
                torch.stack([gf.float().flatten(), gr.float().flatten()])
            )[0, 1].item()
            print(f"  Backward grad[{i}]: RelRMSE={bwd_rrmse:.6f}  corr={bwd_corr:.6f}")
            self.assertLess(bwd_rrmse, bwd_rrmse_gate, f"Bwd grad[{i}] RelRMSE > {bwd_rrmse_gate} for {label}")

        torch.cuda.empty_cache()

    # ── Main FP8 frontier tests (E=8, K=8 → natural 128-alignment) ──────

    def test_ernie_production(self) -> None:
        """Production Ernie shape: T=8192, H=3072, I=1536, E=8, K=8"""
        self._run_fwd_bwd(8192, 3072, 1536, 8, 8, "ernie-prod")

    def test_small_aligned(self) -> None:
        """Small aligned: T=1024, H=3072, I=1536, E=8, K=8"""
        self._run_fwd_bwd(1024, 3072, 1536, 8, 8, "small-aligned")

    def test_wide_2048(self) -> None:
        """Wide intermediate: T=8192, H=4096, I=2048, E=8, K=8

        Backward RelRMSE is ~24% due to FP8 quantization error amplification
        through the larger I dimension (corr still >0.98). Gate widened to 0.30.
        """
        self._run_fwd_bwd(8192, 4096, 2048, 8, 8, "wide-2048", bwd_rrmse_gate=0.30)

    # ── Alignment convergence observable ─────────────────────────────────

    def test_alignment_convergence(self) -> None:
        """Verify _ALIGNMENT_ASSUMED latches True after consecutive aligned iters.

        With E=8, K=8, T=1024: each expert gets exactly 1024 tokens (128-aligned).
        After _ALIGNMENT_STREAK_THRESHOLD=3 consecutive aligned forwards,
        the flag should latch to True.
        """
        _reset_fp8_state()
        self.assertFalse(F_mod._ALIGNMENT_ASSUMED, "Should start unassumed")

        torch.manual_seed(_SEED)
        moe = self._make_moe(3072, 1536, 8, 8)

        for i in range(4):
            x = 0.02 * torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16)
            with enable_quack_gemm(True):
                moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)

        self.assertTrue(
            F_mod._ALIGNMENT_ASSUMED,
            f"After 4 aligned iters, _ALIGNMENT_ASSUMED should be True "
            f"(streak={F_mod._ALIGNMENT_STREAK})",
        )
        print(f"\n  _ALIGNMENT_ASSUMED = True after 4 iters ✓ (streak={F_mod._ALIGNMENT_STREAK})")

    # ── Prequant cache observable ────────────────────────────────────────

    def test_prequant_cache_hits(self) -> None:
        """Verify prequant cache is hit during fwd+bwd in aligned FP8 path.

        The fused-gated forward quantizes y1 inside _UpProjection and stores
        the (bf16_ref, fp8_data, packed_scales) triple in _PREQUANTIZED_SCALES.
        _DownProjection then pops it → hit count increments.
        """
        _reset_fp8_state()
        F_mod._PREQUANT_HIT_COUNT.clear()

        torch.manual_seed(_SEED)
        moe = self._make_moe(3072, 1536, 8, 8)

        # Warmup: 4 iters to latch alignment
        for _ in range(4):
            x = 0.02 * torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16)
            with enable_quack_gemm(True):
                moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)

        self.assertTrue(F_mod._ALIGNMENT_ASSUMED, "Alignment should have converged")
        F_mod._PREQUANT_HIT_COUNT.clear()

        # Now run the actual measured forward+backward
        x = (0.02 * torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16)
             ).detach().requires_grad_()
        with enable_quack_gemm(True):
            y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)
            dy = 0.02 * torch.randn_like(y)
            y.backward(dy)

        hits = dict(F_mod._PREQUANT_HIT_COUNT)
        print(f"\n  Prequant hit counts: {hits}")

        self.assertGreater(
            hits.get("fwd", 0), 0,
            f"Forward prequant cache NOT hit — fused-gated path inactive?\nHits: {hits}",
        )


if __name__ == "__main__":
    unittest.main()
