"""FP8 Frontier Strict Test — no implicit fallback, no skip, fail-loud.

Run:
    source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
    cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
      python -m pytest tests/fp8_frontier_strict_test.py -v --tb=short
"""

import collections
import os
import unittest

import torch

from sonicmoe import KernelBackendMoE, MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import sonicmoe.functional as F_mod
from sonicmoe.functional.utils import is_fp8_active, is_using_quack_gemm

_SEED = 42

# ── Ernie-shape reference (matches profiling baseline) ──────────────────
_SHAPES = [
    # (T, H, I, E, K) — label
    (8192, 3072, 1536, 128, 8, "ernie-3072"),
    (8192, 768, 256, 128, 8, "small-768"),
    (8192, 4096, 2048, 64, 4, "wide-4096"),
]

# Tolerances (aligned with fp8_large_project_contract_test)
_FP8_ATOL = 5e-2
_FP8_RTOL = 5e-2


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
        msg = self.report(label)
        test.assertTrue(
            self.fp8_active_inside_fwd,
            f"FP8 NOT ACTIVE inside forward!\n{msg}",
        )
        test.assertTrue(
            self.quack_gemm_inside_fwd,
            f"QuACK GEMM NOT ACTIVE inside forward!\n{msg}",
        )
        test.assertTrue(
            self.cfg_snapshot["enabled"],
            f"FP8Config.enabled is False — frontier not entered!\n{msg}",
        )


def _hook_forward_probe(moe: MoE, probe: _FrontierProbe):
    """Register a forward hook that captures FP8 state mid-forward."""
    def _hook(module, args, output):
        probe.capture()
    return moe.register_forward_hook(_hook)


class FP8FrontierStrictTest(unittest.TestCase):
    """Every test here MUST enter the FP8 frontier. No skip, no fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        _require_blackwell()
        _require_quack_gemm()

    def _make_moe(self, H: int, I: int, E: int) -> MoE:
        with torch.device("cuda"):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=8,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=torch.bfloat16)
        return moe

    def _run_fwd_bwd(
        self, T: int, H: int, I: int, E: int, K: int, label: str,
    ) -> None:
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)

        moe = self._make_moe(H, I, E)
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
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        moe_ref = self._make_moe(H, I, E)
        # Copy weights from original moe
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
        print(f"  Forward: max_abs_diff={max_abs_diff:.6f}, RelRMSE={rel_rmse:.6f}")
        self.assertLess(
            rel_rmse, 0.10,
            f"RelRMSE {rel_rmse:.4f} > 10% for {label} — precision regression!",
        )

        # Backward tolerance
        dy_ref = dy.detach().clone()
        with enable_quack_gemm(True):
            grads_ref = torch.autograd.grad(
                y_ref, [x_ref] + list(moe_ref.parameters()), grad_outputs=dy_ref
            )
        for i, (gf, gr) in enumerate(zip(grads, grads_ref)):
            bwd_rel = (
                (gf.float() - gr.float()).pow(2).mean().sqrt()
                / gr.float().abs().mean().clamp(min=1e-8)
            ).item()
            print(f"  Backward grad[{i}]: RelRMSE={bwd_rel:.6f}")
            self.assertLess(
                bwd_rel, 0.10,
                f"Backward grad[{i}] RelRMSE {bwd_rel:.4f} > 10% for {label}",
            )

        torch.cuda.empty_cache()

    # ── Parameterized test cases ──────────────────────────────────────────

    def test_ernie_3072(self) -> None:
        """Ernie shape T=8192, H=3072, I=1536, E=128, K=8"""
        self._run_fwd_bwd(8192, 3072, 1536, 128, 8, "ernie-3072")

    def test_small_768(self) -> None:
        """Small shape T=8192, H=768, I=256, E=128, K=8"""
        self._run_fwd_bwd(8192, 768, 256, 128, 8, "small-768")

    def test_wide_4096(self) -> None:
        """Wide shape T=8192, H=4096, I=2048, E=64, K=4"""
        self._run_fwd_bwd(8192, 4096, 2048, 64, 4, "wide-4096")

    def test_alignment_observable(self) -> None:
        """Verify alignment-assumed flag is set after FP8 forward.

        With E=128, K=8, T=8192 → each expert gets ~64 tokens on average,
        which is already 128-aligned for uniform routing. After the
        _ALIGNMENT_STREAK_THRESHOLD (3) consecutive calls, the flag latches.
        """
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        moe = self._make_moe(768, 256, 128)

        # Force alignment assumed via env (test observability, not convergence)
        import sonicmoe.functional as fm
        prev = fm._ALIGNMENT_ASSUMED
        fm._ALIGNMENT_ASSUMED = False
        fm._ALIGNMENT_STREAK = 0

        try:
            for i in range(4):
                x = 0.02 * torch.randn(8192, 768, device="cuda", dtype=torch.bfloat16)
                with enable_quack_gemm(True):
                    moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)
            self.assertTrue(
                fm._ALIGNMENT_ASSUMED,
                "After 4 FP8 forward passes, _ALIGNMENT_ASSUMED should be True "
                "(all expert segments are 128-aligned with T=8192, E=128, K=8).",
            )
            print(f"\n  _ALIGNMENT_ASSUMED = {fm._ALIGNMENT_ASSUMED} after 4 iters ✓")
        finally:
            fm._ALIGNMENT_ASSUMED = prev
            fm._ALIGNMENT_STREAK = 0

    def test_prequant_cache_hits(self) -> None:
        """Verify prequant cache is populated and hit during fwd+bwd."""
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        moe = self._make_moe(768, 256, 128)

        # Force alignment for deterministic prequant path
        import sonicmoe.functional as fm
        prev_assumed = fm._ALIGNMENT_ASSUMED
        fm._ALIGNMENT_ASSUMED = True
        # Reset hit counters
        fm._PREQUANT_HIT_COUNT.clear()

        try:
            x = (0.02 * torch.randn(8192, 768, device="cuda", dtype=torch.bfloat16)
                 ).detach().requires_grad_()
            with enable_quack_gemm(True):
                y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)
                dy = 0.02 * torch.randn_like(y)
                y.backward(dy)

            hits = dict(fm._PREQUANT_HIT_COUNT)
            print(f"\n  Prequant hit counts: {hits}")

            # Fwd prequant should be hit (y1 quantized inside _UpProjection,
            # consumed by _DownProjection).
            self.assertGreater(
                hits.get("fwd", 0), 0,
                "Forward prequant cache was NOT hit — fused gated path may not be active.\n"
                f"Hits: {hits}",
            )
        finally:
            fm._ALIGNMENT_ASSUMED = prev_assumed
            fm._PREQUANT_HIT_COUNT.clear()


if __name__ == "__main__":
    unittest.main()
