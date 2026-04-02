import os

import torch

import sonicmoe.functional as functional
from sonicmoe import KernelBackendMoE, MoE, enable_fp8, enable_quack_gemm, get_default_fp8_protocol, moe_general_routing_inputs
from sonicmoe.enums import ActivationType

from .fp8_operator_options import (
    FP8_WEIGHT_STORAGE,
    MIXED_DTYPE_DOWNPROJ_DW2,
    NATIVE_FP8_DOWNPROJ,
    NATIVE_FP8_UPPROJ,
    RANKFLEX_VARLEN_DOWNPROJ,
    is_operator_opt_enabled,
)
from .test_commons import TestCommons


_SEED = 42


class FP8LargeProjectContractTest(TestCommons):
    def _require_blackwell(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

    def _make_moe(self) -> MoE:
        return MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

    def _make_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)
        return x, dout

    def _make_general_routing_inputs(self, moe: MoE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = torch.nn.functional.linear(x, moe.router.weight)
        topk_scores = torch.empty(x.size(0), moe.top_k, dtype=torch.float32, device=x.device)
        topk_indices = torch.empty(x.size(0), moe.top_k, dtype=torch.int32, device=x.device)
        functional._softmax_topk_fwd(router_logits, topk_scores, topk_indices, moe.num_experts, moe.top_k)
        token_indices = torch.arange(x.size(0), device=x.device, dtype=torch.int32).repeat_interleave(moe.top_k)
        return router_logits, topk_scores.reshape(-1), token_indices, topk_indices.reshape(-1)

    def _run_sonicmoe_path(self, moe: MoE, x: torch.Tensor, *, protocol_enabled: bool) -> torch.Tensor:
        """Run MoE forward and return only the output hidden states."""
        protocol = get_default_fp8_protocol() if protocol_enabled else None
        with enable_quack_gemm(True):
            output, _aux_loss = moe(
                x,
                kernel_backend_moe=KernelBackendMoE.sonicmoe,
                fp8_protocol=protocol,
            )
        return output

    def test_native_fp8_upproj_bf16_gold_contract(self) -> None:
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x, _ = self._make_sample()

        gold_output = self._run_sonicmoe_path(moe, x, protocol_enabled=False)

        self.assertEqual(tuple(gold_output.shape), tuple(x.shape))
        self.assertEqual(gold_output.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_UPPROJ):
            candidate_output = self._run_sonicmoe_path(
                moe, x.detach().clone(), protocol_enabled=True
            )
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)

    def test_native_fp8_downproj_bf16_gold_contract(self) -> None:
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x, _ = self._make_sample()

        gold_output = self._run_sonicmoe_path(moe, x, protocol_enabled=False)

        self.assertEqual(tuple(gold_output.shape), tuple(x.shape))
        self.assertEqual(gold_output.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_DOWNPROJ):
            candidate_output = self._run_sonicmoe_path(
                moe, x.detach().clone(), protocol_enabled=True
            )
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)

    def test_fp8_weight_storage_bf16_gold_contract(self) -> None:
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x, _ = self._make_sample()

        gold_output = self._run_sonicmoe_path(moe, x, protocol_enabled=False)

        self.assertEqual(tuple(gold_output.shape), tuple(x.shape))
        self.assertEqual(gold_output.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(FP8_WEIGHT_STORAGE):
            candidate_output = self._run_sonicmoe_path(
                moe, x.detach().clone(), protocol_enabled=True
            )
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)

    def test_mixed_dtype_downproj_weight_grad_bf16_gold_contract(self) -> None:
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x, dout = self._make_sample()

        gold_output = self._run_sonicmoe_path(moe, x, protocol_enabled=False)
        gold_output.backward(dout)

        gold_dw2 = moe.c_proj.weight.grad.detach().clone()
        self.assertEqual(tuple(gold_dw2.shape), tuple(moe.c_proj.weight.shape))
        self.assertEqual(gold_dw2.dtype, moe.c_proj.weight.dtype)
        self.assertFalse(torch.isnan(gold_dw2.float()).any().item())

        if is_operator_opt_enabled(MIXED_DTYPE_DOWNPROJ_DW2):
            moe.zero_grad(set_to_none=True)
            x_candidate = x.detach().clone().requires_grad_()
            candidate_output = self._run_sonicmoe_path(moe, x_candidate, protocol_enabled=True)
            candidate_output.backward(dout)
            candidate_dw2 = moe.c_proj.weight.grad.detach().clone()
            torch.testing.assert_close(candidate_dw2.float(), gold_dw2.float(), rtol=5e-2, atol=5e-2)

    def test_rank_flexible_varlen_downproj_bf16_gold_contract(self) -> None:
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x, _ = self._make_sample()

        _, router_scores, token_indices, expert_indices = self._make_general_routing_inputs(moe, x)

        with enable_quack_gemm(True):
            gold_output, gold_expert_frequency = moe_general_routing_inputs(
                x,
                router_scores,
                token_indices,
                expert_indices,
                moe.c_fc.weight.permute(1, 2, 0),
                None,
                moe.c_proj.weight.permute(1, 2, 0),
                None,
                moe.num_experts,
                moe.stream_id,
                moe.activation_function,
                is_inference_mode_enabled=False,
            )

        self.assertEqual(tuple(gold_output.shape), tuple(x.shape))
        self.assertEqual(gold_output.dtype, torch.bfloat16)
        self.assertEqual(tuple(gold_expert_frequency.shape), (moe.num_experts,))
        self.assertEqual(gold_expert_frequency.dtype, torch.int32)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(RANKFLEX_VARLEN_DOWNPROJ):
            with enable_quack_gemm(True):
                candidate_output, candidate_expert_frequency = moe_general_routing_inputs(
                    x.detach().clone(),
                    router_scores,
                    token_indices,
                    expert_indices,
                    moe.c_fc.weight.permute(1, 2, 0),
                    None,
                    moe.c_proj.weight.permute(1, 2, 0),
                    None,
                    moe.num_experts,
                    moe.stream_id,
                    moe.activation_function,
                    is_inference_mode_enabled=False,
                )
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(candidate_expert_frequency, gold_expert_frequency, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Backward gradient contracts — required for Projects 1, 2, and 3
    # ------------------------------------------------------------------

    def _collect_all_grads(
        self, moe: MoE, x: torch.Tensor, dout: torch.Tensor, *, protocol_enabled: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fwd+bwd and return (dx, dw1, dw2) gradients."""
        moe.zero_grad(set_to_none=True)
        output = self._run_sonicmoe_path(moe, x, protocol_enabled=protocol_enabled)
        output.backward(dout)
        dx = x.grad.detach().clone()
        dw1 = moe.c_fc.weight.grad.detach().clone()
        dw2 = moe.c_proj.weight.grad.detach().clone()
        x.grad = None
        return dx, dw1, dw2

    def test_native_fp8_upproj_backward_grad_contract(self) -> None:
        """Project 1: when up-proj produces FP8, backward dx and dw1 must stay close to bf16 gold."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x_gold, dout = self._make_sample()

        gold_dx, gold_dw1, gold_dw2 = self._collect_all_grads(moe, x_gold, dout, protocol_enabled=False)

        self.assertFalse(torch.isnan(gold_dx.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw1.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw2.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_UPPROJ):
            x_cand = x_gold.detach().clone().requires_grad_()
            cand_dx, cand_dw1, cand_dw2 = self._collect_all_grads(moe, x_cand, dout, protocol_enabled=True)
            torch.testing.assert_close(cand_dx.float(), gold_dx.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw1.float(), gold_dw1.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw2.float(), gold_dw2.float(), rtol=5e-2, atol=5e-2)

    def test_native_fp8_downproj_backward_grad_contract(self) -> None:
        """Project 2: when down-proj consumes FP8, backward dx and dw2 must stay close to bf16 gold."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x_gold, dout = self._make_sample()

        gold_dx, gold_dw1, gold_dw2 = self._collect_all_grads(moe, x_gold, dout, protocol_enabled=False)

        self.assertFalse(torch.isnan(gold_dx.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw1.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw2.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_DOWNPROJ):
            x_cand = x_gold.detach().clone().requires_grad_()
            cand_dx, cand_dw1, cand_dw2 = self._collect_all_grads(moe, x_cand, dout, protocol_enabled=True)
            torch.testing.assert_close(cand_dx.float(), gold_dx.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw1.float(), gold_dw1.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw2.float(), gold_dw2.float(), rtol=5e-2, atol=5e-2)

    def test_mixed_dtype_backward_all_grads_contract(self) -> None:
        """Project 3: mixed-dtype backward must preserve dx AND dw1 in addition to dw2."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe()
        x_gold, dout = self._make_sample()

        gold_dx, gold_dw1, gold_dw2 = self._collect_all_grads(moe, x_gold, dout, protocol_enabled=False)

        self.assertFalse(torch.isnan(gold_dx.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw1.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw2.float()).any().item())

        if is_operator_opt_enabled(MIXED_DTYPE_DOWNPROJ_DW2):
            x_cand = x_gold.detach().clone().requires_grad_()
            cand_dx, cand_dw1, cand_dw2 = self._collect_all_grads(moe, x_cand, dout, protocol_enabled=True)
            torch.testing.assert_close(cand_dx.float(), gold_dx.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw1.float(), gold_dw1.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dw2.float(), gold_dw2.float(), rtol=5e-2, atol=5e-2)

    # ------------------------------------------------------------------
    # Larger shape — HANDOFF requires real-shape validation, not just toy
    # ------------------------------------------------------------------

    _LARGE_T = 1024
    _LARGE_H = 4096
    _LARGE_I = 1024
    _LARGE_E = 128
    _LARGE_K = 8

    def _make_moe_large(self) -> MoE:
        return MoE(
            num_experts=self._LARGE_E,
            num_experts_per_tok=self._LARGE_K,
            hidden_size=self._LARGE_H,
            intermediate_size=self._LARGE_I,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

    def _make_sample_large(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = (0.02 * torch.randn(self._LARGE_T, self._LARGE_H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)
        return x, dout

    def test_native_fp8_upproj_large_shape_contract(self) -> None:
        """Project 1 at production-scale shape — forward + backward correctness."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe_large()
        x_gold, dout = self._make_sample_large()

        gold_output = self._run_sonicmoe_path(moe, x_gold, protocol_enabled=False)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_UPPROJ):
            x_cand = x_gold.detach().clone().requires_grad_()
            candidate_output = self._run_sonicmoe_path(moe, x_cand, protocol_enabled=True)
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)

            gold_output.backward(dout)
            candidate_output.backward(dout)
            gold_dx = x_gold.grad.float()
            cand_dx = x_cand.grad.float()
            torch.testing.assert_close(cand_dx, gold_dx, rtol=5e-2, atol=5e-2)

    def test_native_fp8_downproj_large_shape_contract(self) -> None:
        """Project 2 at production-scale shape — forward + backward correctness."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe_large()
        x_gold, dout = self._make_sample_large()

        gold_output = self._run_sonicmoe_path(moe, x_gold, protocol_enabled=False)
        self.assertFalse(torch.isnan(gold_output.float()).any().item())

        if is_operator_opt_enabled(NATIVE_FP8_DOWNPROJ):
            x_cand = x_gold.detach().clone().requires_grad_()
            candidate_output = self._run_sonicmoe_path(moe, x_cand, protocol_enabled=True)
            torch.testing.assert_close(candidate_output.float(), gold_output.float(), rtol=5e-2, atol=5e-2)

            gold_output.backward(dout)
            candidate_output.backward(dout)
            gold_dx = x_gold.grad.float()
            cand_dx = x_cand.grad.float()
            torch.testing.assert_close(cand_dx, gold_dx, rtol=5e-2, atol=5e-2)

    def test_mixed_dtype_backward_large_shape_contract(self) -> None:
        """Project 3 at production-scale shape — full backward gradient correctness."""
        self._require_blackwell()
        self.set_seed(_SEED)
        moe = self._make_moe_large()
        x_gold, dout = self._make_sample_large()

        gold_dx, gold_dw1, gold_dw2 = self._collect_all_grads(moe, x_gold, dout, protocol_enabled=False)

        self.assertFalse(torch.isnan(gold_dx.float()).any().item())
        self.assertFalse(torch.isnan(gold_dw2.float()).any().item())

        if is_operator_opt_enabled(MIXED_DTYPE_DOWNPROJ_DW2):
            x_cand = x_gold.detach().clone().requires_grad_()
            cand_dx, cand_dw1, cand_dw2 = self._collect_all_grads(moe, x_cand, dout, protocol_enabled=True)
            torch.testing.assert_close(cand_dw2.float(), gold_dw2.float(), rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(cand_dx.float(), gold_dx.float(), rtol=5e-2, atol=5e-2)


class FP8AlignedContractTest(TestCommons):
    """Contract tests using 128-aligned shapes that ACTUALLY exercise the FP8 GEMM path.

    The original contract tests (FP8LargeProjectContractTest) use shapes where
    tokens_per_expert is 16 or 64 — NOT multiples of 128 — so the FP8 code path
    falls through to BF16 gemm_gated and the tests effectively compare BF16 vs BF16.

    These tests use E=8, K=8 so that tokens_per_expert = T*K/E = T,
    guaranteeing 128-alignment for T that is a multiple of 128.
    """

    # Aligned shape: T=1024, E=8, K=8 → tpe=1024 (128-aligned ✓)
    _ALIGNED_T = 1024
    _ALIGNED_H = 3072
    _ALIGNED_I = 1536
    _ALIGNED_E = 8
    _ALIGNED_K = 8
    _SEEDS = [42, 123, 777]

    def _require_blackwell(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

    def _reset_fp8_state(self) -> None:
        """Reset all FP8 global state to prevent cross-test pollution."""
        # Strip env var that would make BF16 reference use FP8 path
        os.environ.pop("SONIC_MOE_FP8_MODE", None)
        functional.clear_all_fp8_weight_caches()
        functional._ALIGNMENT_ASSUMED = False
        functional._ALIGNMENT_STREAK = 0

    def _make_moe(self) -> MoE:
        return MoE(
            num_experts=self._ALIGNED_E,
            num_experts_per_tok=self._ALIGNED_K,
            hidden_size=self._ALIGNED_H,
            intermediate_size=self._ALIGNED_I,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

    def _make_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = (0.02 * torch.randn(self._ALIGNED_T, self._ALIGNED_H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)
        return x, dout

    def _rrmse(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def _corr(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()

    def test_use_fp8_forward_precision_multi_seed(self) -> None:
        """use_fp8=True forward output matches BF16 within <10% RRMSE across seeds."""
        self._require_blackwell()
        for seed in self._SEEDS:
            with self.subTest(seed=seed):
                self._reset_fp8_state()
                self.set_seed(seed)
                moe = self._make_moe()
                x, _ = self._make_sample()

                with enable_quack_gemm(True):
                    out_bf16, _ = moe(x)

                self._reset_fp8_state()

                with enable_quack_gemm(True):
                    out_fp8, _ = moe(x, use_fp8=True)

                rrmse = self._rrmse(out_fp8, out_bf16)
                corr = self._corr(out_fp8, out_bf16)
                self.assertLess(rrmse, 0.10, f"seed={seed}: fwd RRMSE {rrmse:.4f} >= 10%")
                self.assertGreater(corr, 0.99, f"seed={seed}: fwd corr {corr:.4f} < 0.99")

    def test_use_fp8_backward_precision_multi_seed(self) -> None:
        """use_fp8=True backward gradients match BF16 within <10% RRMSE across seeds."""
        self._require_blackwell()
        for seed in self._SEEDS:
            with self.subTest(seed=seed):
                self._reset_fp8_state()
                self.set_seed(seed)
                moe = self._make_moe()
                x, dout = self._make_sample()

                with enable_quack_gemm(True):
                    out_bf16, _ = moe(x)
                out_bf16.backward(dout)
                dx_bf16 = x.grad.clone()
                dw1_bf16 = moe.c_fc.weight.grad.clone()
                dw2_bf16 = moe.c_proj.weight.grad.clone()

                x.grad = None
                moe.zero_grad(set_to_none=True)
                self._reset_fp8_state()

                with enable_quack_gemm(True):
                    out_fp8, _ = moe(x, use_fp8=True)
                out_fp8.backward(dout)
                dx_fp8 = x.grad.clone()
                dw1_fp8 = moe.c_fc.weight.grad.clone()
                dw2_fp8 = moe.c_proj.weight.grad.clone()

                for name, fp8_g, bf16_g in [("dx", dx_fp8, dx_bf16), ("dw1", dw1_fp8, dw1_bf16), ("dw2", dw2_fp8, dw2_bf16)]:
                    rrmse = self._rrmse(fp8_g, bf16_g)
                    corr = self._corr(fp8_g, bf16_g)
                    self.assertLess(rrmse, 0.10, f"seed={seed} {name}: RRMSE {rrmse:.4f} >= 10%")
                    self.assertGreater(corr, 0.99, f"seed={seed} {name}: corr {corr:.4f} < 0.99")

    def test_enable_fp8_context_manager(self) -> None:
        """enable_fp8() context manager produces identical results to use_fp8=True."""
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        moe = self._make_moe()
        x, _ = self._make_sample()

        with enable_quack_gemm(True):
            out_use_fp8, _ = moe(x, use_fp8=True)

        self._reset_fp8_state()

        with enable_fp8():
            out_ctx, _ = moe(x)

        torch.testing.assert_close(out_ctx, out_use_fp8, rtol=0, atol=0)

    def test_use_fp8_production_shape(self) -> None:
        """FP8 precision at production shape: seqlen=8192, H=3072, I=1536, E=8, K=8."""
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        moe = MoE(
            num_experts=8,
            num_experts_per_tok=8,
            hidden_size=3072,
            intermediate_size=1536,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with enable_quack_gemm(True):
            out_bf16, _ = moe(x)
        out_bf16.backward(dout)
        dx_bf16 = x.grad.clone()
        x.grad = None
        moe.zero_grad(set_to_none=True)

        self._reset_fp8_state()

        with enable_quack_gemm(True):
            out_fp8, _ = moe(x, use_fp8=True)
        out_fp8.backward(dout)
        dx_fp8 = x.grad.clone()

        fwd_rrmse = self._rrmse(out_fp8, out_bf16)
        fwd_corr = self._corr(out_fp8, out_bf16)
        bwd_rrmse = self._rrmse(dx_fp8, dx_bf16)
        bwd_corr = self._corr(dx_fp8, dx_bf16)

        self.assertLess(fwd_rrmse, 0.10, f"fwd RRMSE {fwd_rrmse:.4f}")
        self.assertGreater(fwd_corr, 0.99, f"fwd corr {fwd_corr:.4f}")
        self.assertLess(bwd_rrmse, 0.10, f"bwd RRMSE {bwd_rrmse:.4f}")
        self.assertGreater(bwd_corr, 0.99, f"bwd corr {bwd_corr:.4f}")

    def test_z_fp8_save_precision(self) -> None:
        """z FP8 save in fused gated path preserves precision within contract bounds."""
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        moe = self._make_moe()
        x, dout = self._make_sample()

        # BF16 baseline
        with enable_quack_gemm(True):
            out_bf16, _ = moe(x)
        out_bf16.backward(dout)
        dx_bf16 = x.grad.clone()
        x.grad = None
        moe.zero_grad(set_to_none=True)
        self._reset_fp8_state()

        # FP8 with z save (default: SONIC_MOE_FP8_SAVE_Z_FP8=1)
        os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
        with enable_quack_gemm(True):
            out_fp8, _ = moe(x, use_fp8=True)
        out_fp8.backward(dout)
        dx_fp8 = x.grad.clone()

        fwd_rrmse = self._rrmse(out_fp8, out_bf16)
        fwd_corr = self._corr(out_fp8, out_bf16)
        bwd_rrmse = self._rrmse(dx_fp8, dx_bf16)
        bwd_corr = self._corr(dx_fp8, dx_bf16)
        self.assertLess(fwd_rrmse, 0.10, f"z-fp8-save fwd RRMSE {fwd_rrmse:.4f}")
        self.assertGreater(fwd_corr, 0.99, f"z-fp8-save fwd corr {fwd_corr:.4f}")
        self.assertLess(bwd_rrmse, 0.10, f"z-fp8-save bwd RRMSE {bwd_rrmse:.4f}")
        self.assertGreater(bwd_corr, 0.99, f"z-fp8-save bwd corr {bwd_corr:.4f}")

    def test_fp8_memory_less_than_bf16(self) -> None:
        """FP8 peak memory must be <= BF16 peak memory at production shape."""
        import gc
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)

        T, H, I, E, K = 8192, 3072, 1536, 8, 8
        moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
                  intermediate_size=I, activation_function=ActivationType.SWIGLU,
                  add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

        # FP8 peak
        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        self._reset_fp8_state()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with enable_quack_gemm(True):
            out, _ = moe(x, use_fp8=True)
            out.sum().backward()
        torch.cuda.synchronize()
        fp8_peak = torch.cuda.max_memory_allocated()
        del out, x
        moe.zero_grad(set_to_none=True)
        self._reset_fp8_state()
        gc.collect(); torch.cuda.empty_cache()

        # BF16 peak
        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        with enable_quack_gemm(True):
            out, _ = moe(x, use_fp8=False)
            out.sum().backward()
        torch.cuda.synchronize()
        bf16_peak = torch.cuda.max_memory_allocated()

        ratio = fp8_peak / bf16_peak
        self.assertLessEqual(
            ratio, 1.0,
            f"FP8 peak ({fp8_peak/1024**2:.0f}MiB) must be <= BF16 peak ({bf16_peak/1024**2:.0f}MiB), got ratio={ratio:.3f}",
        )

    def test_weight_cache_dedup(self) -> None:
        """Weight cache dedup: direct_fused_dgated reuses fused_dgated cache."""
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _FUSED_WEIGHT_CACHE,
            precompute_weight_fp8_for_fused_dgated,
            precompute_weight_fp8_for_direct_fused_dgated,
            clear_blockscaled_fp8_weight_cache,
        )
        clear_blockscaled_fp8_weight_cache()

        w = torch.randn(self._ALIGNED_H, self._ALIGNED_I, self._ALIGNED_E,
                        device="cuda", dtype=torch.bfloat16)

        # Call fused_dgated first — populates _FUSED_WEIGHT_CACHE
        w_view, scales_view = precompute_weight_fp8_for_fused_dgated(w)
        self.assertEqual(len(_FUSED_WEIGHT_CACHE), 1)

        # Call direct_fused_dgated — should reuse from _FUSED_WEIGHT_CACHE (no new entry)
        w_cont, scales_cont = precompute_weight_fp8_for_direct_fused_dgated(w)
        self.assertEqual(len(_FUSED_WEIGHT_CACHE), 1,
                        "direct_fused_dgated should reuse fused_dgated cache, not create duplicate")
        # Data must match (same physical storage, different views)
        self.assertEqual(w_cont.data_ptr(), w_view.data_ptr(),
                        "direct and fused dgated should share physical storage")

    def test_fp8_downproj_prequant_precision(self) -> None:
        """FP8 down-proj with pre-quantized y1 preserves precision at I=1536."""
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        moe = self._make_moe()
        x, dout = self._make_sample()

        # BF16 baseline
        with enable_quack_gemm(True):
            out_bf16, _ = moe(x)
        out_bf16.backward(dout)
        dx_bf16 = x.grad.clone()
        x.grad = None
        moe.zero_grad(set_to_none=True)
        self._reset_fp8_state()

        # Reset prequant hit counters
        from sonicmoe.functional import _PREQUANT_HIT_COUNT
        _PREQUANT_HIT_COUNT["fwd"] = 0
        _PREQUANT_HIT_COUNT["bwd"] = 0

        # FP8 with pre-quantized y1 down-proj (default path)
        with enable_quack_gemm(True):
            out_fp8, _ = moe(x, use_fp8=True)
        out_fp8.backward(dout)
        dx_fp8 = x.grad.clone()

        # Verify prequant was actually consumed (not silently falling back to BF16)
        self.assertGreater(_PREQUANT_HIT_COUNT["fwd"], 0,
                           "FP8 down-proj prequant was never consumed — matching bug?")
        self.assertGreater(_PREQUANT_HIT_COUNT["bwd"], 0,
                           "FP8 bwd prequant (dz) was never consumed — matching bug?")

        fwd_rrmse = self._rrmse(out_fp8, out_bf16)
        fwd_corr = self._corr(out_fp8, out_bf16)
        bwd_rrmse = self._rrmse(dx_fp8, dx_bf16)
        bwd_corr = self._corr(dx_fp8, dx_bf16)
        self.assertLess(fwd_rrmse, 0.10, f"downproj-prequant fwd RRMSE {fwd_rrmse:.4f}")
        self.assertGreater(fwd_corr, 0.99, f"downproj-prequant fwd corr {fwd_corr:.4f}")
        self.assertLess(bwd_rrmse, 0.10, f"downproj-prequant bwd RRMSE {bwd_rrmse:.4f}")
        self.assertGreater(bwd_corr, 0.99, f"downproj-prequant bwd corr {bwd_corr:.4f}")

    def test_multi_iteration_cache_consistency(self) -> None:
        """Multiple fwd+bwd iterations don't corrupt weight cache layouts.

        Regression test: precompute_weight_fp8 (row-major for varlen) and
        precompute_weight_fp8_for_fused_dgated (col-major for dgated) must use
        separate caches.  A shared cache causes stride mismatch on iteration 2+.
        """
        self._require_blackwell()
        self._reset_fp8_state()
        torch.manual_seed(42)
        moe = self._make_moe()
        x, dout = self._make_sample()

        with enable_quack_gemm(True):
            for i in range(3):
                moe.zero_grad(set_to_none=True)
                xi = x.clone().detach().requires_grad_(True)
                out, _ = moe(xi, use_fp8=True)
                out.backward(dout)
                # If cache collision exists, iteration ≥1 crashes with
                # RuntimeError: Expected strides[leading_dim] == 1
                torch.cuda.synchronize()

    def test_triton_weight_quant_matches_eager(self) -> None:
        """Triton _quantize_weight_3d_triton matches eager quantize_activation_blockwise.

        Verifies that the optimized single-kernel Triton path produces the same
        FP8 values and ISA-packed scales as the legacy multi-kernel eager path.
        """
        self._require_blackwell()
        torch.manual_seed(42)

        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _quantize_weight_3d_triton,
            quantize_and_pack_activation,
        )

        # Test with Ernie-like weight shapes
        for E, N, K in [(8, 3072, 3072), (8, 1536, 3072), (8, 3072, 1536)]:
            w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            w_fp8, w_scales = _quantize_weight_3d_triton(w)

            # Verify shapes
            self.assertEqual(w_fp8.shape, (E, N, K), f"FP8 shape mismatch for ({E},{N},{K})")
            self.assertEqual(w_fp8.dtype, torch.float8_e4m3fn)

            # Verify per-expert consistency: each expert quantized independently
            # must match the 2D quantize_and_pack_activation result
            for e in range(E):
                ref_fp8, ref_scales = quantize_and_pack_activation(w[e])
                expert_fp8 = w_fp8[e]
                fp8_match = torch.equal(expert_fp8, ref_fp8)
                self.assertTrue(
                    fp8_match,
                    f"Expert {e} FP8 values differ for shape ({E},{N},{K})"
                )

    def test_weight_cache_retention_precision(self) -> None:
        """FP8 precision is maintained with weight cache retention across fwd+bwd.

        Verifies that keeping w1/w2 FP8 caches (instead of evicting after forward)
        does not corrupt backward gradients.  The cache auto-invalidates via
        ``w._version`` when an optimizer updates weights.
        """
        self._require_blackwell()
        self._reset_fp8_state()
        torch.manual_seed(42)

        T, H, I, E, K = 8192, 3072, 1536, 8, 8
        moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
                  intermediate_size=I, activation_function=ActivationType.SWIGLU,
                  add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        # BF16 reference
        with enable_quack_gemm(True):
            out_bf16, _ = moe(x)
        out_bf16.backward(dout)
        dx_bf16 = x.grad.clone()
        dw1_bf16 = moe.c_fc.weight.grad.clone()
        dw2_bf16 = moe.c_proj.weight.grad.clone()
        x.grad = None
        moe.zero_grad(set_to_none=True)
        self._reset_fp8_state()

        # FP8 with cache retention (current production path — no eviction)
        with enable_quack_gemm(True):
            out_fp8, _ = moe(x, use_fp8=True)
        out_fp8.backward(dout)
        dx_fp8 = x.grad.clone()
        dw1_fp8 = moe.c_fc.weight.grad.clone()
        dw2_fp8 = moe.c_proj.weight.grad.clone()

        for name, fp8_g, bf16_g in [
            ("dx", dx_fp8, dx_bf16),
            ("dw1", dw1_fp8, dw1_bf16),
            ("dw2", dw2_fp8, dw2_bf16),
        ]:
            rrmse = self._rrmse(fp8_g, bf16_g)
            corr = self._corr(fp8_g, bf16_g)
            self.assertLess(rrmse, 0.10,
                f"cache-retain {name} RRMSE {rrmse:.4f}")
            self.assertGreater(corr, 0.99,
                f"cache-retain {name} corr {corr:.4f}")

    def test_z_quant_blockscaled_roundtrip(self) -> None:
        """quantize_activation_blockscaled_fast + dequantize_blockscaled_fp8 round-trip.

        Covers the BR=32/GPB=12 tuned kernel grid parameters.
        Tests at both test shape and production shape.
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            quantize_activation_blockscaled_fast,
        )
        from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

        for label, M, K in [("test", 2048, 768), ("prod", 65536, 3072)]:
            with self.subTest(shape=label):
                torch.manual_seed(42)
                x = 0.02 * torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

                fp8_data, scales_e8m0 = quantize_activation_blockscaled_fast(x)
                self.assertEqual(fp8_data.shape, (M, K))
                self.assertEqual(fp8_data.dtype, torch.float8_e4m3fn)

                x_restored = dequantize_blockscaled_fp8(
                    fp8_data, scales_e8m0.view(torch.uint8)
                )
                rrmse = self._rrmse(x_restored, x)
                corr = self._corr(x_restored, x)
                self.assertLess(rrmse, 0.05,
                    f"{label} z-quant roundtrip RRMSE {rrmse:.4f} >= 5%")
                self.assertGreater(corr, 0.999,
                    f"{label} z-quant roundtrip corr {corr:.6f} < 0.999")

    def test_quantize_and_pack_isa_roundtrip(self) -> None:
        """quantize_and_pack_activation produces valid FP8 + ISA-packed scales.

        Verifies the fused quant+ISA-pack kernel at production activation shapes.
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            quantize_and_pack_activation,
        )

        for label, M, K in [("x", 8192, 3072), ("dz", 65536, 3072), ("y1", 65536, 1536)]:
            with self.subTest(activation=label):
                torch.manual_seed(42)
                x = 0.02 * torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

                fp8_data, packed_scales = quantize_and_pack_activation(x)
                self.assertEqual(fp8_data.shape, (M, K))
                self.assertEqual(fp8_data.dtype, torch.float8_e4m3fn)
                self.assertFalse(torch.all(fp8_data.view(torch.uint8) == 0).item(),
                    f"{label}: all-zero FP8 output — quant kernel broken")

                # Verify scale bytes are plausible (not all 0 or all 127)
                scale_bytes = packed_scales.view(torch.uint8)
                unique_scales = scale_bytes.unique().numel()
                self.assertGreater(unique_scales, 1,
                    f"{label}: packed scales have only {unique_scales} unique value(s)")

    def test_production_shape_all_gradients(self) -> None:
        """FP8 precision at production shape for ALL gradients: dx, dw1, dw2.

        Extends test_use_fp8_production_shape to verify weight gradients.
        """
        self._require_blackwell()
        self._reset_fp8_state()
        self.set_seed(42)
        T, H, I, E, K = 8192, 3072, 1536, 8, 8
        moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
                  intermediate_size=I, activation_function=ActivationType.SWIGLU,
                  add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with enable_quack_gemm(True):
            out_bf16, _ = moe(x)
        out_bf16.backward(dout)
        dx_bf16 = x.grad.clone()
        dw1_bf16 = moe.c_fc.weight.grad.clone()
        dw2_bf16 = moe.c_proj.weight.grad.clone()
        x.grad = None
        moe.zero_grad(set_to_none=True)
        self._reset_fp8_state()

        with enable_quack_gemm(True):
            out_fp8, _ = moe(x, use_fp8=True)
        out_fp8.backward(dout)
        dx_fp8 = x.grad.clone()
        dw1_fp8 = moe.c_fc.weight.grad.clone()
        dw2_fp8 = moe.c_proj.weight.grad.clone()

        for name, fp8_g, bf16_g in [
            ("fwd", out_fp8, out_bf16),
            ("dx", dx_fp8, dx_bf16),
            ("dw1", dw1_fp8, dw1_bf16),
            ("dw2", dw2_fp8, dw2_bf16),
        ]:
            rrmse = self._rrmse(fp8_g, bf16_g)
            corr = self._corr(fp8_g, bf16_g)
            self.assertLess(rrmse, 0.10,
                f"prod-shape {name}: RRMSE {rrmse:.4f} >= 10%")
            self.assertGreater(corr, 0.99,
                f"prod-shape {name}: corr {corr:.4f} < 0.99")

    def test_cache_retention_multi_iter_no_drift(self) -> None:
        """Cache retention across 3 iterations: precision must not drift.

        Without eviction, the weight cache persists.  This verifies that
        repeated fwd+bwd passes with the same weights produce consistent
        gradients — no cumulative error from stale cache entries.
        """
        self._require_blackwell()
        self._reset_fp8_state()
        torch.manual_seed(42)
        moe = self._make_moe()
        x_orig, dout = self._make_sample()

        # Collect per-iteration dx gradients (same input, same weights)
        dx_iters = []
        for i in range(3):
            moe.zero_grad(set_to_none=True)
            xi = x_orig.clone().detach().requires_grad_(True)
            with enable_quack_gemm(True):
                out, _ = moe(xi, use_fp8=True)
            out.backward(dout)
            dx_iters.append(xi.grad.clone())
            torch.cuda.synchronize()

        # All iterations should produce identical dx (same weights, same input)
        for i in range(1, 3):
            torch.testing.assert_close(
                dx_iters[i], dx_iters[0], rtol=0, atol=0,
                msg=f"Iter {i} dx differs from iter 0 — cache retention drift",
            )
