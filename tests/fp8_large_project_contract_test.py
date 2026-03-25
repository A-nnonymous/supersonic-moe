import os

import torch

import sonicmoe.functional as functional
from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm, get_default_fp8_protocol, moe_general_routing_inputs
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
