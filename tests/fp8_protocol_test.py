# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from dataclasses import replace

import torch
import torch.nn.functional as F

from sonicmoe import (
    FP8ActivationDType,
    KernelBackendMoE,
    MoE,
    apply_activation_fp8_protocol_cutely_fused,
    apply_preact_activation_fp8_protocol_cutely_fused,
    FP8Protocol,
    FP8ScaleEncoding,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    enable_quack_gemm,
    get_default_fp8_protocol,
    quantize_activation_reference,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
from sonicmoe.enums import ActivationType

from .test_commons import TestCommons


_SEED = 42


class FP8ProtocolTest(TestCommons):
    def test_blackwell_fp8_protocol_defaults(self) -> None:
        protocol = validate_fp8_protocol(get_default_fp8_protocol())

        self.assertEqual(protocol.activation_dtype, FP8ActivationDType.E4M3)
        self.assertEqual(protocol.scale_encoding, FP8ScaleEncoding.E8M0)
        self.assertEqual(protocol.group_size, 128)
        self.assertTrue(protocol.requires_quack_gemm)

    def test_blackwell_fp8_protocol_rejects_unsupported_dtypes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only e4m3 activations"):
            validate_fp8_protocol(replace(FP8Protocol(), activation_dtype=None))

        with self.assertRaisesRegex(ValueError, "Only e8m0 scales"):
            validate_fp8_protocol(replace(FP8Protocol(), scale_encoding=None))

    def test_blackwell_fp8_protocol_runtime_and_reference_quant(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()

        with self.assertRaisesRegex(RuntimeError, "requires the QuACK GEMM path"):
            validate_fp8_runtime_support(protocol, torch.device("cuda"), quack_enabled=False)

        x = (0.25 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).requires_grad_(False)

        with enable_quack_gemm(True):
            validate_fp8_runtime_support(protocol, x.device)
            fp8_tensor = quantize_activation_reference(x, protocol)

        self.assertEqual(fp8_tensor.data.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8_tensor.scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(fp8_tensor.scales.shape), (8, 2))
        self.assertFalse(torch.isnan(fp8_tensor.data.float()).any().item())

        restored = dequantize_activation_reference(fp8_tensor)

        # e8m0 stores power-of-two scales, so the decoded scales should stay exact powers of two.
        log2_scales = torch.log2(fp8_tensor.scales.float())
        rounded_log2_scales = torch.round(log2_scales)
        self.assertTrue(torch.allclose(log2_scales, rounded_log2_scales, atol=0.0, rtol=0.0))

        self.assertEqual(restored.dtype, torch.float32)
        self.assertLess((restored - x.float()).abs().max().item(), 0.35)

    def test_blackwell_fp8_protocol_boundary_keeps_finite_forward_backward(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()

        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with enable_quack_gemm(True):
            y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
            y.backward(dout)

        self.assertFalse(torch.isnan(y.float()).any().item())
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad.float()).any().item())

        sample = 0.1 * torch.randn(4, 130, device="cuda", dtype=torch.bfloat16)
        with enable_quack_gemm(True):
            restored, scales = apply_activation_fp8_protocol(sample, protocol)
        self.assertEqual(tuple(scales.shape), (4, 2))
        self.assertEqual(tuple(restored.shape), (4, 130))
        self.assertFalse(torch.isnan(restored.float()).any().item())

    def test_cutely_fused_adapter_matches_reference_contract(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        sample = 0.1 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

        with enable_quack_gemm(True):
            restored_ref, scales_ref = apply_activation_fp8_protocol(sample, protocol)
            restored_adapter, scales_adapter = apply_activation_fp8_protocol_cutely_fused(sample, protocol)

        self.assertEqual(restored_ref.shape, restored_adapter.shape)
        self.assertEqual(scales_ref.shape, scales_adapter.shape)
        self.assertEqual(restored_ref.dtype, restored_adapter.dtype)
        self.assertEqual(scales_ref.dtype, scales_adapter.dtype)
        torch.testing.assert_close(restored_ref.float(), restored_adapter.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(scales_ref.float(), scales_adapter.float(), atol=0.0, rtol=0.0)

    def test_preact_cutely_fused_path_matches_reference_boundary(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("Blackwell-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        preact = 0.1 * torch.randn(8, 260, device="cuda", dtype=torch.bfloat16)
        postact = (preact[..., 1::2] * F.silu(preact[..., ::2].float()).to(dtype=preact.dtype)).contiguous()

        with enable_quack_gemm(True):
            restored_ref, scales_ref = apply_activation_fp8_protocol(postact, protocol)
            restored_fused, scales_fused = apply_preact_activation_fp8_protocol_cutely_fused(preact, postact, protocol)

        self.assertEqual(restored_ref.shape, restored_fused.shape)
        self.assertEqual(scales_ref.shape, scales_fused.shape)
        self.assertEqual(restored_fused.dtype, torch.bfloat16)
        self.assertEqual(scales_fused.dtype, torch.float8_e8m0fnu)
        diff = (restored_ref.float() - restored_fused.float()).abs()
        self.assertLess(diff.max().item(), 3e-2)
        self.assertLess(torch.sqrt(torch.mean(diff.square())).item(), 6e-3)
        log2_scales = torch.log2(scales_fused.float())
        self.assertTrue(torch.allclose(log2_scales, torch.round(log2_scales), atol=0.0, rtol=0.0))
