# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from dataclasses import replace

import torch

from sonicmoe import (
    FP8ActivationDType,
    FP8Protocol,
    FP8ScaleEncoding,
    dequantize_activation_reference,
    enable_quack_gemm,
    get_default_fp8_protocol,
    quantize_activation_reference,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)

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
