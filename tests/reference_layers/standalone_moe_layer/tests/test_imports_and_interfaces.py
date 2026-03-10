# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Tests for moe_standalone package.

Test levels:
  1. test_import_*          — Verify all modules can be imported (no hidden deps)
  2. test_class_interface_* — Verify class signatures match expected API
  3. test_compat_*          — Verify compat shims work correctly
"""
import importlib
import inspect
import sys

import pytest


# ============================================================================
# Level 1 — Import tests (coldstart, no GPU required)
# ============================================================================
class TestImports:
    """Verify the package and all submodules can be imported cleanly."""

    def test_import_top_level(self):
        import moe_standalone

        assert hasattr(moe_standalone, "__version__")
        assert hasattr(moe_standalone, "DeepEPMOELayer")

    def test_import_compat(self):
        from moe_standalone import compat

        assert hasattr(compat, "deep_ep")
        assert hasattr(compat, "HAVE_DEEP_EP")
        assert hasattr(compat, "TDU")
        assert hasattr(compat, "manual_backward")
        assert hasattr(compat, "FakeGather")
        assert hasattr(compat, "FusedUnpermutation")
        assert hasattr(compat, "fake_scatter_add")
        assert hasattr(compat, "dispatch_to")

    def test_import_moe_statics(self):
        from moe_standalone.moe.moe_statics import MoEStatics

        assert inspect.isclass(MoEStatics)

    def test_import_top2_gate(self):
        from moe_standalone.moe.top2_gate import DeepEPTop2Gate, Top2Gate, TopKGateFused

        assert inspect.isclass(Top2Gate)
        assert inspect.isclass(TopKGateFused)
        assert inspect.isclass(DeepEPTop2Gate)
        assert issubclass(TopKGateFused, Top2Gate)
        assert issubclass(DeepEPTop2Gate, TopKGateFused)

    def test_import_token_dispatcher(self):
        from moe_standalone.token_dispatcher.token_dispatcher import MoEFlexTokenDispatcher

        assert inspect.isclass(MoEFlexTokenDispatcher)

    def test_import_fused_a2a(self):
        from moe_standalone.token_dispatcher import fused_a2a

        assert hasattr(fused_a2a, "DeepEPBuffer")
        assert hasattr(fused_a2a, "barrier_ep")
        # fused_dispatch / fused_combine are None if deep_ep is not installed
        assert hasattr(fused_a2a, "fused_dispatch")
        assert hasattr(fused_a2a, "fused_combine")

    def test_import_fp8_utils(self):
        from moe_standalone.token_dispatcher.fp8_utils import (
            FP8_ALIGN,
            ExpertsGroupGemmContiguousNode,
            ExpertsGroupGemmNode,
            has_config,
        )

        assert FP8_ALIGN == 128

    def test_import_moe_utils(self):
        from moe_standalone.token_dispatcher.moe_utils import (
            UnZipNode,
            ZipNode,
            permute,
            topk_to_permuted_indices_single,
            unpermute,
        )

        assert callable(permute)
        assert callable(unpermute)

    def test_import_deep_ep_moe_layer(self):
        from moe_standalone.moe.deep_ep_moe_layer import (
            DeepEPMOELayer,
            Fp8FusedMoeFunc,
            MlpNode,
            recompute_moe_gate_up_func,
        )

        assert inspect.isclass(DeepEPMOELayer)
        assert inspect.isclass(MlpNode)
        assert callable(recompute_moe_gate_up_func)


# ============================================================================
# Level 2 — Class Interface Verification
# ============================================================================
class TestClassInterfaces:
    """Verify the public API signatures of key classes."""

    def test_deep_ep_moe_layer_init_signature(self):
        from moe_standalone.moe.deep_ep_moe_layer import DeepEPMOELayer

        sig = inspect.signature(DeepEPMOELayer.__init__)
        params = list(sig.parameters.keys())
        expected = [
            "self",
            "gate",
            "experts",
            "layer_idx",
            "shared_experts",
            "group",
            "recompute",
            "enable_logging",
            "k",
            "enable_bpr",
            "all_to_all_dropout",
            "group_experts",
            "moe_statics",
        ]
        assert params == expected, f"Expected {expected}, got {params}"

    def test_deep_ep_moe_layer_forward_signature(self):
        from moe_standalone.moe.deep_ep_moe_layer import DeepEPMOELayer

        sig = inspect.signature(DeepEPMOELayer.forward)
        params = list(sig.parameters.keys())
        expected_subset = ["self", "input", "input_ids"]
        for p in expected_subset:
            assert p in params, f"Missing parameter: {p}"

    def test_deep_ep_moe_layer_has_key_methods(self):
        from moe_standalone.moe.deep_ep_moe_layer import DeepEPMOELayer

        expected_methods = [
            "forward",
            "fp8_quant_weight",
            "gate_score",
            "topk",
            "gate_compute",
            "dispatch_compute",
            "mlp_compute",
            "combine_compute",
            "post_process_compute",
            "forward_experts",
            "fine_grained_forward_experts",
            "calc_router_loss_and_logging",
            "get_num_tokens_per_expert",
            "sharded_state_dict",
        ]
        for method_name in expected_methods:
            assert hasattr(DeepEPMOELayer, method_name), f"Missing method: {method_name}"

    def test_moe_flex_token_dispatcher_interface(self):
        from moe_standalone.token_dispatcher.token_dispatcher import MoEFlexTokenDispatcher

        sig = inspect.signature(MoEFlexTokenDispatcher.__init__)
        params = list(sig.parameters.keys())
        assert "num_local_experts" in params
        assert "num_moe_experts" in params
        assert "ep_group" in params

        expected_methods = [
            "token_permutation",
            "token_unpermutation",
            "cal_final_topk",
        ]
        for method_name in expected_methods:
            assert hasattr(MoEFlexTokenDispatcher, method_name), f"Missing method: {method_name}"

    def test_deep_ep_top2_gate_interface(self):
        from moe_standalone.moe.top2_gate import DeepEPTop2Gate

        expected_methods = [
            "forward",
            "_cal_aux_loss",
            "_cal_z_loss",
            "_cal_orthogonal_loss",
            "_cal_switch_aux_loss",
            "_cal_seq_aux_loss",
        ]
        for method_name in expected_methods:
            assert hasattr(DeepEPTop2Gate, method_name), f"Missing method: {method_name}"


# ============================================================================
# Level 3 — Compat Shim Tests
# ============================================================================
class TestCompat:
    """Verify compat shim functions work correctly."""

    def test_try_import_existing(self):
        from moe_standalone.compat import try_import

        os_mod = try_import(["os"])
        assert os_mod is not None

    def test_try_import_nonexistent(self):
        from moe_standalone.compat import try_import

        result = try_import(["nonexistent_module_xyz123"])
        assert result is None

    def test_profile_context_manager(self):
        from moe_standalone.compat import profile

        with profile("test"):
            pass  # should not raise

    def test_get_env_device(self):
        from moe_standalone.compat import get_env_device

        result = get_env_device()
        assert isinstance(result, str)

    def test_dispatch_to_fallback(self):
        from moe_standalone.compat import dispatch_to

        @dispatch_to(
            lambda *a, **kw: "fast_path",
            cond=lambda *a, **kw: False,
        )
        def my_func():
            return "slow_path"

        assert my_func() == "slow_path"

    def test_global_rr_queue_log(self):
        import queue

        from moe_standalone.compat import global_rr_queue_log

        q = queue.Queue()
        # Should not raise
        global_rr_queue_log.update(q, "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
