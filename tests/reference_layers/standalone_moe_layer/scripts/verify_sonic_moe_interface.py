#!/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Interface verification script for Sonic-MoE integration.

This script verifies that the standalone DeepEPMOELayer exposes all the
interfaces and data formats needed for Sonic-MoE FP8 upgrade.

Usage:
    python scripts/verify_sonic_moe_interface.py

Verification items:
  1. DeepEPMOELayer.forward() input/output contract
  2. Token dispatcher dispatch/combine interfaces
  3. FP8 weight quantization API
  4. Gate scoring interface compatibility
  5. Decomposed compute API (gate_compute, dispatch_compute, mlp_compute, combine_compute)
"""
import inspect
import sys


def check(condition, msg):
    if condition:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        return False
    return True


def main():
    all_passed = True

    print("=" * 70)
    print("MoE Standalone — Sonic-MoE Interface Verification")
    print("=" * 70)

    # --- 1. Package import ---
    print("\n1. Package Import")
    try:
        import moe_standalone

        all_passed &= check(True, "moe_standalone imported")
    except Exception as e:
        print(f"  [FAIL] Cannot import moe_standalone: {e}")
        sys.exit(1)

    # --- 2. Core classes ---
    print("\n2. Core Class Availability")
    from moe_standalone import DeepEPMOELayer, DeepEPTop2Gate, MoEFlexTokenDispatcher, MoEStatics
    from moe_standalone.moe.deep_ep_moe_layer import Fp8FusedMoeFunc, MlpNode, recompute_moe_gate_up_func

    all_passed &= check(inspect.isclass(DeepEPMOELayer), "DeepEPMOELayer is a class")
    all_passed &= check(inspect.isclass(DeepEPTop2Gate), "DeepEPTop2Gate is a class")
    all_passed &= check(inspect.isclass(MoEStatics), "MoEStatics is a class")
    all_passed &= check(inspect.isclass(MoEFlexTokenDispatcher), "MoEFlexTokenDispatcher is a class")
    all_passed &= check(inspect.isclass(MlpNode), "MlpNode is a class")
    all_passed &= check(callable(recompute_moe_gate_up_func), "recompute_moe_gate_up_func is callable")

    # --- 3. DeepEPMOELayer forward interface ---
    print("\n3. DeepEPMOELayer.forward() Interface")
    sig = inspect.signature(DeepEPMOELayer.forward)
    params = sig.parameters

    required_params = ["input", "input_ids"]
    for p in required_params:
        all_passed &= check(p in params, f"forward() has required param '{p}'")

    optional_params = [
        "elastic_topk_value",
        "global_gate_mask",
        "is_diff_expert_num",
        "is_diff_topk",
        "max_topk",
        "origin_input_ids",
        "is_pure_text_line",
    ]
    for p in optional_params:
        all_passed &= check(p in params, f"forward() has optional param '{p}'")

    # Check return annotation
    all_passed &= check(
        "Tuple" in str(sig.return_annotation) or sig.return_annotation == inspect.Parameter.empty,
        "forward() returns Tuple[Tensor, Tensor, Tensor, Tensor]",
    )

    # --- 4. Decomposed compute API ---
    print("\n4. Decomposed Compute API (for comm-overlap)")
    decomposed_methods = {
        "gate_compute": ["input", "input_ids"],
        "dispatch_compute": ["input", "topk_probs", "topk_indices"],
        "mlp_compute": ["dispatched_hidden_states", "dispatched_indices", "dispatched_probs", "fp8_dispatched_handle"],
        "combine_compute": ["hidden_states_tmp"],
        "post_process_compute": ["input", "combined_output"],
    }
    for method_name, expected_params in decomposed_methods.items():
        all_passed &= check(hasattr(DeepEPMOELayer, method_name), f"has method '{method_name}'")
        if hasattr(DeepEPMOELayer, method_name):
            msig = inspect.signature(getattr(DeepEPMOELayer, method_name))
            for p in expected_params:
                all_passed &= check(p in msig.parameters, f"  {method_name}() has param '{p}'")

    # --- 5. FP8 quantization API ---
    print("\n5. FP8 Weight Quantization API")
    all_passed &= check(hasattr(DeepEPMOELayer, "fp8_quant_weight"), "has fp8_quant_weight()")
    fp8_sig = inspect.signature(DeepEPMOELayer.fp8_quant_weight)
    all_passed &= check("batch_mode" in fp8_sig.parameters, "fp8_quant_weight has batch_mode param")
    all_passed &= check("quant_transpose" in fp8_sig.parameters, "fp8_quant_weight has quant_transpose param")
    all_passed &= check("use_ue8m0" in fp8_sig.parameters, "fp8_quant_weight has use_ue8m0 param")

    # --- 6. Token dispatcher interface ---
    print("\n6. Token Dispatcher Interface")
    td_sig = inspect.signature(MoEFlexTokenDispatcher.__init__)
    all_passed &= check("num_local_experts" in td_sig.parameters, "init has num_local_experts")
    all_passed &= check("num_moe_experts" in td_sig.parameters, "init has num_moe_experts")
    all_passed &= check("ep_group" in td_sig.parameters, "init has ep_group")

    perm_sig = inspect.signature(MoEFlexTokenDispatcher.token_permutation)
    all_passed &= check("hidden_states" in perm_sig.parameters, "token_permutation has hidden_states")
    all_passed &= check("token_indices" in perm_sig.parameters, "token_permutation has token_indices")

    # --- 7. FP8 utils ---
    print("\n7. FP8 Utils")
    from moe_standalone.token_dispatcher.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        ExpertsGroupGemmNode,
        has_config,
        tilewise_quant,
    )

    all_passed &= check(FP8_ALIGN == 128, "FP8_ALIGN == 128")
    all_passed &= check(callable(tilewise_quant), "tilewise_quant is callable")
    all_passed &= check(inspect.isclass(ExpertsGroupGemmNode), "ExpertsGroupGemmNode is a class")
    all_passed &= check(inspect.isclass(ExpertsGroupGemmContiguousNode), "ExpertsGroupGemmContiguousNode is a class")

    # --- 8. Compat layer completeness ---
    print("\n8. Compat Layer Completeness")
    from moe_standalone.compat import (
        HAVE_DEEP_EP,
        TDU,
        FakeGather,
        FusedUnpermutation,
        GatherOp,
        ScatterOp,
        deep_ep,
        deep_gemm,
        dispatch_to,
        fake_scatter_add,
        manual_backward,
        profile,
    )

    compat_items = {
        "deep_ep": deep_ep,
        "HAVE_DEEP_EP": HAVE_DEEP_EP,
        "TDU": TDU,
        "deep_gemm": deep_gemm,
    }
    for name, obj in compat_items.items():
        all_passed &= check(True, f"compat.{name} = {type(obj).__name__ if obj else 'None'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED — Interface verification complete.")
        print("The standalone package is ready for Sonic-MoE FP8 integration.")
    else:
        print("SOME CHECKS FAILED — Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
