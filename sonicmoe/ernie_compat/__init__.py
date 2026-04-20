"""ERNIE integration layer for SonicMoE."""

from sonicmoe.ernie_compat.mlp_node import (  # noqa: F401
    SonicMoEFunc,
    invalidate_weight_caches,
    prepare_sonic_inputs,
    stack_ernie_w1,
    stack_ernie_w2,
)
