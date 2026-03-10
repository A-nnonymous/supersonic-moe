# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MoE Standalone — DeepEPMOELayer 独立剥离包

从 ernie-core 中剥离出来的 DeepEPMOELayer 及其完整依赖链，
便于外部库（如 Sonic-MoE）进行 FP8 功能升级和接口对接。
"""

from moe_standalone.moe.deep_ep_moe_layer import DeepEPMOELayer
from moe_standalone.moe.moe_statics import MoEStatics
from moe_standalone.moe.top2_gate import DeepEPTop2Gate, Top2Gate, TopKGateFused
from moe_standalone.token_dispatcher.fused_a2a import barrier_ep, fused_combine, fused_dispatch
from moe_standalone.token_dispatcher.token_dispatcher import MoEFlexTokenDispatcher


__all__ = [
    "DeepEPMOELayer",
    "Top2Gate",
    "TopKGateFused",
    "DeepEPTop2Gate",
    "MoEStatics",
    "MoEFlexTokenDispatcher",
    "fused_dispatch",
    "fused_combine",
    "barrier_ep",
]

__version__ = "0.1.0"
