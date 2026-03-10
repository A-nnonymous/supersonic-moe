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
"""MoEStatics — correction-bias and expert-usage tracking."""

import paddle
from paddle import nn


class MoEStatics(nn.Layer):
    """
    存放 MoE 统计信息
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False
        num_experts = config.n_routed_experts[0] if config.multimodel_experts else config.n_routed_experts
        if config.multimodel_experts:
            assert (
                len(set(config.n_routed_experts)) == 1
            ), "assume expert group has same size, got: {config.n_routed_experts}"

        with paddle.utils.unique_name.guard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = len(config.n_routed_experts) if config.multimodel_experts else 1
            p = self.create_parameter(
                shape=[num_experts_groups, num_experts],
                dtype="float32",
                is_bias=True,
                attr=paddle.ParamAttr(name=paddle.utils.unique_name.generate("corr_bias")),
            )
            p.stop_gradient = False
            self.e_score_correction_bias = p
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.unused_param = True
            if getattr(config, "build_skip_comm_buffer", False):
                self.e_score_correction_bias.color = {
                    "color": "skip_comm",
                    "group": paddle.distributed.new_group([paddle.distributed.get_rank()]),
                }
            p = paddle.zeros(
                shape=[num_experts_groups, num_experts],
                dtype="int64",
            )
            p.stop_gradient = True
            self.expert_usage = p
