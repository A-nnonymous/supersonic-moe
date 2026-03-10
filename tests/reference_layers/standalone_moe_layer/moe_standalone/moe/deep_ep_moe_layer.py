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
DeepEPMOELayer — standalone MoE layer using DeepEP communication.

Extracted from ernie-core moe_layer.py. Contains:
  - recompute_moe_gate_up_func
  - MlpNode
  - Fp8FusedMoeFunc (PyLayer)
  - DeepEPMOELayer (nn.Layer)
"""
import logging
import os
from collections import namedtuple
from functools import partial
from typing import Any, List, Optional, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from moe_standalone.compat import (
    FQO,
    TDU,
    FakeGather,
    FusedMoETopk,
    FusedQuantOps,
    FusedUnpermutation,
    GatherOp,
    PrintOp,
    RefinedRcomputeMoECombine,
    RefinedRcomputeMoEGateDispatch,
    ScatterOp,
    dispatch_to,
    fake_scatter_add,
    get_env_device,
    get_timers,
    global_moe_balance_training_logs_enabled,
    global_training_logs,
    global_training_logs_enabled,
    int_bincount,
    manual_backward,
    moe_router_loss_ops,
    profile,
    routing_map_forward,
)
from moe_standalone.moe.top2_gate import DeepEPTop2Gate, TopKGateFused, cast_if_needed
from moe_standalone.token_dispatcher.fp8_utils import (
    FP8_ALIGN,
    ExpertsGroupGemmContiguousNode,
    ExpertsGroupGemmNode,
    ExpertsGroupGemmWLCHNode,
    has_config,
    tilewise_quant,
)
from moe_standalone.token_dispatcher.moe_utils import (
    UnZipNode,
    ZipNode,
    inplace_offload_if_needed,
    merge_subbatch_cast,
    tokens_zip_unique_add_with_subbatch,
    topk_to_permuted_indices_single,
)
from moe_standalone.token_dispatcher.token_dispatcher import MoEFlexTokenDispatcher
from paddle import Tensor, framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# recompute_moe_gate_up_func
# ---------------------------------------------------------------------------
def recompute_moe_gate_up_func(config, layer_idx):
    """
    根据配置和层索引确定是否重计算。
    """
    if config.recompute_granularity != "selective" or config.recompute_modules is None:
        return False
    if "moe_gate_up" not in config.recompute_modules:
        return False
    if not isinstance(config.recompute_modules, dict):
        return True
    if not isinstance(config.recompute_modules["moe_gate_up"], list):
        return True
    else:
        return layer_idx in config.recompute_modules["moe_gate_up"]
    return False


# ---------------------------------------------------------------------------
# MlpNode
# ---------------------------------------------------------------------------
class MlpNode:
    """
    The FusedMoeLayer class includes operations for unzipping, expert computation, and zipping.
    """

    def __init__(
        self,
        custom_map,
        max_topk,
        recompute_moe_gate_up=False,
        dequant_input=False,
        moe_expert_fusion=True,
        recompute_moe_permute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        fp8=True,
        bypass_expert_output=False,
        use_ue8m0=False,
    ):
        """
        Constructor
        """
        self.token_dispatcher = custom_map.dispatcher
        self.moe_expert_fusion = moe_expert_fusion
        self.experts = custom_map.experts
        if recompute_moe_permute:
            assert not moe_expert_fusion, "moe_expert_fusion must be disabled when recompute_moe_permute = True"
            assert recompute_moe_gate_up, "recompute_moe_gate_up must be enabled when recompute_moe_permute = True"
            assert dequant_input, "dequant_input must be enabled with recompute_moe_permute = True"
        self.recompute_moe_permute = recompute_moe_permute

        self.moe_subbatch_token_num_after_dispatch = moe_subbatch_token_num_after_dispatch

        if self.moe_subbatch_token_num_after_dispatch is not None:
            assert (
                self.moe_subbatch_token_num_after_dispatch > 0
                and self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
            assert (
                not moe_expert_fusion
            ), "moe_expert_fusion must be disabled when moe_subbatch_token_num_after_dispatch > 0"
            # recompute_moe_gate_up will be enabled when moe_subbatch_token_num_after_dispatch > 0
            recompute_moe_gate_up = True
            assert dequant_input, "dequant_input must be enabled when moe_subbatch_token_num_after_dispatch > 0"

        # full recompute下, 用RR的方式cache住combine的输出, 进而可以省下down_proj、zip、combine三者的重计算
        self.bypass_expert_output = bypass_expert_output

        if not self.moe_expert_fusion:
            self.experts_group_gemm_node = [
                ExpertsGroupGemmContiguousNode(
                    custom_map,
                    recompute_moe_gate_up=recompute_moe_gate_up,
                    dequant_input=dequant_input,
                    expert_id=expert_id,
                    moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
                    use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
                    fp8=fp8,
                    bypass_down_proj=bypass_expert_output,
                    use_ue8m0=use_ue8m0,
                )
                for expert_id in range(len(custom_map.experts))
            ]
        else:
            self.experts_group_gemm_node = ExpertsGroupGemmContiguousNode(
                custom_map,
                recompute_moe_gate_up=recompute_moe_gate_up,
                dequant_input=dequant_input,
                moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
                use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
                fp8=fp8,
                bypass_down_proj=bypass_expert_output,
                use_ue8m0=use_ue8m0,
            )
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher, bypass_zip=bypass_expert_output)
        self.hs_2d_dispatched_fp8 = None
        self.hs_2d_dispatched_scale = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = self.token_dispatcher._comm_manager.tokens_per_expert_list
        self.padding_token_per_experts = [(x + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN for x in self.tokens_per_expert]
        self.token_offsets = [0]
        for padding_token in self.padding_token_per_experts:
            self.token_offsets.append(self.token_offsets[-1] + padding_token)
        self.router_topk = max_topk
        self.fp8 = fp8
        self.use_ue8m0 = use_ue8m0

    def cached_tensors(self):
        """
        cached tensors
        """
        if self.experts_group_gemm_node is not None:
            if not self.moe_expert_fusion:
                gemm_node_tensors = []
                for gemm_node in self.experts_group_gemm_node:
                    gemm_node_tensors.extend(gemm_node.cached_tensors())
            else:
                gemm_node_tensors = self.experts_group_gemm_node.cached_tensors()
        else:
            gemm_node_tensors = []

        return (
            gemm_node_tensors
            + self.unzip_node.cached_tensors()
            + self.zip_node.cached_tensors()
            + [
                self.hs_2d_dispatched_fp8,
                self.hs_2d_dispatched_scale,
                self.dispatched_indices,
                self.dispatched_probs,
                self.unzipped_probs,
                self.tokens_per_expert,
                self.router_topk,
            ]
        )

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        idx = 0
        if self.experts_group_gemm_node is not None:
            if not self.moe_expert_fusion:
                for expert_id, gemm_node in enumerate(self.experts_group_gemm_node):
                    num = len(gemm_node.cached_tensors())
                    gemm_node.set_cached_tensors(tensors[idx : idx + num])
                    idx += num
            else:
                num = len(self.experts_group_gemm_node.cached_tensors())
                self.experts_group_gemm_node.set_cached_tensors(tensors[idx : idx + num])
                idx += num

        num = len(self.unzip_node.cached_tensors())
        self.unzip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        num = len(self.zip_node.cached_tensors())
        self.zip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        (
            self.hs_2d_dispatched_fp8,
            self.hs_2d_dispatched_scale,
            self.dispatched_indices,
            self.dispatched_probs,
            self.unzipped_probs,
            self.tokens_per_expert,
            self.router_topk,
        ) = tensors[idx:]

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_statue(self):
        """
        重置所有状态变量。

        Args:
            无。

        Returns:
            无。

        """
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = None
        self.release_mem()

    def release_mem(self):
        """
            释放内存，将变量置为None。
        这个函数应该在程序结束时调用，以便释放不再需要的资源。

        Args:
            无参数。

        Returns:
            无返回值，直接修改了类实例中的变量。
        """
        if not self.moe_expert_fusion:
            for node in self.experts_group_gemm_node:
                node.reset_statue()
        else:
            self.experts_group_gemm_node.reset_statue()
        self.experts_group_gemm_node = None

    def subbatch_unzip_and_prepare_gemm_node(self, hs_2d_dispatched, zipped_expertwise_rowmap, expert_id):
        """
        subbatch_unzip_and_prepare_gemm_node
        """
        hs_2d_dispatched, hs_2d_dispatched_scale = hs_2d_dispatched
        (expert_out, expert_out_scale, expert_unzipped_idx) = TDU.tokens_unzip_gather(
            hs_2d_dispatched,
            hs_2d_dispatched_scale,
            zipped_expertwise_rowmap,
            expert_id,
            self.tokens_per_expert,
            FP8_ALIGN,
        )
        gemm_node = self.experts_group_gemm_node[expert_id]
        if self.fp8 is not None:
            gemm_node.input_fp8 = expert_out
            gemm_node.input_scale = expert_out_scale
        else:
            expert_out = paddle.incubate.nn.functional.fused_act_dequant(expert_out, expert_out_scale)
            gemm_node.input = expert_out
        return expert_unzipped_idx

    def gemm_forward_subbatch(
        self, expert_id, unzipped_probs, unzipped_idx, output, total_zipped_tokens, start_idx=None, end_idx=None
    ):
        """
        gemm_forward_subbatch
        """
        gemm_node = self.experts_group_gemm_node[expert_id]
        if start_idx is not None:
            tokens_per_expert = end_idx - start_idx
            padding_token_per_experts = (tokens_per_expert + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
            padding_end_idx = start_idx + padding_token_per_experts

            unzipped_probs = unzipped_probs._slice(start_idx, padding_end_idx)
            unzipped_idx = unzipped_idx._slice(start_idx, end_idx)
            if self.fp8 is not None:
                origin_input_fp8 = gemm_node.input_fp8
                origin_input_scale = gemm_node.input_scale
                gemm_node.input_fp8 = origin_input_fp8._slice(start_idx, padding_end_idx)
                gemm_node.input_scale = origin_input_scale.contiguous()._slice(start_idx, padding_end_idx)
            else:
                origin_input = gemm_node.input
                gemm_node.input = origin_input._slice(start_idx, padding_end_idx)

            gemm_node.tokens_per_expert = [padding_token_per_experts]
        else:
            tokens_per_expert = self.tokens_per_expert[expert_id]
            padding_token_per_experts = self.padding_token_per_experts[expert_id]

        expert_out = gemm_node.forward(
            None,
            unzipped_probs,
            [padding_token_per_experts],
            tokens_per_expert,
        )

        if start_idx is None and self.recompute_moe_permute:
            gemm_node.input_fp8 = None
            gemm_node.input_scale = None
            gemm_node.input_fp8 = None

        if not self.bypass_expert_output:
            output = tokens_zip_unique_add_with_subbatch(
                output,
                expert_out,
                unzipped_idx,
                zipped_rows=total_zipped_tokens,
                subbatch_rows=self.moe_subbatch_token_num_after_dispatch,
            )
        else:
            assert output.size == 0, "output should be empty when using bypass_zip"

        if start_idx is not None:
            if self.fp8 is not None:
                gemm_node.input_fp8 = origin_input_fp8
                gemm_node.input_scale = origin_input_scale
            else:
                gemm_node.input = origin_input
            gemm_node.tokens_per_expert = [self.padding_token_per_experts[expert_id]]

        return output

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        """
        对输入数据进行前向传播计算。

        Args:
            hs_2d_dispatched (Tensor): 表示被分派到各个专家的输入数据。
            dispatched_indices (Tensor):表示输入数据被分派到的专家索引。
            dispatched_probs (Tensor): 表示输入数据被分派到各个专家的概率。

        Returns:
            Tensor: 经过前向传播计算后的输出数据。

        """
        use_fp8_dispatch_a2a = isinstance(hs_2d_dispatched, tuple)

        num_experts = len(self.tokens_per_expert)
        # 1 unzip
        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_scale) = self.unzip_node.forward(
            hs_2d_dispatched,
            self.dispatched_indices,
            dispatched_probs,
            topk=self.router_topk,
            num_experts=num_experts,
            tokens_per_expert=self.tokens_per_expert,
            fill_output=self.moe_expert_fusion,
        )
        self.unzipped_probs = unzipped_probs
        if not self.moe_expert_fusion:
            unzipped_tokens = None

        if use_fp8_dispatch_a2a:
            total_zipped_tokens = hs_2d_dispatched[0].shape[0]
            hidden_size = hs_2d_dispatched[0].shape[-1]
            hs_2d_dispatched[0]._record_stream()
            hs_2d_dispatched[1]._record_stream()
        else:
            total_zipped_tokens = hs_2d_dispatched.shape[0]
            hidden_size = hs_2d_dispatched.shape[-1]
            hs_2d_dispatched._record_stream()
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()
        if self.dispatched_indices.dtype is not dispatched_indices.dtype:
            dispatched_indices._clear_to_zero_allocation()

        if not self.moe_expert_fusion:
            if use_fp8_dispatch_a2a:
                hs_2d_dispatched_fp8, hs_2d_dispatched_scale = hs_2d_dispatched
            else:
                hs_2d_dispatched_fp8, hs_2d_dispatched_scale = tilewise_quant(hs_2d_dispatched)
                hs_2d_dispatched._clear_to_zero_allocation()

            if self.recompute_moe_permute:
                self.hs_2d_dispatched_fp8 = hs_2d_dispatched_fp8
                self.hs_2d_dispatched_scale = hs_2d_dispatched_scale

            output = paddle.empty([0, hidden_size], dtype=paddle.float32)
            for expert_id, tokens_per_expert in enumerate(self.tokens_per_expert):
                expert_unzipped_idx = self.subbatch_unzip_and_prepare_gemm_node(
                    (hs_2d_dispatched_fp8, hs_2d_dispatched_scale),
                    zipped_expertwise_rowmap,
                    expert_id,
                )

                tokens_per_expert = self.tokens_per_expert[expert_id]
                tmp_unzipped_probs = unzipped_probs[self.token_offsets[expert_id] : self.token_offsets[expert_id + 1]]
                if (
                    self.moe_subbatch_token_num_after_dispatch is not None
                    and self.moe_subbatch_token_num_after_dispatch > 0
                    and tokens_per_expert > self.moe_subbatch_token_num_after_dispatch
                ):
                    nparts = (
                        tokens_per_expert + self.moe_subbatch_token_num_after_dispatch - 1
                    ) // self.moe_subbatch_token_num_after_dispatch
                    for i in range(nparts):
                        start_idx = i * self.moe_subbatch_token_num_after_dispatch
                        end_idx = min(start_idx + self.moe_subbatch_token_num_after_dispatch, tokens_per_expert)
                        output = self.gemm_forward_subbatch(
                            expert_id,
                            tmp_unzipped_probs,
                            expert_unzipped_idx,
                            output,
                            total_zipped_tokens,
                            start_idx=start_idx,
                            end_idx=end_idx,
                        )

                    if self.recompute_moe_permute:
                        gemm_node = self.experts_group_gemm_node[expert_id]
                        gemm_node.input_fp8 = None
                        gemm_node.input_scale = None
                else:
                    nparts = 1
                    output = self.gemm_forward_subbatch(
                        expert_id, tmp_unzipped_probs, expert_unzipped_idx, output, total_zipped_tokens
                    )

                del expert_unzipped_idx
                del tmp_unzipped_probs

            if use_fp8_dispatch_a2a:
                expected_output_dtype = paddle.bfloat16
            else:
                expected_output_dtype = hs_2d_dispatched.dtype

            expert_out = merge_subbatch_cast(output, expected_output_dtype)
            del output
        else:
            if not use_fp8_dispatch_a2a:
                hs_2d_dispatched._clear_to_zero_allocation()
            # 2 experts
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens,
                unzipped_probs,
                self.padding_token_per_experts,
                self.tokens_per_expert,
                output=unzipped_tokens,
                scale=unzipped_scale,  # maybe None
            )

            # 3 zip
            expert_out = expert_out.reshape([-1, expert_out.shape[-1]])

            expert_out = self.zip_node.forward(
                expert_out,
                zipped_expertwise_rowmap,
                self.dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=total_zipped_tokens,
                num_experts=num_experts,
            )

        self.dispatched_probs = dispatched_probs
        expert_out.stop_gradient = False

        return expert_out

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        """
        反向传播函数。

        Args:
            hidden_states_out_grad (Tensor): 隐藏状态梯度。

        Returns:
            Tuple[Tensor, Tensor]: 包含两个元素，分别为hs_fp8_dispatched_grad和dispatched_probs_grad。
                - hs_fp8_dispatched_grad (Tensor): 解压后的隐藏状态梯度。
                - dispatched_probs_grad (Tensor): 分发概率梯度。

        """
        # zip_grad
        hidden_states_out_grad_shape = hidden_states_out_grad.shape
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
            fill_output=self.moe_expert_fusion,
        )
        if get_env_device() != "xpu":
            hidden_states_out_grad._record_stream()

        if not self.moe_expert_fusion:
            output = paddle.empty([0, hidden_states_out_grad_shape[-1]], dtype=paddle.float32)
            probs_grad_list = []

            for expert_id, tokens_per_expert in enumerate(self.tokens_per_expert):
                (unzipped_grad, _, unzipped_grad_idx) = TDU.tokens_unzip_gather(
                    hidden_states_out_grad,
                    None,
                    self.unzip_node.zipped_expertwise_rowmap,
                    expert_id=expert_id,
                    tokens_per_expert=self.tokens_per_expert,
                    padding_multiplex=FP8_ALIGN,
                )

                if self.recompute_moe_permute:
                    self.subbatch_unzip_and_prepare_gemm_node(
                        (self.hs_2d_dispatched_fp8, self.hs_2d_dispatched_scale),
                        self.unzip_node.zipped_expertwise_rowmap,
                        expert_id,
                    )

                gemm_node = self.experts_group_gemm_node[expert_id]
                unzipped_grad, unzipped_probs_grad = self.experts_group_gemm_node[expert_id].backward(
                    unzipped_grad,
                    self.unzipped_probs[self.token_offsets[expert_id] : self.token_offsets[expert_id + 1]],
                )
                output = tokens_zip_unique_add_with_subbatch(
                    output,
                    unzipped_grad,
                    unzipped_grad_idx,
                    zipped_rows=hidden_states_out_grad_shape[0],
                    subbatch_rows=self.moe_subbatch_token_num_after_dispatch,
                )
                if len(unzipped_probs_grad.shape) > 1:
                    unzipped_probs_grad = unzipped_probs_grad.squeeze(-1)
                assert len(unzipped_probs_grad.shape) == 1, unzipped_probs_grad.shape
                probs_grad_list.append(unzipped_probs_grad)
                del unzipped_grad
                del unzipped_grad_idx
                del unzipped_probs_grad

            hidden_states_out_grad._clear_to_zero_allocation()
            hs_fp8_dispatched_grad = merge_subbatch_cast(output, hidden_states_out_grad.dtype)
            del output
            dispatched_probs_grad = TDU.tokens_zip_prob(
                probs_grad_list, self.unzip_node.zipped_expertwise_rowmap, self.dispatched_indices
            )
        else:
            hidden_states_out_grad._clear_to_zero_allocation()

            # expert_grad
            expert_out, probs_grad = self.experts_group_gemm_node.backward(unzipped_grad, self.unzipped_probs)
            del unzipped_grad

            hs_fp8_dispatched_grad, dispatched_probs_grad = self.unzip_node.backward(
                expert_out,
                hidden_states_out_grad_shape,
                probs_grad,
                self.dispatched_indices,
                num_experts=len(self.tokens_per_expert),
            )
        self.reset_statue()
        return hs_fp8_dispatched_grad, dispatched_probs_grad


# ---------------------------------------------------------------------------
# Fp8FusedMoeFunc
# ---------------------------------------------------------------------------
class Fp8FusedMoeFunc(paddle.autograd.PyLayer):
    """
    The Fp8FusedMoeFunc class includes operations for unzipping, expert computation, and zipping.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        dispatched_indices,
        custom_map,
        max_topk,
        recompute_moe_gate_up=False,
        dequant_input=False,
        moe_expert_fusion=True,
        recompute_moe_permute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        is_first_fwd=False,
        fp8_dispatched_handle=None,
        fp8="e4m3",
        bypass_expert_output=False,
        use_ue8m0=False,
    ):
        """
        根据给定的参数执行前向传播操作。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            dispatched_probs (tensor): 分派概率张量。
            dispatched_indices (tensor): 分派索引张量。
            max_topk (int): topk。

        Returns:
            tensor: 前向传播的结果张量。
        """
        ctx.node = MlpNode(
            custom_map,
            max_topk,
            recompute_moe_gate_up=recompute_moe_gate_up,
            dequant_input=dequant_input,
            moe_expert_fusion=moe_expert_fusion,
            recompute_moe_permute=recompute_moe_permute,
            moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
            use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
            fp8=fp8,
            bypass_expert_output=bypass_expert_output and (not is_first_fwd),
            use_ue8m0=use_ue8m0,
        )

        if fp8_dispatched_handle is not None:
            assert hidden_states.dtype == paddle.float8_e4m3fn
            scale = fp8_dispatched_handle["scale"]
            hidden_states = (hidden_states, scale)

        out = ctx.node.forward(hidden_states, dispatched_indices, dispatched_probs)

        if is_first_fwd:
            ctx.node.release_mem()

        cached_tensors = ctx.node.cached_tensors()
        ctx.save_for_backward(cached_tensors)
        ctx.node.clear_cached_tensors()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """
        计算反向传播梯度。

        Args:
            output_grad (Tensor): 输出梯度张量。

        Returns:
            Tuple[Tensor, Tensor, None]: 返回三个梯度张量，前两个分别是隐藏状态和派发概率的梯度，
                                            第三个为None，表示没有需要传递给更前向节点的梯度。

        """
        (cached_tensors,) = ctx.saved_tensor()
        ctx.node.set_cached_tensors(cached_tensors)
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(output_grad)
        return hidden_states_grad, dispatched_probs_grad, None


# ---------------------------------------------------------------------------
# DeepEPMOELayer
# ---------------------------------------------------------------------------
class DeepEPMOELayer(nn.Layer):
    """
    MoE 层，使用 DeepEP 进行通信
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        enable_logging=False,
        k=2,
        enable_bpr=False,
        all_to_all_dropout=False,
        group_experts=False,
        moe_statics=None,
    ):
        """
        初始化MoE层。

        Args:
            gate (nn.Layer): 智能门控层，用于选择需要使用的专家。
            experts (List[nn.Layer]): 需要使用的专家列表。
            layer_idx (int): 当前MoE层的索引。
            group (Group): 分布式通信组。默认值为None。
            recompute (bool): 是否在每个训练迭代中重新计算MoE输出。默认值为False。
        """
        super().__init__()
        self.gate = gate
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)
        self.zero.stop_gradient = True

        if isinstance(self.config.n_routed_experts, (list, tuple)):
            self.global_num_experts = sum(self.config.n_routed_experts)
        else:
            self.global_num_experts = self.config.n_routed_experts
        self.routed_scaling_factor = self.config.routed_scaling_factor  # router_scaling_factor
        logger.info(f"using router_scaling_factor={self.routed_scaling_factor}")

        assert not enable_bpr, "enable bpr is not supported now."
        assert not all_to_all_dropout, "all_to_all_dropout is not supported now."

        self.enable_logging = enable_logging
        self.enable_bpr = enable_bpr
        self.all_to_all_dropout = all_to_all_dropout

        self.layer_idx = layer_idx
        self.recompute = recompute
        logger.info(f"using moe recompute={recompute}")

        self.use_rr_deepep_combine = self.config.use_recompute and self.config.skip_recompute_ops.get(
            "moe_combine", False
        )
        self.use_rr_expert_and_combine = self.config.use_recompute and self.config.skip_recompute_ops.get(
            "expert_and_combine", False
        )
        if self.use_rr_expert_and_combine:
            # Normal forward:
            #   dispatch → unzip -> up_gate_proj → down_proj → zip → combine

            # Recompute with RR:
            #   First Fwd:
            #     dispatch → unzip -> up_gate_proj → down_proj → zip → combine -> cache_combine_outputs
            #   Recompute Fwd:
            #     dispatch → unzip -> up_gate_proj -> use_cached_combine_outputs
            self.use_rr_deepep_combine = True
            self.use_rr_bypass_expert_output = True
        else:
            self.use_rr_bypass_expert_output = False

        if self.config.use_ep_comm_overlap:
            assert (
                not self.use_rr_deepep_combine
            ), "ep_comm_overlap is not supported with rr_deepep_combine, please set use_ep_comm_overlap=False"
            assert (
                not self.use_rr_expert_and_combine
            ), "ep_comm_overlap is not supported with rr_expert_and_combine, please set use_ep_comm_overlap=False"

        for p in self.gate.parameters():
            p.is_gate = True
        if isinstance(experts, nn.LayerList):
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = experts
        self.shared_experts = shared_experts

        self.group = group
        self.k = k
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        self.recompute_decoder_chunk = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "decoder_chunk" in self.config.recompute_modules
        )

        if self.config.moe_subbatch_token_num_after_dispatch is not None:
            logger.info("recompute_moe_gate_up will be enabled when moe_subbatch_token_num_after_dispatch > 0")

        if self.use_correction_bias:
            logger.info(f"using correction bias, aux-coef:{self.gate.config.router_aux_loss_coef}")
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg") and group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        self.is_ep_moe = (
            hasattr(fleet.fleet, "_hcg")
            and hasattr(fleet.get_hybrid_communicate_group(), "get_moe_sharding_parallel_world_size")
            and fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_world_size() > 0
        )
        for p in experts.parameters():
            # p.no_sync 如果设True, 初始化权重时会使用local_seed(处处不一样); 如果设False, 初始化权重时会使用model_parallel_rng(在DP间一样)
            # 所以这里的逻辑是:
            #   如果是tp-moe, 应使用model_parallel_rng, 保证权重在DP间一样(框架保证)
            #   如果是dp-moe, 应使用local_seed, 保证权重处处不一样
            #   如果是ep-moe, 应保证权重在moe-dp组内相同(框架保证), 在ep组内不同
            # 此外, 框架中, 如果参数的no_sync为True, 在sync_params_buffers时会跳过。
            p.no_sync = not self.is_mp_moe
            # 在框架broadcast_moe_sharding_parameter或broadcast_moe_dp_parameter中,
            # 如果参数的expert=True, 会强制在moe_sharding_group/moe_dp_group同步参数
            p.expert = not self.is_mp_moe
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = len(self.experts)

        self.dispatcher = MoEFlexTokenDispatcher(self.num_local_experts, self.global_num_experts, self.group)

        if hasattr(fleet.fleet, "_hcg"):
            hcg = fleet.get_hybrid_communicate_group()
            if hasattr(hcg, "get_moe_sharding_parallel_world_size") and hcg.get_moe_sharding_parallel_world_size() > 0:
                moe_grad_group = hcg.get_moe_sharding_parallel_group()
                for p in self.experts.parameters():
                    setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})

        self.use_node_limit = self.config.n_group != 0 and self.config.topk_group != 0
        if self.use_node_limit:
            self.one = paddle.ones([], dtype="float32")

        if self.use_correction_bias:
            self.zeros_like_correction_bias = paddle.zeros_like(self.moe_statics.e_score_correction_bias[0].detach())

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True, use_ue8m0=False):
        """Quantize weights in FP8 format.

        Args:
            batch_mode: If True, quantize all weights in batch mode using the first expert's weights.
                    If False, quantize each expert's weights individually.
        """

        def quantize_weights(weight_list, weight_obj=None, quant_transpose=True, use_ue8m0=False):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]

            use_pow2_scale = False
            if paddle.device.cuda.get_device_capability()[0] == 10:
                # Blackwell GPUs require the use of pow2_scales quantization.
                use_pow2_scale = True

            # Quantize without transpose
            if hasattr(TDU, "fuse_stack_fp8_quant"):
                fp8_weight, fp8_scale = TDU.fuse_stack_fp8_quant(
                    weight_list,
                    use_pow2_scale,
                    use_ue8m0,
                    use_ue8m0,
                )
                if use_ue8m0:
                    fp8_scale = fp8_scale.T
            else:
                fp8_weight, fp8_scale = FusedQuantOps.fused_stack_quant(weight_list)

            setattr(weight_obj, "fp8_weight_stacked", fp8_weight)
            setattr(weight_obj, "fp8_scale_stacked", fp8_scale)

            if quant_transpose:
                # Quantize with transpose
                if hasattr(TDU, "fuse_stack_transpose_fp8_quant"):
                    fp8_weight_t, fp8_scale_t = TDU.fuse_stack_transpose_fp8_quant(
                        weight_list,
                        use_pow2_scale,
                        use_ue8m0,
                        use_ue8m0,
                    )
                    if use_ue8m0:
                        fp8_scale_t = fp8_scale_t.T
                else:
                    fp8_weight_t, fp8_scale_t = FusedQuantOps.fused_stack_transpose_quant(weight_list)
            else:
                fp8_weight_t, fp8_scale_t = None, None

            setattr(weight_obj, "fp8_weight_stacked_transpose", fp8_weight_t)
            setattr(weight_obj, "fp8_scale_stacked_transpose", fp8_scale_t)

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [expert.up_gate_proj.weight for expert in self.experts if expert is not None]
            expert_w2_list = [expert.down_proj.weight for expert in self.experts if expert is not None]

            if expert_w1_list:
                quantize_weights(
                    expert_w1_list, expert_w1_list[0], quant_transpose=quant_transpose, use_ue8m0=use_ue8m0
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list, expert_w2_list[0], quant_transpose=quant_transpose, use_ue8m0=use_ue8m0
                )
        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight], quant_transpose=quant_transpose, use_ue8m0=use_ue8m0
                    )
                    quantize_weights([expert.down_proj.weight], quant_transpose=quant_transpose, use_ue8m0=use_ue8m0)

    def cal_norm_combine_weights(self, combine_weights_unnorm):
        """
        normalize combine weights
        """
        if self.gate.norm_gate_logits:
            combine_weights = combine_weights_unnorm / paddle.clip(
                combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            combine_weights = combine_weights_unnorm
        return combine_weights

    def calc_and_log_moe_summary(self, key, summary_data, is_numpy=False):
        """算max min等并记录"""
        if not is_numpy:
            summary_data = summary_data.numpy()

        max_value = max(summary_data)
        min_value = min(summary_data)
        var_value = np.var(summary_data)
        median_value = np.median(summary_data)
        mean_value = np.mean(summary_data)
        if mean_value == 0:
            assert max_value == 0, max_value
            assert min_value == 0, min_value

        prefix = f"{key}_layer_{self.layer_idx}"

        _log = {
            f"{prefix}_max": max_value,
            f"{prefix}_min": min_value,
            f"{prefix}_var": var_value,
            f"{prefix}_median": median_value,
            f"{prefix}_mean": mean_value,
            f"{prefix}_max_mean_ratio": max_value / mean_value if mean_value != 0 else 1.0,
            f"{prefix}_min_mean_ratio": min_value / mean_value if mean_value != 0 else 1.0,
        }
        global_training_logs.update(**_log)

    def moe_tokens_per_experts_indicator(self, key, summary_data, count):
        """统计每个专家处理的token数相关指标"""
        dist.all_reduce(summary_data, group=self.group)

        hcg = fleet.get_hybrid_communicate_group()
        tp_world_size = hcg.get_model_parallel_world_size()
        # if tp_group not equal to expert_group, add count inter batch
        if tp_world_size != self.world_size:
            dist.all_reduce(count, group=self.group)
            count /= tp_world_size

        count = paddle.ones_like(count) if count.item() == 0 else count
        summary_data_avg = paddle.cast(summary_data, dtype=count.dtype) / count

        # avg-summary
        self.calc_and_log_moe_summary(key + "_avg", summary_data_avg)

        # origin
        self.calc_and_log_moe_summary(key, summary_data)

    def node_limit_routing(self, gate_probs):
        """
        将所有专家分组, 只在topk_group个group内选择专家
        """
        assert len(gate_probs.shape) == 2
        seq_length, n_experts = gate_probs.shape
        assert n_experts % self.config.n_group == 0, "n_experts must be divisible by n_groups"

        group_scores = (
            gate_probs.reshape([seq_length, self.config.n_group, -1]).topk(2, axis=-1)[0].sum(axis=-1)
        )  # [n, n_group]
        group_idx = paddle.topk(group_scores, k=self.config.topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, self.one, axis=-1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([seq_length, self.config.n_group, n_experts // self.config.n_group])
            .reshape([seq_length, -1])
        )  # [n, e]
        gate_probs = gate_probs.masked_fill(~score_mask.astype(paddle.bool), float("-inf"))
        return gate_probs

    def gate_score(self, input, global_gate_mask=None, is_diff_expert_num=False, input_ids=None):
        """
        Gating function, compute the probs of each token to
        be dispatched to each expert

        Args:
            input: Tensor[S, H]
            global_gate_mask: Tensor[E]
        Returns:
            gate_logits: Tensor[S, E]
            gate_probs: Tensor[S, E]
            topk_normed_probs: Tensor[S, K]
            topk_indices: Tensor[S, K]
            router_loss: Tensor[1]
        """
        assert len(input.shape) == 2
        assert isinstance(self.gate, DeepEPTop2Gate)

        (
            gate_logits,
            router_loss,
        ) = self.gate(input, global_gate_mask=global_gate_mask)

        gate_probs = self.gate.act(gate_logits)

        if input_ids is not None and self.config.gate_force_zero_padding_grad:
            assert input_ids.shape[0] == gate_logits.shape[0], f"check input_ids shape {input_ids.shape}"
            valid_mask = (input_ids != 0).astype(paddle.float32).unsqueeze(-1)
            gate_logits = gate_logits * valid_mask
            gate_probs = gate_probs * valid_mask

        if self.use_correction_bias:
            # NOTE: e_score_correction_bias只能影响topk选择的indices, 而不能影响auxloss、gate_probs等数值
            assert self.moe_statics.e_score_correction_bias is not None, "e_score_correction_bias is None"
            if is_diff_expert_num:
                inf_mask = paddle.isinf(global_gate_mask) & (global_gate_mask < 0)  # shape [E], dtype: bool
                correction_bias = self.moe_statics.e_score_correction_bias[0].detach()
                fixed_correction_bias = paddle.where(inf_mask, self.zeros_like_correction_bias, correction_bias)
                probs_for_choice = gate_probs + fixed_correction_bias

            else:
                probs_for_choice = gate_probs + self.moe_statics.e_score_correction_bias[0].detach()
        else:
            probs_for_choice = gate_probs

        return gate_logits, gate_probs, probs_for_choice, router_loss

    @dispatch_to(
        lambda self, *args, **kwargs: FusedMoETopk.apply(*args, **kwargs),
        cond=lambda self, *args, **kwargs: self.config.fused_moe_topk,
    )
    def calc_topk_probs_indices(
        self, gate_probs, probs_for_choice, moe_k, use_node_limit, n_group, topk_group, norm_gate_logits
    ):
        """
        计算 topk 概率和索引，使用 fused MoE 操作加速。

        Args:
            gate_probs: 原始 gate 概率，shape [seq_len, n_experts]
            probs_for_choice: 用于选择专家的概率（可能包含 correction bias），shape [seq_len, n_experts]
            moe_k: 每个 token 选择的专家数量
            use_node_limit: 是否使用节点限制
            n_group: 专家分组数量
            topk_group: 选择的 topk 分组数量
            norm_gate_logits: 是否对 gate logits 进行归一化

        Returns:
            topk_normed_probs: 归一化后的 topk 概率，shape [seq_len, moe_k]
            topk_indices: topk 专家索引，shape [seq_len, moe_k]
        """
        if use_node_limit:
            # 将所有专家分组, 只在topk_group个group内选择专家
            assert len(probs_for_choice.shape) == 2
            seq_length, n_experts = probs_for_choice.shape
            assert n_experts % n_group == 0, "n_experts must be divisible by n_groups"

            group_scores = (
                probs_for_choice.reshape([seq_length, n_group, -1]).topk(2, axis=-1)[0].sum(axis=-1)
            )  # [n, n_group]
            group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
            group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, 1.0, axis=-1)
            score_mask = (
                group_mask.unsqueeze(-1).expand([seq_length, n_group, n_experts // n_group]).reshape([seq_length, -1])
            )  # [n, e]
            probs_for_choice = probs_for_choice.masked_fill(~score_mask.astype(paddle.bool), float("-inf"))

        _, topk_indices = paddle.topk(probs_for_choice, moe_k, axis=-1)
        topk_probs = paddle.take_along_axis(gate_probs, topk_indices, axis=-1)
        topk_indices.stop_gradient = True

        # normalize combine weights
        if norm_gate_logits:
            topk_normed_probs = topk_probs / paddle.clip(topk_probs.sum(-1, keepdim=True), min=1e-12)
        else:
            topk_normed_probs = topk_probs

        return topk_normed_probs, topk_indices

    @dispatch_to(
        lambda self, *args, **kwargs: routing_map_forward(*args, **kwargs),
        cond=lambda self, *args, **kwargs: self.config.fused_routing_map,
    )
    def routing_map_forward(self, gate_probs, topk_indices, input_ids, is_pure_text_line):
        """
        计算路由映射和分发掩码，使用 fused MoE 操作加速。

        Args:
            gate_probs: gate 概率，shape [seq_len, n_experts]
            topk_indices: topk 专家索引，shape [seq_len, moe_k]
            input_ids: 输入 token ids，用于 padding mask，shape [seq_len]
            is_pure_text_line: 纯文本行 mask，用于多模态场景，shape [seq_len]

        Returns:
            routing_map: 路由映射矩阵，shape [seq_len, n_experts]
            topk_indices: 更新后的 topk 专家索引（可能被 mask）
            dispatch_mask: 分发掩码，shape [n_experts]，表示每个专家的 token 数量
        """
        routing_map = (
            paddle.zeros_like(gate_probs)
            .put_along_axis(topk_indices, paddle.full([], fill_value=1.0, dtype=gate_probs.dtype), axis=1)
            .astype("bool")
        )
        if input_ids is not None:
            # has_padding = (input_ids == 0).any()
            valid_mask = input_ids != 0
            valid_mask = valid_mask.unsqueeze(-1)
            routing_map = routing_map * valid_mask.cast(routing_map.dtype)
            # -1 means neither participates in routing nor expert calculation
            topk_indices = topk_indices.masked_fill(~valid_mask, -1)

        if is_pure_text_line is not None:
            routing_map = routing_map * is_pure_text_line.cast(routing_map.dtype)
            # -1 means neither participates in routing nor expert calculation
            topk_indices = topk_indices.masked_fill(~is_pure_text_line.astype(paddle.bool), -1)

        dispatch_mask = paddle.sum(routing_map.cast(paddle.int64), axis=0)
        return routing_map, topk_indices, dispatch_mask

    def topk(
        self,
        num_experts_per_tok,
        gate_logits,
        gate_probs,
        probs_for_choice,
        is_diff_topk=False,
        is_diff_expert_num=False,
        input_ids=None,
        is_pure_text_line=None,
    ):
        """topk_and_norm"""
        # NOTE: e_score_correction_bias只能影响topk选择的indices, 而不能影响auxloss、gate_probs等数值
        # 所以有gate_probs和probs_for_choice的区别
        topk_normed_probs, topk_indices = self.calc_topk_probs_indices(
            gate_probs,
            probs_for_choice,
            num_experts_per_tok,
            self.use_node_limit,
            self.config.n_group,
            self.config.topk_group,
            self.gate.norm_gate_logits,
        )
        topk_indices.stop_gradient = True
        routing_map, topk_indices, dispatch_mask = self.routing_map_forward(
            gate_probs, topk_indices, input_ids, is_pure_text_line
        )

        if self.routed_scaling_factor:
            # align with ds-v3
            # Ref to https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L471
            topk_normed_probs = topk_normed_probs * self.routed_scaling_factor

        if self.use_correction_bias and not (is_diff_topk or is_diff_expert_num):
            # 避免recompute的时候重复统计
            if framework._dygraph_tracer()._has_grad:
                self.moe_statics.expert_usage[0] += dispatch_mask.detach()
        return topk_normed_probs, topk_indices, routing_map, dispatch_mask

    def pad_for_elastic(self, max_topk, topk_probs, topk_indices):
        """pad_for_elastic"""
        assert len(topk_probs.shape) == 2
        assert len(topk_indices.shape) == 2
        assert topk_probs.shape[-1] == topk_indices.shape[-1]

        if topk_probs.shape[-1] == max_topk:
            return topk_probs, topk_indices

        num_to_pad = max_topk - topk_probs.shape[-1]
        padded_topk_probs = F.pad(topk_probs, (0, 0, 0, num_to_pad), value=0, mode="constant")
        padded_topk_indices = F.pad(topk_indices, (0, 0, 0, num_to_pad), value=-1, mode="constant")
        return padded_topk_probs, padded_topk_indices

    def forward(
        self,
        input: Tensor,
        input_ids: Tensor,
        token_type_ids=None,  # not used
        elastic_topk_value=None,
        global_gate_mask=None,
        is_diff_expert_num=False,
        is_diff_topk=False,
        max_topk=None,  # 保证ep组内topk相等,弹性topk时需padding到max_topk
        origin_input_ids=None,
        is_pure_text_line=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            input (`Tensor`): The input data with shape ``(s, d)``.
                Only one token is supported for now.
        Returns:
            output (`Tensor`): The final output tensor with shape ``(s, d)`` where ``m`` is the
                size of model parameters.
            combine_weights (`Tensor`, optional): A tensor with shape ``(s,)``, which represents weights
                for each expert in MoE.
            router_loss (`Tensor`, optional): A scalar tensor representing the loss of routing function.
        """
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        assert int(self.all_to_all_dropout) == 0, "all_to_all_dropout is not supported yet."
        assert token_type_ids is None, "token_type_ids is not supported yet."
        assert self.gate is not None

        # NOTE: Make sure that the data in each rank within the moe groups is different.

        # gate
        gate_logits, gate_probs, probs_for_choice, router_loss = self.gate_score(
            input, global_gate_mask=global_gate_mask, is_diff_expert_num=is_diff_expert_num, input_ids=input_ids
        )
        if elastic_topk_value is None:
            self.elastic_topk_value = self.k
        else:
            self.elastic_topk_value = elastic_topk_value

        if self.config.context_parallel_size > 1 and self.config.sequence_parallel:
            self.bsz = (
                input.shape[0]
                * self.config.tensor_model_parallel_size
                * self.config.context_parallel_size
                // self.config.seqlen
            )
        elif self.config.context_parallel_size > 1:
            self.bsz = input.shape[0] * self.config.context_parallel_size // self.config.seqlen
        elif self.config.sequence_parallel:
            self.bsz = input.shape[0] * self.config.tensor_model_parallel_size // self.config.seqlen
        else:
            self.bsz = input.shape[0]

        no_elastic_summary, elastic_topk_summary, elastic_expert_summary = (
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
        )
        no_elastic_num, elastic_topk_num, elastic_expert_num = (
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
        )

        ori_topk_probs, ori_topk_indices, routing_map, dispatch_mask = self.topk(
            self.elastic_topk_value,
            gate_logits,
            gate_probs,
            probs_for_choice,
            input_ids=input_ids,
            is_diff_topk=is_diff_topk,
            is_diff_expert_num=is_diff_expert_num,
            is_pure_text_line=is_pure_text_line,
        )

        if max_topk is not None:
            assert max_topk <= 32, f"max_topk {max_topk} exceeds 32, which is not supported yet."
            topk_probs, topk_indices = self.pad_for_elastic(max_topk, ori_topk_probs, ori_topk_indices)
        else:
            topk_probs, topk_indices = ori_topk_probs, ori_topk_indices
        topk_indices.stop_gradient = True

        if not (is_diff_topk or is_diff_expert_num):
            no_elastic_summary += dispatch_mask
            no_elastic_num += self.bsz
        if is_diff_topk:
            elastic_topk_summary += dispatch_mask
            elastic_topk_num += self.bsz
        if is_diff_expert_num:
            elastic_expert_summary += dispatch_mask
            elastic_expert_num += self.bsz

        if self.config.use_ep_comm_overlap:
            dispatch_overlap_handle = {
                "fn": self.calc_router_loss_and_logging,
                "fn_args": (
                    router_loss,
                    gate_logits,
                    gate_probs,
                    routing_map,
                    dispatch_mask,
                    input_ids,
                    is_diff_topk,
                    is_diff_expert_num,
                    origin_input_ids,
                    is_pure_text_line,
                ),
            }
        else:
            router_loss2 = self.calc_router_loss_and_logging(
                router_loss,
                gate_logits=gate_logits,
                gate_probs=gate_probs.clone(),  # 为了确保ep_comm_overlap开和关时, gate_probs梯度累加顺序相同
                routing_map=routing_map,
                dispatch_mask=dispatch_mask,
                input_ids=input_ids,
                is_diff_topk=is_diff_topk,
                is_diff_expert_num=is_diff_expert_num,
                origin_input_ids=origin_input_ids,
                is_pure_text_line=is_pure_text_line,
            )
            dispatch_overlap_handle = None

        if self.shared_experts is not None and self.config.use_ep_comm_overlap:
            if self.recompute_decoder_chunk:
                # chunk_recompute下, mlp不做重计算。为避免oom, 单独给shared_experts使用重计算
                combine_overlap_handle = {"fn": partial(recompute, self.shared_experts), "fn_args": (input,)}
            else:
                combine_overlap_handle = {"fn": self.shared_experts, "fn_args": (input,)}
        else:
            combine_overlap_handle = None

        # calc and log summary
        is_first_fwd = framework._dygraph_tracer()._has_grad
        if is_first_fwd and self.enable_logging and global_moe_balance_training_logs_enabled():
            self.moe_tokens_per_experts_indicator("no_elastic", no_elastic_summary, no_elastic_num)
            self.moe_tokens_per_experts_indicator("elastic_topk", elastic_topk_summary, elastic_topk_num)
            self.moe_tokens_per_experts_indicator("elastic_expert", elastic_expert_summary, elastic_expert_num)

        if max_topk is not None:
            dispatch_topk = max_topk
        else:
            dispatch_topk = self.elastic_topk_value

        if self.config.use_fuse_node:
            with profile("dispatch"):
                # fp8_dispatched_handle里装的是scale, 为了避免污染计算图, 用dict形式的handle进行传递
                (
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    fp8_dispatched_handle,
                ) = self.dispatcher._comm_manager.dispatch(
                    input,
                    topk_indices,
                    topk_probs,
                    fp8_dispatch_a2a=self.config.use_fp8_dispatch_a2a,
                    inner_layer_overlap_handle=dispatch_overlap_handle,
                    use_ue8m0=self.config.use_ue8m0,
                )

            with profile("fusion_mlp"):
                hidden_states_tmp = Fp8FusedMoeFunc.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    dispatch_topk,
                    recompute_moe_gate_up=recompute_moe_gate_up_func(self.config, self.layer_idx),
                    dequant_input=self.config.moe_dequant_input,
                    moe_expert_fusion=self.config.moe_expert_fusion,
                    recompute_moe_permute=(
                        self.config.recompute_granularity == "selective"
                        and self.config.recompute_modules is not None
                        and "moe_permute" in self.config.recompute_modules
                    ),
                    moe_subbatch_token_num_after_dispatch=self.config.moe_subbatch_token_num_after_dispatch,
                    use_bf16_gemm_weight_grad=not self.config.fp8_wgrad,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    fp8=self.config.fp8,
                    bypass_expert_output=self.use_rr_bypass_expert_output,
                    use_ue8m0=self.config.use_ue8m0,
                )

            with profile("combine"):  # combine
                combined_output = self.dispatcher._comm_manager.combine(
                    hidden_states_tmp,
                    inner_layer_overlap_handle=combine_overlap_handle,
                    use_rr_deepep_combine=self.use_rr_deepep_combine,
                )

            self.dispatcher._comm_manager.handle = None
            del hidden_states_tmp
        elif self.config.deepep_fine_grained:
            use_fp8_fuse_node = self.config.use_fuse_node and self.config.fp8 is not None
            assert not use_fp8_fuse_node, "Deepep_fine_grained is not supported for fp8 yet."
            # global dispatch
            dispatched_output, dispatched_indices, dispatched_probs, _ = self.dispatcher._comm_manager.dispatch(
                input,
                topk_indices,
                topk_probs,
                inner_layer_overlap_handle=dispatch_overlap_handle,
            )

            # local dispatch & forward_experts & local combine
            output_tokens = self.fine_grained_forward_experts(
                dispatched_output, dispatched_probs, dispatched_indices, dispatch_topk
            )

            # global combine
            combined_output = self.dispatcher._comm_manager.combine(
                output_tokens, inner_layer_overlap_handle=combine_overlap_handle
            )
        else:
            # dispatch
            (
                dispatched_input,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
            ) = self.dispatcher.token_permutation(
                input, topk_indices, topk_probs, dispatch_topk, inner_layer_overlap_handle=dispatch_overlap_handle
            )

            # ffn
            expert_out = self.forward_experts(dispatched_input)

            # combine
            combined_output = self.dispatcher.token_unpermutation(
                expert_out,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
                deepep_use_fused=self.config.deepep_use_fused,
                inner_layer_overlap_handle=combine_overlap_handle,
            )

        if self.config.use_ep_comm_overlap:
            router_loss2 = dispatch_overlap_handle["fn_out"][0]

        # shared_experts
        if self.shared_experts is not None:
            if self.config.use_ep_comm_overlap:
                shared_out = combine_overlap_handle["fn_out"][0]
            elif self.recompute_decoder_chunk:
                shared_out = recompute(self.shared_experts, input)
            else:
                shared_out = self.shared_experts(input)
            combined_output += shared_out

        if is_first_fwd and self.enable_logging and global_moe_balance_training_logs_enabled():
            num_tokens_per_expert = self.get_num_tokens_per_expert()
            total_num_tokens_per_expert = paddle.full([], fill_value=sum(num_tokens_per_expert))
            total_list = paddle.empty([self.group.world_size], dtype=total_num_tokens_per_expert.dtype)
            dist.stream.all_gather(total_list, total_num_tokens_per_expert, group=self.group)
            self.calc_and_log_moe_summary("local_tokens_per_card", total_list)

        if orig_shape:
            combined_output = combined_output.clone().reshape(orig_shape[:-1] + [combined_output.shape[-1]])
        return combined_output, topk_probs, router_loss2, gate_logits

    def gate_compute(
        self,
        input: Tensor,
        input_ids: Tensor,
        token_type_ids=None,  # not used
        elastic_topk_value=None,
        global_gate_mask=None,
        is_diff_expert_num=False,
        is_diff_topk=False,
        max_topk=None,  # 保证ep组内topk相等,弹性topk时需padding到max_topk
        origin_input_ids=None,
        is_pure_text_line=None,
    ):
        """decomposed gate compute for overlap"""
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        assert int(self.all_to_all_dropout) == 0, "all_to_all_dropout is not supported yet."
        assert token_type_ids is None, "token_type_ids is not supported yet."
        assert self.gate is not None

        # NOTE: Make sure that the data in each rank within the moe groups is different.

        # gate
        gate_logits, gate_probs, probs_for_choice, router_loss = self.gate_score(
            input, global_gate_mask=global_gate_mask, is_diff_expert_num=is_diff_expert_num, input_ids=input_ids
        )
        if elastic_topk_value is None:
            self.elastic_topk_value = self.k
        else:
            self.elastic_topk_value = elastic_topk_value

        if self.config.context_parallel_size > 1 and self.config.sequence_parallel:
            self.bsz = (
                input.shape[0]
                * self.config.tensor_model_parallel_size
                * self.config.context_parallel_size
                // self.config.seqlen
            )
        elif self.config.context_parallel_size > 1:
            self.bsz = input.shape[0] * self.config.context_parallel_size // self.config.seqlen
        elif self.config.sequence_parallel:
            self.bsz = input.shape[0] * self.config.tensor_model_parallel_size // self.config.seqlen
        else:
            self.bsz = input.shape[0]

        no_elastic_summary, elastic_topk_summary, elastic_expert_summary = (
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
            paddle.zeros(self.global_num_experts, dtype=paddle.int64),
        )
        no_elastic_num, elastic_topk_num, elastic_expert_num = (
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
            paddle.zeros(1, dtype=paddle.int64),
        )

        ori_topk_probs, ori_topk_indices, routing_map, dispatch_mask = self.topk(
            self.elastic_topk_value,
            gate_logits,
            gate_probs,
            probs_for_choice,
            input_ids=input_ids,
            is_diff_topk=is_diff_topk,
            is_diff_expert_num=is_diff_expert_num,
            is_pure_text_line=is_pure_text_line,
        )

        if max_topk is not None:
            assert max_topk <= 32, f"max_topk {max_topk} exceeds 32, which is not supported yet."
            topk_probs, topk_indices = self.pad_for_elastic(max_topk, ori_topk_probs, ori_topk_indices)
        else:
            topk_probs, topk_indices = ori_topk_probs, ori_topk_indices
        topk_indices.stop_gradient = True

        if not (is_diff_topk or is_diff_expert_num):
            no_elastic_summary += dispatch_mask
            no_elastic_num += self.bsz
        if is_diff_topk:
            elastic_topk_summary += dispatch_mask
            elastic_topk_num += self.bsz
        if is_diff_expert_num:
            elastic_expert_summary += dispatch_mask
            elastic_expert_num += self.bsz

        router_loss2 = self.calc_router_loss_and_logging(
            router_loss,
            gate_logits=gate_logits,
            gate_probs=gate_probs.clone(),  # 为了确保ep_comm_overlap开和关时, gate_probs梯度累加顺序相同
            routing_map=routing_map,
            dispatch_mask=dispatch_mask,
            input_ids=input_ids,
            is_diff_topk=is_diff_topk,
            is_diff_expert_num=is_diff_expert_num,
            origin_input_ids=origin_input_ids,
            is_pure_text_line=is_pure_text_line,
        )
        dispatch_overlap_handle = None
        combine_overlap_handle = None

        # calc and log summary
        is_first_fwd = framework._dygraph_tracer()._has_grad
        if is_first_fwd and self.enable_logging and global_moe_balance_training_logs_enabled():
            self.moe_tokens_per_experts_indicator("no_elastic", no_elastic_summary, no_elastic_num)
            self.moe_tokens_per_experts_indicator("elastic_topk", elastic_topk_summary, elastic_topk_num)
            self.moe_tokens_per_experts_indicator("elastic_expert", elastic_expert_summary, elastic_expert_num)

        if max_topk is not None:
            dispatch_topk = max_topk
        else:
            dispatch_topk = self.elastic_topk_value

        self.orig_shape = orig_shape
        self.dispatch_topk = dispatch_topk
        self.is_first_fwd = is_first_fwd
        return router_loss2, topk_probs, topk_indices

    def dispatch_compute(self, input, topk_probs, topk_indices, async_finish=False):
        """decomposed dispatch for overlap"""
        # fp8_dispatched_handle里装的是scale, 为了避免污染计算图, 用dict形式的handle进行传递
        (
            dispatched_hidden_states,
            dispatched_indices,
            dispatched_probs,
            fp8_dispatched_handle,
        ) = self.dispatcher._comm_manager.dispatch(
            input,
            topk_indices,
            topk_probs,
            fp8_dispatch_a2a=self.config.use_fp8_dispatch_a2a,
            async_finish=async_finish,
            do_barrier_ep=False,
            use_ue8m0=self.config.use_ue8m0,
        )
        return (
            dispatched_hidden_states,
            dispatched_indices,
            dispatched_probs,
            fp8_dispatched_handle,
        )

    def mlp_compute(self, dispatched_hidden_states, dispatched_indices, dispatched_probs, fp8_dispatched_handle):
        """decomposed mlp compute for overlap"""
        hidden_states_tmp = Fp8FusedMoeFunc.apply(
            dispatched_hidden_states,
            dispatched_probs,
            dispatched_indices,
            self,
            self.dispatch_topk,
            recompute_moe_gate_up=recompute_moe_gate_up_func(self.config, self.layer_idx),
            dequant_input=self.config.moe_dequant_input,
            moe_expert_fusion=self.config.moe_expert_fusion,
            recompute_moe_permute=(
                self.config.recompute_granularity == "selective"
                and self.config.recompute_modules is not None
                and "moe_permute" in self.config.recompute_modules
            ),
            moe_subbatch_token_num_after_dispatch=self.config.moe_subbatch_token_num_after_dispatch,
            use_bf16_gemm_weight_grad=not self.config.fp8_wgrad,
            is_first_fwd=not framework._dygraph_tracer()._has_grad,
            fp8_dispatched_handle=fp8_dispatched_handle,
            fp8=self.config.fp8,
            bypass_expert_output=self.use_rr_bypass_expert_output,
            use_ue8m0=self.config.use_ue8m0,
        )
        return hidden_states_tmp

    def combine_compute(self, hidden_states_tmp, async_finish=False):
        """decomposed combine for overlap"""
        combined_output = self.dispatcher._comm_manager.combine(
            hidden_states_tmp,
            use_rr_deepep_combine=self.use_rr_deepep_combine,
            async_finish=async_finish,
            do_barrier_ep=False,
        )

        self.dispatcher._comm_manager.handle = None
        hidden_states_tmp._clear_to_zero_allocation()

        return combined_output

    def post_process_compute(self, input, combined_output):
        """decomposed shared_experts and misc compute for overlap"""
        # shared_experts
        if self.shared_experts is not None:
            recompute_decoder_chunk = (
                self.config.recompute_granularity == "selective"
                and self.config.recompute_modules is not None
                and "decoder_chunk" in self.config.recompute_modules
            )
            if recompute_decoder_chunk:
                shared_out = recompute(self.shared_experts, input)
            else:
                shared_out = self.shared_experts(input)
            combined_output += shared_out

        if self.is_first_fwd and self.enable_logging and global_moe_balance_training_logs_enabled():
            num_tokens_per_expert = self.get_num_tokens_per_expert()
            total_num_tokens_per_expert = paddle.full([], fill_value=sum(num_tokens_per_expert))
            total_list = paddle.empty([self.group.world_size], dtype=total_num_tokens_per_expert.dtype)
            dist.stream.all_gather(total_list, total_num_tokens_per_expert, group=self.group)
            self.calc_and_log_moe_summary("local_tokens_per_card", total_list)

        if self.orig_shape:
            combined_output = combined_output.clone().reshape(self.orig_shape[:-1] + [combined_output.shape[-1]])
        return combined_output

    def get_num_tokens_per_expert(self):
        """
        获取每个专家处理的 token 数量。
        """
        assert len(self.dispatcher._comm_manager.tokens_per_expert_list) == len(self.experts)
        return self.dispatcher._comm_manager.tokens_per_expert_list

    def maybe_split_subbatch_data(self, permuted_tokens, token_permuted_indices, prob_permuted_indices):
        """maybe_split_subbatch_data"""

        def split_subbatch_data(data, tokens_per_subbatch):
            total_token_num = data.shape[0]

            full_batch_num, remainder = divmod(total_token_num, tokens_per_subbatch)
            num_or_sections = [tokens_per_subbatch] * full_batch_num
            if remainder:
                num_or_sections.append(remainder)

            assert (
                sum(num_or_sections) == total_token_num
            ), f"get_subbatch_data fail, {sum(num_or_sections)}, {total_token_num}"
            # when data is 0-size tensor, we need to compute it and construct the right backward graph.
            if total_token_num == 0:
                return [data]
            return paddle.split(data, num_or_sections=num_or_sections, axis=0)

        if self.config.deepep_tokens_per_subbatch > 0:
            assert (
                permuted_tokens.shape[0] == token_permuted_indices.shape[0]
            ), f"Shape mismatch between {permuted_tokens.shape[0]} and {token_permuted_indices.shape[0]}"
            assert (
                permuted_tokens.shape[0] == prob_permuted_indices.shape[0]
            ), f"Shape mismatch between {permuted_tokens.shape[0]} and {prob_permuted_indices.shape[0]}"
            permuted_tokens_list = split_subbatch_data(permuted_tokens, self.config.deepep_tokens_per_subbatch)
            token_permuted_indices_list = split_subbatch_data(
                token_permuted_indices, self.config.deepep_tokens_per_subbatch
            )
            prob_permuted_indices_list = split_subbatch_data(
                prob_permuted_indices, self.config.deepep_tokens_per_subbatch
            )
        else:
            permuted_tokens_list = [permuted_tokens]
            token_permuted_indices_list = [token_permuted_indices]
            prob_permuted_indices_list = [prob_permuted_indices]
        return permuted_tokens_list, token_permuted_indices_list, prob_permuted_indices_list

    def fine_grained_forward_experts(self, dispatched_output, dispatched_probs, dispatched_indices, dispatch_topk):
        """fine_grained_forward_experts"""
        output_tokens = paddle.zeros(dispatched_output.shape, dispatched_probs.dtype)
        # print("output_tokens:", output_tokens.shape, flush=True)
        total_tokens = sum(self.get_num_tokens_per_expert())
        for expert_id, num_tokens in enumerate(self.get_num_tokens_per_expert()):
            # print("expert_id:", expert_id, ", num_tokens:", num_tokens, flush=True)
            # local dispatch
            token_permuted_indices, prob_permuted_indices = topk_to_permuted_indices_single(
                dispatched_indices, num_tokens, expert_id, dispatch_topk
            )
            permuted_tokens = FakeGather.apply(dispatched_output, token_permuted_indices)
            # 对以下两种情况进行offload
            # 1. 单个tensor达到2GB
            # 2. 专家处理的tokens总数大于阈值，offload至tokens数少于阈值
            if inplace_offload_if_needed(permuted_tokens):
                total_tokens -= num_tokens
            elif self.config.premuted_offload_threshold and total_tokens >= self.config.premuted_offload_threshold:
                inplace_offload_if_needed(permuted_tokens, 0)
                total_tokens -= num_tokens

            # If deepep_tokens_per_subbatch > 0, the data is split into multiple subbatches.
            (
                permuted_tokens_list,
                token_permuted_indices_list,
                prob_permuted_indices_list,
            ) = self.maybe_split_subbatch_data(permuted_tokens, token_permuted_indices, prob_permuted_indices)

            for permuted_tokens_, token_permuted_indices_, prob_permuted_indices_ in zip(
                permuted_tokens_list, token_permuted_indices_list, prob_permuted_indices_list
            ):
                # ffn
                permuted_tokens_ = self.experts[expert_id](permuted_tokens_)

                # local combine
                if self.config.deepep_use_fused and dispatched_probs is not None:
                    output_tokens = FusedUnpermutation.apply(
                        output_tokens,
                        permuted_tokens_,
                        token_permuted_indices_,
                        dispatched_probs.flatten(),
                        prob_permuted_indices_,
                    )
                else:
                    if dispatched_probs is not None:
                        permuted_probs = FakeGather.apply(dispatched_probs.flatten(), prob_permuted_indices_)
                        if permuted_tokens_.dtype != permuted_probs.dtype:
                            new_permuted_tokens = permuted_tokens_.astype(permuted_probs.dtype)
                        else:
                            new_permuted_tokens = permuted_tokens_
                        inplace_offload_if_needed(new_permuted_tokens)
                        permuted_tokens_ = new_permuted_tokens * permuted_probs.unsqueeze(-1)
                    output_tokens = fake_scatter_add(output_tokens, token_permuted_indices_, permuted_tokens_)
        dispatched_output._clear_to_zero_allocation()
        return output_tokens.astype(dispatched_output.dtype)

    def forward_experts(self, dispatched_input):
        """
        each expert gets a chunk of input and runs forward
        """
        num_tokens_per_expert = self.get_num_tokens_per_expert()
        outputs = []
        chunks = paddle.split(dispatched_input, num_or_sections=num_tokens_per_expert, axis=0)
        assert len(chunks) == len(self.experts), (len(chunks), len(self.experts))

        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            outputs += [expert(chunk)]
        return paddle.concat(outputs, axis=0)

    def calc_router_loss_and_logging(
        self,
        router_loss,
        gate_logits,
        gate_probs,
        routing_map,
        dispatch_mask,
        input_ids=None,
        is_diff_topk=False,
        is_diff_expert_num=False,
        origin_input_ids=None,
        is_pure_text_line=None,
    ):
        """calc_router_loss_and_logging"""
        assert isinstance(self.gate, DeepEPTop2Gate)
        l_aux, orthogonal_loss, zloss = None, None, None
        if self.gate.config.router_aux_loss_coef and not (is_diff_topk or is_diff_expert_num):
            if self.gate.act is F.sigmoid:
                gate_probs = self.cal_norm_combine_weights(gate_probs)

            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                sequence_partition_group = hcg.get_model_parallel_group()
            else:
                sequence_partition_group = None

            if is_pure_text_line is not None:
                gate_probs *= is_pure_text_line

            if self.config.moe_router_load_balancing_type == "seq_aux_loss":
                l_aux = self.gate._cal_seq_aux_loss(
                    gate_probs,
                    routing_map,
                    self.bsz,
                    self.config.seqlen,
                    self.elastic_topk_value,
                    sequence_partition_group,
                    input_ids,
                    origin_input_ids,
                )
            elif self.config.moe_router_load_balancing_type == "switch_aux_loss":
                l_aux = self.gate._cal_switch_aux_loss(
                    gate_probs,
                    dispatch_mask,
                    self.elastic_topk_value,
                    sequence_partition_group,
                    input_ids,
                )
            else:
                l_aux = self.gate._cal_aux_loss(gate_probs, dispatch_mask, input_ids)
            router_loss += self.gate.router_aux_loss_coef[0] * l_aux
        else:
            router_loss += self.zero * gate_probs.sum()  # must use gate prob to avoid zero pointer
        if self.gate.config.moe_orthogonal_loss_lambda:
            orthogonal_loss = self.gate._cal_orthogonal_loss()
            router_loss += self.gate.moe_orthogonal_loss_lambda[0] * orthogonal_loss
        if self.gate.config.router_z_loss_coef and not (is_diff_topk or is_diff_expert_num):
            if is_pure_text_line is not None:
                gate_logits *= is_pure_text_line
            zloss = self.gate._cal_z_loss(gate_logits, input_ids, origin_input_ids)
            router_loss += self.gate.router_z_loss_coef[0] * zloss

        # 开启重计算的话只需要logging一次就行
        tracer = framework._dygraph_tracer()
        is_first_fwd = tracer._has_grad
        if self.enable_logging and global_training_logs_enabled() and is_first_fwd:
            log = {}
            if l_aux is not None:
                log[f"aux_loss_layer_{self.layer_idx}"] = l_aux

            if orthogonal_loss is not None:
                log[f"orthogonal_loss_layer_{self.layer_idx}"] = orthogonal_loss

            if zloss is not None:
                log[f"zloss_layer_{self.layer_idx}"] = zloss

            global_training_logs.update(
                **log,
                **{k.replace(f"_layer_{self.layer_idx}", ""): v for k, v in log.items()},
            )

        return router_loss

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """
        sharded_state_dict method for DeepEPMOELayer.

        The global expert ID offset is calculated based on the moe rank information.
        """
        sharded_state_dict = super().sharded_state_dict(structured_name_prefix)
        global_expert_id_offset = self.group.rank * self.num_local_experts
        for k, v in sharded_state_dict.items():
            v.global_expert_id_offset = global_expert_id_offset
            sharded_state_dict[k] = v
        return sharded_state_dict
