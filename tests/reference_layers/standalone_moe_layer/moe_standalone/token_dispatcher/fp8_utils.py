# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
"""FP8 Utils"""

import os
from functools import partial

import numpy
import paddle
import paddle.nn.functional as F
from moe_standalone.compat import TDU, get_env_device, kitchen_fp8_gemm, kitchen_quant
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore


linear_func = F.linear

try:
    from paddle_xpu.layers.nn import xpu_matmul

    if os.getenv("XPU_MOE_USE_FP16") == "1":
        linear_func = xpu_matmul()
except ImportError:
    linear_func = F.linear

try:
    from paddle.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        """
            使用swiglu函数对输入的张量进行Sigmoid-weighted Linear Unit操作，并返回结果。
        如果没有提供y参数，则将输入的张量分割成两个部分，一个是Sigmoid函数的输入，另一个是Linear Unit的输入。
        否则，将x视为Sigmoid函数的输入，y视为Linear Unit的输入。

        Args:
            x (Tensor): 要进行Sigmoid-weighted Linear Unit操作的输入张量，其形状可以是任意维度。（默认值：None）
            y (Tensor, optional): 要与x相乘的常数项，其形状应该和x相同。（默认值：None）

        Returns:
            Tensor: Sigmoid-weighted Linear Unit后的输出张量，其形状与x相同。

        Raises:
            TypeError: 当x不是Tensor类型时会抛出此类型错误。
            ValueError: 当x和y的形状不匹配时会抛出此值错误。
        """
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from moe_standalone.compat import deep_gemm


try:
    import FusedQuantOps as FQO
    import kitchen
except:
    pass

try:
    from paddle.incubate.nn.functional import fused_transpose_wlch_split_quant
except ImportError:
    fused_transpose_wlch_split_quant = None

__all__ = [
    "ExpertsGroupGemmNode",
    "ExpertsGroupGemmContiguousNode",
]


FP8_ALIGN = 128


def _get_fp8_weight_and_scale(weight, transpose=False):
    """_get_fp8_weight_and_scale"""
    fp8_weight, fp8_scale = weight.fp8_weight_stacked, weight.fp8_scale_stacked

    if transpose:
        if hasattr(weight, "fp8_weight_stacked_transpose") and weight.fp8_weight_stacked_transpose is not None:
            fp8_weight = weight.fp8_weight_stacked_transpose
            fp8_scale = weight.fp8_scale_stacked_transpose
        else:

            assert fp8_weight.shape[0] % weight.shape[0] == 0
            assert fp8_weight.ndim == 2, "fp8_weight must be 2 dims"

            expert_num = fp8_weight.shape[0] // weight.shape[0]

            def transpose_tensor(tensor):
                assert tensor.ndim == 2
                h0 = tensor.shape[0] // expert_num
                h1 = tensor.shape[1]
                tensor = tensor.reshape([expert_num, h0, h1])
                return tensor.contiguous().transpose([0, 2, 1]).reshape([-1, h0]).contiguous()

            fp8_weight, fp8_scale = map(lambda x: transpose_tensor(x), [fp8_weight, fp8_scale])

    return fp8_weight, fp8_scale


def fused_stack_transpose_quant(expert_weight_list, use_ue8m0=False):
    """fused_stack_transpose_quant"""
    if hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], transpose=True)
    else:
        if hasattr(TDU, "fuse_stack_transpose_fp8_quant"):
            use_pow2_scale = False
            if paddle.device.cuda.get_device_capability()[0] == 10:
                # Blackwell GPUs require the use of pow2_scales quantization.
                use_pow2_scale = True

            w, scale = TDU.fuse_stack_transpose_fp8_quant(
                expert_weight_list,
                use_pow2_scale,
                use_ue8m0,
                use_ue8m0,
            )
            if use_ue8m0:
                scale = scale.T
        else:
            w, scale = FQO.fused_stack_transpose_quant(expert_weight_list)
    return w, scale


def fused_stack_quant(expert_weight_list, use_ue8m0=False):
    """fused_stack_quant"""
    if hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], transpose=False)
    else:
        if hasattr(TDU, "fuse_stack_fp8_quant"):
            use_pow2_scale = False
            if paddle.device.cuda.get_device_capability()[0] == 10:
                # Blackwell GPUs require the use of pow2_scales quantization.
                use_pow2_scale = True

            w, scale = TDU.fuse_stack_fp8_quant(
                expert_weight_list,
                use_pow2_scale,
                use_ue8m0,
                use_ue8m0,
            )
            if use_ue8m0:
                scale = scale.T
        else:
            w, scale = FQO.fused_stack_quant(expert_weight_list)

    return w, scale


def tilewise_quant(x):
    """
    tilewise_quant
    """
    if x.shape[0] > 0:
        return kitchen_quant(x, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False)
    else:
        shape = x.shape
        x_fp8 = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
        assert shape[-1] % FP8_ALIGN == 0, shape
        shape[-1] //= FP8_ALIGN
        x_scale = paddle.empty(shape, dtype=paddle.float32)
        return x_fp8, x_scale


def split_group_gemm(x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out, use_ue8m0=False):
    """
    将输入的张量分割成多个小的矩阵乘

    Args:
        x_fp8 (torch.Tensor, shape=(N, T)): 需要进行矩阵乘法的FP8格式的张量。
        x_scale (torch.Tensor, shape=(N, T)): 与x_fp8对应的缩放因子。
        w_fp8 (List[torch.Tensor], length=6): 包含6个FP8格式的张量，每个张量代表一个专家的权重。
        w_scale (List[torch.Tensor], length=6): 与w_fp8对应的缩放因子。
        tokens_per_expert (List[int], length=6): 每个专家处理的token数量。
        gemm_out (torch.Tensor, shape=(N, T)): 存储结果的张量。

    Returns:
        torch.Tensor, shape=(N, T): 返回计算结果存储在gemm_out中的张量。
    """
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T
        if use_ue8m0:
            w_scale_tma_align = w_scale[i].T.contiguous().T
        else:
            w_scale_tma_align = w_scale[i]

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x_fp8[start_idx:end_idx], x_scale_tma_align), (w_fp8[i], w_scale_tma_align), gemm_out[start_idx:end_idx]
        )

        start_idx = end_idx

    return gemm_out


def has_config(config_map, key):
    """
    判断给定的配置字典中是否存在指定键，并且该键对应的值不为空。

    Args:
        config_map (Optional[Dict[str, Any]]): 配置字典，可以为None。
        key (str): 需要查找的键名。

    Returns:
        bool: 如果配置字典不为None，且包含指定键，且该键对应的值不为空，则返回True；否则返回False。
    """
    return bool(config_map is not None and key in config_map and config_map[key])


def gen_m_indices(tokens_per_expert):
    tokens = []
    for i in range(len(tokens_per_expert)):
        tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
    out = paddle.concat(tokens, axis=0)
    return out


class ExpertsGroupGemmNode:
    """ExpertsGroupGemmNode"""

    def __init__(self, experts, custom_map, name="moe_experts_node"):
        """
            Initializes the MoEExpertsNode class.

        Args:
            experts (list[tf.keras.layers.Layer]): A list of TensorFlow Keras layers representing the
                expert sub-networks. Each layer should take a tensor as input and produce a tensor as output.
                The number of layers in this list determines the number of experts in the model.
            custom_map (dict): A dictionary mapping from an integer to a string. This is used for
                converting the token indices to their corresponding string labels.
            name (str, optional): An optional string used to name the node. Defaults to "moe_experts_node".

        Raises:
            ValueError: If `experts` is not a list or if it contains any non-layer objects.
        """
        self.o1 = None
        self.unzipped_tokens = None
        self.custom_map = custom_map
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.fp8_fused_ops_configs = custom_map.config.fp8_fused_ops_configs

    def reset_statue(self):
        """
            重置状态，将所有变量设为None。
        包括：o1、unzipped_tokens、unzipped_probs、tokens_per_expert等。

        Returns:
            None, 无返回值，直接修改了类的属性。
        """
        self.o1 = None
        self.unzipped_tokens = None
        self.unzipped_probs = None
        self.tokens_per_expert = None

    def fwd_gate_up(self, x_bf16, expert_w1, expert_w_count, tokens_per_expert):
        """
        前向门上行传播函数，将bfloat16类型的输入x和各个专家的参数w1进行矩阵乘操作，并返回结果。
            该函数使用了深度矩阵乘法算法来提高计算效率。

            Args:
                x_bf16 (Tensor, float32): 形状为[expert_w_count, tokens_per_expert, hidden_size]的bfloat16类型的输入。
                    expert_w_count表示专家的数量，tokens_per_expert表示每个专家处理的token数量，hidden_size表示模型的隐藏单元数。
                expert_w1 (List[Tensor]): 形状为[expert_w_count, tokens_per_expert, hidden_size]的专家参数列表，
                    其中每个元素都是float32类型的Tensor。
                expert_w_count (int): 专家的数量。
                tokens_per_expert (int): 每个专家处理的token数量。

            Returns:
                Tensor, float32: 形状为[expert_w_count, tokens_per_expert, hidden_size]的bfloat16类型的输出，
                    其中每个元素都是由专家参数w1和输入x进行矩阵乘得到的。
        """
        # concat w1
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w1_t_quant, w1_t_scale = fused_stack_transpose_quant(expert_w1)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            stacked_w1_t = paddle.transpose(stacked_w1, [0, 2, 1]).contiguous()
            concated_w1_t = stacked_w1_t.reshape([-1, stacked_w1_t.shape[-1]])

            # quant w1
            w1_t_quant, w1_t_scale = kitchen_quant(
                concated_w1_t,
                backend=kitchen.ops.Backend.CUBLAS,
                is_1d_scaled=False,
                return_transpose=False,
            )

        w1_t_quant = w1_t_quant.reshape([expert_w_count, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([expert_w_count, -1, w1_t_scale.shape[-1]])

        # quant x_bf16
        x_fp8, x_scale = kitchen_quant(
            x_bf16, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        # mask group gemm需要输入x是[group,m,n]
        x_fp8 = x_fp8.reshape([expert_w_count, -1, x_fp8.shape[-1]])
        x_scale = x_scale.reshape([expert_w_count, -1, x_scale.shape[-1]])
        # optimize tma
        x_scale = paddle.transpose(paddle.transpose(x_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        o1 = paddle.zeros([expert_w_count, x_fp8.shape[1], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype)
        if numpy.prod(x_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (x_fp8, x_scale), (w1_t_quant, w1_t_scale), o1, tokens_per_expert, x_fp8.shape[1]
            )
        return o1

    def fwd_swiglu(self, o1):
        """
            将输入的对象转换为SwigLU对象，并返回。
        参数：
            o1 (object) - 需要转换的对象，可以是任何类型的Python对象。
        返回值：
            o2 (object) - SwigLU对象，即使原始对象不支持SwigLU也会返回一个空对象。
        """
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o1, unzipped_probs, expert_w2, expert_w_count, tokens_per_expert):
        """
        前向传播，将输入的o2和expert_w2进行下采样。
            并返回经过量化后的o3。

            Args:
                o2 (Tensor, shape=[expert_w_count, tokens_per_expert, hidden_size]): 输入的特征矩阵。
                expert_w2 (List[Tensor], shape=[num_experts, hidden_size, hidden_size]): 每个专家的权重矩阵。
                expert_w_count (int): 专家数量。
                tokens_per_expert (int): 每个专家处理的token数量。

            Returns:
                Tensor, shape=[expert_w_count, tokens_per_expert, hidden_size]: 经过量化后的o3。
        """
        # concat and transpose w2
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_sacle = fused_stack_transpose_quant(expert_w2)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])

            # quant w2
            w2_quant, w2_sacle = kitchen_quant(
                concated_w2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        w2_quant = w2_quant.reshape([expert_w_count, -1, w2_quant.shape[-1]])
        w2_sacle = w2_sacle.reshape([expert_w_count, -1, w2_sacle.shape[-1]])

        # o2
        o2 = self.fwd_swiglu(o1)

        unzipped_probs = unzipped_probs.unsqueeze(-1).reshape([expert_w_count, -1, 1])

        o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)

        # quant o2
        o2_reshape = o2.reshape([-1, o2.shape[-1]]).contiguous()
        o2_quant, o2_scale = kitchen_quant(
            o2_reshape, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        o2_quant = o2_quant.reshape([expert_w_count, -1, o2_quant.shape[-1]])
        o2_scale = o2_scale.reshape([expert_w_count, -1, o2_scale.shape[-1]])
        # optimize tma
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        # group gemm masked
        o3 = paddle.zeros([expert_w_count, o2_quant.shape[1], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_quant.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (o2_quant, o2_scale), (w2_quant, w2_sacle), o3, tokens_per_expert, o2_quant.shape[1]
            )
        return o3, unzipped_probs

    def fwd_down_no_probs(self, o1, expert_w2, expert_w_count, tokens_per_expert):
        """
        前向传播，将输入的o2和expert_w2进行下采样。
            并返回经过量化后的o3。

            Args:
                o2 (Tensor, shape=[expert_w_count, tokens_per_expert, hidden_size]): 输入的特征矩阵。
                expert_w2 (List[Tensor], shape=[num_experts, hidden_size, hidden_size]): 每个专家的权重矩阵。
                expert_w_count (int): 专家数量。
                tokens_per_expert (int): 每个专家处理的token数量。

            Returns:
                Tensor, shape=[expert_w_count, tokens_per_expert, hidden_size]: 经过量化后的o3。
        """
        # concat and transpose w2
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_sacle = fused_stack_transpose_quant(expert_w2)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])

            # quant w2
            w2_quant, w2_sacle = kitchen_quant(
                concated_w2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        w2_quant = w2_quant.reshape([expert_w_count, -1, w2_quant.shape[-1]])
        w2_sacle = w2_sacle.reshape([expert_w_count, -1, w2_sacle.shape[-1]])

        # o2
        o2 = self.fwd_swiglu(o1)

        # quant o2
        o2_reshape = o2.reshape([-1, o2.shape[-1]]).contiguous()
        o2_quant, o2_scale = kitchen_quant(
            o2_reshape, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        o2_quant = o2_quant.reshape([expert_w_count, -1, o2_quant.shape[-1]])
        o2_scale = o2_scale.reshape([expert_w_count, -1, o2_scale.shape[-1]])
        # optimize tma
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        # group gemm masked
        o3 = paddle.zeros([expert_w_count, o2_quant.shape[1], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_quant.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (o2_quant, o2_scale), (w2_quant, w2_sacle), o3, tokens_per_expert, o2_quant.shape[1]
            )
        return o3

    # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
    def bwd_dowm_input(self, expert_w2, unzipped_grad, tokens_per_expert, expected_m):
        """
            计算反向传播的输入：do2、o2_s、probs_grad。
        其中，do2 和 o2_s 是用于优化 TMA 的量化结果；probs_grad 是用于更新概率分布的反向传播结果。

        Args:
            expert_w2 (List[Tensor]): 每个专家的 weight 列表，形状为 [num_experts, num_tokens, hidden_size]。
                类型为 List[Tensor]，元素类型为 Tensor。
            unzipped_grad (Tensor): 反向传播的梯度张量，形状为 [num_experts, num_tokens, hidden_size]。
                类型为 Tensor。
            tokens_per_expert (int): 每个专家处理的 token 数量。一个 int 值。
            expected_m (int): 期望的 m 值，用于优化 TMA。一个 int 值。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            返回三个 Tensor，分别是 do2、o2_s、probs_grad。
            do2 的形状为 [num_experts, num_tokens, hidden_size]，类型为 Tensor；
            o2_s 的形状为 [num_experts, num_tokens, hidden_size]，类型为 Tensor；
            probs_grad 的形状为 [num_experts, num_tokens, hidden_size]，类型为 Tensor。
        """

        # recompute concated_w2_2d
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_quant(expert_w2)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])

            # quant w2
            bw_w2_quant, bw_w2_scale = kitchen_quant(
                concated_w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = kitchen_quant(
            unzipped_grad, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        unzipped_grad_fp8 = unzipped_grad_fp8.reshape([len(expert_w2), -1, unzipped_grad_fp8.shape[-1]])
        unzipped_grad_scale = unzipped_grad_scale.reshape([len(expert_w2), -1, unzipped_grad_scale.shape[-1]])
        # optimize tma
        unzipped_grad_scale = paddle.transpose(
            paddle.transpose(unzipped_grad_scale, [0, 2, 1]).contiguous(), [0, 2, 1]
        )
        do2_s = paddle.zeros(
            [len(expert_w2), unzipped_grad_fp8.shape[1], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (unzipped_grad_fp8, unzipped_grad_scale),
                (bw_w2_quant, bw_w2_scale),
                do2_s,
                tokens_per_expert,
                expected_m,
            )
        if has_config(self.fp8_fused_ops_configs, "swiglu_probs_bwd"):
            do1, probs_grad, o2_s = FQO.fused_swiglu_probs_bwd(self.o1, do2_s, self.unzipped_probs)
        else:
            # recomput o2
            o2 = self.fwd_swiglu(self.o1)
            o2_s = (o2 * self.unzipped_probs).cast(paddle.bfloat16)
            # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
            do2 = (do2_s.cast(paddle.float32) * self.unzipped_probs).cast(paddle.bfloat16)

            # probs_grad: probs_grad 需要保持 float32（因为 unzipped_probs 是 float32）
            probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
            # do1
            do1 = self.bwd_swiglu(self.o1, do2)

        return do1, o2_s, probs_grad

    # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
    def bwd_dowm_input_no_prob(self, expert_w2, unzipped_grad, tokens_per_expert, expected_m):
        """
        反向传播，下采样输入，不需要修改概率。
            参数：
                expert_w2 (List[paddle.Tensor]): 每个专家的权重，形状为[num_expert, hidden_size, hidden_size]。
                unzipped_grad (paddle.Tensor): 未压缩的梯度，形状为[batch_size, num_expert, hidden_size]。
                tokens_per_expert (int): 每个专家处理的token数量。
                expected_m (int): 期望的最大值。
            返回值：
                Tuple(paddle.Tensor, paddle.Tensor):
                    do2_s (paddle.Tensor): 形状为[num_expert, batch_size, hidden_size]，每个专家对应的梯度。
                    o2_s (paddle.Tensor): 形状为[batch_size, hidden_size]，经过反向传播后的输出。
        """
        # recomput o2
        o2 = self.fwd_swiglu(self.o1)
        o2_s = o2

        # recompute concated_w2_2d
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_quant(expert_w2)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])

            # quant w2
            bw_w2_quant, bw_w2_scale = kitchen_quant(
                concated_w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = kitchen_quant(
            unzipped_grad, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        unzipped_grad_fp8 = unzipped_grad_fp8.reshape([len(expert_w2), -1, unzipped_grad_fp8.shape[-1]])
        unzipped_grad_scale = unzipped_grad_scale.reshape([len(expert_w2), -1, unzipped_grad_scale.shape[-1]])
        do2_s = paddle.zeros(
            [len(expert_w2), unzipped_grad_fp8.shape[1], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (unzipped_grad_fp8, unzipped_grad_scale),
                (bw_w2_quant, bw_w2_scale),
                do2_s,
                tokens_per_expert,
                expected_m,
            )

        return do2_s, o2_s

    # ===== do1 = swiglu_grad(o1, None, do2) =====
    def bwd_swiglu(self, o1, do2):
        """
            反向传播函数，用于计算对输入参数o1的导数do2。
        该函数调用paddle._C_ops.swiglu_grad进行反向传播操作。

        Args:
            o1 (Variable): 需要求导的变量，类型为Variable。
            do2 (Variable): 对输出结果的偏导数，类型为Variable。

        Returns:
            Variable, tuple: 返回一个Variable类型的值，表示对输入参数o1的导数；还会返回一个tuple类型的值，包含一个元素，表示对输出结果的偏导数。
        """
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    # ===== dx = deep_gemm(do1_fp8, w1_fp8)
    def bwd_gate_up_input(self, do1, expert_w1, tokens_per_expert, expected_m):
        """
        计算反向传播过程中，输入的上行网关。
            该函数主要用于计算输入的上行网关，并返回经过反向传播后的结果。

            Args:
                do1 (Tensor, float32): 上行网关的输入张量，形状为（`expected_m`, `tokens_per_expert`）。
                expert_w1 (List[Tensor]): 每个专家的上行网关的输入张量列表，形状为（`len(expert_w1)`, `expected_m`, `tokens_per_expert`）。
                tokens_per_expert (int): 每个专家处理的token数量。
                expected_m (int): 期望的输出维度。

            Returns:
                Tensor, float16: 经过反向传播后的输入的上行网关，形状为（`len(expert_w1)`, `expected_m`, `tokens_per_expert`）。
        """
        # recompute concated_w1_t
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w1_quant, bw_w1_scale = fused_stack_quant(expert_w1)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            concated_w1_t_2d = stacked_w1.reshape([-1, stacked_w1.shape[-1]])

            # quant w1
            bw_w1_quant, bw_w1_scale = kitchen_quant(
                concated_w1_t_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        # quant do1
        do1_fp8_reshape = do1.reshape([-1, do1.shape[-1]]).contiguous()
        do1_fp8, do1_scale = kitchen_quant(
            do1_fp8_reshape, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        do1_fp8 = (do1_fp8.reshape([len(expert_w1), -1, do1_fp8.shape[-1]])).contiguous()
        do1_scale = do1_scale.reshape([len(expert_w1), -1, do1_scale.shape[-1]]).contiguous()
        # optimize tma
        do1_scale = paddle.transpose(paddle.transpose(do1_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        # group gemm
        dx = paddle.zeros(shape=[len(expert_w1), do1_fp8.shape[1], bw_w1_quant.shape[1]], dtype=paddle.bfloat16)
        if numpy.prod(do1_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (do1_fp8, do1_scale), (bw_w1_quant, bw_w1_scale), dx, tokens_per_expert, expected_m
            )
        return dx

    # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
    def bwd_down_weight(self, out_grad, o2, expert_w2):
        """
        计算权重的反向传播，并更新权重。
            参数：
                out_grad (Tensor): 输出的导数，形状为（N, C）或者（N, C, H）。
                o2 (Tensor): 输入的特征，形状为（N, C, H）。
                expert_w2 (List[Tensor]): 每个专家的权重，形状为（H, E）。
            返回值：
                无返回值，直接修改每个专家的权重。
        """
        # transpose o2
        group_num = len(expert_w2)
        H2 = o2.shape[-1]

        o2_t = (
            o2.reshape([group_num, -1, H2])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H2, -1])
            .contiguous()
        )

        o2_t_fp8, o2_t_scale = kitchen_quant(
            o2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        o2_t_fp8 = o2_t_fp8.reshape([group_num, int(o2_t_fp8.shape[0] / group_num), o2_t_fp8.shape[-1]])
        o2_t_scale = paddle.split(o2_t_scale, num_or_sections=group_num, axis=-1)

        # quant out_grad
        H1 = out_grad.shape[-1]
        out_grad = (
            out_grad.reshape([group_num, -1, H1])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H1, -1])
            .contiguous()
        )

        out_grad_fp8, out_grad_scale = kitchen_quant(
            out_grad, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        out_grad_fp8 = out_grad_fp8.reshape([group_num, H1, -1])
        out_grad_scale = paddle.split(out_grad_scale, num_or_sections=group_num, axis=-1)

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                    paddle.float32,
                )
            # 兼容没有_apply_backward_hook方法的旧版本Paddle
            if hasattr(expert_w2[i], "_apply_backward_hook"):
                expert_w2[i]._apply_backward_hook()

    # ===== dw1 = deep_gemm(input_x_t_fp8, do1_t_fp8)
    def bwd_gate_up_weight(self, do1, input_x, expert_w1):
        """
            计算权重的反向传播，对输入进行上采样。
        参数：
            do1 (Tensor): 输出的导数，形状为 [N, H1, H2]。
            input_x (Tensor): 输入特征，形状为 [N, H1, H2]。
            expert_w1 (List[Parameter]): 专家网络的权重列表，每个元素都是 Parameter 类型，形状为 [H1, H2]。
        返回值：
            None，更新了每个专家网络的权重 grad 属性。
        """
        # transpose input_x and quant input_x

        group_num = len(expert_w1)
        H1 = input_x.shape[-1]
        input_x = (
            input_x.reshape([group_num, -1, H1])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H1, -1])
            .contiguous()
        )

        input_x_fp8, input_x_scale = kitchen_quant(
            input_x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        input_x_fp8 = input_x_fp8.reshape([group_num, int(input_x_fp8.shape[0] / group_num), input_x_fp8.shape[-1]])
        input_x_scale = paddle.split(input_x_scale, num_or_sections=group_num, axis=-1)

        # transpose do1 and quant do1
        H2 = do1.shape[-1]
        do1 = (
            do1.reshape([group_num, -1, H2])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H2, -1])
            .contiguous()
        )
        do1_fp8, do1_scale = kitchen_quant(
            do1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        do1_fp8 = do1_fp8.reshape([group_num, int(do1_fp8.shape[0] / group_num), do1_fp8.shape[-1]])
        do1_scale = paddle.split(do1_scale, num_or_sections=group_num, axis=-1)

        # dw1
        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w1[i], "_apply_backward_hook"):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert):
        """
            Forward function of the custom map layer.

        Args:
            hs_out (Tensor): The output of the encoder, shape as (batch_size, seq_len, hidden_dim).
                It's a tensor with dtype float32 or float64.
            unzipped_probs (Tensor): The probability of each token being chosen by an expert.
                Shape as (batch_size, num_experts), it's a tensor with dtype float32 or float64.
            tokens_per_expert (List[int]): A list of integers representing the number of tokens per expert.
                Each integer represents the number of tokens assigned to one expert.

        Returns:
            Tensor: The output of the custom map layer, shape as (batch_size, seq_len, hidden_dim).
                It's a tensor with dtype float32 or float64.
        """
        # get w1
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w_count = len(expert_w1)

        # get w2
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, expert_w_count, tokens_per_expert)
        self.o1 = o1

        # o3
        o3, unzipped_probs = self.fwd_down(o1, unzipped_probs, expert_w_count, tokens_per_expert)

        # save for bwd
        self.unzipped_probs = unzipped_probs
        self.unzipped_tokens = hs_out
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, tokens_per_expert, dispatched_indices, expected_m):
        """
            Backward function of the custom mapping layer. It computes the gradients of the input, output,
        weights, and bias of the custom mapping layer.

        Args:
            out_grad (Tensor): The gradient tensor of the output of the custom mapping layer. Its shape should be
                `[batch_size, hidden_size]`.
            tokens_per_expert (List[int]): A list containing the number of tokens per expert for each batch.
            dispatched_indices (Tensor): The indices of the tokens that are assigned to each expert. Its shape should
                be `[batch_size, max_seq_len]`.
            expected_m (int): The expected number of experts to be used.

        Returns:
            Tuple[Tensor, Tensor]:
                - dx (Tensor): The gradient tensor of the input of the custom mapping layer. Its shape should be
                  `[batch_size, hidden_size]`.
                - probs_grad (Tensor): The gradient tensor of the probability distribution of the custom mapping layer.
                  Its shape should be `[batch_size, num_experts]`.
        """
        # recompute expert_w2 and expert_w1
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]

        # do2
        do1, o2_s, probs_grad = self.bwd_dowm_input(expert_w2, out_grad, tokens_per_expert, expected_m)

        # dx
        dx = self.bwd_gate_up_input(do1, expert_w1, tokens_per_expert, expected_m)
        dx = dx.reshape([-1, dx.shape[-1]])

        # dw2
        self.bwd_down_weight(out_grad, o2_s, expert_w2)

        # dw1
        self.bwd_gate_up_weight(do1, self.unzipped_tokens, expert_w1)

        self.reset_statue()
        return dx, probs_grad

    @paddle.no_grad()
    def forward_no_prob(self, hs_out, tokens_per_expert):
        """
        在不使用概率的情况下，进行前向传播。
            将输入的hs_out和tokens_per_expert作为参数，返回经过自定义映射网络的结果。
            Args:
                hs_out (Tensor): 形状为[batch_size, seq_len, hidden_size]的隐藏状态。
                tokens_per_expert (List[int]): 每个专家处理的token数量列表。
            返回值（Tuple[Tensor, Tensor]）：
                - o3 (Tensor): 形状为[batch_size, seq_len, hidden_size]的输出特征。
                  该特征经过了自定义映射网络的转换。
                - unzipped_tokens (Tensor): 形状为[batch_size, seq_len, hidden_size]的原始隐藏状态。
                  该变量用于反向传播时保存原始隐藏状态。
        """
        # get w1
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w_count = len(expert_w1)

        # get w2
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, expert_w_count, tokens_per_expert)
        self.o1 = o1

        # o3
        o3 = self.fwd_down_no_probs(o1, expert_w2, expert_w_count, tokens_per_expert)

        # save for bwd
        self.unzipped_tokens = hs_out
        return o3

    @paddle.no_grad()
    def backward_no_prob(self, out_grad, tokens_per_expert):
        """
        Backward function without probability computation.
            Args:
                out_grad (Tensor): The output gradient Tensor, has shape [batch_size, hidden_dim].
                tokens_per_expert (List[int]): A list of integers representing the number of tokens per expert.

            Returns:
                Tensor: The input gradient Tensor, has shape [batch_size, hidden_dim].

            Raises:
                None.
        """
        # recompute expert_w2 and expert_w1
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]

        # ("out grad", out_grad)
        expected_m = int(numpy.prod(out_grad.shape[:-1]) // len(expert_w1))

        out_grad = out_grad.reshape([-1, out_grad.shape[-1]])
        # do2
        do2, o2_s = self.bwd_dowm_input_no_prob(expert_w2, out_grad, tokens_per_expert, expected_m)

        # do1
        do1 = self.bwd_swiglu(self.o1, do2)

        # dx
        dx = self.bwd_gate_up_input(do1, expert_w1, tokens_per_expert, expected_m)
        dx = dx.reshape([-1, dx.shape[-1]])

        # dw2
        self.bwd_down_weight(out_grad, o2_s, expert_w2)

        # dw1
        self.bwd_gate_up_weight(do1, self.unzipped_tokens, expert_w1)

        self.reset_statue()
        return dx


class ExpertsGroupGemmContiguousNode:
    """ExpertsGroupGemmContiguousNode"""

    def __init__(
        self,
        custom_map,
        recompute_moe_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_contiguous_node",
        expert_id=None,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        fp8="e4m3",
        bypass_down_proj=False,
        use_ue8m0=False,
    ):
        """
            Initializes the experts group gemm contiguous node.

        Args:
            custom_map (CustomMapping): Custom mapping for the model.
            recompute_moe_gate_up (bool, optional): Whether to recompute forward gate up. Defaults to False.
            dequant_input (bool, optional): Whether to dequantize input. Defaults to False.
            name (str, optional): Name of the node. Defaults to "experts_group_gemm_contiguous_node".
        """
        if expert_id is None:
            self.experts = custom_map.experts
        else:
            self.experts = [custom_map.experts[expert_id]]
        self.expert_id = expert_id
        self.recompute_moe_gate_up = recompute_moe_gate_up
        self.dequant_input = dequant_input
        self.tokens_per_expert = None
        self.m_indices = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.fp8_fused_ops_configs = custom_map.config.fp8_fused_ops_configs
        self.is_split_group_gemm = not custom_map.config.moe_grouped_gemm
        self.group = group
        self.moe_subbatch_token_num_after_dispatch = moe_subbatch_token_num_after_dispatch
        if self.moe_subbatch_token_num_after_dispatch is not None:
            assert (
                self.moe_subbatch_token_num_after_dispatch > 0
                and self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
        self.use_bf16_gemm_weight_grad = use_bf16_gemm_weight_grad
        self.fp8 = fp8
        self.bypass_down_proj = bypass_down_proj
        self.use_ue8m0 = use_ue8m0

        if self.use_ue8m0:
            assert paddle.device.cuda.get_device_capability()[0] == 10, "use_ue8m0 requires Blackwell GPU"

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        (
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ) = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors))

    def reset_statue(self):
        """
        reset_statue
        """
        self.tokens_per_expert = None
        self.m_indices = None
        self.clear_activation_tensors()

    def clear_activation_tensors(self):
        """
        clear_activation_tensors
        """
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def gen_m_indices(self, tokens_per_expert):
        """
        generate m indices
        """
        tokens = []
        for i in range(len(tokens_per_expert)):
            tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
        out = paddle.concat(tokens, axis=0)
        return out

    def fwd_gate_up_bf16(self, x, expert_w1):
        """
        fwd_gate_up bf16
        """
        if self.is_split_group_gemm is False:
            raise NotImplementedError("fuse node do not support group gemm currently")

        if x is None:
            assert self.input is not None
            x = self.input

        if numpy.prod(x.shape) != 0:
            expert_output_list = []
            start_idx = 0
            for i, token_num in enumerate(self.tokens_per_expert):
                if token_num == 0:
                    continue
                end_idx = start_idx + token_num
                x_i = x[start_idx:end_idx].contiguous()
                expert_w1_i = expert_w1[i]
                expert_output_list.append(linear_func(x_i, expert_w1_i))
                start_idx = end_idx
            o1 = paddle.concat(expert_output_list, axis=0)
        else:
            o1 = paddle.empty([x.shape[0], expert_w1[0].shape[1]], dtype=expert_w1[0].dtype)
        self.input = x
        return o1

    def fwd_gate_up(self, x, expert_w1, num_expert, tokens_per_expert, scale=None):
        self.tokens_per_expert = tokens_per_expert
        if self.fp8 is None:
            return self.fwd_gate_up_bf16(x, expert_w1)
        else:
            return self.fwd_gate_up_fp8(x, expert_w1, num_expert, tokens_per_expert, scale)

    def fwd_gate_up_fp8(self, x, expert_w1, num_expert, tokens_per_expert, scale=None):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """

        if not self.is_split_group_gemm:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w1_t_quant, w1_t_scale = fused_stack_transpose_quant(expert_w1, use_ue8m0=self.use_ue8m0)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            stacked_w1_t = paddle.transpose(stacked_w1, [0, 2, 1]).contiguous()
            concated_w1_t = stacked_w1_t.reshape([-1, stacked_w1_t.shape[-1]])
            # quant w1
            w1_t_quant, w1_t_scale = kitchen_quant(
                concated_w1_t,
                backend=kitchen.ops.Backend.CUBLAS,
                is_1d_scaled=False,
                return_transpose=False,
            )
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if x is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
        elif scale is not None:
            x_fp8, x_scale = x, scale
            assert self.dequant_input, "如果传入了scale, 说明a2a使用了fp8,。必须开启dequant_input"
        else:
            # quant x_bf16
            x_fp8, x_scale = kitchen_quant(
                x, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
            )
        x_scale = paddle.transpose(paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0])

        # compute gemm
        o1 = paddle.empty([x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype)
        if numpy.prod(x_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    x_fp8, x_scale, w1_t_quant, w1_t_scale, tokens_per_expert, o1, use_ue8m0=self.use_ue8m0
                )
            else:
                if self.use_ue8m0:
                    w1_t_scale = w1_t_scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (x_fp8, x_scale), (w1_t_quant, w1_t_scale), o1, m_indices=self.m_indices
                )

        if self.dequant_input:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down_bf16(self, o1, unzipped_probs, expert_w2, clear_o1=False):
        """
        fwd_down_bf16
        """
        if self.is_split_group_gemm is False:
            raise NotImplementedError("fuse node do not support group gemm currently")

        # swiglu
        o2 = self.fwd_swiglu(o1)

        unzipped_probs = unzipped_probs.unsqueeze(-1)
        o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)

        if clear_o1:
            o1._clear_to_zero_allocation()

        # down proj
        if numpy.prod(o2.shape) != 0:
            expert_output_list = []
            start_idx = 0
            for i, token_num in enumerate(self.tokens_per_expert):
                if token_num == 0:
                    continue
                end_idx = start_idx + token_num
                o1_i = o2[start_idx:end_idx].contiguous()
                expert_w2_i = expert_w2[i]
                expert_output_list.append(linear_func(o1_i, expert_w2_i))
                start_idx = end_idx
            o3 = paddle.concat(expert_output_list, axis=0)
        else:
            o3_shape = [o2.shape[0], expert_w2[0].shape[1]]
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        return o3, unzipped_probs

    def fwd_down(self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False):
        if self.fp8 is None:
            return self.fwd_down_bf16(o1, unzipped_probs, expert_w2, clear_o1)
        else:
            return self.fwd_down_fp8(o1, unzipped_probs, expert_w2, num_expert, o3, clear_o1)

    def fwd_down_fp8(self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False):
        """
        o3 = o2 * w2
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # concat and transpose w2
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_sacle = fused_stack_transpose_quant(expert_w2, use_ue8m0=self.use_ue8m0)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])
            # quant w2
            w2_quant, w2_sacle = kitchen_quant(
                concated_w2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_sacle = w2_sacle.reshape([num_expert, -1, w2_sacle.shape[-1]])

        if has_config(self.fp8_fused_ops_configs, "spaq"):
            if hasattr(TDU, "fuse_weighted_swiglu_fp8_quant"):
                o2_fp8, o2_scale = TDU.fuse_weighted_swiglu_fp8_quant(
                    o1, unzipped_probs, using_pow2_scaling=True, use_ue8m0=self.use_ue8m0
                )
            else:
                o2_fp8, o2_scale = FQO.fused_spaq(o1, unzipped_probs, using_pow2_scaling=True)

            o2_scale = paddle.transpose(paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0])
            unzipped_probs = unzipped_probs.unsqueeze(-1)
        else:
            # o2
            o2 = self.fwd_swiglu(o1)

            unzipped_probs = unzipped_probs.unsqueeze(-1)
            o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)

            # quant o2
            o2_fp8, o2_scale = kitchen_quant(
                o2, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
            )

        if clear_o1:
            o1._clear_to_zero_allocation()

        # compute gemm
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, "{} vs {}".format(o3.shape, o3_shape)
            o3.zero_()
        else:
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    o2_fp8, o2_scale, w2_quant, w2_sacle, self.tokens_per_expert, o3, use_ue8m0=self.use_ue8m0
                )
            else:
                if self.use_ue8m0:
                    w2_sacle = w2_sacle.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (o2_fp8, o2_scale), (w2_quant, w2_sacle), o3, m_indices=self.m_indices
                )
        return o3, unzipped_probs

    def bwd_down_input_bf16(self, expert_w2, unzipped_grad, o1, unzipped_probs):
        """
        bwd_down_input_bf16
        """
        if numpy.prod(unzipped_grad.shape) != 0:
            do2_s_list = []
            start_idx = 0
            for i, token_num in enumerate(self.tokens_per_expert):
                if token_num == 0:
                    continue
                end_idx = start_idx + token_num
                unzipped_grad_i = unzipped_grad[start_idx:end_idx].contiguous()
                expert_w2_i = expert_w2[i].T.contiguous()
                do2_s_list.append(linear_func(unzipped_grad_i, expert_w2_i))
                start_idx = end_idx
            do2_s = paddle.concat(do2_s_list, axis=0)
        else:
            do2_s_shape = [unzipped_grad.shape[0], expert_w2[0].shape[1]]
            do2_s = paddle.empty(do2_s_shape, dtype=unzipped_grad.dtype)

        # recomput o2
        o2 = self.fwd_swiglu(o1)
        o2_s = (o2 * unzipped_probs).cast(paddle.bfloat16)
        # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
        do2 = (do2_s.cast(paddle.float32) * unzipped_probs).cast(paddle.bfloat16)

        # probs_grad: probs_grad 需要保持 float32（因为 unzipped_probs 是 float32）
        probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
        # do1
        do1 = self.bwd_swiglu(o1, do2)

        return do1, o2_s, probs_grad

    def bwd_dowm_input_fp8(self, expert_w2, unzipped_grad, o1, unzipped_probs, inplace_swiglu_prob=False):
        """
        do2 = do3 * w2_t
        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        """
        # recompute concated_w2_2d
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_quant(expert_w2, use_ue8m0=self.use_ue8m0)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])
            # quant w2
            bw_w2_quant, bw_w2_scale = kitchen_quant(
                concated_w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        # compute gemm
        if self.use_ue8m0:
            if self.is_split_group_gemm:
                unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    unzipped_grad,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
            else:
                unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    unzipped_grad,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
                unzipped_grad_scale = unzipped_grad_scale.T
        else:
            unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                unzipped_grad,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
                using_ue8m0_scale=False,
            )

        do2_s = paddle.empty([unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype)
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    unzipped_grad_fp8,
                    unzipped_grad_scale,
                    bw_w2_quant,
                    bw_w2_scale,
                    self.tokens_per_expert,
                    do2_s,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                if self.use_ue8m0:
                    bw_w2_scale = bw_w2_scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=self.m_indices,
                )

        if has_config(self.fp8_fused_ops_configs, "swiglu_probs_bwd"):
            do1, probs_grad, o2_s = FQO.fused_swiglu_probs_bwd(o1, do2_s, unzipped_probs, inplace_swiglu_prob)
        else:
            # recomput o2
            o2 = self.fwd_swiglu(o1)
            o2_s = (o2 * unzipped_probs).cast(paddle.bfloat16)
            # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
            do2 = (do2_s.cast(paddle.float32) * unzipped_probs).cast(paddle.bfloat16)

            # probs_grad: probs_grad 需要保持 float32（因为 unzipped_probs 是 float32）
            probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
            # do1
            do1 = self.bwd_swiglu(o1, do2)

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input_bf16(self, do1, expert_w1):
        """
        bwd_gate_up_input_bf16
        """
        if numpy.prod(do1.shape) != 0:
            dx_list = []
            start_idx = 0
            for i, token_num in enumerate(self.tokens_per_expert):
                if token_num == 0:
                    continue
                end_idx = start_idx + token_num
                do1_i = do1[start_idx:end_idx].contiguous()
                expert_w1_i = expert_w1[i].T.contiguous()
                dx_list.append(linear_func(do1_i, expert_w1_i))
                start_idx = end_idx
            dx = paddle.concat(dx_list, axis=0)
        else:
            dx_shape = [do1.shape[0], expert_w1[0].shape[0]]
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        return dx

    def bwd_gate_up_input_fp8(self, do1, expert_w1, dx=None):
        """
        dx = do1 * w1_t
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # recompute concated_w1_t
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w1_quant, bw_w1_scale = fused_stack_quant(expert_w1, use_ue8m0=self.use_ue8m0)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            concated_w1_t_2d = stacked_w1.reshape([-1, stacked_w1.shape[-1]])
            # quant w1
            bw_w1_quant, bw_w1_scale = kitchen_quant(
                concated_w1_t_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        # quant do1
        if self.use_ue8m0:
            if self.is_split_group_gemm:
                do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    do1,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
            else:
                do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    do1,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    using_ue8m0_scale=True,
                )
                do1_scale = do1_scale.T
        else:
            do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do1, output_scale_transpose=False, quant_method="1x128", input_transpose=False, using_ue8m0_scale=False
            )

        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
            dx.zero_()
        if numpy.prod(do1_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    do1_fp8, do1_scale, bw_w1_quant, bw_w1_scale, self.tokens_per_expert, dx, use_ue8m0=self.use_ue8m0
                )
            else:
                if self.use_ue8m0:
                    bw_w1_scale = bw_w1_scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (do1_fp8, do1_scale), (bw_w1_quant, bw_w1_scale), dx, m_indices=self.m_indices
                )

        return dx

    def fused_transpose_split_quant(self, x, tokens_per_expert, pow_2_scales):
        """
        Quantize on dim[0] of X, transpose dim[0] and dim[1] of X, then
        split the result into out and scale.

        Inputs:
          X     : [SUM(M_1...M_N), K], bfloat16
          tokens_per_expert : list of int

        Outputs:
          out   : {[K, M_1], [K, M_2], ..., [K, M_N]}, float8_e4m3fn
          scale : {[M_1/FP8_ALIGN, K], [M_2/FP8_ALIGN, K], ..., [M_N/FP8_ALIGN, K]}, float

        Attrs:
          pow_2_scales
                : bool that indicates whether to use power-of-2 scaling
        use_ue8m0
                : bool that indicates whether to use ue8m0 scaling

        Requirements:
          1) M_i % FP8_ALIGN == 0 for M_i in [M_1, M_2, ..., M_N]
          2) K <= 65535 if pow_2_scales == False* FP8_ALIGN
        """

        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def align(x: int, y: int) -> int:
            return ceil_div(x, y) * y

        def get_tma_aligned_size(x: int, element_size: int) -> int:
            kNumTMAAlignmentBytes = 16
            assert kNumTMAAlignmentBytes % element_size == 0
            return align(x, kNumTMAAlignmentBytes // element_size)

        out, scale = [], []
        for tokens in tokens_per_expert:
            out.append(paddle.empty([x.shape[1], tokens], dtype="float8_e4m3fn"))
            if self.use_ue8m0:
                aligned_mn = get_tma_aligned_size(x.shape[1], 4)
                aligned_k = ceil_div(tokens, 128 * 4)
                scale.append(paddle.empty([aligned_k, aligned_mn], dtype="int32"))
            else:
                scale.append(paddle.empty([tokens // FP8_ALIGN, x.shape[1]], dtype="float32"))
        if hasattr(TDU, "fuse_transpose_split_fp8_quant"):
            TDU.fuse_transpose_split_fp8_quant(
                x, None, out, scale, tokens_per_expert, pow_2_scales, use_ue8m0=self.use_ue8m0
            )
        else:
            FQO.fused_transpose_split_quant(x, out, scale, pow_2_scales)
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2):
        """
        dw2 = do2_t * do3
        [n, k] = [n, m_sum] * [m_sum, k] (m_sum = sum(tokens_per_expert))
        """
        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(o2, self.tokens_per_expert, True)
        else:
            o2_t = o2.transpose([1, 0]).contiguous()
            o2_t_fp8, o2_t_scale = kitchen_quant(
                o2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            o2_t_fp8 = paddle.split(o2_t_fp8, num_or_sections=self.tokens_per_expert, axis=-1)
            o2_t_scale = paddle.split(
                o2_t_scale, num_or_sections=[int(x / FP8_ALIGN) for x in self.tokens_per_expert], axis=0
            )

        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(do3, self.tokens_per_expert, True)
        else:
            do3_t = do3.transpose([1, 0]).contiguous()
            do3_t_fp8, do3_t_scale = kitchen_quant(
                do3_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            do3_t_fp8 = paddle.split(do3_t_fp8, num_or_sections=self.tokens_per_expert, axis=-1)
            do3_t_scale = paddle.split(
                do3_t_scale, num_or_sections=[int(x / FP8_ALIGN) for x in self.tokens_per_expert], axis=0
            )

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)

                if self.use_ue8m0:
                    if o2_t_fp8[i].shape[1] != 0:
                        deep_gemm.gemm_fp8_fp8_bf16_nt(
                            (o2_t_fp8[i], o2_t_scale[i].T),
                            (do3_t_fp8[i], do3_t_scale[i].T),
                            expert_w2[i].main_grad,
                            expert_w2[i].main_grad,
                        )
                else:
                    kitchen_fp8_gemm(
                        o2_t_fp8[i],
                        o2_t_scale[i],
                        do3_t_fp8[i],
                        do3_t_scale[i],
                        True,
                        True,
                        expert_w2[i].main_grad,
                        paddle.float32,
                    )

            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)

                if self.use_ue8m0:
                    if o2_t_fp8[i].shape[1] != 0:
                        deep_gemm.gemm_fp8_fp8_bf16_nt(
                            (o2_t_fp8[i], o2_t_scale[i].T),
                            (do3_t_fp8[i], do3_t_scale[i].T),
                            expert_w2[i].grad,
                            expert_w2[i].grad,
                        )
                else:
                    kitchen_fp8_gemm(
                        o2_t_fp8[i],
                        o2_t_scale[i],
                        do3_t_fp8[i],
                        do3_t_scale[i],
                        True,
                        True,
                        expert_w2[i].grad,
                        paddle.float32,
                    )

            if hasattr(expert_w2[i], "_apply_backward_hook"):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1, clear_input=False):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """

        if input_x is None:
            if self.dequant_input:
                input_x = paddle.incubate.nn.functional.fused_act_dequant(self.input_fp8, self.input_scale)
            else:
                input_x = self.input
        if clear_input:
            self.input = None
            self.input_fp8 = None
            self.input_scale = None
        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(input_x, self.tokens_per_expert, True)
        else:
            input_x_t = input_x.transpose([1, 0]).contiguous()
            input_x_t_fp8, input_x_t_scale = kitchen_quant(
                input_x_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            input_x_t_fp8 = paddle.split(input_x_t_fp8, num_or_sections=self.tokens_per_expert, axis=-1)
            input_x_t_scale = paddle.split(
                input_x_t_scale, num_or_sections=[int(x / FP8_ALIGN) for x in self.tokens_per_expert], axis=0
            )

        del input_x

        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(do1, self.tokens_per_expert, True)
        else:
            do1_t = do1.transpose([1, 0]).contiguous()
            do1_t_fp8, do1_t_scale = kitchen_quant(
                do1_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            do1_t_fp8 = paddle.split(do1_t_fp8, num_or_sections=self.tokens_per_expert, axis=-1)
            do1_t_scale = paddle.split(
                do1_t_scale, num_or_sections=[int(x / FP8_ALIGN) for x in self.tokens_per_expert], axis=0
            )

        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                if self.use_ue8m0:
                    if input_x_t_fp8[i].shape[1] != 0:
                        deep_gemm.gemm_fp8_fp8_bf16_nt(
                            (input_x_t_fp8[i], input_x_t_scale[i].T),
                            (do1_t_fp8[i], do1_t_scale[i].T),
                            expert_w1[i].main_grad,
                            expert_w1[i].main_grad,
                        )
                else:
                    kitchen_fp8_gemm(
                        input_x_t_fp8[i],
                        input_x_t_scale[i],
                        do1_t_fp8[i],
                        do1_t_scale[i],
                        True,
                        True,
                        expert_w1[i].main_grad,
                        paddle.float32,
                    )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                if self.use_ue8m0:
                    if input_x_t_fp8[i].shape[1] != 0:
                        deep_gemm.gemm_fp8_fp8_bf16_nt(
                            (input_x_t_fp8[i], input_x_t_scale[i].T),
                            (do1_t_fp8[i], do1_t_scale[i].T),
                            expert_w1[i].grad,
                            expert_w1[i].grad,
                        )
                else:
                    kitchen_fp8_gemm(
                        input_x_t_fp8[i],
                        input_x_t_scale[i],
                        do1_t_fp8[i],
                        do1_t_scale[i],
                        True,
                        True,
                        expert_w1[i].grad,
                        paddle.float32,
                    )

            if hasattr(expert_w1[i], "_apply_backward_hook"):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert, origin_token_per_experts, output=None, scale=None):
        """如果传入了scale, 说明在a2a之前就做了quant, 这里的hs_out就是fp8。否则, hs_out是bf16"""
        self.origin_token_per_experts = origin_token_per_experts
        if hs_out is None:
            if self.fp8 is not None:
                assert self.input_fp8 is not None
                assert self.input_scale is not None
                shape = self.input_fp8.shape
            else:
                shape = self.input.shape
            dtype = paddle.bfloat16
        elif scale is not None:
            shape = hs_out.shape
            dtype = paddle.bfloat16
        else:
            shape = hs_out.shape
            dtype = hs_out.dtype

        if shape[0] == 0:
            o3 = paddle.zeros(shape, dtype=dtype)
            return o3
        # get w1/w2
        expert_w1 = [x.up_gate_proj.weight for x in self.experts if x is not None]
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, num_expert, tokens_per_expert, scale=scale)
        if not self.recompute_moe_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        if not self.bypass_down_proj:
            o3, _ = self.fwd_down(o1, unzipped_probs, expert_w2, num_expert)
        else:
            # 通过RR保存combine的输出, 避免了down_proj、zip、combine的重计算
            # 返回0size Tensor而不是None，以满足PyLayer的限制
            o3 = paddle.empty([0], "bfloat16")

        return o3

    @paddle.no_grad()
    def backward(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        反向传播函数，用于计算输入的梯度和参数的梯度。
            该函数会根据输出梯度更新模型的参数，并返回输入的梯度和隐藏状态的梯度。

            Args:
                out_grad (Tensor, optional): 输出梯度张量，默认为None，表示没有输出梯度。
                    shape为（batch_size, ...），dtype为float32。如果不为None，则需要保证batch_size大于等于1。

            Returns:
                tuple (dx, probs_grad) (Tensor, Tensor):
                    - dx (Tensor) - 输入的梯度张量，shape为（batch_size, ...），dtype为float32。
                    - probs_grad (Tensor) - 隐藏状态的梯度张量，shape为（batch_size, hidden_size），dtype为float32。
        """
        unzipped_probs = unzipped_probs.unsqueeze(-1)
        if out_grad.shape[0] == 0:
            # for cornet case, Get 0 teken in full train step
            dx = paddle.zeros_like(out_grad)
            probs_grad = paddle.zeros_like(unzipped_probs)

            for expert in self.experts:
                if expert is None:
                    continue

                if hasattr(expert.down_proj.weight, "main_grad"):
                    if expert.down_proj.weight.main_grad is None:
                        expert.down_proj.weight.main_grad = paddle.zeros(
                            shape=expert.down_proj.weight.shape, dtype=paddle.float32
                        )
                else:
                    if expert.down_proj.weight.grad is None:
                        expert.down_proj.weight.grad = paddle.zeros(
                            shape=expert.down_proj.weight.shape, dtype=paddle.float32
                        )

                if hasattr(expert.up_gate_proj.weight, "main_grad"):
                    if expert.up_gate_proj.weight.main_grad is None:
                        expert.up_gate_proj.weight.main_grad = paddle.zeros(
                            shape=expert.up_gate_proj.weight.shape, dtype=paddle.float32
                        )
                else:
                    if expert.up_gate_proj.weight.grad is None:
                        expert.up_gate_proj.weight.grad = paddle.zeros(
                            shape=expert.up_gate_proj.weight.shape, dtype=paddle.float32
                        )

            if a2a_async_fn:
                dx, task = a2a_async_fn(dx)
                task.wait()
            return dx, probs_grad

        subbatch_rows = self.moe_subbatch_token_num_after_dispatch
        if subbatch_rows is None:
            return self.backward_impl(out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn)

        assert (
            a2a_async_fn is None
        ), "a2a_async_fn should be None when moe_subbatch_token_num_after_dispatch is not None"
        assert self.expert_id is not None, self.expert_id

        rows, _ = out_grad.shape
        nparts = (rows + subbatch_rows - 1) // subbatch_rows
        if nparts <= 1:
            return self.backward_impl(out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn)

        input = self.input
        input_fp8 = self.input_fp8
        input_scale = self.input_scale.contiguous()
        o1 = self.o1
        tokens_per_expert = self.tokens_per_expert

        probs_grad = []
        for i in range(nparts):
            s_idx = subbatch_rows * i
            e_idx = min(rows, subbatch_rows * (i + 1))
            if input is not None:
                self.input = input._slice(s_idx, e_idx)

            if input_fp8 is not None:
                self.input_fp8 = input_fp8._slice(s_idx, e_idx)
                self.input_scale = input_scale._slice(s_idx, e_idx)

            if o1 is not None:
                self.o1 = o1._slice(s_idx, e_idx)
            self.tokens_per_expert = [e_idx - s_idx]

            tmp_out_grad = out_grad._slice(s_idx, e_idx)
            tmp_unzipped_probs = unzipped_probs._slice(s_idx, e_idx)

            tmp_dx, tmp_probs_grad = self.backward_impl(tmp_out_grad, tmp_unzipped_probs)
            assert tmp_dx is tmp_out_grad
            probs_grad.append(tmp_probs_grad)

        if self.input is not None:
            self.input = input

        if self.input_fp8 is not None:
            self.input_fp8 = input_fp8
            self.input_scale = input_scale

        if self.o1 is not None:
            self.o1 = o1

        self.tokens_per_expert = tokens_per_expert
        probs_grad = paddle.concat(probs_grad, axis=0)
        return out_grad, probs_grad

    def backward_impl_bf16(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl_bf16
        """
        if a2a_async_fn is not None:
            raise NotImplementedError("bf16 fuse node do not support a2a_async_fn currently")
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.experts if x is not None]
        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(None, expert_w1, len(expert_w1), self.tokens_per_expert)
        else:
            o1 = self.o1

        do1, o2_s, probs_grad = self.bwd_down_input_bf16(expert_w2, out_grad, o1, unzipped_probs)
        del o1
        self.o1 = None

        # dw1
        self.bf16_weight_grad(do1, self.input, expert_w1)
        self.input = None

        # dw2
        self.bf16_weight_grad(out_grad, o2_s, expert_w2)

        # dx
        dx = self.bwd_gate_up_input_bf16(do1, expert_w1)
        del do1
        self.reset_statue()
        return dx, probs_grad

    def backward_impl(self, out_grad, unzipped_probs, a2a_async_fn=None):
        if self.fp8 is None:
            return self.backward_impl_bf16(out_grad, unzipped_probs, a2a_async_fn)
        else:
            return self.backward_impl_fp8(out_grad, unzipped_probs, a2a_async_fn)

    def backward_impl_fp8(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl
        """
        # recompute expert_w2 and expert_w1
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.experts if x is not None]

        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(None, expert_w1, len(expert_w1), self.tokens_per_expert)
        else:
            o1 = self.o1

        # do2
        do1, o2_s, probs_grad = self.bwd_dowm_input_fp8(
            expert_w2, out_grad, o1, unzipped_probs, inplace_swiglu_prob=True
        )
        del o1
        self.o1 = None

        if a2a_async_fn is None:
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            self.input_fp8 = None
            self.input_scale = None
            self.input = None

            # dw2
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2)
            else:
                self.bwd_down_weight(out_grad, o2_s, expert_w2)

            # dx
            dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)
            del do1
        else:
            # 为了更充分地overlap, 将dx提前。不过这样可能会增加峰值显存。

            # dx
            dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)

            dx, task = a2a_async_fn(dx)
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            self.input_fp8 = None
            self.input_scale = None
            self.input = None
            del do1

            # dw2
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2)
            else:
                self.bwd_down_weight(out_grad, o2_s, expert_w2)

            task.wait()

        self.reset_statue()
        return dx, probs_grad

    def bf16_weight_grad(self, dy, x, weights):
        """
        BF16 GEMM for weight grad
        """
        if x is None:
            if self.dequant_input:
                x = paddle.incubate.nn.functional.fused_act_dequant(self.input_fp8, self.input_scale)
            else:
                x = self.input

        start_idx = 0
        for i, n in enumerate(self.tokens_per_expert):
            if weights[i].main_grad is None:
                weights[i].main_grad = paddle.zeros(weights[i].shape, dtype=paddle.float32)
            if n > 0:
                n = (n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
                end_idx = start_idx + n

                if get_env_device() == "xpu":
                    x_in = x._slice(start_idx, end_idx)
                    dy_in = dy._slice(start_idx, end_idx)
                    grad = weights[i].main_grad
                    # print(f"(lilujia): x_in: {x_in}")
                    # print(f"(lilujia): dy_in: {dy_in}")
                    # print(f"(lilujia): grad: {grad}")
                    paddle._C_ops.fused_linear_param_grad_add(
                        paddle.cast(x_in, paddle.float32),
                        paddle.cast(dy_in, paddle.float32),
                        grad,
                        None,
                        True,
                        False,
                    )
                else:

                    def grad_func(x, dy, weight, start_idx, end_idx):
                        # We disable auto_cast here because this kernel natually has
                        # inputs of different dtypes, which may confuse auto_cast.
                        with paddle.amp.auto_cast(False):
                            paddle._C_ops.fused_linear_param_grad_add(
                                x._slice(start_idx, end_idx),
                                dy._slice(start_idx, end_idx),
                                weight.main_grad,
                                None,
                                True,
                                False,
                            )

                    if WeightGradStore.enabled:
                        WeightGradStore.put(partial(grad_func, x, dy, weights[i], start_idx, end_idx))
                    else:
                        grad_func(x, dy, weights[i], start_idx, end_idx)
                start_idx = end_idx

            if hasattr(weights[i], "_apply_backward_hook"):
                weights[i]._apply_backward_hook()


class ExpertsGroupGemmWLCHNode(ExpertsGroupGemmContiguousNode):
    """ExpertsGroupGemmWLCHNod"""

    def __init__(
        self,
        custom_map,
        recompute_moe_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_WLCH_node",
    ):
        """
            Initializes the experts group gemm WLCH node.

        Args:
            custom_map (CustomMapping): Custom mapping for the model.
            recompute_moe_gate_up (bool, optional): Whether to recompute forward gate up. Defaults to False.
            dequant_input (bool, optional): Whether to dequantize input. Defaults to False.
            name (str, optional): Name of the node. Defaults to "experts_group_gemm_contiguous_node".
        """

        super().__init__(
            custom_map,
            recompute_moe_gate_up=recompute_moe_gate_up,
            dequant_input=dequant_input,
            group=group,
            name=name,
        )

        self.fp8_fused_ops_configs["transpose_split_quant"] = True
        self.moe_grouped_gemm = True

        self.w = custom_map.world_size
        self.l = custom_map.num_local_experts

    def gen_m_indices(self, tokens_per_expert):
        """
        generate m indices
        """

        m_indices = paddle.arange(self.l, dtype=paddle.int32).repeat_interleave(tokens_per_expert[0])
        m_indices = m_indices.reshape([self.w, self.l, -1]).transpose([1, 0, 2]).contiguous().reshape([-1])

        return m_indices

    def fused_transpose_split_quant(self, x, tokens_per_expert, pow_2_scales):
        """
        Quantize on dim[0] of X, transpose dim[0] and dim[1] of X, then
        split the result into out and scale.

        Inputs:
          X     : [SUM(M_1...M_N), K], bfloat16
          tokens_per_expert : list of int

        Outputs:
          out   : {[K, M_1], [K, M_2], ..., [K, M_N]}, float8_e4m3fn
          scale : {[M_1/FP8_ALIGN, K], [M_2/FP8_ALIGN, K], ..., [M_N/FP8_ALIGN, K]}, float

        Attrs:
          pow_2_scales
                : bool that indicates whether to use power-of-2 scaling

        Requirements:
          1) M_i % FP8_ALIGN == 0 for M_i in [M_1, M_2, ..., M_N]
          2) K <= 65535 if pow_2_scales == False* FP8_ALIGN
        """
        if fused_transpose_wlch_split_quant is None:
            s, h = x.shape
            x = x.reshape([self.w, self.l, -1, h]).transpose([1, 0, 2, 3]).contiguous().reshape([s, h])

            # print( "x shape", x.shape)
            # print("token per expert", tokens_per_expert)
            out, scale = [], []
            for tokens in tokens_per_expert:
                out.append(paddle.empty([x.shape[1], tokens], dtype="float8_e4m3fn"))
                scale.append(paddle.empty([tokens // FP8_ALIGN, x.shape[1]], dtype="float32"))
            if hasattr(TDU, "fuse_transpose_split_fp8_quant"):
                TDU.fuse_transpose_split_fp8_quant(
                    x, None, out, scale, tokens_per_expert, pow_2_scales, use_ue8m0=self.use_ue8m0
                )
            else:
                FQO.fused_transpose_split_quant(x, out, scale, pow_2_scales)
        else:
            s, h = x.shape
            x = x.reshape([self.w, self.l, -1, h])
            out, scale = fused_transpose_wlch_split_quant(x, tokens_per_expert, pow_2_scales=pow_2_scales)

        return out, scale
