# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

from functools import wraps
from typing import Optional

import torch

from megatron.training import get_args
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region
)
from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
    ColumnParallelLinear
)
from megatron.legacy.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.core import parallel_state

from mindspeed.utils import get_actual_seq_len, set_actual_seq_len


def vocab_embedding_init_wrapper(fn):
    """Patch for legacy norm."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        if parallel_state.is_pipeline_first_stage() and args.embed_layernorm:
            norm = MixedFusedLayerNorm(args.hidden_size)
            self.norm = norm
    return wrapper


def vocab_embedding_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        output = fn(self, *args, **kwargs)
        args_ = get_args()
        if hasattr(self, 'norm'):
            output = self.norm(output)
        return output * args_.embedding_multiplier_scale if args_.embedding_multiplier_scale else output
    return wrapper


class SegmentedColumnParallelLinear(ColumnParallelLinear):
    def __int__(self):
        super(ColumnParallelLinear, self).__init__()

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        args_ = get_args()
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                        self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None

        if (
                self.async_tensor_model_parallel_allreduce
                or self.sequence_parallel
                or self.explicit_expert_comm
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        weight = torch.split(weight, weight.shape[0] // args_.output_layer_slice_num, dim=0)

        output_parallel = []
        for i in range(args_.output_layer_slice_num):
            output_parallel.append(self._forward_impl(
                input=input_parallel,
                weight=weight[i],
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=False
                if self.explicit_expert_comm
                else self.async_tensor_model_parallel_allreduce,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None,
            ))
        output_parallel = torch.cat(output_parallel, dim=2)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def checkpoint_forward_wrapper(fn):
    def wrapper(ctx, run_function, distribute_saved_activations, *args):
        ctx.actual_seq_len = get_actual_seq_len()
        return fn(ctx, run_function, distribute_saved_activations, *args)

    return wrapper


def checkpoint_backward_wrapper(fn):
    def wrapper(ctx, *args):
        set_actual_seq_len(ctx.actual_seq_len)
        return fn(ctx, *args)

    return wrapper
