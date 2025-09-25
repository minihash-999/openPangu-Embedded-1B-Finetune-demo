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


import torch
import torch_npu
from .ascend_turbo_cfg import ascend_turbo_cfg


class ColumnSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None

        rank = torch.distributed.get_rank(group)
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )

        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        world_size = ascend_turbo_cfg.get_world_size()
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight.t(),
            hcomm_info,
            world_size,
            bias=bias,
            gather_index=0,
            gather_output=(not ascend_turbo_cfg.all_gather_recomputation),
        )

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )

        ctx.all_gather_output = all_gather_grad_output
        ctx.world_size = world_size
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        if ascend_turbo_cfg.all_gather_recomputation:
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * ctx.world_size
            all_gather_output = torch.empty(
                dim_size,
                dtype=input_.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            all_gather_work = torch.distributed._all_gather_base(
                all_gather_output, input_.contiguous(), group=ctx.group, async_op=True
            )
        else:
            all_gather_output = ctx.all_gather_output

        grad_input = grad_output_.matmul(weight)
        grad_input = grad_input.reshape(
            grad_output.shape[0], grad_output.shape[1], weight.shape[1]
        )

        sub_grad_input = torch.empty(
            list(input_.size()), dtype=input_.dtype, device=torch.cuda.current_device()
        )
        reduce_scatter_work = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=ctx.group, async_op=True
        )

        if ascend_turbo_cfg.all_gather_recomputation:
            all_gather_work.wait()
        all_gather_output = all_gather_output.reshape(
            all_gather_output.shape[0] * all_gather_output.shape[1],
            all_gather_output.shape[2],
        )

        grad_weight = grad_output_.t().matmul(all_gather_output)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output_.sum(dim=0)
                if grad_output_.is_contiguous()
                else grad_output_.t().sum(dim=1)
            )
        else:
            grad_bias = None

        reduce_scatter_work.wait()
        return sub_grad_input, grad_weight, grad_bias, None


class RowSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None

        rank = torch.distributed.get_rank(group)
        world_size = ascend_turbo_cfg.get_world_size()
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=bias
        )

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        grad_input, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )

        grad_input = grad_input.view_as(input_)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        grad_weight = all_gather_grad_output.t().matmul(x)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output.sum(dim=0)
                if grad_output.is_contiguous()
                else grad_output.t().sum(dim=1)
            )
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None
