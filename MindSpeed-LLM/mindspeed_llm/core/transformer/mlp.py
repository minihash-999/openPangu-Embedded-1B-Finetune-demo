# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import math
import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig


def should_recompute_activation(self):
    args = get_args()
    if not args.recompute_activation_function or self.layer_number is None:
        return False

    activation_recompute_layers = args.recompute_activation_function_num_layers
    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = args.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = args.num_layers // pp_size
    else:
        layer_per_chunk = args.num_layers

    if vpp_rank is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1
    recompute_priority = ((self.layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = args.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full re-computation when both full re-computation and activation re-computation are enabled
            return False
        elif activation_recompute_layers is None:
            # Do activation function re-computation
            return True
        elif recompute_priority < full_recompute_layers + activation_recompute_layers:
            # Do activation function re-computation
            return True
        else:
            # No recomputation
            return False

    if activation_recompute_layers is None:
        # Do activation function re-computation
        return True
    else:
        return recompute_priority < activation_recompute_layers


def core_mlp_init(self, config, submodules, is_expert=False, input_size=None):
    super(MLP, self).__init__(config=config)

    self.config: TransformerConfig = config

    self.input_size = input_size if input_size != None else self.config.hidden_size
    # geglu activation function
    _args = get_args()
    if _args.geglu:
        self.config.gated_linear_unit = True
        self.config.activation_func = F.gelu
        self.config.bias_gelu_fusion = False

    if _args.gelu_tanh:
        def gelu_tanh_approximation(x):
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        self.config.gated_linear_unit = True
        self.config.activation_func = gelu_tanh_approximation
        self.config.bias_gelu_fusion = False

    ffn_hidden_size = self.config.ffn_hidden_size
    if self.config.gated_linear_unit:
        ffn_hidden_size *= 2

    self.linear_fc1 = build_module(
        submodules.linear_fc1,
        self.input_size,
        ffn_hidden_size,
        config=self.config,
        init_method=self.config.init_method,
        gather_output=False,
        bias=self.config.add_bias_linear,
        skip_bias_add=True,
        is_expert=is_expert,
        tp_comm_buffer_name='fc1',
    )

    self.activation_func = self.config.activation_func

    self.linear_fc2 = build_module(
        submodules.linear_fc2,
        self.config.ffn_hidden_size,
        self.config.hidden_size,
        config=self.config,
        init_method=self.config.output_layer_init_method,
        bias=self.config.add_bias_linear,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=is_expert,
        tp_comm_buffer_name='fc2',
    )
