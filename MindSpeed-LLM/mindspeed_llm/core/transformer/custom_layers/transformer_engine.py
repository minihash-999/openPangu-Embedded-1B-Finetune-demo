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

import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model.rms_norm import RMSNorm


class PTNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
            cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            instance = nn.LayerNorm(
                normalized_shape=hidden_size,
                eps=eps,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(
                dim=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
