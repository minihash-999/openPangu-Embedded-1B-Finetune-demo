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

from .gpt_model import GPTModel, post_language_model_processing
from .transformer import (
    parallel_transformer_init, state_dict_for_save_checkpoint_wrapper,
    core_attention_wrapper, FlashSelfAttention, core_attention_forward,
    ParallelAttention_wrapper, ParallelAttentionForward,
    parallel_transformer_forward, parallel_mlp_init_wrapper
)
from .language_model import transformer_language_model_forward_wrapper
from .rms_norm import rms_norm_init_wrapper, rms_norm_forward
