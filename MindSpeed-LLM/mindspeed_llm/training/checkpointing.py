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
from megatron.training import get_args
from megatron.training.utils import print_rank_0
from megatron.training.checkpointing import _load_base_checkpoint
from mindspeed_llm.tasks.finetune.lora.utils import is_enable_lora, merge_dicts, modify_keys_with_dict
from mindspeed_llm.tasks.post_train.utils import load_checkpoint_loosely


def _load_base_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        state_dict, checkpoint_name, release = fn(*args, **kwargs)
        rank0 = kwargs.pop('rank0')
        if is_enable_lora() and state_dict is not None:
            args_ = get_args()
            words_to_match = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
            exclude_words = ['base_layer', 'lora_', 'norm']
            state_dict = modify_keys_with_dict(state_dict, words_to_match, exclude_words)

            if not args_.lora_load:
                return state_dict, checkpoint_name, release

            # Read the tracker file and set the iteration.
            state_dict_lora, checkpoint_name_lora, release_lora = fn(args_.lora_load, rank0)
            if state_dict_lora is not None:
                merge_dicts(state_dict, state_dict_lora)
                checkpoint_name = checkpoint_name_lora
                release = release_lora
        return state_dict, checkpoint_name, release
    return wrapper


def load_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_enable_lora() or load_checkpoint_loosely():
            kwargs['strict'] = False
        return fn(*args, **kwargs)

    return wrapper


def load_args_from_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not isinstance(res, tuple):
            return res
        args, checkpoint_args = res
        
        def _set_arg(arg_name, old_arg_name=None, force=False):
            if not force and getattr(args, arg_name, None) is not None:
                return
            if old_arg_name is not None:
                checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
            else:
                checkpoint_value = getattr(checkpoint_args, arg_name, None)
            if checkpoint_value is not None:
                print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
                setattr(args, arg_name, checkpoint_value)
            else:
                print_rank_0(f"Checkpoint did not provide arguments {arg_name}")
        
        _set_arg('num_layer_list', force=True)
        _set_arg('post_norm', force=True)
        _set_arg('num_experts')
        _set_arg('sequence_parallel', force=True)
        _set_arg('n_shared_experts', force=True)
        _set_arg('qk_layernorm', force=True)
        _set_arg('moe_intermediate_size', force=True)
        _set_arg('first_k_dense_replace', force=True)
        _set_arg('moe_layer_freq', force=True)
        _set_arg('multi_head_latent_attention', force=True)
        _set_arg('qk_rope_head_dim', force=True)
        _set_arg('qk_nope_head_dim', force=True)
        _set_arg('q_lora_rank', force=True)
        _set_arg('kv_lora_rank', force=True)
        _set_arg('v_head_dim', force=True)
        _set_arg('shared_expert_gate', force=True)

        state_dict, checkpoint_name, release = _load_base_checkpoint(
            getattr(args, kwargs.get('load_arg', 'load')),
            rank0=True,
            exit_on_missing_checkpoint=kwargs.get('exit_on_missing_checkpoint', False),
            checkpoint_step=args.ckpt_step
        )
        checkpoint_version = state_dict.get('checkpoint_version', 0)
        if checkpoint_version >= 3.0:
            _set_arg('expert_model_parallel_size', force=True)
            
        return args, checkpoint_args
    
    return wrapper
