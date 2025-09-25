# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
import importlib
import os
import sys
from functools import wraps
import torch.multiprocessing as mp
from mindspeed_llm import megatron_adaptor
from pretrain_gpt import model_provider

MODULE_ROOT = "mindspeed_llm.tasks.checkpoint"


def load_plugin(plugin_type, name):
    if name == '':
        module_name = f"{MODULE_ROOT}.{plugin_type}"
    else:
        module_name = f"{MODULE_ROOT}.{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = f"{MODULE_ROOT}.{name}"
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


def main():

    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default=None, const=None, choices=['hf', 'mg'],
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')
    parser.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                            'that returns a spec to customize transformer layer, depending on the use case.')
    parser.add_argument('--model-type-hf', type=str, default="llama2",
                        choices=['baichuan', 'baichuan2', 'llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2', 'bloom',
                                 'qwen', 'internlm2', 'deepseek2', 'minicpm', 'minicpm-moe', 'deepseek2-lite', 'qwen2-moe'],
                        help='model type of huggingface')
    known_args, _ = parser.parse_known_args()

    use_saver = known_args.load_model_type is None
    if use_saver:
        loader = load_plugin('loader', known_args.loader)
        saver = load_plugin('saver', known_args.saver)
    else:
        loader = load_plugin('loader', known_args.load_model_type)
        saver = load_plugin('saver', '')

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_model_checkpoint, args=(model_provider, queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(model_provider, queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()


if __name__ == '__main__':
    main()

