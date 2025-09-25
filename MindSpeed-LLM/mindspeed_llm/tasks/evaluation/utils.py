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

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task-data-path",
                       nargs='*',
                       default=[],
                       help='Path to the training dataset. Accepted format:'
                            '1) a single data path, 2) multiple datasets in the'
                            'form: dataset1-path dataset2-path ...')
    group.add_argument("--temperature", type=float, default=0.5,
                       help='Sampling temperature.')
    group.add_argument("--evaluation-batch-size", type=int, default=1,
                       help='Size of evaluation batch')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top-p", type=float, default=0.9,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--max-new-tokens", type=int, default=128,
                       help='Size of the output generated text.')
    group.add_argument("--task", nargs='*', default=[], help='Choose one task from mmlu, boolq and gsm8k')
    group.add_argument("--instruction-template", type=str, default="",
                       help="Instruction template for the evaluation task.")
    group.add_argument("--no-chat-template", action="store_true", default=False,
                       help="Disable Huggingface chat template")
    group.add_argument('--use-kv-cache', action="store_true", default=False,
                       help="Use kv cache to accelerate inference")
    group.add_argument('--hf-chat-template', action='store_true', default=False,
                        help="Using Huggingface chat template")
    group.add_argument('--eval-language', type=str, default='en',
                        choices=['en', 'zh'], help="Language used by evaluation")
    group.add_argument('--max-eval-samples', type=int, default=None,
                        help="Max sample each dataset, for debug")
    return parser
