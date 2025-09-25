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


def transformer_language_model_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if global_args.variable_seq_lengths:
            self.seq_length = args[0].shape[1]
        else:
            self.seq_length = global_args.seq_length
        return fn(self, *args, **kwargs)

    return wrapper
