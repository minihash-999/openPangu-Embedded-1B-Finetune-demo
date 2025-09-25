# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses


def attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        if args.context_parallel_size > 1 and args.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo', 'hybrid_adaptive_cp_algo']:
            ulysses_group = mpu.get_context_parallel_group()
            if args.context_parallel_algo == 'hybrid_cp_algo' or args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)

    return wrapper
