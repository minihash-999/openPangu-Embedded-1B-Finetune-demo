# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import os
from functools import wraps
import torch
from megatron.training import get_args


def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operation
    for this bucket.

    When overlap_grad_reduce is set to True, dispatches an asynchronous
    communication call. When overlap_grad_reduce is set to False, makes
    synchronous call.
    """
    assert (
            self.communication_handle is None and not self.communication_issued
    ), 'Should not have multiple communication calls in flight at once'

    # Make sure norm of grads in bucket are not NaN
    # prior to data-parallel all-reduce / reduce-scatter.
    if self.check_for_nan_in_grad:
        global_rank = torch.distributed.get_rank()
        norm = self.grad_data.norm(p=2)
        assert not norm.isnan(), (
            f'Rank {global_rank}: found NaN in local grad norm in '
            f'backward pass before data-parallel communication collective. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    self.grad_data *= self.gradient_scaling_factor
    # Use async_op only when overlap_grad_reduce is True.
    self.communication_handle = torch.distributed.all_reduce(
        self.grad_data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
    )
    self.communication_issued = True


def start_grad_sync_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        args = get_args()
        if args.enable_high_availability:
            start_grad_sync(self)
        else:
            fn(self)
    return wrapper