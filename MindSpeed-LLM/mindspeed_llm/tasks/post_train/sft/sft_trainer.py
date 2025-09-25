# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from functools import partial
import torch
from megatron.training import get_args
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training import get_timers
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank, generate_actual_seq_len
from mindspeed_llm.tasks.post_train.base import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_provider=self.model_provider,
            get_batch_func=self.get_batch,
            loss_func=self.loss_func,
            forward_step_func=self.forward_step)
    
    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        args = get_args()
        if args.reset_position_ids:
            keys += ['position_ids']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                return tokens, None, None, attention_mask, None
            else:
                if args.reset_position_ids:
                    # Broadcast data.
                    data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                    generate_actual_seq_len(data_b)
                    attention_mask = None
                batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == -100, 0, 1)

        if args.reset_position_ids:
            position_ids = data_b.get('position_ids').long()
            generate_actual_seq_len(data_b)
            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
            }
            batch = get_batch_on_this_cp_rank(batch)
            batch['attention_mask'] = None
            batch['position_ids'] = position_ids
            return batch.values()

        attention_mask = get_tune_attention_mask(attention_mask_1d)
        position_ids = None
        batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()


    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        loss_mask = input_tensor

        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].view(-1).float()
        if self.args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss[0] / loss[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if self.args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

        return output_tensor, partial(self.loss_func, loss_mask)

