# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from typing import Dict, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from megatron.training import get_args, get_model
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_llm.tasks.post_train.base import BaseTrainer
from mindspeed_llm.tasks.post_train.dpo.dpo_model import DPOModel
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank


class DPOTrainer(BaseTrainer):
    """
    A trainer class for Direct Preference Optimization (DPO).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """
    IGNORE_INDEX = -100

    def __init__(self):
        """
        Initializes the DPOTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the DPO model.
        """
        super().__init__(
            model_provider=self.model_provider,
            get_batch_func=self.get_batch,
            loss_func=self.loss_func,
            forward_step_func=self.forward_step)

        self.args.actual_micro_batch_size = self.args.micro_batch_size * 4
        self.hyper_model = DPOModel(
            self.train_args[1][0],
            self._init_reference_model()
        )   

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch identical to Llama factory"""
        args = get_args()

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                return tokens, None, attention_mask, None
            else:
                return None, None, attention_mask, None
                
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        data_type = torch.int64

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        position_ids = None

        all_results = (tokens, labels, attention_mask, position_ids)
        return all_results

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """DPO Loss function.

        Args:
            input_tensor (torch.Tensor): The tensor with the labels (repeated in double)
            output_tensor (torch.Tensor): The tensor with the Policy and Reference Model's Logits
        """
        args = get_args()
        labels = input_tensor[:args.actual_micro_batch_size // 2, ...]

        all_policy_logits, all_reference_logits = torch.chunk(output_tensor, 2, dim=0)
        all_reference_logits = all_reference_logits.detach()

        loss, metrics = self.get_batch_loss_metrics(
            all_policy_logits,
            all_reference_logits,
            labels
        )

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        metrics['lm loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        return loss, metrics

    def forward_step(self, data_iterator, model):
        """DPO Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, attention_mask, position_ids = self.get_batch(data_iterator)
        self.timers('batch-generator').stop()

        output_tensor = self.hyper_model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, labels)

    @staticmethod
    def vocab_parallel_log_softmax(logits):
        """
        Compute log softmax values for each sets of scores in vocab parallel.

        Args:
            logits (Tensor): Input logits.

        Returns:
            Tensor: Log softmax values.
        """
        # Step 1: Compute the local max value for numerical stability
        z_max = logits.max(dim=-1, keepdim=True).values

        # Step 2: Perform all-reduce to get the global max across all processes
        torch.distributed.all_reduce(
            z_max,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group()
        )

        # Step 3: Compute the log(exp(x - z_max)) for local logits
        local_exp = torch.exp(logits - z_max)

        # Step 4: Compute local sum of exp(x - z_max)
        local_sum_exp = local_exp.sum(dim=-1, keepdim=True)

        # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
        torch.distributed.all_reduce(
            local_sum_exp,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        # Step 6: Compute the log of the global sum of exp(x - z_max)
        log_sum_exp = local_sum_exp.log()

        # Step 7: Compute and return the log softmax values
        return logits - z_max - log_sum_exp

    def dpo_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        reference_chosen_log_probs: torch.Tensor,
        reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs:
            Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_log_probs:
            Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_log_probs:
            Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_log_probs:
            Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
        ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs
        logits = pi_log_ratios - ref_log_ratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0.
        # The label_smoothing parameter encodes our uncertainty about the labels and calculates a conservative DPO loss.
        if self.args.dpo_loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.args.dpo_beta * logits) * (1 - self.args.dpo_label_smoothing)
                - F.logsigmoid(-self.args.dpo_beta * logits) * self.args.dpo_label_smoothing
            )
        elif self.args.dpo_loss_type == "hinge":
            losses = torch.relu(1 - self.args.dpo_beta * logits)
        elif self.args.dpo_loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.args.dpo_beta)) ** 2
        elif self.args.dpo_loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_kl = (policy_chosen_log_probs - reference_chosen_log_probs).mean().clamp(min=0)
            rejected_kl = (policy_rejected_log_probs - reference_rejected_log_probs).mean().clamp(min=0)

            chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
            rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_log_ratios - rejected_kl)),
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_kl - rejected_log_ratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.args.dpo_loss_type}."
                f" Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.args.dpo_beta
            * (
                    policy_chosen_log_probs - reference_chosen_log_probs
            ).detach()
        )
        rejected_rewards = (
            self.args.dpo_beta
            * (
                    policy_rejected_log_probs - reference_rejected_log_probs
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        reference_chosen_log_probs: torch.Tensor,
        reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the preference loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs: Log probabilities of the policy model for the chosen responses.
            policy_rejected_log_probs: Log probabilities of the policy model for the rejected responses.
            reference_chosen_log_probs: Log probabilities of the reference model for the chosen responses.
            reference_rejected_log_probs: Log probabilities of the reference model for the rejected responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the preference loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs
        )
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            all_policy_logits,
            all_reference_logits,
            label
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes the sum log probabilities of the labels under the given logits.

        Otherwise, the average log probabilities.

        Args:
            all_policy_logits: Logits of the policy model.
            all_reference_logits: Logits of the reference model.
            label: The label tensor.

        Returns:
            A tuple containing the loss tensor and a dictionary of metrics.
        """
        metrics = {}

        (
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            policy_chosen_log_probs_avg,
        ) = self._compute_log_probs(all_policy_logits, label)

        reference_chosen_log_probs, reference_rejected_log_probs, *_ = self._compute_log_probs(
            all_reference_logits,
            label
        )

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs,
        )

        sft_loss = -policy_chosen_log_probs_avg
        if self.args.pref_ftx > 1e-6:
            losses += self.args.pref_ftx * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = ""
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()
        if self.args.dpo_loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean()
            metrics["{}odds_ratio_loss".format(prefix)] = (
                    (losses - sft_loss) / self.args.dpo_beta).detach().mean()

        return losses.mean(), metrics

    def _init_reference_model(self):
        """
        Initializes the reference model frozen.

        Returns:
            The initialized reference model.
        """
        model = get_model(self.model_provider, ModelType.encoder_or_decoder)

        self.args.load = self.args.ref_model if self.args.ref_model is not None else self.args.load
        if self.args.load:
            strict = False if self.args.lora_load else True
            # to avoid assert error
            consumed_train_samples = self.args.consumed_train_samples
            self.args.consumed_train_samples = 0
            load_checkpoint(model, None, None, 'load', strict)
            self.args.consumed_train_samples = consumed_train_samples

        return model[0].eval()

    def _compute_log_probs(self, all_logits, label) -> Tuple[torch.Tensor, ...]:
        """
        Computes the sum log probabilities of the labels under given logits if loss_type.
        Otherwise, the average log probabilities.
        Assuming IGNORE_INDEX is all negative numbers, the default is -100.

        Args:
            all_logits: The logits tensor.
            label: The label tensor.

        Returns:
            A tuple containing the log probabilities and other tensors.
        """
        label = label[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        batch_size = all_logits.size(0) // 2

        all_log_probs, valid_length = self._get_batch_log_probs(
            all_logits,
            label
        )

        if self.args.dpo_loss_type in ["ipo", "orpo", "simpo"]:
            all_log_probs = all_log_probs / torch.clamp(valid_length, min=1)

        chosen_log_probs, rejected_log_probs = all_log_probs.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        all_results = (chosen_log_probs, rejected_log_probs, chosen_log_probs / chosen_length)

        return all_results

    def _get_batch_log_probs(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log probabilities of the given labels under the given logits.

        In the tensor parallelism case, it takes into account the vocab parallelism and
        performs the necessary adjustments to the labels and logits.

        Args:
            logits: The logits tensor.
            labels: The label tensor.

        Returns:
            A tuple containing the log probabilities and the valid length.
        """
        if mpu.get_tensor_model_parallel_world_size() > 1:
            tp_vocab_size = logits.size(2)

            labels -= mpu.get_tensor_model_parallel_rank() * tp_vocab_size
            labels = labels.masked_fill(torch.logical_or(labels < 0, labels >= tp_vocab_size), 0)
            loss_mask = labels != 0

            per_token_log_probs = torch.gather(
                self.vocab_parallel_log_softmax(logits), dim=2, index=labels.unsqueeze(2)).squeeze(2)

            all_log_probs = (per_token_log_probs * loss_mask).sum(-1)
            valid_length = loss_mask.sum(-1)

            torch.distributed.all_reduce(
                all_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

            torch.distributed.all_reduce(
                valid_length,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

        else:
            label_pad_token_id = self.IGNORE_INDEX
            loss_mask = labels != label_pad_token_id
            labels[labels == label_pad_token_id] = 0  # dummy token
            per_token_log_probs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            all_log_probs = (per_token_log_probs * loss_mask).sum(-1)
            valid_length = loss_mask.sum(-1)

        return all_log_probs, valid_length
