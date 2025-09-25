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

import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, save_to_aux_losses_tracker
from megatron.core import parallel_state

from .moe_utils import topk_softmax_with_capacity, switch_load_balancing_loss_func


def group_limited_greedy_topKgating(self, logits: torch.Tensor):
    args = get_args()
    seq_length = logits.shape[0]
    
    scores = F.softmax(logits, dim=1)
    group_scores = (
        scores.view(args.micro_batch_size * seq_length, args.expert_model_parallel_size, -1).max(dim=-1).values
    )  # [n, EP]

    group_idx = torch.topk(group_scores, k=args.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]

    group_mask = torch.zeros_like(group_scores)  # [n, EP]
    group_mask.scatter_(1, group_idx, 1)  # [n, EP]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            args.micro_batch_size * seq_length, args.expert_model_parallel_size, args.num_experts // args.expert_model_parallel_size
        )
        .reshape(args.micro_batch_size * seq_length, -1)
    )  # [n, e]

    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    topk_weight, topk_idx = torch.topk(
        tmp_scores, k=args.moe_router_topk, dim=-1, sorted=False
    )

    ### norm gate to sum 1
    if args.moe_router_topk > 1 and args.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    else:
        topk_weight = topk_weight * args.routed_scaling_factor

    if not self.training:
        l_aux = None
        self.l_aux = l_aux
        return topk_weight, topk_idx

    scores_for_aux = scores  # [s*b, n_global_experts]
    topk_idx_for_aux_loss = topk_idx.view(args.micro_batch_size, -1)  # [b, s*top_k]
    topk_group_idx_for_aux_loss = group_idx.view(args.micro_batch_size, -1)  # [b, s*topk_group]
    fi, Pi, l_aux = None, None, 0

    #########################################################
    ################ Expert-Level Balance Loss #############
    #########################################################
    if self.config.moe_aux_loss_coeff > 0:
        l_expert_aux = 0
        # always compute aux loss based on the naive greedy topk method
        if args.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)
            # [b, s, n_global_experts]

            ce = torch.zeros(
                args.micro_batch_size, args.num_experts, device=logits.device
            )  # [b, n_global_experts]
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(args.micro_batch_size, seq_length * args.moe_router_topk, device=logits.device),
            )
            fi = ce.div(seq_length * args.moe_router_topk / args.num_experts)  # [b, n_global_experts]
            Pi = scores_for_seq_aux.mean(dim=1)  # [b, n_global_experts]
            l_expert_aux = (Pi * fi).sum(dim=1).mean() * self.config.moe_aux_loss_coeff
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
            )
            ce = mask_ce.to(logits.dtype).mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * args.num_experts
            l_expert_aux = (Pi * fi).sum() * self.config.moe_aux_loss_coeff

        self.l_expert_aux = l_expert_aux
        l_aux += l_expert_aux

    #########################################################
    ################ Device-Level Balance Loss ##############
    #########################################################
    P_devi = None
    args.n_group = args.expert_model_parallel_size
    if args.moe_device_level_aux_loss_coeff > 0:
        l_device_aux = 0
        if args.seq_aux:
            if fi is None:
                scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)

                ce = torch.zeros(
                    args.micro_batch_size, args.num_experts, device=logits.device
                )  # [b, n_global_experts]
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(args.micro_batch_size, seq_length * args.moe_router_topk, device=logits.device),
                )
                fi = ce.div(seq_length * args.moe_router_topk / args.num_experts)  # [b, n_global_experts]
                Pi = scores_for_seq_aux.mean(dim=1)  # [b, n_global_experts]

            P_devi = Pi.view(args.micro_batch_size, args.n_group, -1).sum(-1)  # [b, n_group]
            f_devi = fi.view(args.micro_batch_size, args.n_group, -1).mean(-1)
            l_device_aux = (f_devi * P_devi).sum(dim=1).mean() * args.moe_device_level_aux_loss_coeff

        else:
            if fi is None:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
                )
                ce = mask_ce.to(logits.dtype).mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * args.num_experts

            P_devi = Pi.view(args.n_group, -1).sum(-1)
            f_devi = fi.view(args.n_group, -1).mean(-1)
            l_device_aux = (f_devi * P_devi).sum() * args.moe_device_level_aux_loss_coeff

        self.l_device_aux = l_device_aux
        l_aux += l_device_aux

    ##########################################################
    ################ Communication Balance Loss ##############
    ##########################################################
    if args.moe_comm_aux_loss_coeff > 0:
        l_comm_aux = 0
        if args.seq_aux:
            if P_devi is None:
                if Pi is None:
                    scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)
                    Pi = scores_for_seq_aux.mean(dim=1)

                P_devi = Pi.view(args.micro_batch_size, args.n_group, -1).sum(-1)  # [b, n_group]

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = (ge.view(args.micro_batch_size, seq_length, args.n_group, -1).sum(-1) > 0).to(logits.dtype).sum(dim=1)
            ge.div_(seq_length * args.topk_group / args.n_group)

            l_comm_aux = (ge * P_devi).sum(dim=1).mean() * args.moe_comm_aux_loss_coeff

        else:
            if P_devi is None:
                if Pi is None:
                    Pi = scores_for_aux.mean(0)

                P_devi = Pi.view(args.n_group, -1).sum(-1)

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = rearrange(ge, 'b s (ng gs) -> (b s) ng gs', ng=args.n_group, gs=args.num_experts // args.n_group)
            ge = (ge.sum(dim=-1) > 0).to(logits.dtype).mean(0).div(args.topk_group / args.n_group)

            l_comm_aux = (ge * P_devi).sum() * args.moe_comm_aux_loss_coeff

        self.l_comm_aux = l_comm_aux
        l_aux += l_comm_aux

    self.l_aux = l_aux

    return topk_weight, topk_idx


def pai_megatron_aux_loss(self, logits: torch.Tensor):
    routing_weights = torch.softmax(logits, dim=1, dtype=torch.float32).type_as(logits)
    scores, indices = torch.topk(routing_weights, k=self.topk, dim=-1)

    # Apply load balancing loss
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    scores = self.apply_load_balancing_loss(probs, indices, activation=scores)
    args = get_args()
    global_indices = indices
    if args.moe_token_dispatcher_type == "allgather":
        if args.moe_permutation_async_comm and (self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1)):
            from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
            with torch.no_grad():
                global_indices = gather_from_sequence_parallel_region_to_moe_async(indices)
    return scores, global_indices


def topk_router_routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
    """
    logits = logits.view(-1, self.config.num_moe_experts)
    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    if (
        self.config.tensor_model_parallel_size > 1
        and self.config.moe_token_dispatcher_type == "alltoall"
    ):
        # Gather the logits from the TP region
        logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, indices = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, indices = self.aux_loss_load_balancing(logits)
    # add softmax_topk for softmax before topk that difference form routing_type is none
    elif self.routing_type == "softmax_topk":
        logits_ = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores, indices = torch.topk(logits_, k=self.topk, dim=1)
    elif self.routing_type == "pai_megatron_aux_loss":
        scores, indices = pai_megatron_aux_loss(self, logits)
    elif self.routing_type == "group_limited_greedy":
        scores, indices = group_limited_greedy_topKgating(self, logits)
    elif self.routing_type == "none":
        # A naive top-k routing without load balancing
        scores, indices, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

    return scores, indices


def topk_router_forward(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scores and indices.
    """
    args = get_args()
    self.hidden = input.shape[-1]

    # add input_jitter to distinguish whether to use
    if args.input_jitter:
        input = self.apply_input_jitter(input)
    logits = self.gating(input)
    logits = logits.view(-1, self.config.num_moe_experts)

    scores, indices = self.routing(logits)

    return scores, indices


def aux_loss_load_balancing(self, logits: torch.Tensor):
    """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
    """
    probs, indices, tokens_per_expert = topk_softmax_with_capacity(
        logits,
        self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        drop_policy=self.config.moe_token_drop_policy,
    )

    # Apply load balancing loss
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
    return probs, indices


def apply_load_balancing_loss(
    self,
    probs: torch.Tensor,
    num_local_tokens_per_expert: torch.Tensor,
    activation: torch.Tensor,
):
    """Applies auxiliary loss to the MoE layer.

    Args:
        probs (torch.Tensor): The probabilities output by the MoE layer.
        num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
        activation (torch.Tensor): The activation tensor to attach the gradient function to.

    Returns:
        torch.Tensor: The activation tensor with the attached gradient function.
    """
    moe_aux_loss_coeff = (
            self.config.moe_aux_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()
    )
    aux_loss = switch_load_balancing_loss_func(
        probs, num_local_tokens_per_expert, self.topk, moe_aux_loss_coeff
    )
    save_to_aux_losses_tracker(
        "load_balancing_loss",
        aux_loss / moe_aux_loss_coeff,
        self.layer_number,
        self.config.num_layers,
    )
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)

    return activation

