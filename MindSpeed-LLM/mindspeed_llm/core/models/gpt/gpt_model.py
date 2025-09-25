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

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args


from mindspeed_llm.core.tensor_parallel.layers import SegmentedColumnParallelLinear


def gpt_model_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        post_layer_norm = kwargs.pop('post_layer_norm', True)
        fn(self, *args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs['config']
        if self.post_process and get_args().output_layer_slice_num > 1:
            self.output_layer = SegmentedColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
        if not post_layer_norm:
            self.decoder.post_layer_norm = False

    return wrapper


def gpt_model_forward(self, input_ids: Tensor,
                      position_ids: Tensor, attention_mask: Tensor,
                      decoder_input: Tensor = None,
                      labels: Tensor = None,
                      inference_params: InferenceParams = None,
                      packed_seq_params: PackedSeqParams = None,
                      extra_block_kwargs: dict = None,
                      tokentype_ids=None) -> Tensor:
    """
    Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units
    add output_multiplier_scale to scale logits
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
    args = get_args()
    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        if args.scale_emb is not None:
            decoder_input = decoder_input * args.scale_emb
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    if self.position_embedding_type == 'rope':
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_params, self.decoder, decoder_input, self.config
        )
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

    # Run decoder.
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if args.dim_model_base is not None:
        hidden_states = hidden_states / (args.hidden_size / args.dim_model_base)
    logits, _ = self.output_layer(hidden_states, weight=output_weight)
    # new add to scale logits
    if args.output_multiplier_scale:
        logits = logits * args.output_multiplier_scale

    if args.output_logit_softcapping:
        logits = logits / args.output_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * args.output_logit_softcapping

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    if args.is_instruction_dataset:
        labels = labels[:, 1:].contiguous()
        logits = logits[:-1, :, :].contiguous()

    loss = self.compute_language_model_loss(labels, logits)

    return loss