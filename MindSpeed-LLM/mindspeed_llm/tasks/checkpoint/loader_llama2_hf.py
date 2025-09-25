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

import json
import os
import sys
import types
import torch
import transformers
from tqdm import tqdm


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                            'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    group.add_argument("--w-pack", type=bool,
                       help='True is w_pack weight for llm',
                       default=False)
    parser.add_argument('--add-qkv-bias', action='store_true',
                        help='Add bias for attention qkv',
                        default=False)
    parser.add_argument('--add-dense-bias', action='store_true',
                        help='Add bias for attention dense',
                        default=False)
    parser.add_argument('--params-dtype', type=str,
                        help='Set weight dtype',
                        default='fp16')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=1,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    if major < 4 or minor < 31:
        raise ValueError("the version transformers should greater or equal 4.31")


def load_args_from_checkpoint(args):
    # Read Llama args.
    llama_args_path = os.path.join(args.load, "config.json")
    with open(llama_args_path) as f:
        llama_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = 4096
    args.hidden_size = llama_args["hidden_size"]
    args.num_attention_heads = llama_args["num_attention_heads"]
    args.num_layers = llama_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = llama_args["rms_norm_eps"]
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = True
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.tokenizer_type = "Llama2Tokenizer"
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = not llama_args.get("tie_word_embeddings", False)
    args.vocab_size = llama_args["vocab_size"]
    args.padded_vocab_size = llama_args["vocab_size"]
    args.llama = llama_args
    args.ffn_hidden_size = llama_args["intermediate_size"]
    args.gradient_accumulation_fusion = False
    if args.add_dense_bias:
        args.skip_bias_add = False

    if "num_key_value_heads" in llama_args \
            and llama_args["num_attention_heads"] != llama_args["num_key_value_heads"] \
            and llama_args["num_key_value_heads"] != 1:
        args.group_query_attention = True
        args.num_query_groups = llama_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    nh = args.num_attention_heads
    ng = (args.num_query_groups if args.group_query_attention \
              else args.num_attention_heads)
    dim = args.kv_channels
    if not nh % ng == 0:
        raise ValueError("nh % ng should equal 0")

    if args.w_pack:
        w_pack = hf_attn.W_pack.weight
        wq, wk, wv = w_pack.chunk(3, dim=0)
        attn.query_key_value.weight.data.copy_(torch.cat([
            wq.reshape((ng, dim * nh // ng, -1)),
            wk.reshape((ng, dim, -1)),
            wv.reshape((ng, dim, -1)),
        ], dim=1).reshape((-1, args.hidden_size)))

    else:
        attn.query_key_value.weight.data.copy_(torch.cat([
            hf_attn.q_proj.weight.reshape((ng, dim * nh // ng, -1)),
            hf_attn.k_proj.weight.reshape((ng, dim, -1)),
            hf_attn.v_proj.weight.reshape((ng, dim, -1)),
        ], dim=1).reshape((-1, args.hidden_size)))

    if args.add_qkv_bias:
        attn.query_key_value.bias.data.copy_(torch.cat([
            hf_attn.q_proj.bias.reshape((ng, dim * nh // ng)),
            hf_attn.k_proj.bias.reshape((ng, dim)),
            hf_attn.v_proj.bias.reshape((ng, dim)),
        ], dim=1).reshape((-1)))

    if args.add_dense_bias:
        attn.dense.bias.data.copy_(hf_attn.o_proj.bias)

    attn.dense.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
        hf_mlp.gate_proj.weight,
        hf_mlp.up_proj.weight,
    ], dim=0))
    mlp.dense_4h_to_h.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(model_provider, args):
    '''Set model params.'''

    from transformers import AutoModelForCausalLM

    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(args.load, device_map="cpu", trust_remote_code=True)

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(model_provider, queue, args):
    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    from megatron.training.arguments import validate_args
    from mindspeed_llm.training.utils import parse_args
    from megatron.training.global_vars import set_global_variables
    from megatron.legacy.model import module
    from megatron.core import mpu
    from megatron.core.enums import ModelType

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.w_pack = args.w_pack
    margs.add_qkv_bias = args.add_qkv_bias
    margs.add_dense_bias = args.add_dense_bias
    margs.tokenizer_model = args.tokenizer_model
    margs.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    if args.params_dtype == 'bf16':
        margs.bf16 = True
    elif args.params_dtype == 'fp16':
        margs.fp16 = True
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    if not args.model_type == 'GPT':
        raise ValueError("Llama-2 is a GPT model.")
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.spec = args.spec
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    if args.loader in ['loader_bloom_hf', 'bloom_hf']:
        md.norm_has_bias = True
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = margs.vocab_size  # skips padding in saver
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.embed_layernorm = margs.embed_layernorm

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(model_provider, margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    }

    # bloom
    if hasattr(model.language_model.embedding.word_embeddings, 'norm'):
        message["word embeddings norm_w"] = model.language_model.embedding.word_embeddings.norm.weight.data
        message["word embeddings norm_b"] = model.language_model.embedding.word_embeddings.norm.bias.data

    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    else:
        if hasattr(model.language_model.embedding, 'position_embeddings'):
            raise ValueError("model should have position_embeddings")

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.encoder.layers[layer_num]
        message["input norm weight"] = layer.input_norm.weight.data
        message["post norm weight"] = layer.post_attention_norm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
            message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

        if md.norm_has_bias:
            message["input norm bias"] = layer.input_norm.bias.data
            message["post norm bias"] = layer.post_attention_norm.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.query_key_value.weight.data)
        dense_weight.append(layer.self_attention.dense.weight.data)
        mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
        mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)
        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)
        if args.add_qkv_bias:
            message["qkv bias"] = layer.self_attention.query_key_value.bias.data
        if args.add_dense_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
        # Handle gated linear units.
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.language_model.encoder.final_norm.weight.data,
    }
    if md.norm_has_bias:
        message["bias"] = model.language_model.encoder.final_norm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.language_model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(model_provider, queue, args):
    try:
        _load_checkpoint(model_provider, queue, args)
    except:
        queue.put("exit")
        raise

