import unittest
import sys
import os
import subprocess
from pathlib import Path
import torch

from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import judge_expression
from tests.test_tools.utils import weight_compare


class CovertCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "8"
    load_dir = "/data/llama-2-7b-hf"
    save_dir = "/data/llama2-7B-tp8-pp1"
    tokenizer_model = "/data/llama-2-7b-hf/tokenizer.model"


class CovertVppCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/data/llama-2-7b-hf"
    save_dir = "/data/llama2-7B-tp2-pp4-vpp4"
    tokenizer_model = "/data/llama-2-7b-hf/tokenizer.model"
    num_layers_per_virtual_pipeline_stage = "2"


class CovertDynamicCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/data/llama-2-7b-hf/"
    save_dir = "/data/llama-2-7b-mg-tp2-pp4-dynamic-test/"
    base_dir = "/data/llama-2-7b-mg-tp2-pp4-dynamic-base/"
    tokenizer_model = "/data/llama-2-7b-hf/tokenizer.model"
    num_layer_list = '6,8,8,10'


class CovertMCoreDynamicCkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/data/llama-2-7b-hf/"
    save_dir = "/data/llama-2-7b-mg-tp2-pp4-mcore-dynamic-test/"
    base_dir = "/data/llama-2-7b-mg-tp2-pp4-mcore-dynamic/"
    tokenizer_model = "/data/llama-2-7b-hf/tokenizer.model"
    num_layer_list = "6,8,8,10"


class CovertVPPCkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "2"
    num_vpp_layer = "1"
    load_dir = "/data/llama-3-8b-hf/"
    save_dir = "/data/llama-3-8b-hf-test/"
    base_dir = "/data/llama-3-8b-tp2-pp2-vpp1/"
    tokenizer_model = "/data/llama-3-8b-hf/tokenizer.json"


class CovertMCoreVPPCkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/data/llama-2-7b-hf/"
    save_dir = "/data/llama-2-7b-mg-tp2-pp4-mcore-vpp2-test/"
    base_dir = "/data/llama-2-7b-mg-tp2-pp4-mcore-vpp2/"
    tokenizer_model = "/data/llama-2-7b-hf/tokenizer.model"
    num_layers_per_virtual_pipeline_stage = "2"


class CovertLegacyChatGLM3CkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    model_type_hf = "chatglm3"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/data/chatglm3-6b-base-hf/"
    save_dir = "/data/chatglm3-6b-base-mg-tp2pp4-legacy-test/"
    base_dir = "/data/chatglm3-6b-base-mg-tp2pp4-legacy-base/"
    tokenizer_model = "/data/chatglm3-6b-base-hf/tokenizer.model"


class CovertMCoreChatGLM3CkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    target_tensor_parallel_size = "1"
    target_pipeline_parallel_size = "2"
    load_dir = "/data/chatglm3-6b-base-hf/"
    save_dir = "/data/chatglm3-6b-base-mg-tp1pp2-mcore-test/"
    base_dir = "/data/chatglm3-6b-base-mg-tp1pp2-mcore-base/"
    tokenizer_model = "/data/chatglm3-6b-base-hf/tokenizer.model"


class CovertMCoreQwen2CkptFromHuggingfaceArgs:
    model_type = "GPT"
    load_model_type = "hf"
    save_model_type = "mg"
    target_tensor_parallel_size = "1"
    target_pipeline_parallel_size = "1"
    load_dir = "/data/Qwen2-1.5B/"
    save_dir = "/data/qwen2-1.5b-base-mg-tp1pp1-mcore-test/"
    base_dir = "/data/qwen2-1.5b-hf-v0.1-tp1-pp1/"
    tokenizer_model = "/data/Qwen2-1.5B/tokenizer.model"



class TestConvertCkptFromHuggingface:

    def test_file_exsit(self):
        args = CovertCkptFromHuggingfaceArgs()
        """
        Test if the file in the `--load-dir` exsit, including `.bin`, `.json`...
        """
        bin_file = 0
        for file_name in os.listdir(args.load_dir):
            if file_name.endswith(".bin"):
                bin_file += 1
        judge_expression(bin_file == 2)
        judge_expression(os.path.exists(os.path.join(args.load_dir, "pytorch_model.bin.index.json")))

    def test_convert_mcore_dynamic_weights_form_huggingface(self):
        args = CovertMCoreDynamicCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model,
            "--use-mcore-models",
            "--model-type-hf", "llama2",
            "--num-layer-list", args.num_layer_list,

        ]
        exit_code = subprocess.run(["python3", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_mcore_dynamic_weights_form_huggingface failed!"

    def test_convert_mcore_vpp_weights_form_huggingface(self):
        args = CovertVPPCkptFromHuggingfaceArgs()
        """
        Test case for pp2vpp.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--model-type-hf", "llama2",
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--num-layers-per-virtual-pipeline-stage", args.num_vpp_layer,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model,
            "--use-mcore-models",
        ]
        exit_code = subprocess.run(["python3", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_mcore_vpp_weights_form_huggingface failed!"


    def test_convert_mcore_vpp_weights_form_huggingface(self):
        args = CovertMCoreVPPCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model,
            "--use-mcore-models",
            "--model-type-hf", "llama2",
            "--num-layers-per-virtual-pipeline-stage", args.num_layers_per_virtual_pipeline_stage
        ]
        exit_code = subprocess.run(["python3", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_mcore_vpp_weights_form_huggingface failed!"

    def test_convert_dynamic_weights_form_huggingface(self):
        args = CovertDynamicCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--num-layer-list", args.num_layer_list,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model
        ]
        exit_code = subprocess.run(["python", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_legacy_dynamic_weights_form_huggingface failed!"

    def test_convert_mcore_pack_gqa_form_huggingface(self):
        args = CovertMCoreChatGLM3CkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--use-mcore-models",
            "--add-qkv-bias",
            "--model-type-hf", "chatglm3",
            "--tokenizer-model", args.tokenizer_model
        ]
        exit_code = subprocess.run(["python", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_mcore_pack_gqa_form_huggingface failed!"

    def test_convert_legacy_pack_gqa_form_huggingface(self):
        args = CovertLegacyChatGLM3CkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--model-type", args.model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--add-qkv-bias",
            "--tokenizer-model", args.tokenizer_model,
            "--model-type-hf", args.model_type_hf
        ]
        exit_code = subprocess.run(["python", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_legacy_pack_gqa_form_huggingface failed!"

    def test_convert_weights_form_huggingface(self):
        args = CovertCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model
        ]
        subprocess.run(["python", file_path] + arguments)
        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([4000, 4096]))
        judge_expression(weight_common_content['encoder']['final_norm.weight'].size() == torch.Size([4096]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([1536, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 512]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([2752, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 1376]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))

        judge_expression(weight_common_content['output_layer']['weight'].size() == torch.Size([4000, 4096]))

    def test_convert_weights_vpp_form_huggingface(self):
        args = CovertVppCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model,
            "--num-layers-per-virtual-pipeline-stage", args.num_layers_per_virtual_pipeline_stage
        ]
        subprocess.run(["python", file_path] + arguments)
        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00_000/model_optim_rng.pt"))
        weight_common_content = weight_content['model0']['language_model']  # extract commmon content

        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size) * int(args.target_pipeline_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([16000, 4096]))

        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([6144, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 2048]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([11008, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 5504]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))

    def test_convert_weights_qwen2_mcore_form_huggingface(self):
        args = CovertMCoreQwen2CkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name,
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--load-model-type", args.load_model_type,
            "--save-model-type", args.save_model_type,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--use-mcore-models",
            "--add-qkv-bias",
            "--model-type-hf", "llama2",
            "--params-dtype", "bf16",
            "--tokenizer-model", args.tokenizer_model,
        ]
        exit_code = subprocess.run(["python", file_path] + arguments).returncode
        assert exit_code == 0 and weight_compare(args.base_dir, args.save_dir), "convert_weights_qwen2_mcore_form_huggingface failed!"
