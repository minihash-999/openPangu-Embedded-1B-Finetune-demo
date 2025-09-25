import unittest
import sys
import os
import subprocess
from pathlib import Path
import torch

from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import judge_expression


class CovertCkptFromMegatronArgs:
    model_type = "GPT"
    loader = "megatron"
    saver = "megatron"
    target_tensor_parallel_size = "1"
    save_dir = "/data/llama2-7B-tp1-pp1"
    load_dir = "/data/llama2-7B-tp8-pp1"


class TestConvertCkptFromMegatron:

    def test_combine_lora_weights_from_megatron(self):
        args = CovertCkptFromMegatronArgs()

        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--save-model-type", "megatron",
            "--load-dir", args.load_dir,
            "--target-tensor-parallel-size", "1",
            "--target-pipeline-parallel-size", "1",
            "--save-dir", args.save_dir
        ]

        exit_code = subprocess.run(["python3", file_path] + arguments).returncode
        assert exit_code == 0
        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([32000, 4096]))
        judge_expression(weight_common_content['encoder']['final_norm.weight'].size() == torch.Size([4096]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([12288, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([22016, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 11008]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))

        judge_expression(weight_common_content['output_layer']['weight'].size() == torch.Size([32000, 4096]))