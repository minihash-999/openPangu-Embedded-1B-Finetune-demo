import unittest
import sys
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import judge_expression


class CovertCkptToHuggingfaceArgs:
    model_type = "GPT"
    loader = "megatron"
    saver = "megatron"
    save_dir = "/data/llama-2-7b-hf"
    lora_dir = "/data/llama2-7B-lora-ckpt"
    load_dir = "/data/llama2-7B-tp8-pp1"


class TestConvertCkptFromHuggingface:

    def test_combine_lora_weights_to_huggingface(self):
        """
        Test whether the weight to be converted as we want in `--lora-load`. We will check the combine weight
        in huggingface equals loraB @ loraA * rate + base in megatron.
        """
        args = CovertCkptToHuggingfaceArgs()
        rate = 2
        hidden_layer = 4096
        num_head = 32
        tp = 8
        dk = 128
        
        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--save-model-type", "save_huggingface_llama",
            "--load-dir", args.load_dir,
            "--lora-load", args.lora_dir,
            "--target-tensor-parallel-size", "1",
            "--target-pipeline-parallel-size", "1",
            "--save-dir", args.save_dir
        ]
        
        exit_code = subprocess.run(["python3", file_path] + arguments).returncode
        assert exit_code == 0
        output_dir = os.path.join(args.save_dir, "mg2hg")
        
        model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True, low_cpu_mem_usage=True)
        q_hf = model.state_dict()["model.layers.0.self_attn.q_proj.weight"]

        judge_expression(q_hf.size() == torch.Size([4096, 4096]))
        
        base_dir = os.path.join(args.load_dir, "iter_0000001")
        weight_base = torch.load(os.path.join(base_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_base_content = weight_base['model']['language_model']['encoder'] # extract commmon content
        base_qkv = weight_base_content['layers.0.self_attention.query_key_value.weight']

        lora_dir = os.path.join(args.lora_dir, "iter_0000010")
        weight_lora = torch.load(os.path.join(lora_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_lora_content = weight_lora['model']['language_model']['encoder'] # extract commmon content
        loraB_qkv = weight_lora_content['layers.0.self_attention.query_key_value.lora_B.default.weight']
        loraA_qkv = weight_lora_content['layers.0.self_attention.query_key_value.lora_A.default.weight']
        
        res_qkv = loraB_qkv.cpu().float() @ loraA_qkv.cpu().float() * rate + base_qkv

        gp1_q_mg = res_qkv.reshape(num_head // tp, 3, dk, hidden_layer)[:1, :1, :, :].reshape(dk, hidden_layer)
        gp1_q_hf = q_hf.reshape(num_head, dk, hidden_layer)[:1, :, :].reshape(dk, hidden_layer)
        judge_expression(np.allclose(gp1_q_mg.cpu(), gp1_q_hf.cpu(), rtol=0.001, atol=0.001))