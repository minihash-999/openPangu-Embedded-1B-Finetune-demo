# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
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
"""Tests of Evaluation"""

import sys
import os
import shutil
from pathlib import Path
import logging
import re
import math
import pytest
from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import create_testconfig, weight_compare, run_cmd


BASE_DIR = Path(__file__).absolute().parents[3]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_mixtral_hf2mcore_tp2pp2ep2dypp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mixtral_hf2mcore_tp2pp2ep2dypp'])
        assert exit_code == 0
        base_dir = '/data/wttest/base/hf2mc_mixtral_tp2pp2ep2dypp'
        save_dir = self.test_config['test_mixtral_hf2mcore_tp2pp2ep2dypp'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)

    def test_mixtral_mcore2hf_tp1pp4ep2vpp2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mixtral_mcore2hf_tp1pp4ep2vpp2'])
        assert exit_code == 0
        base_dir = '/data/Mixtral-8x7B-v0.1/base_mg2hf_tp1pp4ep2vpp2'
        save_dir = os.path.join(self.test_config['test_mixtral_mcore2hf_tp1pp4ep2vpp2'][0]['save-dir'], 'mg2hf')
        assert weight_compare(base_dir, save_dir, suffix="safetensors", use_md5=True)
        shutil.rmtree(save_dir)

    def test_deepseek2_hf2mcore_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_hf2mcore_tp1pp4ep8'])
        assert exit_code == 0
        base_dir = '/data/ci/deepseek2/mg_base/deepseek2-l8-t1p4e8-gemm'
        save_dir = self.test_config['test_deepseek2_hf2mcore_tp1pp4ep8'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)

    def test_deepseek2_mcore2hf_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_mcore2hf_tp1pp4ep8'])
        assert exit_code == 0
        base_dir = '/data/ci/deepseek2/hf_base/deepseek2_hf_base'
        save_dir = os.path.join(self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][0]['save-dir'], 'mg2hf')
        assert weight_compare(base_dir, save_dir, suffix="safetensors", use_md5=True)
        shutil.rmtree(save_dir)

    def test_deepseek2_lite_hf2mcore_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_hf2mcore_tp1pp1ep8'])
        assert exit_code == 0
        base_dir = '/data/ci/deepseek2_lite/mg_base/deepseek2_lite_l3_t1p1e8'
        save_dir = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)

    def test_deepseek2_lite_mcore2hf_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_mcore2hf_tp1pp1ep8'])
        assert exit_code == 0
        base_dir = '/data/ci/deepseek2_lite/hf_base/deepseek2_lite_hf_base'
        save_dir = os.path.join(self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][0]['save-dir'], 'mg2hf')
        assert weight_compare(base_dir, save_dir, suffix="safetensors", use_md5=True)

    def test_gemma2_hf2mcore_tp8pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_gemma2_hf2mcore_tp8pp1'])
        assert exit_code == 0
        base_dir = '/data/gemma2-9b-mg-tp8pp1-mcore-base/'
        save_dir = self.test_config['test_gemma2_hf2mcore_tp8pp1'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)

    def test_qwen2_moe_hf2mcore_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp1pp2ep2'])
        assert exit_code == 0
        base_dir = "/data/ci/qwen2_moe/mg_base/qwen2_moe_l2_t1p2e2"
        save_dir = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)

    def test_qwen2_moe_mcore2hf_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_mcore2hf_tp1pp2ep2'])
        assert exit_code == 0
        base_dir = '/data/ci/qwen2_moe/hf_base/qwen2_moe_hf_base'
        save_dir = os.path.join(self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]['save-dir'], 'mg2hf')
        assert weight_compare(base_dir, save_dir, suffix="safetensors", use_md5=True)
        shutil.rmtree(save_dir)
