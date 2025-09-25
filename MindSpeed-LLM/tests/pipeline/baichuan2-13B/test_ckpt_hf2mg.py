import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import mindspeed_llm
from convert_ckpt import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (build_args, create_testconfig, run_cmd,
                                    weight_compare)

PATTERN = r"acc = (.*)"


def init_process_group(backend='nccl'):
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group(backend)

init_process_group()


class TestCovertBaichuan2CkptHf2mg():
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_baichuan2_hf2mg_tp8pp1"])
    def test_baichuan2_hf2mg_tp8pp1(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        BASE_DIR = Path(__file__).absolute().parents[3]
        CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")
        arguments = [f"--{k}={v}" for k, v in params.items()]
        exit_code = run_cmd(["python3", CKPT_PYPATH] + arguments)
        assert exit_code == 0
        base_dir = '/data/pipe/baichuan2-13b-tp8pp1-legacy-base'
        save_dir = '/data/pipe/Baichuan2-legacy-test/'
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)