"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""
import os
import hashlib
import logging
import re
import json
import glob
import sys
import subprocess
import pytest
import torch
import torch_npu
import megatron.core.parallel_state as mpu


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def judge_expression(expression):
    if not expression:
        raise AssertionError


def compare_state_dicts(state_dict1, state_dict2):
    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1.keys():
        value1 = state_dict1[key]
        value2 = state_dict2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.equal(value1, value2):
                print(f"Difference found in key: {key}")
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_state_dicts(value1, value2):
                return False
        else:
            pass

    return True


def weight_compare(dir_1, dir_2, suffix="pt", use_md5=False):
    models_path = glob.glob(os.path.join(dir_1, '**', f'*.{suffix}'), recursive=True)
    if not models_path:
        print(f"Can't find any weight files in {dir_1}.")
        return False
    for path_1 in models_path:
        path_1 = os.path.normpath(path_1)
        path_2 = path_1.replace(os.path.normpath(dir_1), os.path.normpath(dir_2))
        if use_md5:
            are_equal = (get_md5sum(path_1) == get_md5sum(path_2))
        else:
            state_dict1 = torch.load(path_1)
            state_dict2 = torch.load(path_2)
            are_equal = compare_state_dicts(state_dict1, state_dict2)
        if not are_equal:
            return False

    return True


def compare_file_md5_same(file1, file2):
    return get_md5sum(file1) == get_md5sum(file2)


def get_md5sum(fpath):
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"{fpath} is not a file or not exists !")
    md5sum = hashlib.md5()
    with open(fpath, 'rb') as f:
        md5sum.update(f.read())
        return md5sum.hexdigest()


@pytest.fixture
def build_args(request, monkeypatch):
    params = request.getfixturevalue("params")
    argv = [sys.argv[0]]
    for k, v in params.items():
        if v is None:
            argv.append(f"--{k}")
        elif isinstance(v, list):
            argv.extend([f"--{k}"] + [str(value) for value in v])
        else:
            argv.extend([f"--{k}", str(v)])
    monkeypatch.setattr(sys, "argv", argv)


def create_testconfig(path: str, cmd: bool = False):
    with open(path) as f:
        raw_data = json.load(f)
    
    res = {k: [tuple(s.values()) if len(s) > 1 else tuple(s.values())[0] for s in v] for k, v in raw_data.items()}

    if not cmd:
        return res

    def __dict2cmdlist(param_value):
        cmdlsts = []
        cmdlst = []
        for target in param_value:
            for k, v in target.items():
                cmdlst.append(f"--{k}")
                if v is not None:
                    cmdlst.extend(v.split())
        cmdlsts.extend(cmdlst)
        return cmdlsts

    res_cmd = {key: __dict2cmdlist(value) for key, value in res.items()}
    return res_cmd


class ListHandler(logging.Handler):
    # Extract inference log, the regular expression is universal.
    # Just pass the pattern you want.
    def __init__(self, pattern):
        super().__init__()
        self.log_capture = []
        self.pattern = pattern
    
    def emit(self, record):
        log_entry = self.format(record)
        if re.search(self.pattern, log_entry, re.DOTALL):
            self.log_capture.append(log_entry)


def setup_logger(pattern):
    # Set the logger and the handler.
    # Different tasks will not form interference, feel relieved to use. 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = ListHandler(pattern)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return handler, handler.log_capture


def run_cmd(cmd_strlist):
    return subprocess.run(cmd_strlist).returncode
