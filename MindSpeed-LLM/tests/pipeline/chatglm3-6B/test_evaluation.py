import sys
import os
from pathlib import Path
import logging
import re
import pytest
import math
import torch.distributed as dist
from evaluation import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger
from tests.ut.evaluation.test_evaluate import acquire_score


PATTERN = r"acc = (.*)"


class TestEvaluate(DistributedTest):
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_chatglm3_6B_mmlu_evaluate"])
    def test_chatglm_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        
        if dist.get_rank() == 0:
            print("=================== chatglm3_6B MMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.6257, abs_tol=1e-2), expected_score
